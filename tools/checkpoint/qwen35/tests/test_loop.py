"""Tests for Loop Transformer checkpoint conversion logic."""

import copy
import os
import sys
import tempfile

import pytest
import torch
import yaml

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwen35.config import Config
from qwen35.converter import MoEConverter
from qwen35.gdn import is_gdn_layer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_BASE_CFG = {
    "num_layers": 40,
    "hidden_size": 2560,
    "ffn_hidden_size": 1024,
    "num_attention_heads": 20,
    "num_query_groups": 4,
    "group_query_attention": True,
    "kv_channels": 256,
    "attention_output_gate": True,
    "untie_embeddings_and_output_weights": True,
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "expert_model_parallel_size": 1,
    "num_experts": 8,
    "moe_router_topk": 2,
    "moe_ffn_hidden_size": 1024,
    "moe_shared_expert_intermediate_size": 2560,
    "moe_grouped_gemm": True,
    "moe_use_shared_expert_gate": True,
    "expert_tensor_parallel_size": 1,
    "linear_attention_freq": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "no_enable_vision": True,
    "mtp_num_layers": 0,
}

_LOOP_FIELDS = {
    "loop_start_layer": 10,
    "loop_end_layer": 30,
    "num_loop_iterations": 2,
    "loop_residual_scale": 0.5,
}


def _make_cfg(extra=None):
    """Write a temp YAML and return a Config."""
    d = {**_BASE_CFG}
    if extra:
        d.update(extra)
    fd, path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(d, f)
        return Config(path)
    finally:
        os.unlink(path)


def _make_cfg_expect_error(extra=None):
    """Write a temp YAML and return path (caller handles error)."""
    d = {**_BASE_CFG}
    if extra:
        d.update(extra)
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.dump(d, f)
    return path


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------
class TestLoopConfig:
    def test_is_loop_true(self):
        cfg = _make_cfg(_LOOP_FIELDS)
        assert cfg.is_loop is True

    def test_is_loop_false(self):
        cfg = _make_cfg()
        assert cfg.is_loop is False

    def test_hf_num_layers_loop(self):
        cfg = _make_cfg(_LOOP_FIELDS)
        # 10 prelude + 20*2 loop + 10 coda = 60
        assert cfg.hf_num_layers == 60

    def test_hf_num_layers_nonloop(self):
        cfg = _make_cfg()
        assert cfg.hf_num_layers == 40

    def test_validation_mismatched_start_end(self):
        path = _make_cfg_expect_error({"loop_start_layer": 10})
        try:
            with pytest.raises(ValueError, match="both be None or both be set"):
                Config(path)
        finally:
            os.unlink(path)

    def test_validation_bad_range(self):
        path = _make_cfg_expect_error(
            {
                "loop_start_layer": 30,
                "loop_end_layer": 10,
                "num_loop_iterations": 2,
            }
        )
        try:
            with pytest.raises(ValueError, match="loop_start_layer < loop_end_layer"):
                Config(path)
        finally:
            os.unlink(path)

    def test_validation_iterations_too_small(self):
        path = _make_cfg_expect_error(
            {
                "loop_start_layer": 10,
                "loop_end_layer": 30,
                "num_loop_iterations": 1,
            }
        )
        try:
            with pytest.raises(ValueError, match="num_loop_iterations must be >= 2"):
                Config(path)
        finally:
            os.unlink(path)

    def test_validation_negative_scale(self):
        path = _make_cfg_expect_error(
            {
                "loop_start_layer": 10,
                "loop_end_layer": 30,
                "num_loop_iterations": 2,
                "loop_residual_scale": -0.5,
            }
        )
        try:
            with pytest.raises(ValueError, match="loop_residual_scale must be positive"):
                Config(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Execution plan tests
# ---------------------------------------------------------------------------
class TestExecutionPlan:
    def test_plan_length(self):
        cfg = _make_cfg(_LOOP_FIELDS)
        conv = MoEConverter(cfg)
        plan = conv._build_loop_execution_plan()
        assert len(plan) == 60

    def test_prelude_layers(self):
        cfg = _make_cfg(_LOOP_FIELDS)
        conv = MoEConverter(cfg)
        plan = conv._build_loop_execution_plan()
        for hf, phys, in_loop, iteration in plan[:10]:
            assert hf == phys
            assert in_loop is False
            assert iteration == -1

    def test_loop_layers(self):
        cfg = _make_cfg(_LOOP_FIELDS)
        conv = MoEConverter(cfg)
        plan = conv._build_loop_execution_plan()
        loop_entries = plan[10:50]
        assert len(loop_entries) == 40
        # First iteration: phys 10-29
        for i, (hf, phys, in_loop, iteration) in enumerate(loop_entries[:20]):
            assert hf == 10 + i
            assert phys == 10 + i
            assert in_loop is True
            assert iteration == 0
        # Second iteration: phys 10-29 again
        for i, (hf, phys, in_loop, iteration) in enumerate(loop_entries[20:]):
            assert hf == 30 + i
            assert phys == 10 + i
            assert in_loop is True
            assert iteration == 1

    def test_coda_layers(self):
        cfg = _make_cfg(_LOOP_FIELDS)
        conv = MoEConverter(cfg)
        plan = conv._build_loop_execution_plan()
        coda = plan[50:]
        assert len(coda) == 10
        for i, (hf, phys, in_loop, iteration) in enumerate(coda):
            assert hf == 50 + i
            assert phys == 30 + i
            assert in_loop is False


# ---------------------------------------------------------------------------
# Scale absorption tests
# ---------------------------------------------------------------------------
def _build_fake_meg_sd(cfg):
    """Build a minimal Megatron-like state dict for conversion testing."""
    sd = {}
    freq = cfg.linear_attention_freq
    h = cfg.hidden_size
    sd["language_model.embedding.word_embeddings.weight"] = torch.randn(1000, h)

    for layer_idx in range(cfg.num_layers):
        pfx = f"language_model.decoder.layers.{layer_idx}"

        if is_gdn_layer(layer_idx, freq):
            # GDN layer: in_proj uses linear attention dims
            num_qk_heads = cfg.linear_num_key_heads
            num_v_heads = cfg.linear_num_value_heads
            qk_head_dim = cfg.linear_key_head_dim
            v_head_dim = cfg.linear_value_head_dim
            v_per_group = num_v_heads // num_qk_heads
            q_size = num_qk_heads * qk_head_dim
            k_size = num_qk_heads * qk_head_dim
            v_size = num_qk_heads * v_per_group * v_head_dim
            z_size = num_qk_heads * v_per_group * v_head_dim
            b_size = num_qk_heads * v_per_group
            a_size = num_qk_heads * v_per_group
            in_proj_size = q_size + k_size + v_size + z_size + b_size + a_size
            sd[f"{pfx}.self_attention.in_proj.layer_norm_weight"] = torch.randn(h)
            sd[f"{pfx}.self_attention.in_proj.weight"] = torch.randn(in_proj_size, h)
            sd[f"{pfx}.self_attention.conv1d.weight"] = torch.randn(h, 1, 4)
            sd[f"{pfx}.self_attention.out_proj.weight"] = torch.randn(h, h)
            sd[f"{pfx}.self_attention.out_norm.weight"] = torch.randn(h)
            sd[f"{pfx}.self_attention.A_log"] = torch.randn(num_qk_heads)
            sd[f"{pfx}.self_attention.dt_bias"] = torch.randn(num_qk_heads)
        else:
            # SDPA layer
            n_heads = cfg.num_attention_heads
            n_groups = cfg.num_query_groups
            kv_channels = cfg.kv_channels
            heads_per_group = n_heads // n_groups
            if cfg.attention_output_gate:
                total_hpg = 2 * heads_per_group + 2
            else:
                total_hpg = heads_per_group + 2
            qkv_size = n_groups * total_hpg * kv_channels
            sd[f"{pfx}.self_attention.linear_qkv.layer_norm_weight"] = torch.randn(h)
            sd[f"{pfx}.self_attention.linear_qkv.weight"] = torch.randn(qkv_size, h)
            sd[f"{pfx}.self_attention.linear_proj.weight"] = torch.randn(h, h)
            # q_norm: with output gate, q and z are both num_heads*kv_channels
            if cfg.attention_output_gate:
                q_norm_size = 2 * n_heads * kv_channels
            else:
                q_norm_size = n_heads * kv_channels
            sd[f"{pfx}.self_attention.q_layernorm.weight"] = torch.randn(q_norm_size)
            sd[f"{pfx}.self_attention.k_layernorm.weight"] = torch.randn(n_groups * kv_channels)

        # MLP: MoE with stacked experts
        sd[f"{pfx}.pre_mlp_layernorm.weight"] = torch.randn(h)
        sd[f"{pfx}.mlp.router.weight"] = torch.randn(cfg.num_experts, h)
        moe_ffn = cfg.moe_ffn_hidden_size
        for e in range(cfg.num_experts):
            sd[f"{pfx}.mlp.experts.linear_fc1.weight{e}"] = torch.randn(2 * moe_ffn, h)
            sd[f"{pfx}.mlp.experts.linear_fc2.weight{e}"] = torch.randn(h, moe_ffn)
        se_ffn = cfg.moe_shared_expert_intermediate_size
        sd[f"{pfx}.mlp.shared_experts.linear_fc1.weight"] = torch.randn(2 * se_ffn, h)
        sd[f"{pfx}.mlp.shared_experts.linear_fc2.weight"] = torch.randn(h, se_ffn)
        sd[f"{pfx}.mlp.shared_experts.gate_weight"] = torch.randn(1, h)

    sd["language_model.decoder.final_layernorm.weight"] = torch.randn(h)
    sd["language_model.output_layer.weight"] = torch.randn(1000, h)
    return sd


class TestScaleAbsorption:
    def test_loop_model_has_60_layers(self):
        cfg = _make_cfg(_LOOP_FIELDS)
        conv = MoEConverter(cfg)
        meg_sd = _build_fake_meg_sd(cfg)
        hf_sd = conv.convert_to_hf(meg_sd)
        # Should have keys for layers 0..59
        layer_keys = [k for k in hf_sd if k.startswith("model.language_model.layers.")]
        layer_indices = set()
        for k in layer_keys:
            idx = int(k.split(".")[3])
            layer_indices.add(idx)
        assert max(layer_indices) == 59
        assert len(layer_indices) == 60

    def test_nonloop_model_has_40_layers(self):
        cfg = _make_cfg()
        conv = MoEConverter(cfg)
        meg_sd = _build_fake_meg_sd(cfg)
        hf_sd = conv.convert_to_hf(meg_sd)
        layer_keys = [k for k in hf_sd if k.startswith("model.language_model.layers.")]
        layer_indices = set()
        for k in layer_keys:
            idx = int(k.split(".")[3])
            layer_indices.add(idx)
        assert max(layer_indices) == 39
        assert len(layer_indices) == 40

    def test_scale_applied_to_loop_layers(self):
        """GDN out_proj in loop region should be scaled by 0.5."""
        cfg = _make_cfg(_LOOP_FIELDS)
        conv = MoEConverter(cfg)
        meg_sd = _build_fake_meg_sd(cfg)

        # Also create a scale=1.0 reference
        cfg_noscale = _make_cfg({**_LOOP_FIELDS, "loop_residual_scale": 1.0})
        conv_noscale = MoEConverter(cfg_noscale)

        hf_scaled = conv.convert_to_hf(copy.deepcopy(meg_sd))
        hf_noscale = conv_noscale.convert_to_hf(copy.deepcopy(meg_sd))

        # HF layer 10 is phys 10, in_loop=True.
        # is_gdn_layer(10, 4) => 10%4=2, 2!=3 => True => GDN
        hf_pfx = "model.language_model.layers.10"
        attn_key = f"{hf_pfx}.linear_attn.out_proj.weight"
        assert torch.allclose(hf_scaled[attn_key], hf_noscale[attn_key] * 0.5)

        # Check shared expert down_proj also scaled
        se_key = f"{hf_pfx}.mlp.shared_expert.down_proj.weight"
        assert torch.allclose(hf_scaled[se_key], hf_noscale[se_key] * 0.5)

    def test_scale_not_applied_to_prelude(self):
        """Prelude layers should NOT be scaled."""
        cfg = _make_cfg(_LOOP_FIELDS)
        conv = MoEConverter(cfg)
        meg_sd = _build_fake_meg_sd(cfg)

        cfg_noscale = _make_cfg({**_LOOP_FIELDS, "loop_residual_scale": 1.0})
        conv_noscale = MoEConverter(cfg_noscale)

        hf_scaled = conv.convert_to_hf(copy.deepcopy(meg_sd))
        hf_noscale = conv_noscale.convert_to_hf(copy.deepcopy(meg_sd))

        # HF layer 0 is prelude, not in loop
        # is_gdn_layer(0, 4) => 0%4=0, 0!=3 => True => GDN
        hf_pfx = "model.language_model.layers.0"
        attn_key = f"{hf_pfx}.linear_attn.out_proj.weight"
        assert torch.equal(hf_scaled[attn_key], hf_noscale[attn_key])

    def test_scale_not_applied_to_coda(self):
        """Coda layers should NOT be scaled."""
        cfg = _make_cfg(_LOOP_FIELDS)
        conv = MoEConverter(cfg)
        meg_sd = _build_fake_meg_sd(cfg)

        cfg_noscale = _make_cfg({**_LOOP_FIELDS, "loop_residual_scale": 1.0})
        conv_noscale = MoEConverter(cfg_noscale)

        hf_scaled = conv.convert_to_hf(copy.deepcopy(meg_sd))
        hf_noscale = conv_noscale.convert_to_hf(copy.deepcopy(meg_sd))

        # HF layer 50 is coda (phys 30), not in loop
        hf_pfx = "model.language_model.layers.50"
        se_key = f"{hf_pfx}.mlp.shared_expert.down_proj.weight"
        assert torch.equal(hf_scaled[se_key], hf_noscale[se_key])

    def test_sdpa_layer_scale(self):
        """SDPA layer o_proj in loop region should be scaled."""
        cfg = _make_cfg(_LOOP_FIELDS)
        conv = MoEConverter(cfg)
        meg_sd = _build_fake_meg_sd(cfg)

        cfg_noscale = _make_cfg({**_LOOP_FIELDS, "loop_residual_scale": 1.0})
        conv_noscale = MoEConverter(cfg_noscale)

        hf_scaled = conv.convert_to_hf(copy.deepcopy(meg_sd))
        hf_noscale = conv_noscale.convert_to_hf(copy.deepcopy(meg_sd))

        # phys 11: is_gdn_layer(11, 4) => 11%4=3, 3!=3 => False => SDPA
        # HF idx 11 maps to phys 11 (first iteration), in_loop=True
        hf_pfx = "model.language_model.layers.11"
        attn_key = f"{hf_pfx}.self_attn.o_proj.weight"
        assert torch.allclose(hf_scaled[attn_key], hf_noscale[attn_key] * 0.5)

    def test_loop_weight_sharing(self):
        """Iteration 0 and iteration 1 of same physical layer share weights (before scale)."""
        cfg_noscale = _make_cfg({**_LOOP_FIELDS, "loop_residual_scale": 1.0})
        conv_noscale = MoEConverter(cfg_noscale)
        meg_sd = _build_fake_meg_sd(cfg_noscale)
        hf_noscale = conv_noscale.convert_to_hf(copy.deepcopy(meg_sd))

        # HF 10 = iter0 of phys 10, HF 30 = iter1 of phys 10
        # Without scale they should be identical
        pfx0 = "model.language_model.layers.10"
        pfx1 = "model.language_model.layers.30"
        key_suffix = ".linear_attn.in_proj_qkv.weight"
        assert torch.equal(
            hf_noscale[pfx0 + key_suffix],
            hf_noscale[pfx1 + key_suffix],
        )

    def test_expert_down_proj_scaled(self):
        """Stacked expert down_proj tensors in loop region should be scaled."""
        cfg = _make_cfg(_LOOP_FIELDS)
        conv = MoEConverter(cfg)
        meg_sd = _build_fake_meg_sd(cfg)

        cfg_noscale = _make_cfg({**_LOOP_FIELDS, "loop_residual_scale": 1.0})
        conv_noscale = MoEConverter(cfg_noscale)

        hf_scaled = conv.convert_to_hf(copy.deepcopy(meg_sd))
        hf_noscale = conv_noscale.convert_to_hf(copy.deepcopy(meg_sd))

        # HF layer 10 (in loop): check stacked experts.down_proj
        hf_pfx = "model.language_model.layers.10"
        # Converter produces stacked format when all experts present
        stacked_key = f"{hf_pfx}.mlp.experts.down_proj"
        if stacked_key in hf_scaled:
            assert torch.allclose(hf_scaled[stacked_key], hf_noscale[stacked_key] * 0.5), (
                "Stacked experts down_proj not scaled"
            )
        else:
            # Per-expert format fallback
            for e in range(cfg.num_experts):
                key = f"{hf_pfx}.mlp.experts.{e}.down_proj.weight"
                if key in hf_scaled:
                    assert torch.allclose(hf_scaled[key], hf_noscale[key] * 0.5), (
                        f"Expert {e} down_proj not scaled"
                    )


# ---------------------------------------------------------------------------
# sdpa-wise loop tests
# ---------------------------------------------------------------------------
_SDPA_LOOP_FIELDS = {
    "loop_start_layer": 10,
    "loop_end_layer": 30,
    "num_loop_iterations": 2,
    "loop_residual_scale": 0.5,
    "loop_type": "sdpa-wise",
}


class TestSdpaWiseConfig:
    def test_num_looped_sdpa_layers(self):
        cfg = _make_cfg(_SDPA_LOOP_FIELDS)
        # freq=4 -> SDPA at indices 3,7,11,...; in [10,30): 11,15,19,23,27
        assert cfg.num_looped_sdpa_layers == 5

    def test_hf_num_layers(self):
        cfg = _make_cfg(_SDPA_LOOP_FIELDS)
        # only the 5 SDPA layers double: 40 + 5*(2-1) = 45
        assert cfg.hf_num_layers == 45

    def test_hf_num_layers_k3(self):
        cfg = _make_cfg({**_SDPA_LOOP_FIELDS, "num_loop_iterations": 3})
        assert cfg.hf_num_layers == 40 + 5 * 2

    def test_no_block_size_required(self):
        # span 20 is not required to divide by anything for sdpa-wise
        cfg = _make_cfg({**_SDPA_LOOP_FIELDS, "loop_start_layer": 10, "loop_end_layer": 29})
        assert cfg.loop_type == "sdpa-wise"

    def test_validation_no_sdpa_in_range(self):
        # [8,11) holds SDPA only at 11, which is excluded -> no-op
        path = _make_cfg_expect_error(
            {**_SDPA_LOOP_FIELDS, "loop_start_layer": 8, "loop_end_layer": 11}
        )
        try:
            with pytest.raises(ValueError, match="no SDPA layers"):
                Config(path)
        finally:
            os.unlink(path)

    def test_invalid_loop_type_still_rejected(self):
        path = _make_cfg_expect_error({**_LOOP_FIELDS, "loop_type": "bogus-wise"})
        try:
            with pytest.raises(ValueError, match="loop_type must be one of"):
                Config(path)
        finally:
            os.unlink(path)


class TestLayerWiseAlias:
    """'layer-wise' is an alias for block-wise with block_size=1 (ABC -> AABBCC)."""

    def test_normalised_to_block_wise(self):
        cfg = _make_cfg({**_LOOP_FIELDS, "loop_type": "layer-wise"})
        assert cfg.loop_type == "block-wise"
        assert cfg.loop_block_size == 1

    def test_block_size_1_is_explicitly_allowed(self):
        cfg = _make_cfg({**_LOOP_FIELDS, "loop_type": "block-wise", "loop_block_size": 1})
        assert cfg.loop_block_size == 1

    def test_plan_matches_block_wise_1(self):
        a = _make_cfg({**_LOOP_FIELDS, "loop_type": "layer-wise"})
        b = _make_cfg({**_LOOP_FIELDS, "loop_type": "block-wise", "loop_block_size": 1})
        assert (
            MoEConverter(a)._build_loop_execution_plan()
            == MoEConverter(b)._build_loop_execution_plan()
        )

    def test_repeats_are_adjacent(self):
        cfg = _make_cfg({**_LOOP_FIELDS, "loop_type": "layer-wise"})
        plan = MoEConverter(cfg)._build_loop_execution_plan()
        for layer in range(cfg.loop_start_layer, cfg.loop_end_layer):
            idxs = [i for i, (_, p, _, _) in enumerate(plan) if p == layer]
            assert idxs == list(range(idxs[0], idxs[0] + cfg.num_loop_iterations))

    def test_hf_num_layers_unaffected_by_block_size(self):
        cfg = _make_cfg({**_LOOP_FIELDS, "loop_type": "layer-wise"})
        span = cfg.loop_end_layer - cfg.loop_start_layer
        assert cfg.hf_num_layers == cfg.num_layers + span * (cfg.num_loop_iterations - 1)

    def test_conflicting_block_size_rejected(self):
        path = _make_cfg_expect_error(
            {**_LOOP_FIELDS, "loop_type": "layer-wise", "loop_block_size": 4}
        )
        try:
            with pytest.raises(ValueError, match="implies loop_block_size=1"):
                Config(path)
        finally:
            os.unlink(path)


class TestSdpaWiseExecutionPlan:
    def test_plan_length(self):
        cfg = _make_cfg(_SDPA_LOOP_FIELDS)
        plan = MoEConverter(cfg)._build_loop_execution_plan()
        assert len(plan) == cfg.hf_num_layers == 45

    def test_hf_indices_contiguous(self):
        cfg = _make_cfg(_SDPA_LOOP_FIELDS)
        plan = MoEConverter(cfg)._build_loop_execution_plan()
        assert [hf for hf, _, _, _ in plan] == list(range(45))

    def test_layer_order_preserved(self):
        """ABC -> ABBC must never reorder physical layers."""
        cfg = _make_cfg(_SDPA_LOOP_FIELDS)
        plan = MoEConverter(cfg)._build_loop_execution_plan()
        phys = [p for _, p, _, _ in plan]
        assert phys == sorted(phys)

    def test_gdn_layers_execute_once(self):
        cfg = _make_cfg(_SDPA_LOOP_FIELDS)
        plan = MoEConverter(cfg)._build_loop_execution_plan()
        counts = {}
        for _, phys, _, _ in plan:
            counts[phys] = counts.get(phys, 0) + 1
        for layer in range(cfg.num_layers):
            if is_gdn_layer(layer, cfg.linear_attention_freq):
                assert counts[layer] == 1, f"GDN layer {layer} repeated"

    def test_sdpa_in_range_execute_k_times(self):
        cfg = _make_cfg(_SDPA_LOOP_FIELDS)
        plan = MoEConverter(cfg)._build_loop_execution_plan()
        counts = {}
        for _, phys, _, _ in plan:
            counts[phys] = counts.get(phys, 0) + 1
        for layer in range(cfg.loop_start_layer, cfg.loop_end_layer):
            if not is_gdn_layer(layer, cfg.linear_attention_freq):
                assert counts[layer] == cfg.num_loop_iterations

    def test_sdpa_outside_range_execute_once(self):
        cfg = _make_cfg(_SDPA_LOOP_FIELDS)
        plan = MoEConverter(cfg)._build_loop_execution_plan()
        counts = {}
        for _, phys, _, _ in plan:
            counts[phys] = counts.get(phys, 0) + 1
        for layer in (3, 7, 31, 35, 39):
            assert counts[layer] == 1

    def test_repeats_are_adjacent(self):
        """Each SDPA layer's copies must be consecutive (in place, ABBC not ABCB)."""
        cfg = _make_cfg(_SDPA_LOOP_FIELDS)
        plan = MoEConverter(cfg)._build_loop_execution_plan()
        for layer in (11, 15, 19, 23, 27):
            idxs = [i for i, (_, p, _, _) in enumerate(plan) if p == layer]
            assert idxs == list(range(idxs[0], idxs[0] + len(idxs)))

    def test_gdn_in_range_not_marked_in_loop(self):
        """GDN layers get in_loop=False so loop scale is not absorbed into them."""
        cfg = _make_cfg(_SDPA_LOOP_FIELDS)
        plan = MoEConverter(cfg)._build_loop_execution_plan()
        for _, phys, in_loop, iteration in plan:
            if is_gdn_layer(phys, cfg.linear_attention_freq):
                assert in_loop is False
                assert iteration == -1

    def test_iterations_numbered_per_sdpa_layer(self):
        cfg = _make_cfg(_SDPA_LOOP_FIELDS)
        plan = MoEConverter(cfg)._build_loop_execution_plan()
        for layer in (11, 15, 19, 23, 27):
            iters = [it for _, p, il, it in plan if p == layer and il]
            assert iters == list(range(cfg.num_loop_iterations))


class TestKeyLoraAbsorption:
    """Per-iteration key LoRA is folded into each iteration's in_proj copy."""

    FIELDS = {**_LOOP_FIELDS, "loop_type": "layer-wise", "loop_gdn_key_lora_rank": 4}

    def _sd(self, cfg, mg_pfx, scale=1.0):
        qk = cfg.linear_key_head_dim * cfg.linear_num_key_heads
        v = cfg.linear_value_head_dim * cfg.linear_num_value_heads
        dim = qk * 2 + v * 2 + cfg.linear_num_value_heads * 2
        r = cfg.loop_gdn_key_lora_rank
        R = cfg.num_loop_iterations
        return {
            f"{mg_pfx}.self_attention.key_lora_down": torch.full((R, cfg.hidden_size, r), scale),
            f"{mg_pfx}.self_attention.key_lora_up": torch.full((R, r, qk), scale),
        }, torch.zeros(dim, cfg.hidden_size)

    def test_no_op_when_disabled(self):
        cfg = _make_cfg(_LOOP_FIELDS)
        conv = MoEConverter(cfg)
        w = torch.randn(8, cfg.hidden_size)
        assert conv._absorb_key_lora({}, "x", w, 0) is w

    def test_no_op_for_unlooped_layer(self):
        cfg = _make_cfg(self.FIELDS)
        conv = MoEConverter(cfg)
        sd, w = self._sd(cfg, "x")
        # iteration < 0 marks a layer outside the loop
        assert conv._absorb_key_lora(sd, "x", w, -1) is w

    def test_missing_params_raise(self):
        cfg = _make_cfg(self.FIELDS)
        conv = MoEConverter(cfg)
        _, w = self._sd(cfg, "x")
        with pytest.raises(KeyError, match="key LoRA"):
            conv._absorb_key_lora({}, "x", w, 0)

    def test_delta_lands_only_on_key_rows(self):
        cfg = _make_cfg(self.FIELDS)
        conv = MoEConverter(cfg)
        sd, w = self._sd(cfg, "x")
        out = conv._absorb_key_lora(sd, "x", w, 0)
        qk = cfg.linear_key_head_dim * cfg.linear_num_key_heads
        tp = cfg.tp
        per_rank = w.shape[0] // tp
        qk_local = qk // tp
        expected = torch.zeros(w.shape[0], dtype=torch.bool)
        for rank in range(tp):
            base = rank * per_rank
            expected[base + qk_local : base + 2 * qk_local] = True
        touched = (out - w).abs().sum(dim=1) > 0
        assert torch.equal(touched, expected), "delta hit non-key rows"

    def test_iterations_give_different_deltas(self):
        cfg = _make_cfg(self.FIELDS)
        conv = MoEConverter(cfg)
        qk = cfg.linear_key_head_dim * cfg.linear_num_key_heads
        v = cfg.linear_value_head_dim * cfg.linear_num_value_heads
        dim = qk * 2 + v * 2 + cfg.linear_num_value_heads * 2
        r, R = cfg.loop_gdn_key_lora_rank, cfg.num_loop_iterations
        torch.manual_seed(0)
        sd = {
            "x.self_attention.key_lora_down": torch.randn(R, cfg.hidden_size, r),
            "x.self_attention.key_lora_up": torch.randn(R, r, qk),
        }
        w = torch.zeros(dim, cfg.hidden_size)
        a = conv._absorb_key_lora(sd, "x", w, 0)
        b = conv._absorb_key_lora(sd, "x", w, 1)
        assert not torch.allclose(a, b), "iterations must differ -- that is the point"

    def test_zero_up_absorbs_to_identity(self):
        """A zero `up` must fold to a no-op.

        Training seeds `up` randomly, so this is not the init any more; it stays as
        a check that absorption introduces nothing of its own when the delta is zero.
        """
        cfg = _make_cfg(self.FIELDS)
        conv = MoEConverter(cfg)
        qk = cfg.linear_key_head_dim * cfg.linear_num_key_heads
        v = cfg.linear_value_head_dim * cfg.linear_num_value_heads
        dim = qk * 2 + v * 2 + cfg.linear_num_value_heads * 2
        r, R = cfg.loop_gdn_key_lora_rank, cfg.num_loop_iterations
        sd = {
            "x.self_attention.key_lora_down": torch.randn(R, cfg.hidden_size, r),
            "x.self_attention.key_lora_up": torch.zeros(R, r, qk),
        }
        w = torch.randn(dim, cfg.hidden_size)
        assert torch.allclose(conv._absorb_key_lora(sd, "x", w, 0), w)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
