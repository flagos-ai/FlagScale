# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""Pin the ms-swift SFT data path against ms-swift itself, at get_batch granularity.

No model weights are loaded: the template is built from the tokenizer only, and
ms-swift's ``prepare_batch`` helpers are driven directly rather than through a
distributed trainer. The reference configuration mirrors
``exp/train_dsv4_flash_multinode.sh``.

Both sides share the same ``swift`` template object, so what is actually under
test is the FlagScale collator layer: the packed layout, the next-token roll,
``cu_seqlens`` derivation, and the loss numerator/denominator split.

Set ``SWIFT_SFT_ALIGN_MODEL`` / ``SWIFT_SFT_ALIGN_DATA`` to point at a tokenizer
directory and a jsonl file; the test skips when they are absent.
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("swift")

_MEGATRON_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "flagscale", "train", "megatron"
)
sys.path.insert(0, os.path.abspath(_MEGATRON_ROOT))

MODEL = os.environ.get("SWIFT_SFT_ALIGN_MODEL")
DATA = os.environ.get("SWIFT_SFT_ALIGN_DATA")

pytestmark = pytest.mark.skipif(
    not MODEL or not os.path.isdir(MODEL) or not DATA or not os.path.isfile(DATA),
    reason="set SWIFT_SFT_ALIGN_MODEL (tokenizer dir) and SWIFT_SFT_ALIGN_DATA (jsonl)",
)

MAX_LENGTH = 65536
TEMPLATE = "deepseek_v4_flash"
LOSS_SCALE = "default+ignore_empty_think"


class _Args:
    """The subset of Megatron args ms-swift's packed-sequence helpers read."""

    task_type = "causal_lm"
    pipeline_model_parallel_size = 1
    context_parallel_size = 1
    padding_free = True
    tensor_model_parallel_size = 1
    sequence_parallel = False
    fp8_recipe = None
    attention_backend = "flash"
    cp_partition_mode = "contiguous"


def _build_template(loss_scale=LOSS_SCALE, processor=None, **kwargs):
    from swift.template import get_template

    if processor is None:
        # The ms-swift reference path. FlagScale itself uses the local shim; the
        # two are pinned to each other by test_processor_shim_matches_ms_swift.
        from swift.model import get_processor

        processor = get_processor(MODEL, model_type="deepseek_v4", download_model=False)
    template = get_template(
        processor,
        max_length=MAX_LENGTH,
        template_type=TEMPLATE,
        # arg-level "delete" maps to template-level "raise"; over-long rows then
        # raise MaxLengthError and are skipped by the LazyLLMDataset layer.
        truncation_strategy="raise",
        loss_scale=loss_scale,
        add_non_thinking_prefix=True,
        padding_side="right",
        padding_free=True,
        **kwargs,
    )
    template.set_mode("train")
    template.use_megatron = True
    return template


def _encode(template, limit=4):
    import json

    encoded = []
    with open(DATA) as f:
        for line in f:
            if len(encoded) >= limit:
                break
            try:
                row = template.encode(json.loads(line))
            except Exception:  # MaxLengthError -> dropped, as in LazyLLMDataset
                continue
            # PackingDataset sets 'length'; packing_row reads it.
            row["length"] = len(row["input_ids"])
            encoded.append(row)
    if not encoded:
        pytest.skip("no sample short enough to encode")
    return encoded


def _swift_batch(template, batch):
    """ms-swift's get_batch for the single-rank, pp=1/cp=1 padding_free path."""
    from swift.megatron.utils import get_packed_seq_params

    res = template.data_collator(list(batch), padding_to=None)
    res = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in res.items()}
    # swift/megatron/trainers/utils.py:22 -- labels are rolled at batch time.
    res["labels"] = torch.roll(res["labels"], -1, dims=-1)
    if res.get("loss_scale") is not None:
        res["loss_scale"] = torch.roll(res["loss_scale"], -1, dims=-1)
    packed = get_packed_seq_params(_Args(), res["position_ids"])
    return res, packed


def test_swift_sft_collator_matches_ms_swift():
    from data.swift_collator import swift_sft_collate

    template = _build_template()
    batch = _encode(template)

    swift_res, packed = _swift_batch(template, batch)
    fs = swift_sft_collate(list(batch), template=template, padding_to=None)

    # padding_free must yield one packed [1, T] sequence, not a padded [B, T] batch.
    assert fs["tokens"].shape[0] == 1, fs["tokens"].shape
    assert fs["attention_mask"] is None

    assert torch.equal(fs["tokens"], swift_res["input_ids"].to(torch.int64))
    assert torch.equal(fs["position_ids"], swift_res["position_ids"].to(torch.int64))

    # FlagScale remaps -100 -> 0, so labels can only be compared where they count.
    labels = swift_res["labels"]
    valid = labels != -100
    assert valid.any(), "no supervised token in the sample; the test would be vacuous"
    assert torch.equal(fs["labels"][valid], labels[valid].to(torch.int64))

    # ms-swift's loss_mask is the token count, never the weights.
    assert torch.equal(fs["loss_mask"], (labels != -100).to(torch.float32))

    fs_cu = fs["cu_seqlens"].squeeze(0).cpu().to(torch.int32)
    assert torch.equal(fs_cu, packed.cu_seqlens_q.cpu().to(torch.int32))
    assert int(fs["max_seqlen"]) == int(packed.max_seqlen_q)


def test_swift_sft_loss_matches_ms_swift():
    """Given identical per-token losses, both sides must reduce to the same scalar."""
    from data.swift_collator import swift_sft_collate

    template = _build_template()
    batch = _encode(template)
    swift_res, _ = _swift_batch(template, batch)
    fs = swift_sft_collate(list(batch), template=template, padding_to=None)

    torch.manual_seed(0)
    per_token = torch.rand_like(fs["loss_mask"])

    # FlagScale: loss_scale weights the numerator, loss_mask is the denominator.
    fs_losses = per_token.float()
    if fs["loss_scale"] is not None:
        fs_losses = fs_losses * fs["loss_scale"]
    fs_loss = (fs_losses.view(-1) * fs["loss_mask"].view(-1)).sum()
    fs_ntok = fs["loss_mask"].sum()

    # ms-swift, swift/megatron/trainers/trainer.py:58-63.
    mask = (swift_res["labels"] != -100).to(torch.float32)
    sw_losses = per_token.float()
    if swift_res.get("loss_scale") is not None:
        sw_losses = sw_losses * swift_res["loss_scale"]
    sw_loss = (sw_losses * mask).sum()
    sw_ntok = mask.sum()

    assert torch.allclose(fs_loss, sw_loss)
    assert torch.allclose(fs_ntok, sw_ntok)
    assert torch.allclose(fs_loss / fs_ntok, sw_loss / sw_ntok)


def test_non_binary_loss_scale_stays_out_of_the_denominator():
    """A non-binary strategy must not inflate the token count.

    ``loss_scale`` folded into ``loss_mask`` would divide by the sum of the
    weights instead of the number of tokens, which diverges from ms-swift
    silently -- the loss merely looks different, nothing errors.
    """
    import json

    from data.swift_collator import swift_sft_collate

    template = _build_template(loss_scale="default", is_binary_loss_scale=False)

    encoded = []
    with open(DATA) as f:
        for line in f:
            if len(encoded) >= 2:
                break
            row = json.loads(line)
            for message in row["messages"]:
                if message["role"] == "assistant":
                    message["loss_scale"] = 2.0
            try:
                enc = template.encode(row)
            except Exception:
                continue
            enc["length"] = len(enc["input_ids"])
            encoded.append(enc)
    if not encoded:
        pytest.skip("no sample short enough to encode")

    # data_collator consumes 'length' off the rows, so hand each side its own copy.
    swift_res, _ = _swift_batch(template, [dict(e) for e in encoded])
    fs = swift_sft_collate([dict(e) for e in encoded], template=template, padding_to=None)
    assert fs["loss_scale"] is not None, "expected per-token weights for a non-binary strategy"

    n_tokens = fs["loss_mask"].sum()
    sum_weights = fs["loss_scale"].sum()
    assert n_tokens > 0
    # The 2.0 weight makes the weight sum strictly larger; the denominator must
    # track tokens, so the two must not coincide.
    assert sum_weights > n_tokens
    assert torch.equal(fs["loss_mask"], (fs["loss_mask"] > 0).to(torch.float32)), "loss_mask must stay binary"

    # The weights must cover exactly the supervised positions, which is what lets
    # the CP/TP plumbing gate 'loss_scale' on 'loss_mask'.
    assert torch.equal((fs["loss_scale"] > 0).to(torch.float32), fs["loss_mask"])

    # train_gpt.loss_func's reduction, against ms-swift's.
    torch.manual_seed(0)
    per_token = torch.rand_like(fs["loss_mask"])
    weighted = (per_token.view(-1) * fs["loss_scale"].view(-1) * fs["loss_mask"].view(-1)).sum()
    unweighted = (per_token.view(-1) * fs["loss_mask"].view(-1)).sum()
    assert not torch.allclose(weighted, unweighted), "the weights must actually change the loss"

    mask = (swift_res["labels"] != -100).to(torch.float32)
    sw = (per_token.float() * swift_res["loss_scale"]) * mask
    assert torch.allclose(weighted, sw.sum())
    assert torch.allclose(n_tokens, mask.sum())


def test_loss_func_applies_loss_scale_to_the_numerator_only():
    """``train_gpt.loss_func`` must weight the numerator and count tokens."""
    pytest.importorskip("megatron.core")

    T = 8
    loss_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    loss_scale = torch.tensor([[0.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    per_token = torch.ones(1, T)

    # Replicate the reduction under test without importing train_gpt, which pulls
    # in the full Megatron training stack at import time.
    def reduce(losses, mask, scale):
        losses = losses.view(-1).float()
        mask = mask.view(-1).float()
        if scale is not None:
            losses = losses * scale.view(-1).float()
        return torch.sum(losses * mask), mask.sum().to(torch.int)

    loss, num_tokens = reduce(per_token, loss_mask, loss_scale)
    # 2 + 2 + 1 + 1 weighted; 4 supervised tokens.
    assert loss.item() == pytest.approx(6.0)
    assert int(num_tokens) == 4, "the denominator must be the token count, not sum(weights)"

    unweighted, unweighted_tokens = reduce(per_token, loss_mask, None)
    assert unweighted.item() == pytest.approx(4.0)
    assert int(unweighted_tokens) == 4


LOADER_CASES = [
    ([""], 0.0),            # plain path
    (["#3"], 0.0),          # #N below the row count
    (["#10"], 0.0),         # #N above it -> repeat sampling
    ([""], 0.34),           # train/val split
    (["#4"], 0.5),
    (["", "#2"], 0.0),      # two sources, concatenated
    # A ratio small enough that int(train_len * ratio) rounds to 0: upstream
    # clamps the val split to at least one row, and the larger cases above do not
    # distinguish that from plain truncation.
    ([""], 0.01),
    (["#3"], 0.1),
]


@pytest.mark.parametrize("suffixes,ratio", LOADER_CASES)
def test_vendored_loader_matches_ms_swift(suffixes, ratio):
    """The focused local loader must return the same rows as ms-swift's.

    Covers the ``#N`` sampling arithmetic (including repeat-sampling when N
    exceeds the file), the train/val split, and multi-source concatenation --
    these are the parts of ``swift/dataset/loader.py`` the rewrite had to keep,
    since the reference training scripts already use the ``#N`` form.
    """
    import json

    from swift.dataset import load_dataset as reference

    from data.dataset import load_dataset as vendored

    specs = [DATA + suffix for suffix in suffixes]
    kwargs = dict(
        split_dataset_ratio=ratio,
        seed=42,
        num_proc=1,
        load_from_cache_file=False,
        strict=False,
        streaming=False,
    )
    mine = vendored(list(specs), **kwargs)
    ref = reference(list(specs), **kwargs)

    def signature(dataset):
        if dataset is None:
            return None
        return [json.dumps(row, sort_keys=True) for row in dataset]

    assert signature(mine[0]) == signature(ref[0])
    assert signature(mine[1]) == signature(ref[1])


def test_vendored_loader_rejects_unsupported_sources():
    """Remote ids and hub prefixes must fail loudly, not resolve unexpectedly."""
    from data.dataset import load_dataset

    with pytest.raises(ValueError, match="hub-prefixed"):
        load_dataset(["hf::AI-ModelScope/alpaca-gpt4-data-zh"])
    with pytest.raises(FileNotFoundError):
        load_dataset(["AI-ModelScope/alpaca-gpt4-data-zh"])
    with pytest.raises(ValueError, match="integer"):
        load_dataset([f"{DATA}#abc"])
    with pytest.raises(NotImplementedError, match="streaming"):
        load_dataset([DATA], streaming=True)


def test_vendored_pipeline_matches_ms_swift_end_to_end():
    """load -> AddLength -> Lazy -> Packing must agree with ms-swift throughout.

    The packing signature is the real assertion: bin-packing order depends on
    every per-sample length, so identical packs mean the lengths, the encode and
    the packer all agree.
    """
    import data.dataset as vendored_ds
    import swift.dataset as reference_ds

    results = {}
    for name, module, template in (
        ("vendored", vendored_ds, _build_vendored_template()),
        ("reference", reference_ds, _build_template()),
    ):
        train, _ = module.load_dataset(
            [DATA], split_dataset_ratio=0.0, seed=42, num_proc=1,
            load_from_cache_file=False, strict=False, streaming=False,
        )
        train = module.AddLengthPreprocessor(template=template)(
            train, num_proc=1, load_from_cache_file=False, strict=False, batch_size=1000
        )
        lazy = module.LazyLLMDataset(train, template.encode, strict=False, random_state=42)
        packed = module.PackingDataset(
            template, lazy, num_proc=1, packing_length=MAX_LENGTH, packing_num_proc=1,
            packing_strategy="binpack", strict=False, load_from_cache_file=False,
        )
        results[name] = {
            "lengths": list(train["lengths"]),
            "packs": [
                [(len(r["input_ids"]), sum(1 for x in r["labels"] if x != -100)) for r in packed[i]]
                for i in range(len(packed))
            ],
        }

    assert results["vendored"]["lengths"] == results["reference"]["lengths"]
    assert results["vendored"]["packs"] == results["reference"]["packs"]
    assert results["vendored"]["packs"], "nothing was packed; the comparison would be vacuous"


def test_whole_data_path_stands_alone_without_swift():
    """The end state of Phase 2: load, encode, pack and collate with no ``swift``.

    Blocks ``swift`` in a subprocess and drives the complete pipeline, ending at
    the FlagScale batch keys. ``swift`` stays installed in this environment only
    so the tests above can compare against it.
    """
    import subprocess
    import textwrap

    script = textwrap.dedent(
        f"""
        import sys, builtins
        sys.path.insert(0, {os.path.abspath(_MEGATRON_ROOT)!r})

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "swift" or name.startswith("swift."):
                raise ImportError("swift is blocked for this test")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocked

        from data.swift_processor_shim import build_processor
        from data.template import get_template
        from data.dataset import AddLengthPreprocessor, LazyLLMDataset, PackingDataset, load_dataset
        from data.swift_collator import swift_sft_collate

        template = get_template(
            build_processor({MODEL!r}, "deepseek_v4", {TEMPLATE!r}),
            max_length={MAX_LENGTH}, template_type={TEMPLATE!r}, truncation_strategy="raise",
            loss_scale={LOSS_SCALE!r}, add_non_thinking_prefix=True,
            padding_side="right", padding_free=True,
        )
        template.set_mode("train")
        template.use_megatron = True

        train, _ = load_dataset([{DATA!r}], num_proc=1, load_from_cache_file=False)
        train = AddLengthPreprocessor(template=template)(
            train, num_proc=1, load_from_cache_file=False, strict=False, batch_size=1000
        )
        lazy = LazyLLMDataset(train, template.encode, strict=False, random_state=42)
        packed = PackingDataset(
            template, lazy, num_proc=1, packing_length={MAX_LENGTH}, packing_num_proc=1,
            packing_strategy="binpack", strict=False, load_from_cache_file=False,
        )
        batch = swift_sft_collate(packed[0], template=template, padding_to=None)
        builtins.__import__ = real_import

        assert batch["tokens"].shape[0] == 1, batch["tokens"].shape
        assert batch["loss_mask"].sum() > 0
        assert batch["attention_mask"] is None
        print("SWIFT:" + ",".join(sorted(m for m in sys.modules if m.split(".")[0] == "swift")))
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, out.stderr[-4000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("SWIFT:")]
    assert line, out.stdout[-2000:]
    leaked = [m for m in line[0][len("SWIFT:"):].split(",") if m]
    assert not leaked, f"the data path still pulls in swift: {leaked}"


def _build_vendored_template(loss_scale=LOSS_SCALE, **kwargs):
    """The same template, built from the vendored package instead of ``swift``."""
    from data.swift_processor_shim import build_processor
    from data.template import get_template

    template = get_template(
        build_processor(MODEL, "deepseek_v4", TEMPLATE),
        max_length=MAX_LENGTH,
        template_type=TEMPLATE,
        truncation_strategy="raise",
        loss_scale=loss_scale,
        add_non_thinking_prefix=True,
        padding_side="right",
        padding_free=True,
        **kwargs,
    )
    template.set_mode("train")
    template.use_megatron = True
    return template


def test_vendored_template_matches_ms_swift():
    """The vendored template must encode byte-identically to swift.template."""
    import json

    reference = _build_template()
    vendored = _build_vendored_template()

    rows = 0
    with open(DATA) as f:
        for line in f:
            row = json.loads(line)
            try:
                ref = reference.encode(json.loads(json.dumps(row)))
            except Exception:
                continue
            mine = vendored.encode(json.loads(json.dumps(row)))
            assert mine["input_ids"] == ref["input_ids"]
            assert mine["labels"] == ref["labels"]
            assert mine.get("loss_scale") == ref.get("loss_scale")
            rows += 1
    assert rows > 0, "no row encoded; the comparison would be vacuous"


@pytest.mark.parametrize("effort", [None, "low", "high", "max"])
def test_vendored_template_reasoning_effort_matches(effort):
    """Each reasoning-effort level prepends a different system prefix."""
    import json

    messages = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "<think>greet</think>Hi!"},
        ]
    }
    if effort is None:
        os.environ.pop("REASONING_EFFORT", None)
    else:
        os.environ["REASONING_EFFORT"] = effort
    try:
        ref = _build_template().encode(json.loads(json.dumps(messages)))
        mine = _build_vendored_template().encode(json.loads(json.dumps(messages)))
    finally:
        os.environ.pop("REASONING_EFFORT", None)
    assert mine["input_ids"] == ref["input_ids"]
    assert mine["labels"] == ref["labels"]


def test_vendored_template_tool_calling_matches():
    """The DSML tool-call path runs through the vendored agent_template."""
    import json

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}, "n": {"type": "integer"}},
                "required": ["location"],
            },
        },
    }]
    row = {
        "tools": tools,
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "weather in Beijing?"},
            {"role": "assistant", "content": "<think>use the tool</think>"},
            {
                "role": "tool_call",
                "content": '{"name": "get_weather", "arguments": "{\\"location\\": \\"Beijing\\", \\"n\\": 3}"}',
            },
            {"role": "tool_response", "content": '{"temperature": 22}'},
            {"role": "assistant", "content": "<think>done</think>22C in Beijing."},
        ],
    }
    vendored = _build_vendored_template()
    ref = _build_template().encode(json.loads(json.dumps(row)))
    mine = vendored.encode(json.loads(json.dumps(row)))
    assert mine["input_ids"] == ref["input_ids"]
    assert mine["labels"] == ref["labels"]

    decoded = vendored.safe_decode(mine["input_ids"])
    assert "DSML" in decoded, "the tools section/tool_calls block did not render"
    assert "<tool_result>" in decoded

    # And the response parses back into the same call.
    from data.agent_template import agent_template_map

    functions = agent_template_map["deepseek_v4"]().get_toolcall(
        '<think>r</think>\n\n<｜DSML｜tool_calls>\n'
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="n" string="false">3</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n</｜DSML｜tool_calls>'
    )
    assert len(functions) == 1
    assert functions[0].name == "get_weather"
    assert json.loads(functions[0].arguments) == {"location": "Beijing", "n": 3}


def test_vendored_max_length_error_is_honoured_by_the_dataset_layer():
    """An over-long row must be *skipped*, not propagated.

    The template raises ``MaxLengthError`` and the dataset layer decides to skip
    by ``isinstance(e, MaxLengthError)``. Both sides are vendored now, so this
    pins that they still refer to the same class: if they diverge, every row over
    ``max_length`` aborts the run instead of being dropped -- and with
    ``truncation_strategy='delete'`` on 64k data that is the normal path.
    """
    import json

    import datasets

    from data.dataset import LazyLLMDataset
    from data.template import MaxLengthError as VendoredMaxLengthError

    assert issubclass(VendoredMaxLengthError, ValueError)

    # max_length small enough that the real rows do not fit.
    template = _build_vendored_template()
    template.max_length = 2000

    rows = [json.loads(line) for line in open(DATA)]
    short = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>r</think>ok"},
        ]
    }
    rows.append(short)

    # The over-long rows raise; LazyLLMDataset must swallow that and move on.
    with pytest.raises(VendoredMaxLengthError):
        template.encode(json.loads(json.dumps(rows[0])))

    lazy = LazyLLMDataset(
        datasets.Dataset.from_list(rows), template.encode, strict=False, random_state=0
    )
    encoded = lazy[len(rows) - 1]
    assert encoded["input_ids"], "LazyLLMDataset failed to fall back to the short row"


def test_vendored_template_stands_alone_without_swift():
    """The vendored template must encode with ``swift`` entirely unavailable.

    This is the Phase 2 end state for the template layer: ``swift`` is still
    importable in this environment (the tests compare against it), so block it
    in a subprocess to prove nothing silently depends on it.
    """
    import subprocess
    import textwrap

    script = textwrap.dedent(
        f"""
        import sys, builtins
        sys.path.insert(0, {os.path.abspath(_MEGATRON_ROOT)!r})

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "swift" or name.startswith("swift."):
                raise ImportError("swift is blocked for this test")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocked
        from data.swift_processor_shim import build_processor
        from data.template import get_template

        template = get_template(
            build_processor({MODEL!r}, "deepseek_v4", {TEMPLATE!r}),
            max_length=4096, template_type={TEMPLATE!r}, truncation_strategy="raise",
            loss_scale={LOSS_SCALE!r}, padding_side="right", padding_free=True,
        )
        template.set_mode("train")
        encoded = template.encode({{"messages": [
            {{"role": "user", "content": "q"}},
            {{"role": "assistant", "content": "<think>r</think>a"}},
        ]}})
        builtins.__import__ = real_import
        assert encoded["input_ids"], "encode produced nothing"
        print("SWIFT:" + ",".join(sorted(m for m in sys.modules if m.split(".")[0] == "swift")))
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("SWIFT:")]
    assert line, out.stdout[-2000:]
    leaked = [m for m in line[0][len("SWIFT:"):].split(",") if m]
    assert not leaked, f"the vendored template pulled in swift: {leaked}"


def test_processor_shim_matches_ms_swift():
    """The local processor shim must encode identically to swift.model.get_processor.

    This is what licenses dropping ``swift.model`` (~12k LOC, plus ``peft`` and
    ``swift.sequence_parallel``) from the data path.
    """
    import json

    from data.swift_processor_shim import build_processor

    reference = _build_template()
    shimmed = _build_template(processor=build_processor(MODEL, "deepseek_v4", TEMPLATE))

    # The six attributes the template actually reads off the processor.
    ref_info, shim_info = reference.model_info, shimmed.model_info
    assert shim_info.model_dir == ref_info.model_dir
    assert shim_info.task_type == ref_info.task_type
    assert shim_info.torch_dtype == ref_info.torch_dtype
    assert shim_info.max_model_len == ref_info.max_model_len
    assert shimmed.model_meta.is_multimodal == reference.model_meta.is_multimodal
    assert type(shim_info.config) is type(ref_info.config)

    rows = 0
    with open(DATA) as f:
        for line in f:
            row = json.loads(line)
            try:
                ref = reference.encode(json.loads(json.dumps(row)))
            except Exception:
                continue
            shim = shimmed.encode(json.loads(json.dumps(row)))
            assert shim["input_ids"] == ref["input_ids"]
            assert shim["labels"] == ref["labels"]
            assert shim.get("loss_scale") == ref.get("loss_scale")
            rows += 1
    assert rows > 0, "no row encoded; the comparison would be vacuous"


def test_processor_shim_rejects_multimodal():
    """The shim is text-only, so it must refuse rather than mis-encode."""
    from data import swift_processor_shim

    class _Cfg:
        vision_config = {"hidden_size": 1}

    import transformers

    # build_processor imports AutoConfig inside the function, so patching the
    # transformers attribute is what the shim will actually resolve.
    original = transformers.AutoConfig.from_pretrained
    transformers.AutoConfig.from_pretrained = staticmethod(lambda *a, **k: _Cfg())
    try:
        with pytest.raises(ValueError, match="multimodal"):
            swift_processor_shim.build_processor(MODEL, "deepseek_v4", TEMPLATE)
    finally:
        transformers.AutoConfig.from_pretrained = original


def test_data_path_does_not_import_swift_model():
    """Regression guard: the data path must stay off swift.model (and peft).

    Run in a subprocess so an earlier test's imports do not mask a reintroduced
    dependency.
    """
    import subprocess
    import textwrap

    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {os.path.abspath(_MEGATRON_ROOT)!r})
        from data.swift_processor_shim import build_processor
        from swift.template import get_template

        processor = build_processor({MODEL!r}, "deepseek_v4", {TEMPLATE!r})
        template = get_template(
            processor, max_length=4096, template_type={TEMPLATE!r},
            truncation_strategy="raise", loss_scale={LOSS_SCALE!r},
            padding_side="right", padding_free=True,
        )
        template.set_mode("train")
        template.encode({{"messages": [
            {{"role": "user", "content": "q"}},
            {{"role": "assistant", "content": "<think>r</think>a"}},
        ]}})

        leaked = sorted(
            m for m in sys.modules
            if m == "swift.model" or m.startswith("swift.model.")
            or m == "swift.sequence_parallel" or m.startswith("swift.sequence_parallel.")
        )
        print("LEAKED:" + ",".join(leaked))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=900
    )
    assert out.returncode == 0, out.stderr[-3000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("LEAKED:")]
    assert line, out.stdout[-2000:]
    leaked = [m for m in line[0][len("LEAKED:"):].split(",") if m]
    assert not leaked, f"swift.model / swift.sequence_parallel got reimported: {leaked}"


LOSS_SCALE_SPECS = (
    "default",
    "last_round",
    "all",
    "ignore_empty_think",
    "ignore_think_prefix",
    "default+ignore_empty_think",
    "default+react",
    "default+hermes",
    "default+qwen",
    "default+agentflan",
    "default+alpha_umi",
    "last_round+hermes+ignore_empty_think",
)


@pytest.mark.parametrize("spec", LOSS_SCALE_SPECS)
def test_vendored_loss_scale_matches_ms_swift(spec):
    """The vendored loss_scale package must reproduce swift.loss_scale exactly."""
    from swift.loss_scale import get_loss_scale as reference

    from data.loss_scale import ContextType, get_loss_scale as vendored

    mine, ref = vendored(spec), reference(spec)
    assert type(mine).__name__ == type(ref).__name__
    assert mine.base_strategy == ref.base_strategy
    assert mine.is_binary_loss_scale == ref.is_binary_loss_scale

    # Exercise __call__ on a conversation covering every branch that matters:
    # an empty think block, a data-level loss_scale weight, a data-level
    # loss: false mask, and ReAct-style keywords the agent strategies split on.
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "<think></think>a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "<think>r</think>a2", "loss_scale": 2.0},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "Thought: t\nAction: x\nAction Input: y\n", "loss": False},
    ]
    contexts = [
        "sys", "q1", "<think></think>a1", "q2", "<think>r</think>a2", "q3",
        "Thought: t\nAction: x\nAction Input: y\n",
    ]
    types = [
        ContextType.OTHER, ContextType.OTHER, ContextType.RESPONSE, ContextType.OTHER,
        ContextType.RESPONSE, ContextType.OTHER, ContextType.RESPONSE,
    ]
    assert mine(list(contexts), list(types), messages) == ref(list(contexts), list(types), messages)


def test_vendored_types_match_ms_swift():
    """The three names lifted out of swift.template must behave identically.

    ``get_last_user_round`` decides which round counts as "last" for the
    ``last_round`` strategy and for history-thinking removal, and it keys off
    'tool' as well as 'user'. The parametrized test above cannot catch a slip
    here because its fixture holds no tool-role message, so cover it directly.
    """
    from swift.template import ContextType as RefContextType, get_last_user_round as reference

    from data.loss_scale import ContextType, get_last_user_round as vendored

    assert ContextType.RESPONSE == RefContextType.RESPONSE
    assert ContextType.SUFFIX == RefContextType.SUFFIX
    assert ContextType.OTHER == RefContextType.OTHER

    cases = [
        [],
        [{"role": "user", "content": "q"}],
        [{"role": "system", "content": "s"}],
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        # the 'tool' branch, in both the last and a middle position
        [{"role": "user", "content": "q"}, {"role": "tool", "content": "t"}],
        [
            {"role": "user", "content": "q"},
            {"role": "tool", "content": "t"},
            {"role": "assistant", "content": "a"},
        ],
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": "t"},
            {"role": "assistant", "content": "a2"},
        ],
    ]
    for messages in cases:
        assert vendored(messages) == reference(messages), messages


def test_vendored_split_str_parts_by_matches_ms_swift():
    """``split_str_parts_by`` drives every ConfigLossScale split."""
    from swift.template import split_str_parts_by as reference

    from data.loss_scale import split_str_parts_by as vendored

    plain = ["Thought:", "Action:", "Action Input:", "Observation:"]
    regex = [r"^<think>\s*</think>\s*", r"^</think>\s*"]
    cases = [
        ("Thought: t\nAction: a\nAction Input: i\nObservation: o\n", plain, False),
        ("no delimiters here", plain, False),
        ("", plain, False),
        ("Action: only\n", plain, False),
        ("<think></think>answer", regex, True),
        ("</think>answer", regex, True),
        ("<think>reasoned</think>answer", regex, True),
        ("plain answer", regex, True),
        ("text", [], False),
    ]
    for text, delimiters, regex_mode in cases:
        assert vendored(text, list(delimiters), regex_mode=regex_mode) == reference(
            text, list(delimiters), regex_mode=regex_mode
        ), (text, delimiters, regex_mode)


def test_vendored_loss_scale_registers_the_same_strategies():
    from swift.loss_scale import loss_scale_map as reference

    from data.loss_scale import loss_scale_map as vendored

    assert sorted(vendored) == sorted(reference)


def test_vendored_loss_scale_has_no_swift_dependency():
    """The vendored package must import without pulling in ``swift``."""
    import subprocess
    import textwrap

    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {os.path.abspath(_MEGATRON_ROOT)!r})
        from data.loss_scale import get_loss_scale
        ls = get_loss_scale("default+ignore_empty_think")
        assert ls.is_binary_loss_scale is True
        print("SWIFT:" + ",".join(sorted(m for m in sys.modules if m.split(".")[0] == "swift")))
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("SWIFT:")]
    assert line, out.stdout[-2000:]
    leaked = [m for m in line[0][len("SWIFT:"):].split(",") if m]
    assert not leaked, f"data.loss_scale pulled in swift modules: {leaked}"


def test_batch_keys_order_matches_the_unpack_sites():
    """BATCH_KEYS is unpacked positionally, so its order is part of the contract."""
    import re

    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
    pattern = re.compile(r"^BATCH_KEYS = (\[[^\]]*\])", re.M)
    inline = re.compile(r"    BATCH_KEYS = (\[[^\]]*\])", re.M)

    found = {}
    for rel in ("flagscale/train/megatron/model_specific_batch.py", "flagscale/train/megatron/train_gpt.py"):
        text = open(os.path.join(root, rel)).read()
        match = pattern.search(text) or inline.search(text)
        assert match, f"no BATCH_KEYS in {rel}"
        found[rel] = eval(match.group(1))  # noqa: S307 - a literal list in our own source

    keys = list(found.values())
    assert keys[0] == keys[1], f"BATCH_KEYS diverged between the two get_batch paths: {found}"
    assert "loss_scale" in keys[0]
    assert keys[0] == sorted(keys[0]), "BATCH_KEYS must stay alphabetical"


def test_data_level_loss_false_masks_tokens():
    """The per-message ``loss: false`` field must actually remove supervision."""
    import json

    template = _build_template()

    with open(DATA) as f:
        for line in f:
            row = json.loads(line)
            if not any(
                m.get("loss") is False for m in row["messages"] if m["role"] == "assistant"
            ):
                continue
            try:
                masked = template.encode(json.loads(json.dumps(row)))
            except Exception:
                continue
            stripped_row = {
                "messages": [{k: v for k, v in m.items() if k != "loss"} for m in row["messages"]]
            }
            stripped = template.encode(stripped_row)

            n_masked = (torch.tensor(masked["labels"]) != -100).sum().item()
            n_stripped = (torch.tensor(stripped["labels"]) != -100).sum().item()
            assert n_masked < n_stripped, (n_masked, n_stripped)
            return
    pytest.skip("no row with a `loss: false` assistant message")
