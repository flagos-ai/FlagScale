# Copyright (c) 2026, BAAI. All rights reserved.

"""CPU unit tests for flagscale.models.mimo.bridge.contracts.

Covers the grid communicator contract registry:

- register/get round-trip (keyed and single-registration default),
- fail-fast paths: zero registrations, multiple registrations without an
  explicit key, unknown key, duplicate key, non-contract payload,
- the Qwen3.5 provider's import-time registration: the registered contract
  must carry the values previously hardcoded in ``bridge.training`` (pure
  refactor - no behavioral change).

Run (inside the container, from the FlagScale repo root):

    source /root/miniconda3/bin/activate flagscale
    PYTHONPATH=/workspace/multimodal/Megatron-LM-FL python -m pytest \
        tests/unit_tests/models/test_mimo_grid_contracts.py
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_MEGATRON_REPO = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir, "Megatron-LM-FL"))
if os.path.isdir(_MEGATRON_REPO) and _MEGATRON_REPO not in sys.path:
    sys.path.insert(0, _MEGATRON_REPO)

# Importing the provider registers its contract at module import time.
import flagscale.models.mimo.bridge.providers.qwen35  # noqa: F401  (registration side effect)
from flagscale.models.mimo.bridge import contracts
from flagscale.models.mimo.bridge.contracts import (
    SBH_DIM_MAPPING,
    GridCommunicatorContract,
    get_grid_communicator_contract,
    register_grid_communicator_contract,
)

_CONTRACT_A = GridCommunicatorContract(
    topology={"enc": ["language"], "language": []},
    dim_mapping=dict(SBH_DIM_MAPPING),
    module_output_ndim={"enc": 2, "language": 3},
)
_CONTRACT_B = GridCommunicatorContract(
    topology={"audio": ["language"], "language": []},
    dim_mapping=dict(SBH_DIM_MAPPING),
    module_output_ndim={"audio": 2, "language": 3},
)


class _RegistryIsolatedTestCase(unittest.TestCase):
    """Save/clear/restore the process-global registry around each test."""

    def setUp(self):
        self._saved = dict(contracts._GRID_COMMUNICATOR_CONTRACTS)
        contracts._GRID_COMMUNICATOR_CONTRACTS.clear()

    def tearDown(self):
        contracts._GRID_COMMUNICATOR_CONTRACTS.clear()
        contracts._GRID_COMMUNICATOR_CONTRACTS.update(self._saved)


class TestRegistryMechanics(_RegistryIsolatedTestCase):
    def test_keyed_roundtrip(self):
        register_grid_communicator_contract("model_a", _CONTRACT_A)
        self.assertIs(get_grid_communicator_contract("model_a"), _CONTRACT_A)

    def test_single_registration_returned_without_key(self):
        register_grid_communicator_contract("model_a", _CONTRACT_A)
        self.assertIs(get_grid_communicator_contract(), _CONTRACT_A)

    def test_zero_registrations_fail_fast(self):
        with self.assertRaisesRegex(RuntimeError, "import the model's providers module"):
            get_grid_communicator_contract()

    def test_multiple_registrations_require_key(self):
        register_grid_communicator_contract("model_a", _CONTRACT_A)
        register_grid_communicator_contract("model_b", _CONTRACT_B)
        with self.assertRaisesRegex(RuntimeError, "multiple grid communicator contracts"):
            get_grid_communicator_contract()
        self.assertIs(get_grid_communicator_contract("model_a"), _CONTRACT_A)
        self.assertIs(get_grid_communicator_contract("model_b"), _CONTRACT_B)

    def test_unknown_key_fail_fast(self):
        register_grid_communicator_contract("model_a", _CONTRACT_A)
        with self.assertRaisesRegex(KeyError, "model_x"):
            get_grid_communicator_contract("model_x")

    def test_duplicate_key_fail_fast(self):
        register_grid_communicator_contract("model_a", _CONTRACT_A)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_grid_communicator_contract("model_a", _CONTRACT_A)

    def test_non_contract_payload_fail_fast(self):
        with self.assertRaises(TypeError):
            register_grid_communicator_contract("model_a", {"topology": {}})


class TestQwen35RegisteredContract(unittest.TestCase):
    """The provider's import-time registration carries the values that used to
    be hardcoded as ``QWEN35_GRID_*`` constants in ``bridge.training``."""

    def test_default_lookup_returns_qwen35_contract(self):
        contract = get_grid_communicator_contract()
        self.assertIs(contract, get_grid_communicator_contract("qwen35"))

    def test_contract_values_match_legacy_constants(self):
        contract = get_grid_communicator_contract("qwen35")
        self.assertEqual(contract.topology, {"images": ["language"], "language": []})
        self.assertEqual(contract.dim_mapping, {"s": 0, "b": 1, "h": 2})
        self.assertIs(contract.dim_mapping, SBH_DIM_MAPPING)
        self.assertEqual(contract.module_output_ndim, {"images": 2, "language": 3})


if __name__ == "__main__":
    unittest.main()
