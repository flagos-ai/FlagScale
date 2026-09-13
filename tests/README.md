# FlagScale Test Suite

## Prepare a Local Training Runtime

Run these commands from the FlagScale repository root, inside a Linux machine or
container with the target accelerators and their drivers available. Activate the
platform's training Python environment first. The test runners do not build
Megatron-LM-FL or TransformerEngine-FL for you.

- Use `.github/configs/<platform>.yml` for the CI image, dependency refs and
  build/runtime environment settings. Apply the configured runtime environment
  for your platform; do not reuse another platform's wheel or vendor settings.
- Use `tests/test_utils/config/platforms/<platform>.yaml` for device names,
  test selections and process counts. Supported platforms include `cuda`,
  `ascend`, `enflame`, `hygon`, `metax`, `musa` and `kunlunxin`.
- The active environment must contain compatible PyTorch, vendor plugins,
  TransformerEngine-FL and the test dependencies. For a new environment, the
  installation entry point is `bash tools/install/install.sh --help`.
- Megatron-LM-FL must be available as a prepared install directory or a compatible
  local source checkout. The directory below must contain `megatron/core`.
  To reproduce a CI failure, use the exact dependency commits and runtime from
  that run, not an arbitrary preinstalled version.

```bash
export PROJECT_ROOT="$PWD"
export PLATFORM=cuda
export DEVICE=a100
export CI_PYTHON_BIN="$(command -v python)"

# Replace this path with your prepared install directory or source checkout.
export MEGATRON_INSTALL_DIR=/absolute/path/to/megatron-lm-fl-install
test -d "$MEGATRON_INSTALL_DIR/megatron/core"
export PYTHONNOUSERSITE=1
source .github/scripts/set_env_common.sh
ci_resolve_python_bin
ci_configure_training_pythonpath

# Both commands must resolve to the activated training environment.
"$CI_PYTHON_BIN" --version
command -v torchrun
```

For local runs, keep these exports in the same shell used to launch tests.
Running a setup script with `bash` does not export variables back to its parent
shell. In CI, setup steps instead persist variables through `GITHUB_ENV`.

The functional runner automatically selects
`${GITHUB_WORKSPACE:-$PROJECT_ROOT/..}/megatron-lm-fl-install` when that directory
exists. Ensure it is the intended runtime; otherwise it takes precedence over
your local source selection. With no prepared directory there, the local runner
retains the dependency paths configured above. CI workflows require restored
prepared dependencies and fail before testing if they are missing.

Check the actual imports before starting distributed tests:

```bash
"$CI_PYTHON_BIN" - <<'PY'
import megatron.core
import megatron.training
import transformer_engine
from megatron.training.arguments_fs import FSTrainArguments

print("training:", megatron.training.__file__)
print("core:", megatron.core.__file__)
print("TE-FL:", transformer_engine.__file__)
PY
```

`megatron.training` should resolve inside this checkout's
`flagscale/train/megatron/training`; `megatron.core` should resolve inside the
selected Megatron-LM-FL runtime. Import training modules through
`megatron.training.*`, not `flagscale.train.megatron.training.*`: loading the
same files under a second package name changes relative import resolution and
can duplicate global state.

## Run Tests

`--platform` is required by the unified runner. Use a device listed in your
platform YAML; omitting `--device` runs the configured devices for that platform.
The examples below use the `PLATFORM` and `DEVICE` exports above.

```bash
# Run only unit tests
bash tests/test_utils/runners/run_tests.sh \
  --platform "$PLATFORM" --device "$DEVICE" --type unit

# Run configured training functional tests
bash tests/test_utils/runners/run_tests.sh \
  --platform "$PLATFORM" --device "$DEVICE" --type functional --task train

# Run one configured CUDA/A100 case (requires enough devices for TP2/PP2)
bash tests/test_utils/runners/run_functional_tests.sh \
  --platform cuda --device a100 --task train --model aquila --list tp2_pp2

# Run all configured tests, including non-training tasks if present in the YAML
bash tests/test_utils/runners/run_tests.sh --platform "$PLATFORM" --device "$DEVICE"
```

Before functional tests, inspect the selected case under
`tests/functional_tests/<task>/<model>/conf/` and provide its datasets, tokenizer
files, checkpoints and output directories. CI-specific absolute paths must be
mounted on the local machine or changed in the local case configuration. The
runner does not download these assets. Match the case's parallelism to the
available device count. Heterogeneous cases also require their configured hosts
and communication setup.

For unit-test coverage, use the unit runner directly:

```bash
bash tests/test_utils/runners/run_unit_tests.sh \
  --platform "$PLATFORM" --device "$DEVICE" --coverage-dir "$PROJECT_ROOT/coverage"
```

CI script-only regression tests can also run without accelerator initialization:

```bash
python3 -m pytest --noconftest --import-mode=importlib \
  tests/unit_tests/test_ci_training_runtime_scripts.py -q
```

This narrow check requires `pytest` and `PyYAML`; it does not validate the
training runtime or replace distributed unit/functional tests.

## Directory Structure

```
tests/
├── functional_tests/
│   ├── train/                  # Training tests
│   │   ├── aquila/
│   │   │   ├── conf/           # Test configs (*.yaml)
│   │   │   └── gold_values/    # Expected results (*.json)
│   │   ├── deepseek/
│   │   └── mixtral/
│   └── hetero_train/           # Heterogeneous training tests
├── unit_tests/                 # Unit tests (test_*.py)
└── test_utils/
    ├── config/platforms/       # Platform configs (cuda.yaml, default.yaml)
    └── runners/                # Test runners (*.sh, *.py)
```

## Adding Tests

### Functional Test
1. Add config: `functional_tests/<task>/<model>/conf/<test_name>.yaml`
2. Add gold values: `functional_tests/<task>/<model>/gold_values/<test_name>.json`

### Unit Test
Add test file: `unit_tests/test_<name>.py`
