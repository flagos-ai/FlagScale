# FlagScale Test Suite

## Quick Start

```bash
# Run all tests
bash tests/test_utils/runners/run_tests.sh

# Run only unit tests
bash tests/test_utils/runners/run_tests.sh --type unit

# Run only functional tests
bash tests/test_utils/runners/run_tests.sh --type functional

# Run specific functional test
bash tests/test_utils/runners/run_functional_tests.sh --task train --model aquila
bash tests/test_utils/runners/run_functional_tests.sh --task train --model aquila --list tp2_pp2,tp4_pp2
```

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
