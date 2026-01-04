## Overview

The test module supports:

1. Unit testing for different backends and operation modes.
2. Functional testing revolves around the training, compression, inference, and service of large language models.
3. Ensuring code style consistency.
4. The test function has been added to the flagscale instruction system.

This section introduces how to use these features.

> **Note**: Please run the test commands in a non-Conda environment or in the `base` Conda environment. During test execution, the system will automatically switch to the corresponding Conda environment, such as `flagscale-train` or `flagscale-inference`.

---

## Unit Testing

### Run Specific Unit Tests Locally

Use script to run:

```bash
tests/scripts/unit_tests/test_subset.sh --backend ${BACKEND} --subset ${SUBSET}
```

Use 'test' command to run:

```bash
flagscale test --unit --backend ${BACKEND} --subset ${SUBSET}
```

Please set the following variables:

- `BACKEND`: Specifies the backend for unit testing, either `megatron` or `flagscale`.
- `SUBSET`: The directory for unit tests. Check the directories within `tests/unit_tests` for specific folders. Note: `./` represents the root directory of the above folders.

### Run All Unit Tests Locally

Use script to run:

```bash
tests/scripts/unit_tests/test_all.sh
```

Use 'test' command to run:

```bash
flagscale test --unit-all
```

### Run Unit Tests Online

When you create a PR using your forked repository, the testing workflow will automatically trigger. Find the corresponding action for your [PR](https://github.com/FlagOpen/FlagScale/pulls) in [All Tests Nvidia](https://github.com/FlagOpen/FlagScale/actions/workflows/all-tests-nvidia.yml) to view the results. The process for other chips is similar.

### Adding Unit Tests

For `flagscale`, the test path is `tests/unit_tests`. For `megatron`, it's the native unit tests of Megatron-LM-FL.

- **Adding a Single Test Function**

  - Directly add a function named `test_${NEW_FUNCTION}` inside the appropriate test file. `NEW_FUNCTION` refers to the name of the new test function.

- **Adding a Single Unit Test File**

  - Directly add a file named `test_${NEW_FILE}.py` in the appropriate directory. `NEW_FILE` refers to the name of the new test, which should include the `test_${NEW_FUNCTION}` function.

- **Adding a Unit Test Directory**

  1. Add a test directory named `${NEW_FOLD}` and the corresponding files in the appropriate location, which should include `test_${NEW_FILE}.py` and the `test_${NEW_FUNCTION}` function.

  2. Update the configuration file `tests/scripts/unit_tests/config.yml` to include configuration for the directory, specifying `ignore`, `type`, and `depth` as needed. Unspecified parameters will default to pre-defined settings. Below is the **configuration file explanation:**

     ```yaml
     # backend: The backend for unit testing, either flagscale or megatron
     megatron:
       # Set the environment required before running unit tests
       set_environment:
         - source /root/miniconda3/etc/profile.d/conda.sh
         - conda activate flagscale-train
         - python tools/patch/unpatch.py --backend Megatron-LM
         - cd flagscale/train
         - export PYTHONPATH=../..:$PYTHONPATH
         - export NVTE_FLASH_ATTN=0
         - export NVTE_FUSED_ATTN=0
         - ulimit -n 65535
       # Specify the target folder for test coverage
       coverage_fold:
         core
       # Select different tests for different test directories
       subset:
         ...
         # Use default configuration if not shown
         dist_checkpointing:
           # Files to ignore during testing
           ignore: models/test_mamba.py
         models:
           ignore: test_mamba_model.py
         transformer/moe:
           # Test mode:
           # batch (default): Run all test files at once
           # single: Run each test file individually
           # NOTE: Batch mode runs faster, single mode avoids environment interference among tests
           type: single
           ignore: test_upcycling.py
         transformer:
           # Test depth
           # all (default): All test files within the directory
           # Integer: Test files at the specified path depth
           # NOTE: Useful for running test files within a folder, rather than in subdirectories
           depth: 1
         ...
     ```

  3. Online Test Configuration

     Modify the workflow configuration in `.github/workflows/all-tests-nvidia.yml` to activate online testing, the process for other chips is similar.:

     ```yaml
     ...

     # Megatron Unit Tests with Matrix
     megatron-unit-tests:
       needs:
         - set-env
         uses: ./.github/workflows/unit-tests-nvidia.yml
       strategy:
         matrix:
           subset:
             # Add your new folder if you have a new test directory
             - {NEW_FOLD}
             - data
             - ...
             - ./
       name: "megatron-${{ matrix.subset == './' && 'root' || matrix.subset }}"
       with:
         backend: megatron
         subset: ${{ matrix.subset }}

     ...
     ```

---

## Functional Testing

### Run Specific Functional Tests Locally

Use script to run:

```bash
tests/scripts/functional_tests/test_task.sh --type ${TYPE} --task ${TASK}
```

Use 'test' command to run:

```bash
flagscale test --functional --type ${TYPE} --task ${TASK}
```

Please set the following variables:

- `TYPE`: The type of functional testing, supporting `train` or `hetero_train` or `inference` or `serve`.
- `TASK`: The task used for functional testing, in conjunction with `TYPE`. Specific tasks can be found under the `tests/functional_tests/test_cases` directory.

### Run All Functional Tests Locally

Use script to run:

```bash
tests/scripts/functional_tests/test_all.sh
```

Use 'test' command to run:

```bash
flagscale test --functional-all
```

### Run Functional Tests Online

Find the corresponding action for your [PR](https://github.com/FlagOpen/FlagScale/pulls) in [All Tests Nvidia](https://github.com/FlagOpen/FlagScale/actions/workflows/all-tests-nvidia.yml) to view the results. The process for other chips is similar.

### Adding Functional Tests

The following shows how to add a functional test for training. Other similar tasks are similar.

1. Update the functional test configuration file `tests/scripts/functional_tests/config.yml` to include relevant experiment configurations:

   ```yaml
   ...
   train:
     # Models used
     aquila:
       # Parallel modes
       - tp2_pp2
       - tp4_pp2
   ...
   ```

2. Add configuration files and test results in the appropriate directory. Directory file structure:

   ```bash
   tests/functional_tests/test_cases/train/aquila
   ├── conf
   │   ├── tp2_pp2.yaml # Environment configuration file
   │   ├── tp4_pp2.yaml
   │   └── train
   │       ├── tp2_pp2.yaml # Model configuration file
   │       └── tp4_pp2.yaml
   ├── results_gold
   │   ├── tp2_pp2.json # Test result file
   │   └── tp4_pp2.json
   └── results_test
   ```

   *Note: We have included data and task files that you can use. For more details, consult the training configuration file of the respective test case. If you need to add your own test data or task files, please contact us.*

3. Modify the yml configuration file in the workflow to enable online testing:

   ```yaml
   ...

   # Functional Tests with Model and Type Matrix
   functional-tests-train:
     needs:
       - megatron-unit-tests
       - flagscale-unit-tests
     uses: ./.github/workflows/functional-tests.yml
     strategy:
       matrix:
         task:
           # Add the new task if applicable
           - {NEW_TASK}
           - aquila
           - mixtral
     name: "train-${{ matrix.task }}"
     with:
       task: ${{ matrix.task }}
       type: train

   ...
   ```

---

## Code Style Check

1. **Run via Pre-commit:**

   ```bash
   pre-commit install
   ```
   Code format checks will run automatically upon committing.

2. **Online Format Check:**

   Find the corresponding action for your [PR](https://github.com/FlagOpen/FlagScale/pulls) in the [Format Check](https://github.com/FlagOpen/FlagScale/actions/workflows/format.yml).
