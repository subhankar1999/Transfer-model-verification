# Practical Transfer-and-Verification Workflow

This document describes the purpose and execution flow of `abstract_reuse/practical_test.py`.

## Overview

`practical_test.py` is a full experimental pipeline for testing the abstract-reuse idea on ACAS Xu-style neural networks. The script:

1. loads an original network and a transferred network,
2. perturbs edge weights using a configurable strategy,
3. creates an abstract network from the source model,
4. reuses that abstraction to build an abstract transfer network,
5. verifies the transfer model with a CEGAR-style refinement loop,
6. compares the result with a fresh CEGAR verification of the transferred network.

The script is designed to evaluate whether a reused abstraction can verify a transfer network efficiently while preserving correctness.

---

## Purpose

The program implements a research workflow for the following scenario:

- Start from a validated source network.
- Apply a transfer or perturbation to network weights.
- Reuse the structure or abstraction learned from the original network.
- Verify the transferred model without rebuilding the entire abstraction from scratch.
- Compare against the baseline of verifying the transfer model independently.

This is useful when testing transfer robustness, abstraction reuse, and CEGAR-based verification acceleration.

---

## Script structure

The file contains the following major parts:

### 1. Utility and timeout helpers

The script defines:

- `TimeoutException`
- `timeout_handler()`
- `run_with_timeout()`

These are used to stop long-running verification steps and continue the process without hanging the whole run.

### 2. `NetworkTransferer`

This is the main class in the file. Its responsibilities include:

- loading the original and transferred .nnet networks,
- perturbing weights with configurable strategies,
- extracting abstract-network node mappings,
- constructing the abstract transfer network,
- running verification loops,
- saving intermediate and final results.

### 3. CLI entry point

The script exposes a command-line interface via `parse_args()` and `main()`. It is intended to be run as a standalone program.


## Command-line arguments

The script accepts the following arguments:

```bash
-nn, --nnet_file
    Path to the original network file.

--tnnet_file
    Path to the transferred network file.

-pid, --property_id
    Property ID to verify, such as property_1.

-alpha, --alpha
    Maximum perturbation magnitude used in transfer step.

-strategy
    Weight-perturbation strategy. Choices:
    random, worst_case, best_case, random-discrete

-r, --results_dir
    Directory used for CEGAR result artifacts.

-o, --output_dir
    Directory for saved network artifacts.

-max_refine, --max_refinement_steps
    Maximum refinement iterations during verification.

-log, --log_csv
    CSV log file for execution statistics.

-layer, --last_layers
    Number of layers changed in transfer.

-v, --verbose
    Enable more verbose logging.
```

---

## Example usage

Run from the repository root:

```bash
python abstract_reuse/practical_test.py \
  --nnet_file nnet/ACASXU_run2a_1_1_batch_2000.nnet \
  --tnnet_file nnet/ACASXU_run2a_1_6_batch_2000.nnet \
  --property_id property_1 \
  --alpha 0.05 \
  --strategy random-discrete \
  --results_dir CEGAR_NN/experiments/dev_exp_results_archive \
  --output_dir abstract_reuse/transferred_networks \
  --max_refinement_steps 10 \
  --log_csv sample_result.csv
```

