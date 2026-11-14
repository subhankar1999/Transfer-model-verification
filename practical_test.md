# Practical Effectiveness Test of Abstract Reuse

This document describes the usage of `abstract_reuse/practical_test.py`.

## Overview

`practical_test.py` is a full experimental pipeline for testing the abstract-reuse idea on ACAS Xu-style neural networks. The script:

1. loads an original network and a transferred network,
2. creates an abstract network from the source model,
3. reuses that abstraction to build an abstract transfer network,
4. verifies the transfer model with a CEGAR-style refinement loop,
5. compares the result with a fresh CEGAR verification of the transferred network.


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

