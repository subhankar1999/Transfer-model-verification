# Installation Guide

This guide explains how to set up the project environment needed to run `abstract_reuse/transfer_abstract.py`.

## 1. Project Location

Download or clone the project, then open a terminal in the project root directory.

For example, if the project was extracted into a directory named `artifact`, run:

```bash
cd path/to/artifact
```

Replace `path/to/artifact` with the actual location on your machine.

The main executable file is:

```text
abstract_reuse/transfer_abstract.py
```

The script should be run from the project root so that relative paths such as `nnet/`, `CEGAR_NN/`, and `abstract_reuse/` resolve correctly.

## 2. Required Components

The project expects these directories to exist in the project root:

```text
CEGAR_NN/
Marabou/
abstract_reuse/
nnet/
```

`CEGAR_NN` provides the abstraction/refinement verification framework.

`Marabou` and `maraboupy` provide the neural network verification backend used by the project.

## 3. Conda Environment

This project is known to run successfully in a Python 3.8 conda environment named:

```bash
conda activate marabou_py3.8
```

The recommended way to create the environment is to use the included `conda_marabou_py3.8.yml` file. It installs the required Python version and packages.

After creating and activating the environment, verify that Python 3.8 is being used:

```bash
python --version
```

## 4. Required Packages

The required pip packages are listed in:

```text
requirements.txt
```

After activating the conda environment, install them with:

```bash
pip install -r requirements.txt
```

The main packages are:

```text
maraboupy==2.0.0
numpy==1.24.4
pandas==2.0.3
networkx==3.1
matplotlib==3.7.5
pillow==10.4.0
packaging==25.0
python-dateutil==2.9.0.post0
pytz==2025.2
tzdata==2025.2
```

The `requirements.txt` file also includes supporting packages required by these libraries:

```text
contourpy==1.1.1
cycler==0.12.1
fonttools==4.57.0
importlib-resources==6.4.5
kiwisolver==1.4.7
pyparsing==3.1.4
six==1.17.0
zipp==3.20.2
```

## 5. Creating the Environment From File

An exported conda environment file is included:

```text
conda_marabou_py3.8.yml
```

If the environment does not already exist, create it with:

```bash
conda env create -f conda_marabou_py3.8.yml
```

Then activate it:

```bash
conda activate marabou_py3.8
```

This is the preferred installation method because it installs the necessary packages automatically.

## 6. Manual Environment and Package Installation

If you do not use the full YAML file, create a Python 3.8 conda environment manually:

```bash
conda create -n marabou_py3.8 python=3.8.2 pip
conda activate marabou_py3.8
```

Then install all required packages with one command:

```bash
pip install -r requirements.txt
```

## 7. Verify Python Imports

From the project root, run:

```bash
python -c "import numpy, pandas, networkx; from maraboupy import MarabouNetworkNNet; print('Environment OK')"
```

If this prints `Environment OK`, the main Python dependencies are available.

Then verify that project-local imports work:

```bash
python -c "from CEGAR_NN.core.nnet.read_nnet import network_from_nnet_file; from abstract_reuse.transfer_logging import TransferExecutionLogger; print('Project imports OK')"
```

## 8. Run the Main Program

Use the default configuration:

```bash
python abstract_reuse/transfer_abstract.py
```

Or run with explicit arguments:

```bash
python abstract_reuse/transfer_abstract.py \
  -nn 1_6 \
  -pid property_1 \
  -alpha 0.05 \
  -strategy random-discrete \
  -max_refine 10 \
  -log results_new/transfer_execution_log_1.csv
```

The `-nn 1_6` argument resolves to:

```text
nnet/ACASXU_run2a_1_6_batch_2000.nnet
```

## 9. Expected Outputs

Successful runs may generate or update:

```text
transfer_network/
abstract_reuse/transferred_networks/
results_new/
CEGAR_NN/experiments/dev_exp_results_archive/
```

These are generated artifacts and are ignored by the root `.gitignore`.

## 10. Troubleshooting

If `maraboupy` cannot be imported, make sure the `marabou_py3.8` conda environment is active.

If `CEGAR_NN` imports fail, make sure the command is being run from the project root.

If a network file is not found, check that the expected `.nnet` file exists in `nnet/`.

If verification takes a long time, try a smaller network/property combination or reduce refinement settings such as `-max_refine`.
