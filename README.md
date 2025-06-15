# Foundational SSM

A framework for analyzing neural data using State Space Models (SSM)

## Setup Guide

### Prerequisites

- Anaconda or Miniconda
- Git
- Access to the project repository

### Environment Setup

1. Clone the repository (if not already done):
   ```bash
   git clone <repository-url>
   cd foundational_ssm
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate foundational_ssm
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```
   

### Project Structure

- `/src/foundational_ssm/`: Core package code
  - `/models/`: Model definitions including S4D implementations
  - `/utils/`: Utility functions
  - `/data_preprocessing/`: Data processing tools
  - `/trainer/`: Training loops and optimization

- `/scripts/`: Executable scripts for data processing and training
- `/notebooks/`: Jupyter notebooks for analysis and visualization



### Troubleshooting
- **Import errors**: Make sure the conda environment is activated and the package is installed
- **CUDA errors**: Check GPU availability with `nvidia-smi`
- **Data not found**: Verify paths in configuration files
- **WandB issues**: Make sure you have proper access to the WandB project
- **Command not found**: After pip installing, make sure your Python's scripts directory is in PATH

If you encounter problems with command-line tools, you may need to add the Python scripts directory to your path:

```bash
# For bash
PYTHON_SCRIPTS_DIR=$(python -c "import site; print(site.getsitepackages()[0] + '/../../bin')")
export PATH=$PATH:$PYTHON_SCRIPTS_DIR
# For csh/tcsh
set PYTHON_SCRIPTS_DIR=`python -c "import site; print(site.getsitepackages()[0] + '/../../bin')"`
setenv PATH "${PATH}:${PYTHON_SCRIPTS_DIR}"
```