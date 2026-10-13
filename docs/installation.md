# Installation Guide

### Option 1: Basic Installation (pip)

```bash
pip install PyMut-Library
```

**Note**: The pip installation provides core functionality for mutation data visualization, but some advanced features may be limited as certain bioinformatics tools are not available through PyPI.

### Option 2: Main environment (Conda)

The main environment is defined in `PyMut_env.yml`:

```bash
# Download the environment file
curl -fsSL https://raw.githubusercontent.com/Molecular-Oncology-Group-UFV/PyMut/main/PyMut_env.yml -o PyMut_env.yml
# or: wget -O PyMut_env.yml https://raw.githubusercontent.com/Molecular-Oncology-Group-UFV/PyMut/main/PyMut_env.yml

# Create and activate the main environment
conda env create -f PyMut_env.yml
conda activate PyMut_env

# Install pyMut from PyPI in this environment
python -m pip install PyMut-Library
```

This environment includes `bcftools`, `htslib`, and `tabix` for workflows that use those command-line tools. It does not include Ensembl VEP.

### VEP annotation environment

Because of dependency conflicts, run pyMut's VEP annotation functions in the separate `pyMut_vep` environment:

```bash
curl -fsSL https://raw.githubusercontent.com/Molecular-Oncology-Group-UFV/PyMut/main/pyMut_vep.yml -o pyMut_vep.yml
conda env create -f pyMut_vep.yml
conda activate pyMut_vep
python -m pip install PyMut-Library
```

Keep `pyMut_vep` active while running VEP annotation; use `conda activate PyMut_env` to return to the main environment.
