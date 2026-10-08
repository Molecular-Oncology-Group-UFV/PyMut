# pyMut 🧬

[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/pymut-bio.svg)](https://badge.fury.io/py/pymut-bio)

A Python library designed for the preprocessing, analysis and visualization of somatic genetic variants in standard formats such as VCF and MAF.

## 🚀 Quick Start

### Installation

#### Option 1: Basic Installation (pip)

```bash
pip install PyMut-Library
```

**Note**: The pip installation provides core functionality for mutation data visualization, but some advanced features may be limited as certain bioinformatics tools are not available through PyPI.

#### Option 2: Main environment (Conda)

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

#### VEP annotation environment

Because of dependency conflicts, run pyMut's VEP annotation functions in the separate `pyMut_vep` environment:

```bash
curl -fsSL https://raw.githubusercontent.com/Molecular-Oncology-Group-UFV/PyMut/main/pyMut_vep.yml -o pyMut_vep.yml
conda env create -f pyMut_vep.yml
conda activate pyMut_vep
python -m pip install PyMut-Library
```

Keep `pyMut_vep` active while running VEP annotation; use `conda activate PyMut_env` to return to the main environment.

## 📚 Documentation

- **[Complete Documentation](https://Molecular-Oncology-Group-UFV.github.io/PyMut/)** - Comprehensive guides and API reference
- **[Installation Guide](https://Molecular-Oncology-Group-UFV.github.io/PyMut/installation/)** - Detailed installation instructions
- **[API Reference](https://Molecular-Oncology-Group-UFV.github.io/PyMut/api/Core/pymutation_class/)** - Complete API documentation
- **[Examples](https://Molecular-Oncology-Group-UFV.github.io/PyMut/examples/data/input_read_maf/#example-loading-tcga-laml-maf-file)** - Real-world usage examples


## 📋 Requirements

| Package | Version declared in `pyproject.toml` |
| ------- | ------------------------------------ |
| duckdb | `>=1.3.2, <2.0.0` |
| fastparquet | `>=2024.11.0, <2025.0.0` |
| matplotlib | `>=3.10.3, <4.0.0` |
| mkdocs | `>=1.6.1, <2.0.0` |
| numpy | `==1.26.*` |
| pandas | `>=2.3.1, <3.0.0` |
| pyarrow | `>=14.0.2, <15.0.0` |
| pyensembl | `>=2.3.13, <3.0.0` |
| pyfaidx | `>=0.8.1.4, <0.9.0.0` |
| requests | `>=2.32.4, <3.0.0` |
| scikit-learn | `>=1.7.1, <2.0.0` |
| scipy | `==1.11.*` |
| seaborn | `>=0.13.2, <0.14.0` |
| urllib3 | `>=2.5.0, <3.0.0` |


## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🎯 Comparison with Other Tools

| FUNCTIONAL CRITERIA                         | PYMUT (PROPOSAL)   | MUTSCAPE              | MAFTOOLS              |
|---------------------------------------------|--------------------|-----------------------|-----------------------|
| Input formats                               | VCF & MAF (native) | MAF                   | MAF                   |
| VEP annotation                              | ✓                  |                       |                       |
| Genomic range filtering                     | ✓                  | ✓                     | ✓                     |
| PASS category variant filtering             | ✓                  | ✓                     |                       |
| Sample filtering                            | ✓                  |                       | ✓                     |
| Tissue expression filtering                 | ✓                  | ✓                     |                       |
| File format transformation                  | ✓                  | ✓ *(VCF to MAF only)* | ✓ *(VCF to MAF only)* |
| File combination                            | ✓                  | ✓                     |                       |
| Significantly mutated genes (SMG) detection | ✓                  | ✓                     |                       |
| Cancer-related gene annotation              | ✓                  | ✓                     |                       |
| Tumor mutational burden (TMB) calculation   | ✓                  | ✓                     |                       |
| Mutational signature identification         | ✓                  |                       |                       |
| Medical implications mutation annotation    | ✓                  | ✓                     |                       |
| PFAM annotation support                     | ✓                  |                       | ✓                     |
| Summary Plot                                | ✓                  |                       | ✓                     |
| Oncoplot                                    | ✓                  |                       | ✓                     |
| Lollipop Plot                               | ✓                  |                       | ✓                     |
| Somatic Interactions Plot                   | ✓                  |                       | ✓                     |
| Mutational Signature Analysis               | ✓                  | ✓                     |                       |
| CoMut Plot                                  | ✓                  | ✓                     |                       |
