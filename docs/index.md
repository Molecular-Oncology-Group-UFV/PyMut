# pyMut 🧬

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/pymut-bio.svg)](https://badge.fury.io/py/pymut-bio)

A Python library for gene mutation analysis and visualisation

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
| File output                                 | ✓                  |                       |                       |
| File combination                            | ✓                  | ✓                     |                       |
| Significantly mutated genes (SMG) detection |                    | ✓                     |                       |
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

This project is licensed under the **MIT License** - see the [LICENSE](https://github.com/Molecular-Oncology-Group-UFV/PyMut/blob/main/LICENSE) file for details.
