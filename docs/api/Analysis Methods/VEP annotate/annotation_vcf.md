### wrap_vcf_vep_annotate_unified

#### Short description

Unified VEP runner for VCFs that lets you combine protein, gene, and variant-class annotations in one call. Creates a VCF annotated by VEP with options like `--protein`, `--symbol`, `--nearest`, and `--variant_class` as requested.

#### Signature

```python
def wrap_vcf_vep_annotate_unified(
    vcf_file: Union[str, Path],
    cache_dir: Union[str, Path],
    fasta: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    synonyms_file: Optional[Union[str, Path]] = None,
    assembly: Optional[str] = None,
    version: Optional[str] = None,
    no_stats: bool = True,
    annotate_protein: bool = False,
    annotate_gene: bool = False,
    annotate_variant_class: bool = False,
    distance: Optional[int] = None
) -> Tuple[bool, str]:
```

#### Parameters

| Parameter                 | Type                 | Required | Description |
| ------------------------- | -------------------- | -------- | ----------- |
| `vcf_file`                | `str \| Path`        | Yes      | Path to the input VCF/VCF.GZ to annotate. |
| `cache_dir`               | `str \| Path`        | Yes      | Path to the VEP cache directory. If `assembly`/`version` are not provided, they are auto-extracted from the cache directory name (`homo_sapiens_vep_{version}_{assembly}`). |
| `fasta`                   | `str \| Path`        | Yes      | Path to the reference FASTA used by VEP. |
| `output_file`             | `str \| Path`        | No       | Output VCF path. If not provided, a time-stamped folder `vep_annotation_<HHMMDDMM>` is created next to the VCF, and a descriptive filename `<vcf_stem>_vep_<annotations>.vcf` is used. |
| `synonyms_file`           | `str \| Path`        | No       | Path to chromosome synonyms file. If not provided, defaults to `<cache_dir>/homo_sapiens/{version}_{assembly}/chr_synonyms.txt`. |
| `assembly`                | `str`                | No       | Genome assembly (e.g., `GRCh38`). If not provided, extracted from `cache_dir`. |
| `version`                 | `str`                | No       | VEP cache version (e.g., `110`). If not provided, extracted from `cache_dir`. |
| `no_stats`                | `bool`               | No       | If `True`, passes `--no_stats` to VEP to disable statistics generation. Default `True`. |
| `annotate_protein`        | `bool`               | No       | If `True`, includes protein-level annotation (`--protein --uniprot --domains --symbol`). Default `False`. |
| `annotate_gene`           | `bool`               | No       | If `True`, adds gene symbol annotation (`--symbol`). If `distance` is set, adds nearest-gene search (`--nearest symbol --distance <N>`). Default `False`. |
| `annotate_variant_class`  | `bool`               | No       | If `True`, adds variant class annotation (`--variant_class`). Default `False`. |
| `distance`                | `int`                | No       | Optional distance (in bp) for nearest-gene search. Only used when `annotate_gene=True`. |

#### Return value

Returns a tuple `(success: bool, info: str)`. On success, `success=True` and `info` includes the path to the VEP-annotated VCF (e.g., `"VEP output file: <path>"`).

#### Exceptions

List only those the user should handle:

* `ValueError`: if none of the annotation toggles are enabled (`annotate_protein`, `annotate_gene`, `annotate_variant_class`).
* `FileNotFoundError`: if any required path (`vcf_file`, `cache_dir`, `fasta`) does not exist.
* `ValueError`: if `assembly`/`version` cannot be extracted from `cache_dir` and were not provided.
* `subprocess.CalledProcessError`: VEP returned a non-zero exit code (captured and reported; function returns `(False, <output_path>)`).

#### Minimal usage example

```python
>>> from pyMut.analysis.vep_annotate import wrap_vcf_vep_annotate_unified
>>> ok, info = wrap_vcf_vep_annotate_unified(
...     vcf_file="tumor.vcf.gz",
...     cache_dir="/data/vep_cache/homo_sapiens_vep_110_GRCh38",
...     fasta="/data/reference/GRCh38.fa",
...     annotate_protein=True,
...     annotate_gene=True,
...     distance=5000,
...     annotate_variant_class=True,
...     no_stats=True
... )
>>> print(ok, info)
```
