### wrap_maf_vep_annotate_protein

#### Short description

Run VEP on a MAF file (converted internally to region format) and merge the VEP annotations back into the original MAF, producing an annotated MAF with `VEP_`-prefixed columns.

#### Signature

```python
def wrap_maf_vep_annotate_protein(
    maf_file: Union[str, Path],
    cache_dir: Union[str, Path],
    fasta: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    synonyms_file: Optional[Union[str, Path]] = None,
    assembly: Optional[str] = None,
    version: Optional[str] = None,
    compress: bool = True,
    no_stats: bool = True
) -> Tuple[bool, str]:
```

#### Parameters

| Parameter       | Type                 | Required | Description |
| --------------- | -------------------- | -------- | ----------- |
| `maf_file`      | `str \| Path`        | Yes      | Path to the input MAF file (`.maf` or `.maf.gz`). |
| `cache_dir`     | `str \| Path`        | Yes      | Path to the VEP cache directory. If `assembly`/`version` are not provided, they are auto-extracted from the cache directory name (`homo_sapiens_vep_{version}_{assembly}`). |
| `fasta`         | `str \| Path`        | Yes      | Path to the reference FASTA file used by VEP. |
| `output_file`   | `str \| Path`        | No       | Output VEP annotation file path. If not provided, a time-stamped folder `vep_annotation_<HHMMDDMM>` is created next to the MAF, and a default filename `<maf_stem>_vep_protein.txt` is used. |
| `synonyms_file` | `str \| Path`        | No       | Path to chromosome synonyms file. If not provided, defaults to `<cache_dir>/homo_sapiens/{version}_{assembly}/chr_synonyms.txt`. |
| `assembly`      | `str`                | No       | Genome assembly (e.g., `GRCh38`). If not provided, extracted from `cache_dir`. |
| `version`       | `str`                | No       | VEP cache version (e.g., `110`). If not provided, extracted from `cache_dir`. |
| `compress`      | `bool`               | No       | If `True`, compresses the merged output MAF (`.gz`). Default `True`. |
| `no_stats`      | `bool`               | No       | If `True`, passes `--no_stats` to VEP to disable statistics generation. Default `True`. |

#### Return value

Returns a tuple `(success: bool, info: str)`. On success, `success=True` and `info` contains paths to the VEP output and the merged MAF (e.g., `"VEP folder: <vep_output>, Merged file: <merged_maf>"`). If the merge step fails but VEP ran correctly, `success=True` and `info` includes the merge error message.

#### Exceptions

List only those the user should handle:

* `FileNotFoundError`: if any required path (`maf_file`, `cache_dir`, `fasta`) does not exist.
* `ValueError`: if `assembly`/`version` cannot be extracted from `cache_dir` and were not provided, or if the MAF lacks required columns for region conversion.
* `subprocess.CalledProcessError`: VEP returned a non-zero exit code (captured and reported; function returns `(False, <output_path>)`).

#### Minimal usage example

```python
>>> from pyMut.analysis.vep_annotate import wrap_maf_vep_annotate_protein
>>> ok, info = wrap_maf_vep_annotate_protein(
...     maf_file="mutations.maf.gz",
...     cache_dir="/data/vep_cache/homo_sapiens_vep_110_GRCh38",
...     fasta="/data/reference/GRCh38.fa"
... )
>>> print(ok, info)
```
