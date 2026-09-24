#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flow_pymut.py

pyMut side of the filter-concordance benchmark.

For EACH of two independent TCGA cohorts (TCGA-LAML and TCGA-PAAD), this
script applies two INDEPENDENT filters and writes each result to its own
MAF file:

  1) Chromosome-based filter: keep chromosomes 1, 2, 3, 17 (same filter,
     applied independently to each cohort).
  2) Gene-locus / position-based filter (GRCh37 coordinates):
       - TCGA-LAML  -> DNMT3A locus, chromosome 2
       - TCGA-PAAD  -> TP53 locus,   chromosome 17
     A different gene was chosen per cohort because it is a clinically
     relevant, recurrently mutated gene for that tumor type (DNMT3A in
     AML, TP53 in pancreatic adenocarcinoma), which keeps the region
     filter non-trivial (i.e. it actually removes variants) in both
     cohorts.

Counterpart: flow_maftools.R (same cohorts, same two filters, done with
maftools).
Comparison + figure: compare_and_plot.py (combines both cohorts into a
single publication figure).

Paths (resolved relative to this script, regardless of the working dir):
  - Input MAFs:   <project_root>/data/
  - Output MAFs:  validation/filter_concordance/results/
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

# pyMut is assumed to be installed in the active environment
# (e.g. `pip install -e .` from the pyMut source tree).
from pyMut import read_maf  # noqa: E402

# --------------------------------------------------------------------------
# Compatibility shim
# --------------------------------------------------------------------------
# The installed pyMut version calls pyarrow.compute.utf8_replace_substring()
# inside to_maf()'s large-dataset (PyArrow) export path. That function does
# not exist in some pyarrow releases -- only the equivalent
# pyarrow.compute.replace_substring() does, with the same argument order
# (array, pattern, replacement). Without this shim, to_maf() crashes with
# "AttributeError: module 'pyarrow.compute' has no attribute
# 'utf8_replace_substring'" as soon as a filtered MAF is large enough to
# trigger that code path (e.g. TCGA-PAAD, but not the much smaller
# TCGA-LAML). This only adds the missing alias if it is not already
# present, so it is a no-op on pyarrow versions that do have the function.
import pyarrow.compute as _pc  # noqa: E402

if not hasattr(_pc, "utf8_replace_substring"):
    _pc.utf8_replace_substring = _pc.replace_substring

# The same large-dataset (>10,000 rows) PyArrow branch inside to_maf() also
# crashes with "ValueError: Found non-unique column index" when the source
# MAF already contains columns that to_maf() re-adds via
# pyarrow.Table.append_column() (e.g. Chromosome, Start_Position,
# NCBI_Build, dbSNP_RS) -- append_column() does not check for existing
# names, so pandas ends up with duplicate columns and refuses to convert.
# Rather than patching that branch function by function, force to_maf() to
# always take its pandas-based export path instead, which is the one
# already exercised successfully for the smaller TCGA-LAML cohort and is
# only marginally slower for datasets in the tens-of-thousands range.
import pyMut.output as _pymut_output  # noqa: E402

_pymut_output.HAS_PYARROW = False

# --------------------------------------------------------------------------
# Paths (relative to the validation/ folder -- run from there)
# --------------------------------------------------------------------------

# Folder containing this script (validation/filter_concordance/) and the
# project root (two levels up). All paths below are resolved from here, so
# the script works no matter which directory it is launched from.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Directory containing the raw input MAF files for both cohorts.
DATA_DIR = PROJECT_ROOT / "data"

# Directory where the filtered MAF files are written. Shared with
# flow_maftools.R and consumed by compare_and_plot.py.
OUTPUT_DIR = SCRIPT_DIR / "results"

# Genome assembly used to interpret the region filter below. Both input
# MAFs are annotated on GRCh37/hg19 coordinates.
ASSEMBLY = "37"

# Chromosome-based filter: identical set of chromosomes for both cohorts.
# Keep this in sync with flow_maftools.R (CHROM_FILTER).
CHROM_FILTER = ["1", "2", "3", "17"]


@dataclass(frozen=True)
class CohortConfig:
    """Configuration for one TCGA cohort processed by this benchmark.

    Keep the values below in sync with the matching ``COHORTS`` entry in
    flow_maftools.R.
    """

    name: str  # short id used in output filenames, e.g. "laml"
    label: str  # display name, e.g. "TCGA-LAML"
    maf_input: Path  # path to the raw input MAF for this cohort
    chrom_filter: Sequence[str]  # chromosomes kept by the chromosome filter
    region_gene: str  # gene symbol used for the region filter, for labeling
    region_chrom: str  # chromosome of the region filter (GRCh37)
    region_start: int  # region filter start position (GRCh37, 1-based)
    region_end: int  # region filter end position (GRCh37, 1-based)


# Cohort definitions. GRCh37/hg19 gene-body coordinates:
#   - DNMT3A: chr2:25,455,845-25,565,459 (RefSeq NM_022552, minus strand)
#   - TP53:   chr17:7,571,720-7,590,868  (RefSeq NM_000546, minus strand)
COHORTS: List[CohortConfig] = [
    CohortConfig(
        name="laml",
        label="TCGA-LAML",
        maf_input=DATA_DIR / "tcga_laml.maf.gz",
        chrom_filter=CHROM_FILTER,
        region_gene="DNMT3A",
        region_chrom="2",
        region_start=25_455_845,
        region_end=25_565_459,
    ),
    CohortConfig(
        name="paad",
        label="TCGA-PAAD",
        maf_input=DATA_DIR / "PAAD-TB.final_analysis.clean.maf",
        chrom_filter=CHROM_FILTER,
        region_gene="TP53",
        region_chrom="17",
        region_start=7_571_720,
        region_end=7_590_868,
    ),
]


def process_cohort(cohort: CohortConfig) -> None:
    """Apply the chromosome filter and the gene-locus filter to one cohort
    using pyMut, and write both filtered MAFs to ``OUTPUT_DIR``.
    """
    print(f"\n=== {cohort.label} (pyMut) ===")

    if not cohort.maf_input.exists():
        raise FileNotFoundError(
            f"Input MAF not found for cohort '{cohort.label}': {cohort.maf_input}"
        )

    # 1) Load ----------------------------------------------------------
    print(f"Loading: {cohort.maf_input}")
    py_mut = read_maf(str(cohort.maf_input), consolidate_variants= False, assembly=ASSEMBLY)
    print(f"Total variants loaded: {len(py_mut.data)}")

    # 2) Chromosome-based filter ----------------------------------------
    print(f"Applying chromosome filter: {list(cohort.chrom_filter)}")
    chrom_filtered = py_mut.filter_by_chrom_sample(chrom=list(cohort.chrom_filter))
    print(f"Variants after chromosome filter: {len(chrom_filtered.data)}")
    chrom_out = OUTPUT_DIR / f"tcga_{cohort.name}_chr_filter_py.maf"
    chrom_filtered.to_maf(output_path=str(chrom_out))
    print(f"Written: {chrom_out}")

    # 3) Gene-locus / position-based filter ------------------------------
    print(
        f"Applying {cohort.region_gene} region filter: "
        f"chr{cohort.region_chrom}:{cohort.region_start}-{cohort.region_end}"
    )
    region_filtered = py_mut.region(
        chrom=cohort.region_chrom,
        start=cohort.region_start,
        end=cohort.region_end,
    )
    print(f"Variants after {cohort.region_gene} region filter: {len(region_filtered.data)}")
    region_out = OUTPUT_DIR / f"tcga_{cohort.name}_{cohort.region_gene.lower()}_region_py.maf"
    region_filtered.to_maf(output_path=str(region_out))
    print(f"Written: {region_out}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for cohort in COHORTS:
        process_cohort(cohort)
    print("\nDone. Now run flow_maftools.R, then compare_and_plot.py.")


if __name__ == "__main__":
    main()
