#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_and_plot.py

Compares pyMut vs maftools filtering outputs for TWO independent filters,
applied INDEPENDENTLY to TWO TCGA cohorts (TCGA-LAML and TCGA-PAAD),
i.e. four filtering "cases" in total:

  Cohort      Filter 1 (shared)              Filter 2 (gene-specific)
  ----------  -----------------------------  ---------------------------
  TCGA-LAML   Chromosomes 1, 2, 3, 17        DNMT3A locus, chr2 (GRCh37)
  TCGA-PAAD   Chromosomes 1, 2, 3, 17        TP53 locus,   chr17 (GRCh37)

The gene-specific filter uses DNMT3A for LAML and TP53 for PAAD: both are
recurrently mutated, clinically relevant genes for their respective tumor
type (DNMT3A in AML, TP53 in pancreatic adenocarcinoma), and both were
picked to keep the filter non-trivial in each cohort's data.

Inputs are the MAF files written by flow_pymut.py and flow_maftools.R.
Each case is compared independently (normalized key columns, Jaccard
index over row keys, diff TSVs), then all four cases are combined into a
single publication figure grouped by cohort.

Paths (resolved relative to this script, regardless of the working dir):
  - Input MAFs (--data-dir):  validation/filter_concordance/results/ (default)
  - Outputs (--out-dir):      same folder by default

Outputs (into --out-dir):
  - report.json                                  overall + per-case stats
  - report.md                                     human-readable summary
  - diff_<case>.tsv                               only written if non-empty
  - pymut_vs_maftools_filter_concordance.png/.pdf  the publication figure
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------- Utils --------------------------------
# (same normalization logic as compare_flows.py, so results stay consistent
# with the existing pyMut-vs-R comparison harness)

def read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def norm_chr(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"^chr", "", regex=True)


def norm_int_str(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    try:
        s = s.astype("Int64")
        return s.astype(object).apply(lambda x: "" if pd.isna(x) else str(int(x)))
    except Exception:
        return series.astype(str)


def build_key_df(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    columns_by_lower: Dict[str, List[str]] = {}
    for column in df.columns:
        columns_by_lower.setdefault(column.lower(), []).append(column)

    resolved_keys = {}
    missing = []
    for key in keys:
        if key in df.columns:
            resolved_keys[key] = key
            continue
        matches = columns_by_lower.get(key.lower(), [])
        if not matches:
            missing.append(key)
        elif len(matches) > 1:
            raise KeyError(f"Key '{key}' is ambiguous across columns: {matches}")
        else:
            resolved_keys[key] = matches[0]

    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    out = df[[resolved_keys[key] for key in keys]].copy()
    out.columns = list(keys)
    if "Chromosome" in out.columns:
        out["Chromosome"] = norm_chr(out["Chromosome"])
    for c in ("Start_Position", "End_Position"):
        if c in out.columns:
            out[c] = norm_int_str(out[c])
    for c in out.columns:
        out[c] = out[c].astype(str).str.strip()
    return out


def as_key_tuples(df: pd.DataFrame) -> List[Tuple[str, ...]]:
    return [tuple(v) for v in df.astype(str).itertuples(index=False, name=None)]


# ------------------------------ Report ---------------------------------

@dataclass
class CaseReport:
    name: str
    label: str
    cohort: str
    filter_kind: str
    maf_py: str
    maf_maftools: str
    n_rows_py: int
    n_rows_maftools: int
    n_unique_py: int
    n_unique_maftools: int
    n_shared: int
    n_py_only: int
    n_maftools_only: int
    n_union: int
    jaccard_index: float
    pass_threshold: bool
    pass_exact: bool


@dataclass
class GlobalReport:
    settings: Dict
    thresholds: Dict[str, float]
    cases: List[Dict] = field(default_factory=list)
    pass_all: bool = False


# --------------------------- Execution helpers --------------------------

def run_rscript(script: Path, cwd: Optional[Path] = None) -> None:
    subprocess.run(["Rscript", str(script)], check=True, cwd=str(cwd) if cwd else None)


def run_pyscript(script: Path, cwd: Optional[Path] = None) -> None:
    subprocess.run([sys.executable, str(script)], check=True, cwd=str(cwd) if cwd else None)


# --------------------------------- Core ---------------------------------

KEY_COLS = ["Hugo_Symbol", "Chromosome", "Start_Position", "End_Position", "Tumor_Sample_Barcode"]


def compare_case(
    name: str,
    label: str,
    cohort: str,
    filter_kind: str,
    maf_py_path: Path,
    maf_r_path: Path,
    keys: Sequence[str],
    threshold: float,
    out_dir: Path,
) -> CaseReport:
    if not maf_py_path.exists():
        raise FileNotFoundError(f"pyMut MAF not found for case '{name}': {maf_py_path}")
    if not maf_r_path.exists():
        raise FileNotFoundError(f"maftools MAF not found for case '{name}': {maf_r_path}")

    df_py_raw = read_tsv(maf_py_path)
    df_r_raw = read_tsv(maf_r_path)

    df_py_k = build_key_df(df_py_raw, keys)
    df_r_k = build_key_df(df_r_raw, keys)

    set_py = set(as_key_tuples(df_py_k))
    set_r = set(as_key_tuples(df_r_k))

    inter = set_py & set_r
    union = set_py | set_r
    only_py = sorted(set_py - set_r)
    only_r = sorted(set_r - set_py)

    jaccard = (len(inter) / len(union)) if union else float("nan")

    print(f"[{label.replace(chr(10), ' ')}]")
    print(f"  pyMut rows:      {df_py_raw.shape[0]} ({len(set_py)} unique keys)")
    print(f"  maftools rows:   {df_r_raw.shape[0]} ({len(set_r)} unique keys)")
    print(f"  Shared keys:     {len(inter)}")
    print(f"  pyMut only:      {len(only_py)}")
    print(f"  maftools only:   {len(only_r)}")
    print(f"  Jaccard index:   {jaccard:.4f}")

    # Diff TSV (only if there is any discordance)
    diff_rows = []
    for t in only_py:
        row = dict(zip(keys, t))
        row["present_py"] = True
        row["present_maftools"] = False
        diff_rows.append(row)
    for t in only_r:
        row = dict(zip(keys, t))
        row["present_py"] = False
        row["present_maftools"] = True
        diff_rows.append(row)

    diff_path = out_dir / f"diff_{name}.tsv"
    if diff_rows:
        diff_df = pd.DataFrame(diff_rows)
        diff_df = diff_df[list(keys) + ["present_py", "present_maftools"]]
        diff_df.to_csv(diff_path, sep="\t", index=False)
        print(f"  Discordant keys written to: {diff_path}")
    elif diff_path.exists():
        diff_path.unlink()

    return CaseReport(
        name=name,
        label=label,
        cohort=cohort,
        filter_kind=filter_kind,
        maf_py=str(maf_py_path),
        maf_maftools=str(maf_r_path),
        n_rows_py=int(df_py_raw.shape[0]),
        n_rows_maftools=int(df_r_raw.shape[0]),
        n_unique_py=len(set_py),
        n_unique_maftools=len(set_r),
        n_shared=len(inter),
        n_py_only=len(only_py),
        n_maftools_only=len(only_r),
        n_union=len(union),
        jaccard_index=float(jaccard),
        pass_threshold=bool(jaccard >= threshold) if jaccard == jaccard else False,
        pass_exact=bool(jaccard == 1.0 and len(only_py) == 0 and len(only_r) == 0),
    )


# ------------------------------- Figure ----------------------------------

def build_concordance_figure(cases: List[CaseReport], figsize=(24, 5.5)):
    """Build the three-panel publication figure: concordance counts for
    TCGA-LAML, concordance counts for TCGA-PAAD (each with its OWN y-axis
    scale, since LAML has far fewer records than PAAD), and the Jaccard
    index for all four cases together.
    """
    jaccard = [c.jaccard_index for c in cases]

    cohorts_in_order: List[str] = []
    for c in cases:
        if c.cohort not in cohorts_in_order:
            cohorts_in_order.append(c.cohort)

    within_gap = 6
    between_gap = 6
    x = []
    pos = 0.0
    prev_cohort = None
    for c in cases:
        if prev_cohort is not None and c.cohort != prev_cohort:
            pos += between_gap
        elif prev_cohort is not None:
            pos += within_gap
        x.append(pos)
        prev_cohort = c.cohort
    x = np.array(x)

    # x position of the boundary between cohort groups (used only for the
    # Jaccard panel, which still shows all cases together).
    boundaries = []
    for i in range(1, len(cases)):
        if cases[i].cohort != cases[i - 1].cohort:
            boundaries.append((x[i - 1] + x[i]) / 2)

    fig, axes = plt.subplots(
        1, 3, figsize=figsize,
        gridspec_kw={"width_ratios": [1, 1, 1.3]},
        constrained_layout=True,
    )
    fig.get_layout_engine().set(wspace=0.15, w_pad=0.1)
    width = 0.75  # wider bars

    # Panels A/B: stacked concordance bars, one panel per cohort so each
    # gets its own y-axis scale.
    for ax, cohort in zip(axes[:2], cohorts_in_order):
        cohort_cases = [c for c in cases if c.cohort == cohort]
        c_labels = [c.label for c in cohort_cases]
        c_shared = [c.n_shared for c in cohort_cases]
        c_py_only = [c.n_py_only for c in cohort_cases]
        c_maftools_only = [c.n_maftools_only for c in cohort_cases]
        c_x = np.arange(len(cohort_cases))

        ax.bar(c_x, c_shared, width, label="Shared (concordant)", color="#2C7BB6")
        ax.bar(c_x, c_py_only, width, bottom=c_shared, label="pyMut only", color="#D7191C")
        bottom2 = [s + p for s, p in zip(c_shared, c_py_only)]
        ax.bar(c_x, c_maftools_only, width, bottom=bottom2, label="maftools only", color="#FDAE61")
        ax.set_xticks(c_x)
        ax.set_xticklabels(c_labels, fontsize=14)
        ax.set_ylabel("Variant records\n(Hugo_Symbol, Chromosome, Start/End, sample)", fontsize=14)
        ax.set_title(
            f"Variant-record concordance between pyMut and maftools\n{cohort}",
            fontsize=17, fontweight="bold", pad=12,
        )
        ax.tick_params(axis="both", labelsize=13)
        totals = [s + p + m for s, p, m in zip(c_shared, c_py_only, c_maftools_only)]
        max_total = max(totals) if totals else 1
        ax.set_ylim(0, max_total * 1.28)
        ax.legend(frameon=False, fontsize=15, loc="upper right")
        for xi, total in zip(c_x, totals):
            ax.text(xi, total + max_total * 0.04, f"n={total}", ha="center", fontsize=15)
        ax.spines[["top", "right"]].set_visible(False)

    # Panel C: Jaccard index (still shown for all cases together, since
    # jaccard is already a bounded 0-1 metric and comparable across cohorts).
    ax2 = axes[2]
    ax2.bar(x, jaccard, width=width +1, color="#2C7BB6")
    labels = [c.label for c in cases]
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=14)
    ax2.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Jaccard index", fontsize=14)
    ax2.set_title(
        "Filter concordance (Jaccard index)\n"
        "pyMut vs maftools, TCGA-LAML and TCGA-PAAD",
        fontsize=17, fontweight="bold", pad=12,
    )
    ax2.tick_params(axis="both", labelsize=13)
    for boundary in boundaries:
        ax2.axvline(boundary, color="grey", linestyle=":", linewidth=1, zorder=0)
    for xi, j in zip(x, jaccard):
        ax2.text(xi, j + 0.03, f"{j:.3f}", ha="center", fontsize=15, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)

    return fig


# --------------------------------- Main -----------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    this_dir = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--run-py", action="store_true", help="Run flow_pymut.py before comparing")
    p.add_argument("--run-r", action="store_true", help="Run flow_maftools.R before comparing")
    p.add_argument("--py-script", default=str(this_dir / "flow_pymut.py"))
    p.add_argument("--r-script", default=str(this_dir / "flow_maftools.R"))

    # Default directories, resolved from this script's location:
    # the input MAFs are the outputs of flow_pymut.py / flow_maftools.R,
    # written by default to <this folder>/results/.
    default_data_dir = this_dir / "results"
    p.add_argument("--data-dir", default=str(default_data_dir),
                    help="Folder containing the input MAFs (output of flow_pymut.py / flow_maftools.R)")
    p.add_argument("--out-dir", default=str(default_data_dir),
                    help="Output folder for the report and the figure")

    p.add_argument("--keys", nargs="+", default=KEY_COLS, help="Key columns used for the comparison")
    p.add_argument("--jaccard-min", type=float, default=0.995, help="Minimum Jaccard index required to PASS")

    args = p.parse_args(argv)

    py_script = Path(args.py_script)
    r_script = Path(args.r_script)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_py:
        print(f"Running: {py_script}")
        run_pyscript(py_script, cwd=py_script.parent)

    if args.run_r:
        print(f"Running: {r_script}")
        run_rscript(r_script, cwd=r_script.parent)

    # Four cases: {LAML, PAAD} x {chromosome filter, gene-locus filter}.
    # Filenames must match those written by flow_pymut.py / flow_maftools.R.
    cases_config = [
        {
            "name": "laml_chromosome_filter",
            "label": "TCGA-LAML\nChromosome filter",
            "cohort": "TCGA-LAML",
            "filter_kind": "chromosome",
            "maf_py": data_dir / "tcga_laml_chr_filter_py.maf",
            "maf_r": data_dir / "tcga_laml_chr_filter_maftools.maf",
        },
        {
            "name": "laml_dnmt3a_region_filter",
            "label": "TCGA-LAML\nDNMT3A region",
            "cohort": "TCGA-LAML",
            "filter_kind": "gene_locus",
            "maf_py": data_dir / "tcga_laml_dnmt3a_region_py.maf",
            "maf_r": data_dir / "tcga_laml_dnmt3a_region_maftools.maf",
        },
        {
            "name": "paad_chromosome_filter",
            "label": "TCGA-PAAD\nChromosome filter",
            "cohort": "TCGA-PAAD",
            "filter_kind": "chromosome",
            "maf_py": data_dir / "tcga_paad_chr_filter_py.maf",
            "maf_r": data_dir / "tcga_paad_chr_filter_maftools.maf",
        },
        {
            "name": "paad_tp53_region_filter",
            "label": "TCGA-PAAD\nTP53 region",
            "cohort": "TCGA-PAAD",
            "filter_kind": "gene_locus",
            "maf_py": data_dir / "tcga_paad_tp53_region_py.maf",
            "maf_r": data_dir / "tcga_paad_tp53_region_maftools.maf",
        },
    ]

    case_reports: List[CaseReport] = []
    for cfg in cases_config:
        case_reports.append(
            compare_case(
                name=cfg["name"],
                label=cfg["label"],
                cohort=cfg["cohort"],
                filter_kind=cfg["filter_kind"],
                maf_py_path=cfg["maf_py"],
                maf_r_path=cfg["maf_r"],
                keys=args.keys,
                threshold=args.jaccard_min,
                out_dir=out_dir,
            )
        )

    pass_all = all(c.pass_threshold for c in case_reports)

    report = GlobalReport(
        settings={"keys": list(args.keys), "cohorts": ["TCGA-LAML", "TCGA-PAAD"]},
        thresholds={"global_match_min": args.jaccard_min},
        cases=[asdict(c) for c in case_reports],
        pass_all=pass_all,
    )

    json_path = out_dir / "report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    print(f"Report written: {json_path}")

    md_lines = [
        "# pyMut vs maftools — Filter Concordance Report",
        "",
        "Cohorts: TCGA-LAML, TCGA-PAAD",
        "",
    ]
    md_lines.append(f"Key columns: {', '.join(args.keys)}")
    md_lines.append(f"Jaccard threshold: >= {args.jaccard_min}")
    md_lines.append("")
    for cohort_name in ("TCGA-LAML", "TCGA-PAAD"):
        md_lines.append(f"## {cohort_name}")
        md_lines.append("")
        for c in case_reports:
            if c.cohort != cohort_name:
                continue
            md_lines.append(f"### {c.label.replace(chr(10), ' ')}")
            md_lines.append(f"- pyMut rows: {c.n_rows_py} ({c.n_unique_py} unique keys)")
            md_lines.append(f"- maftools rows: {c.n_rows_maftools} ({c.n_unique_maftools} unique keys)")
            md_lines.append(f"- Shared keys: {c.n_shared}")
            md_lines.append(f"- pyMut only: {c.n_py_only}")
            md_lines.append(f"- maftools only: {c.n_maftools_only}")
            md_lines.append(f"- Jaccard index: {c.jaccard_index*100:.3f}%")
            md_lines.append(f"- PASS (>= threshold): {'YES' if c.pass_threshold else 'NO'}")
            md_lines.append(f"- Exact match (Jaccard = 1, 0 discordant): {'YES' if c.pass_exact else 'NO'}")
            md_lines.append("")
    md_lines.append(f"**OVERALL: {'PASS' if pass_all else 'FAIL'}**")

    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Report written: {md_path}")

    fig = build_concordance_figure(case_reports)
    png_path = out_dir / "pymut_vs_maftools_filter_concordance.png"
    pdf_path = out_dir / "pymut_vs_maftools_filter_concordance.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Figure written: {png_path}")
    print(f"Figure written: {pdf_path}")

    return 0 if pass_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
