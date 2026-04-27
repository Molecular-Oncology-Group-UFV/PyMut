#!/usr/bin/env python3
"""
Compare TMB outputs from PyMut and maftools and generate a reproducible report.

- Inputs (TSV):
  PyMut:    --pymut-analysis, --pymut-stats
  maftools: --maftools-analysis, --maftools-stats

- CLI options:
  --keys Sample
  --num-cols TMB Total_Mutations NonSyn_TMB
  --atol 1e-6 --rtol 1e-6
  --json-out report_tmb.json
  --md-out report_tmb.md
  --diff-out diff_tmb.tsv

- Comparison rules:
  categorical = exact equality
  numerical   = numpy.isclose with atol/rtol

- Metric:
  Global cell match %

- PASS/FAIL criteria:
  - Global match ≥ 99.5%
  Exit code: 0 on PASS, 1 on FAIL
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ALIASES = {
    # aliases → canonical column names found in provided TSVs
    "TMB": "TMB_Total_Normalized",
    "NonSyn_TMB": "TMB_Non_Synonymous_Normalized",
    # allow some common variations
    "TMB_Total": "TMB_Total_Normalized",
    "TMB_NonSyn": "TMB_Non_Synonymous_Normalized",
}


def _canon(col: str) -> str:
    return ALIASES.get(col, col)




@dataclass
class GlobalReport:
    settings: Dict
    files: Dict[str, str]
    shape: Dict[str, int]
    compared_columns: Dict[str, List[str]]
    global_match_rate: float
    thresholds: Dict[str, float]
    pass_global: bool
    pass_all: bool


def read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    # Strip whitespace from headers and string values
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")  # undefined when zero variance
    return float(np.corrcoef(x, y)[0, 1])


def compute_metrics_numeric(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    diff = x - y
    mae = float(np.nanmean(np.abs(diff)))
    rmse = float(math.sqrt(np.nanmean(diff ** 2)))
    # Avoid division by zero in bias percentage: only where y (maftools) != 0
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(y != 0, diff / y, np.nan)
    bias_pct = float(np.nanmean(rel) * 100.0)
    r = pearsonr_safe(x, y)
    return r, mae, rmse, bias_pct




def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pymut-analysis", required=True, help="Path to PyMut TMB_analysis.tsv")
    p.add_argument("--pymut-stats", required=True, help="Path to PyMut TMB_statistics.tsv")
    p.add_argument("--maftools-analysis", required=True, help="Path to maftools TMB_analysis.tsv")
    p.add_argument("--maftools-stats", required=True, help="Path to maftools TMB_statistics.tsv")

    p.add_argument("--keys", nargs="+", default=["Sample"], help="Key columns to join on")
    p.add_argument("--num-cols", nargs="+", default=[
        "TMB_Total_Normalized", "Total_Mutations", "TMB_Non_Synonymous_Normalized"
    ], help="Numeric columns to compare (aliases allowed)")
    p.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for numeric comparison")
    p.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for numeric comparison")

    p.add_argument("--json-out", default="report_tmb.json", help="Output JSON report path")
    p.add_argument("--md-out", default="report_tmb.md", help="Output Markdown report path")
    p.add_argument("--diff-out", default="diff_tmb.tsv", help="Output TSV with discrepancies")
    p.add_argument("--fig-scatter", nargs='?', const="scatter_tmb.png", help="[DEPRECATED] No-op; figures are no longer generated")
    p.add_argument("--fig-bland", nargs='?', const="bland_altman_tmb.png", help="[DEPRECATED] No-op; figures are no longer generated")

    args = p.parse_args(argv)

    pa = Path(args.pymut_analysis)
    ps = Path(args.pymut_stats)
    ma = Path(args.maftools_analysis)
    ms = Path(args.maftools_stats)

    # Read inputs
    df_py = read_tsv(pa)
    df_ma = read_tsv(ma)

    # Harmonize numeric columns via aliases
    numeric_cols = [_canon(c) for c in args.num_cols]

    # Ensure keys exist
    for k in args.keys:
        if k not in df_py.columns:
            p.error(f"Key '{k}' not found in PyMut analysis columns: {list(df_py.columns)}")
        if k not in df_ma.columns:
            p.error(f"Key '{k}' not found in maftools analysis columns: {list(df_ma.columns)}")

    # Inner join on keys to compare common samples
    merged = df_py.merge(df_ma, on=args.keys, suffixes=("_pymut", "_maftools"), how="inner")

    if merged.empty:
        print("No overlapping keys between inputs. Nothing to compare.", file=sys.stderr)
        return 1

    # Determine comparable columns (intersection excluding keys)
    cols_py = [c for c in df_py.columns if c not in args.keys]
    cols_ma = [c for c in df_ma.columns if c not in args.keys]
    common_cols = sorted(set(cols_py).intersection(cols_ma))

    # Prepare masks and diffs
    total_cells = 0
    total_matches = 0

    # Build diff frame rows for mismatches
    diff_rows: List[Dict[str, object]] = []

    for col in common_cols:
        left = merged[f"{col}_pymut"]
        right = merged[f"{col}_maftools"]
        is_numeric = _canon(col) in numeric_cols

        if is_numeric:
            x = to_numeric(left).to_numpy(dtype=float)
            y = to_numeric(right).to_numpy(dtype=float)
            comp = np.isclose(x, y, atol=args.atol, rtol=args.rtol)
            # record diffs for mismatches
            mism_idx = np.where(~comp)[0]
            for i in mism_idx:
                row = {k: merged.iloc[i][k] for k in args.keys}
                row[f"{col}_pymut"] = x[i]
                row[f"{col}_maftools"] = y[i]
                if np.isfinite(x[i]) and np.isfinite(y[i]):
                    row[f"{col}_delta"] = float(x[i] - y[i])
                else:
                    row[f"{col}_delta"] = np.nan
                row[f"{col}_match"] = bool(comp[i])
                diff_rows.append(row)
            matches = int(np.sum(comp))
            total_cells += len(comp)
            total_matches += matches
        else:
            comp = (left == right).to_numpy()
            mism_idx = np.where(~comp)[0]
            for i in mism_idx:
                row = {k: merged.iloc[i][k] for k in args.keys}
                row[f"{col}_pymut"] = left.iloc[i]
                row[f"{col}_maftools"] = right.iloc[i]
                row[f"{col}_match"] = bool(comp[i])
                diff_rows.append(row)
            matches = int(np.sum(comp))
            total_cells += len(comp)
            total_matches += matches

    global_match_rate = total_matches / total_cells if total_cells else float("nan")

    # Build JSON thresholds
    thresholds = {
        "global_match_min": 0.995,
    }

    pass_global = (global_match_rate >= thresholds["global_match_min"]) if not math.isnan(global_match_rate) else False
    pass_all = bool(pass_global)

    report = GlobalReport(
        settings={
            "keys": args.keys,
            "numeric_columns": numeric_cols,
            "atol": args.atol,
            "rtol": args.rtol,
        },
        files={
            "pymut_analysis": str(pa),
            "pymut_stats": str(ps),
            "maftools_analysis": str(ma),
            "maftools_stats": str(ms),
        },
        shape={
            "n_rows_compared": int(merged.shape[0]),
            "n_cols_compared": int(len(common_cols)),
            "n_cells_compared": int(total_cells),
        },
        compared_columns={
            "common": common_cols,
            "numeric": [c for c in common_cols if _canon(c) in numeric_cols],
            "categorical": [c for c in common_cols if _canon(c) not in numeric_cols],
        },
        global_match_rate=float(global_match_rate),
        thresholds=thresholds,
        pass_global=pass_global,
        pass_all=pass_all,
    )

    # Write JSON
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)

    # Write Markdown summary
    md_path = Path(args.md_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt_pct(x: float) -> str:
        return f"{x*100:.3f}%" if x == x else "nan"

    lines = []
    lines.append("# TMB Comparison Report (PyMut vs maftools)")
    lines.append("")
    lines.append(f"Compared rows (intersection on {args.keys}): {report.shape['n_rows_compared']}")
    lines.append(f"Compared columns (excluding keys): {report.shape['n_cols_compared']}")
    lines.append(f"Global cell match: {fmt_pct(report.global_match_rate)}")
    lines.append("")
    lines.append("## PASS/FAIL Criteria")
    lines.append(f"- Global match ≥ {thresholds['global_match_min']*100:.2f}% → {'PASS' if pass_global else 'FAIL'}")
    lines.append("")
    lines.append(f"OVERALL: {'PASS' if pass_all else 'FAIL'}")
    lines.append("")
    lines.append("### Files")
    lines.append(f"- PyMut analysis: {pa}")
    lines.append(f"- maftools analysis: {ma}")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Write diff TSV only if there are discrepancies
    diff_df = pd.DataFrame(diff_rows)
    diff_path = Path(args.diff_out)
    if not diff_df.empty:
        # Ensure keys are first
        key_cols = args.keys
        other_cols = [c for c in diff_df.columns if c not in key_cols]
        diff_df = diff_df[key_cols + other_cols]
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_df.to_csv(diff_path, sep="\t", index=False)
    else:
        # No discrepancies: remove any stale diff file if present
        try:
            if diff_path.exists():
                diff_path.unlink()
        except Exception:
            pass


    # Exit code
    return 0 if pass_all else 1


if __name__ == "__main__":
    sys.exit(main())
