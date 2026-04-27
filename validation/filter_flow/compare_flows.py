#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare filtering flows between PyMut (Python) and R.

This script:
  1) Runs validation/filter_flow/flow.R (via Rscript) to generate an R-filtered MAF.
  2) Runs validation/filter_flow/flow.py (via Python) to generate a PyMut-filtered MAF.
  3) Compares the results between:
       - tcga_laml_filtered_py.maf
       - tcga_laml_filtered_R.maf
     using the key columns:
       [Hugo_Symbol, Chromosome, Start_Position, End_Position, Tumor_Sample_Barcode]
  4) Produces a JSON and Markdown report like in validation/TMB:
       - report_tmb.json
       - report_tmb.md
  5) If there are discrepancies, writes a TSV with keys present only on one side:
       - diff_filter_flow.tsv

Exit code 0 if thresholds are met, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# ------------------------------- Utils ------------------------------------

def read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    # Strip whitespace from headers and string values
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def norm_chr(series: pd.Series) -> pd.Series:
    # Remove 'chr' prefix (if present) and normalize to plain string
    return series.astype(str).str.replace(r"^chr", "", regex=True)


def norm_int_str(series: pd.Series) -> pd.Series:
    # Convert to numeric (coerce NaN), then to string without leading zeros
    s = pd.to_numeric(series, errors="coerce")
    # Convert to pandas Int64 then to python-int backed string where possible
    try:
        s = s.astype("Int64")
        return s.astype(object).apply(lambda x: "" if pd.isna(x) else str(int(x)))
    except Exception:
        return series.astype(str)


# ------------------------------ Report ------------------------------------

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


# ------------------------------- Core -------------------------------------

def build_key_df(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}. Disponibles: {list(df.columns)}")

    out = df[list(keys)].copy()
    # Specific normalizations
    if "Chromosome" in out.columns:
        out["Chromosome"] = norm_chr(out["Chromosome"])
    for c in ("Start_Position", "End_Position"):
        if c in out.columns:
            out[c] = norm_int_str(out[c])
    # Ensure general string dtype
    for c in out.columns:
        out[c] = out[c].astype(str).str.strip()
    return out


def as_key_tuples(df: pd.DataFrame) -> List[Tuple[str, ...]]:
    # Return immutable tuples per row, in the DataFrame column order
    return [tuple(v) for v in df.astype(str).itertuples(index=False, name=None)]


# --------------------------- Execution helpers ----------------------------

def run_rscript(script: Path, *, cwd: Optional[Path] = None) -> None:
    cmd = ["Rscript", str(script)]
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def run_pyscript(script: Path, *, cwd: Optional[Path] = None) -> None:
    # Use the current Python interpreter to guarantee the environment
    cmd = [sys.executable, str(script)]
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


# --------------------------------- Main -----------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    this_dir = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Flow execution
    p.add_argument("--run-r", action="store_true", help="Ejecutar el flujo R (flow.R) antes de comparar")
    p.add_argument("--run-py", action="store_true", help="Ejecutar el flujo Python (flow.py) antes de comparar")
    p.add_argument("--r-script", default=str(this_dir / "flow.R"), help="Ruta al script de R a ejecutar")
    p.add_argument("--py-script", default=str(this_dir / "flow.py"), help="Ruta al script de Python a ejecutar")

    # Output files to compare
    p.add_argument("--maf-py", default=str(this_dir / "tcga_laml_filtered_py.maf"), help="MAF filtrado de PyMut")
    p.add_argument("--maf-r", default=str(this_dir / "tcga_laml_filtered_R.maf"), help="MAF filtrado de R")

    # Key columns
    p.add_argument("--keys", nargs="+", default=[
        "Hugo_Symbol", "Chromosome", "Start_Position", "End_Position", "Tumor_Sample_Barcode"
    ], help="Columnas clave para comparar")

    # Report outputs (same names as in TMB by requirement)
    p.add_argument("--json-out", default=str(this_dir / "report_tmb.json"), help="Ruta de salida del reporte JSON")
    p.add_argument("--md-out", default=str(this_dir / "report_tmb.md"), help="Ruta de salida del reporte Markdown")
    p.add_argument("--diff-out", default=str(this_dir / "diff_filter_flow.tsv"), help="Ruta TSV de discrepancias")

    args = p.parse_args(argv)

    r_script = Path(args.r_script)
    py_script = Path(args.py_script)

    maf_py_path = Path(args.maf_py)
    maf_r_path = Path(args.maf_r)

    # 1) Run R ---------------------------------------------------------
    if args.run_r:
        print(f"▶ Ejecutando Rscript: {r_script}")
        run_rscript(r_script, cwd=r_script.parent)
        # The R script saves by default to src/pyMut/data/examples/MAF/tcga_laml_filtered_R.maf
        r_default_out = repo_root / "src" / "pyMut" / "data" / "examples" / "MAF" / "tcga_laml_filtered_R.maf"
        if r_default_out.exists():
            maf_r_path.parent.mkdir(parents=True, exist_ok=True)
            # Copy/update to the expected destination if different
            if r_default_out.resolve() != maf_r_path.resolve():
                maf_r_path.write_text(r_default_out.read_text(encoding="utf-8"), encoding="utf-8")
        # If r_default_out does not exist, we expect flow.R to have written directly to maf_r_path

    # 2) Run Python ----------------------------------------------------
    if args.run_py:
        print(f"▶ Ejecutando Python: {py_script}")
        # Run from the script directory so its output lands there
        run_pyscript(py_script, cwd=py_script.parent)
        # flow.py writes tcga_laml_filtered.maf in its CWD by default; rename to *_py.maf if it exists
        default_py_out = py_script.parent / "tcga_laml_filtered.maf"
        if default_py_out.exists():
            maf_py_path.parent.mkdir(parents=True, exist_ok=True)
            if default_py_out.resolve() != maf_py_path.resolve():
                # move/rename (overwrite)
                try:
                    default_py_out.replace(maf_py_path)
                except Exception:
                    # fall back to copying content if replace fails across devices
                    maf_py_path.write_text(default_py_out.read_text(encoding="utf-8"), encoding="utf-8")
        # If flow.py already produced maf_py_path, do nothing

    # 3) Read MAFs ----------------------------------------------------------
    if not maf_py_path.exists():
        raise FileNotFoundError(f"No se encontró el MAF de PyMut: {maf_py_path}")
    if not maf_r_path.exists():
        raise FileNotFoundError(f"No se encontró el MAF de R: {maf_r_path}")

    df_py_raw = read_tsv(maf_py_path)
    df_r_raw = read_tsv(maf_r_path)

    # 4) Build normalized keys -------------------------------------
    df_py_k = build_key_df(df_py_raw, args.keys)
    df_r_k = build_key_df(df_r_raw, args.keys)

    # 5) Compare as sets of tuples ----------------------------------
    py_keys = as_key_tuples(df_py_k)
    r_keys = as_key_tuples(df_r_k)

    set_py = set(py_keys)
    set_r = set(r_keys)

    inter = set_py & set_r
    union = set_py | set_r

    jaccard = (len(inter) / len(union)) if union else float("nan")

    # 6) Prepare diffs -----------------------------------------------------
    only_py = sorted(list(set_py - set_r))
    only_r = sorted(list(set_r - set_py))

    diff_rows: List[Dict[str, object]] = []
    for t in only_py:
        row = {k: v for k, v in zip(args.keys, t)}
        row["present_py"] = True
        row["present_r"] = False
        diff_rows.append(row)
    for t in only_r:
        row = {k: v for k, v in zip(args.keys, t)}
        row["present_py"] = False
        row["present_r"] = True
        diff_rows.append(row)

    # 7) Thresholds and report -------------------------------------------------
    thresholds = {
        "global_match_min": 0.995,  # same as in TMB for consistency
    }
    pass_global = (jaccard >= thresholds["global_match_min"]) if jaccard == jaccard else False
    pass_all = bool(pass_global)

    report = GlobalReport(
        settings={
            "keys": list(args.keys),
        },
        files={
            "r_script": str(r_script),
            "py_script": str(py_script),
            "maf_py": str(maf_py_path),
            "maf_r": str(maf_r_path),
        },
        shape={
            "n_rows_py": int(df_py_raw.shape[0]),
            "n_rows_r": int(df_r_raw.shape[0]),
            "n_unique_py": int(len(set_py)),
            "n_unique_r": int(len(set_r)),
            "n_common": int(len(inter)),
            "n_union": int(len(union)),
        },
        compared_columns={
            "keys": list(args.keys),
        },
        global_match_rate=float(jaccard),
        thresholds=thresholds,
        pass_global=pass_global,
        pass_all=pass_all,
    )

    # JSON
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)

    # Markdown (format similar to TMB)
    md_path = Path(args.md_out)
    md_lines = []
    md_lines.append("# Filter Flow Comparison Report (PyMut vs R)")
    md_lines.append("")
    md_lines.append(f"Columnas clave: {', '.join(args.keys)}")
    md_lines.append(f"Filas PyMut: {report.shape['n_rows_py']}")
    md_lines.append(f"Filas R: {report.shape['n_rows_r']}")
    md_lines.append(f"Claves únicas PyMut: {report.shape['n_unique_py']}")
    md_lines.append(f"Claves únicas R: {report.shape['n_unique_r']}")
    md_lines.append(f"Intersección de claves: {report.shape['n_common']}")
    md_lines.append(f"Unión de claves: {report.shape['n_union']}")
    md_lines.append(f"Global match (Jaccard): {report.global_match_rate*100:.3f}%" if report.global_match_rate == report.global_match_rate else "Global match: nan")
    md_lines.append("")
    md_lines.append("## PASO/ERROR (criterios)")
    md_lines.append(f"- Global match ≥ {thresholds['global_match_min']*100:.2f}% → {'PASS' if pass_global else 'FAIL'}")
    md_lines.append("")
    md_lines.append(f"OVERALL: {'PASS' if pass_all else 'FAIL'}")
    md_lines.append("")
    md_lines.append("### Archivos")
    md_lines.append(f"- PyMut MAF: {maf_py_path}")
    md_lines.append(f"- R MAF: {maf_r_path}")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # Differences TSV
    diff_df = pd.DataFrame(diff_rows)
    diff_path = Path(args.diff_out)
    if not diff_df.empty:
        # Ensure keys go first
        key_cols = list(args.keys)
        other_cols = [c for c in diff_df.columns if c not in key_cols]
        diff_df = diff_df[key_cols + other_cols]
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_df.to_csv(diff_path, sep="\t", index=False)
    else:
        try:
            if diff_path.exists():
                diff_path.unlink()
        except Exception:
            pass

    return 0 if pass_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
