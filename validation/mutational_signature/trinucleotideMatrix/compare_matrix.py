#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare trinucleotide 96xN matrices between PyMut (Python) and R.

This script:
  1) Optionally runs the producers in this folder:
       - trinucleotide.R  → r_trinuc_matrix.csv
       - trinucleotide.py → py_trinuc_matrix.csv
  2) Reads both matrices (rows=context labels, columns=samples), aligns them on the
     union of contexts and samples. Missing samples (or contexts) on either side
     are filled with zeros as requested.
  3) Compares cell-by-cell equality and reports a global match rate
     (proportion of identical cells across the aligned matrices).
  4) Produces a JSON and Markdown report with the same filenames used in other
     validations (report_tmb.json, report_tmb.md).

Exit code 0 if thresholds are met, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ------------------------------- Helpers ------------------------------------

def _run_rscript(script: Path, *, cwd: Optional[Path] = None) -> None:
    cmd = ["Rscript", str(script)]
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _run_pyscript(script: Path, *, cwd: Optional[Path] = None) -> None:
    cmd = [sys.executable, str(script)]
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _read_matrix_csv(path: Path) -> pd.DataFrame:
    """
    Read a contexts-by-samples CSV written by either pandas (Python) or write.csv (R).
    - First column is the row index (context labels), but may come quoted by R.
    - Column headers (sample barcodes) may be quoted by R.
    The function strips quotes and coerces data to integers.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {path}")

    # Read with the first column as index (context labels)
    df = pd.read_csv(path, index_col=0, dtype=str)

    # Strip quotes and whitespace from index and columns
    df.index = df.index.astype(str).str.strip().str.strip('"').str.strip("'")
    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]

    # Coerce values to numeric, fill NaN with 0, and cast to Int64
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("Int64")

    return df


# ------------------------------ Report model --------------------------------

@dataclass
class GlobalReport:
    settings: Dict
    files: Dict[str, str]
    shape: Dict[str, int]
    compared_axes: Dict[str, List[str]]
    global_match_rate: float
    thresholds: Dict[str, float]
    pass_global: bool
    pass_all: bool


# --------------------------------- Main -------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    this_dir = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Optional producers
    p.add_argument("--run-r", action="store_true", help="Ejecutar trinucleotide.R antes de comparar")
    p.add_argument("--run-py", action="store_true", help="Ejecutar trinucleotide.py antes de comparar")
    p.add_argument("--r-script", default=str(this_dir / "trinucleotide.R"), help="Ruta al script de R a ejecutar")
    p.add_argument("--py-script", default=str(this_dir / "trinucleotide.py"), help="Ruta al script de Python a ejecutar")

    # Inputs to compare
    p.add_argument("--csv-py", default=str(this_dir / "py_trinuc_matrix.csv"), help="Matriz 96xN de PyMut (CSV)")
    p.add_argument("--csv-r", default=str(this_dir / "r_trinuc_matrix.csv"), help="Matriz 96xN de R (CSV)")

    # Reports (same names as other validations)
    p.add_argument("--json-out", default=str(this_dir / "report_tmb.json"), help="Salida JSON del reporte")
    p.add_argument("--md-out", default=str(this_dir / "report_tmb.md"), help="Salida Markdown del reporte")

    args = p.parse_args(argv)

    r_script = Path(args.r_script)
    py_script = Path(args.py_script)

    csv_py_path = Path(args.csv_py)
    csv_r_path = Path(args.csv_r)

    # 1) Optionally run producers ------------------------------------------------
    if args.run_r:
        print(f"▶ Ejecutando Rscript: {r_script}")
        _run_rscript(r_script, cwd=r_script.parent)
    if args.run_py:
        print(f"▶ Ejecutando Python: {py_script}")
        _run_pyscript(py_script, cwd=py_script.parent)

    # 2) Read matrices -----------------------------------------------------------
    df_py = _read_matrix_csv(csv_py_path)
    df_r = _read_matrix_csv(csv_r_path)

    # 3) Align on union of contexts and samples, fill missing with 0 -------------
    contexts_union = sorted(set(df_py.index) | set(df_r.index))
    samples_union = sorted(set(df_py.columns) | set(df_r.columns))

    df_py_a = (df_py.reindex(index=contexts_union, columns=samples_union).fillna(0)).astype("Int64")
    df_r_a = (df_r.reindex(index=contexts_union, columns=samples_union).fillna(0)).astype("Int64")

    # 4) Compare cell-by-cell ----------------------------------------------------
    # Use equality comparison; Int64 vs Int64 yields boolean with NA for NA, but we filled with zeros
    eq = (df_py_a.values == df_r_a.values)
    total_cells = int(df_py_a.size)
    n_equal = int(eq.sum()) if total_cells else 0
    match_rate = (n_equal / total_cells) if total_cells else float("nan")

    # Thresholds and flags
    thresholds = {
        "global_match_min": 0.995,
    }
    pass_global = (match_rate >= thresholds["global_match_min"]) if match_rate == match_rate else False
    pass_all = bool(pass_global)

    # 5) Build report ------------------------------------------------------------
    report = GlobalReport(
        settings={
            "fill_missing_with_zero": True,
        },
        files={
            "r_script": str(r_script),
            "py_script": str(py_script),
            "csv_py": str(csv_py_path),
            "csv_r": str(csv_r_path),
        },
        shape={
            "n_contexts_py": int(df_py.shape[0]),
            "n_contexts_r": int(df_r.shape[0]),
            "n_samples_py": int(df_py.shape[1]),
            "n_samples_r": int(df_r.shape[1]),
            "n_contexts_union": int(len(contexts_union)),
            "n_samples_union": int(len(samples_union)),
            "total_cells": int(total_cells),
            "n_equal_cells": int(n_equal),
        },
        compared_axes={
            "contexts": contexts_union,
            "samples": samples_union,
        },
        global_match_rate=float(match_rate),
        thresholds=thresholds,
        pass_global=pass_global,
        pass_all=pass_all,
    )

    # 6) Write JSON --------------------------------------------------------------
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)

    # 7) Write Markdown ----------------------------------------------------------
    md_path = Path(args.md_out)
    md_lines: List[str] = []
    md_lines.append("# Trinucleotide Matrix Comparison Report (PyMut vs R)")
    md_lines.append("")
    md_lines.append(f"Contexts (rows) PyMut: {report.shape['n_contexts_py']}")
    md_lines.append(f"Contexts (rows) R: {report.shape['n_contexts_r']}")
    md_lines.append(f"Samples (cols) PyMut: {report.shape['n_samples_py']}")
    md_lines.append(f"Samples (cols) R: {report.shape['n_samples_r']}")
    md_lines.append(f"Union contexts: {report.shape['n_contexts_union']}")
    md_lines.append(f"Union samples: {report.shape['n_samples_union']}")
    md_lines.append(f"Total cells compared: {report.shape['total_cells']}")
    if report.global_match_rate == report.global_match_rate:
        md_lines.append(f"Global cell-wise match: {report.global_match_rate*100:.3f}%")
    else:
        md_lines.append("Global cell-wise match: nan")
    md_lines.append("")
    md_lines.append("## PASO/ERROR (criterios)")
    md_lines.append(f"- Global match ≥ {thresholds['global_match_min']*100:.2f}% → {'PASS' if pass_global else 'FAIL'}")
    md_lines.append("")
    md_lines.append(f"OVERALL: {'PASS' if pass_all else 'FAIL'}")
    md_lines.append("")
    md_lines.append("### Archivos")
    md_lines.append(f"- PyMut CSV: {csv_py_path}")
    md_lines.append(f"- R CSV: {csv_r_path}")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return 0 if pass_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
