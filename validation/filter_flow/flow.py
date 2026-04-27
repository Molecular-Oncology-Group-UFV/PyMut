#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure we can import pyMut from the repository tree
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import the public pyMut API
from pyMut import read_maf  # noqa: E402


def main() -> None:
    # 1) Input --------------------------------------------------------------
    maf_path = REPO_ROOT / "src" / "pyMut" / "data" / "examples" / "MAF" / "tcga_laml.maf.gz"
    if not maf_path.exists():
        raise FileNotFoundError(f"No se encontró el MAF de ejemplo en: {maf_path}")

    print(f"📂 Cargando archivo: {maf_path}")
    py_mut = read_maf(str(maf_path), assembly="37")

    # 2) Filtering -------------------------------------------------------------
    start, end = 20818769, 133551224
    print(f"🔎 Filtrando por rango genómico {start}-{end} en chr1, chr2 y chr3…")

    # Usar GenomicRangeMixin.region con múltiples cromosomas a la vez
    filtered = py_mut.region(chrom=["1", "2", "3"], start=start, end=end)

    print(f"✅ Variantes tras filtro: {len(filtered.data)}")

    # 3) To
    print(filtered.samples)
    filtered.to_maf(output_path=os.path.join(os.getcwd(), "tcga_laml_filtered_py.maf"))

if __name__ == "__main__":
    main()
