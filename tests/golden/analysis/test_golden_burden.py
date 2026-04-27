import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pyMut.input import read_vcf, read_maf

# The MutationBurden functionality is provided as a mixin on the returned PyMutation object
# via the method `calculate_tmb_analysis`.

GOLDEN_ROOT = Path(__file__).parent
VCF_DIR = GOLDEN_ROOT / "vcf_burden"
MAF_DIR = GOLDEN_ROOT / "maf_burden"


def _load_expected(expected_dir: Path):
    """Load expected analysis parquet and optional meta.json. Skip if missing."""
    analysis_path = expected_dir / "analysis.parquet"
    meta_path = expected_dir / "meta.json"

    if not analysis_path.exists():
        pytest.skip(
            f"Golden snapshots not present in {expected_dir}. "
            f"Generate them offline as documented and commit analysis.parquet (and meta.json)."
        )

    analysis_exp = pd.read_parquet(analysis_path)

    meta = {}
    if meta_path.exists():
        # Always use std json to preserve list types precisely
        import json
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

    return analysis_exp, meta


def _assert_df_equal_numeric(actual: pd.DataFrame, expected: pd.DataFrame, rtol=1e-12, atol=1e-12):
    # Ejes idénticos en orden
    assert list(actual.index) == list(expected.index), "Row index mismatch"
    assert list(actual.columns) == list(expected.columns), "Columns mismatch"
    # Valores (floats: usar tolerancia; ints: exacto). Convertimos a float por simplicidad.
    np.testing.assert_allclose(actual.values.astype(float), expected.values.astype(float), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "loader, expected_dir, assembly",
    [
        (read_vcf, VCF_DIR, "38"),
        (read_maf, MAF_DIR, "37"),
    ],
)
def test_golden_mutation_burden(loader, expected_dir, assembly):
    analysis_exp, meta = _load_expected(expected_dir)

    # Cargar dataset
    data_path = os.environ.get(
        "PYMUT_VCF_PATH" if assembly == "38" else "PYMUT_MAF_PATH",
        None,
    )
    if data_path is None:
        # Fijar rutas a fixtures del repo
        if assembly == "38":
            data_path = "tests/unit/io/fixtures/data/test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        else:
            data_path = "tests/unit/io/fixtures/data/test_tcga_laml_100variants.maf"

    assert Path(data_path).exists(), f"Missing input file: {data_path}"

    # Leer PyMutation
    pm = loader(data_path, assembly=assembly)

    # Parámetros congelados
    genome_size_bp = int(meta.get("genome_size_bp", 60456963))  # WES default
    variant_classification_column = meta.get("variant_classification_column")  # allow None for autodetect

    # Ejecutar análisis TMB
    res = pm.calculate_tmb_analysis(
        variant_classification_column=variant_classification_column,
        genome_size_bp=genome_size_bp,
        save_files=False,
    )

    analysis_act = res["analysis"].copy()

    # Asegurar índice por muestra
    if "Sample" in analysis_act.columns:
        analysis_act = analysis_act.set_index("Sample")

    if "Sample" in analysis_exp.columns:
        analysis_exp = analysis_exp.set_index("Sample")

    # Reordenar filas a samples_order si está en meta para comparación 1:1
    samples_order = meta.get("samples_order")
    if samples_order is not None:
        # Validar que todas las muestras existen
        missing = [s for s in samples_order if s not in analysis_act.index]
        assert not missing, f"Missing samples in actual analysis: {missing}"
        analysis_act = analysis_act.loc[samples_order]

    # Alinear columnas al esperado (mismo orden)
    analysis_act = analysis_act[analysis_exp.columns]

    # Asersiones de esquema
    assert analysis_act.shape == analysis_exp.shape, "Analysis shape changed; regenerate snapshots si es intencional"

    # Asersiones numéricas
    # Columnas esperadas: ['Total_Mutations', 'Non_Synonymous_Mutations', 'TMB_Total_Normalized', 'TMB_Non_Synonymous_Normalized']
    # - Para los contadores enteros, exigir igualdad exacta
    # - Para TMB (float), tolerancia estricta
    int_cols = [c for c in analysis_exp.columns if c in ("Total_Mutations", "Non_Synonymous_Mutations")]
    float_cols = [c for c in analysis_exp.columns if c in ("TMB_Total_Normalized", "TMB_Non_Synonymous_Normalized")]

    if int_cols:
        _assert_df_equal_numeric(analysis_act[int_cols], analysis_exp[int_cols], rtol=0, atol=0)
    if float_cols:
        _assert_df_equal_numeric(analysis_act[float_cols], analysis_exp[float_cols], rtol=1e-12, atol=1e-12)

    # Invariantes útiles
    assert (analysis_act[[c for c in int_cols]].values >= 0).all(), "Counts must be non-negative"
    if float_cols:
        assert (analysis_act[float_cols].values >= 0).all(), "TMB must be non-negative"
        # Chequeo razonable: TMB_Total_Normalized ≈ Total_Mutations / genome_size_bp * 1e6
        if "Total_Mutations" in analysis_act.columns and "TMB_Total_Normalized" in analysis_act.columns and genome_size_bp > 0:
            expected_tmb = analysis_act["Total_Mutations"].astype(float) / float(genome_size_bp) * 1_000_000
            np.testing.assert_allclose(
                analysis_act["TMB_Total_Normalized"].values,
                expected_tmb.values,
                rtol=1e-12,
                atol=1e-12,
            )
