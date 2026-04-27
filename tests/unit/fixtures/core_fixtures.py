import pandas as pd
import pytest

from src.pyMut.utils.constants import (
    GENE_COLUMN,
    VARIANT_CLASSIFICATION_COLUMN,
    REF_COLUMN,
    ALT_COLUMN,
    SAMPLE_COLUMN,
    FUNCOTATION_COLUMN,
    VARIANT_TYPE_COLUMN,
)


@pytest.fixture
def df_minimo():
    # Minimal dataframe with required columns and a few rows
    data = {
        GENE_COLUMN: ["TP53", "PIK3CA", "BRCA1", "TP53"],
        VARIANT_CLASSIFICATION_COLUMN: [
            "Missense_Mutation",
            "Nonsense_Mutation",
            "Frame_Shift_Del",
            "Missense_Mutation",
        ],
        VARIANT_TYPE_COLUMN: ["SNP", "SNP", "DEL", "INS"],
        REF_COLUMN: ["A", "C", "AG", "T"],
        ALT_COLUMN: ["G", "T", "-", "TT"],
        SAMPLE_COLUMN: ["S1", "S1", "S2", "S3"],
        FUNCOTATION_COLUMN: [
            "Gene:TP53|Variant_Classification:Missense_Mutation",
            "Gene:PIK3CA|Variant_Classification:Nonsense_Mutation",
            "Gene:BRCA1|Variant_Classification:Frame_Shift_Del",
            "Gene:TP53|Variant_Classification:Missense_Mutation",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def df_wide_format():
    # Wide format: gene-wise rows and samples as columns
    # Include TCGA-like and hyphenated sample IDs to trigger detection
    data = {
        GENE_COLUMN: ["TP53", "PIK3CA", "BRCA1"],
        "TCGA-01-0001": [1, 0, 1],
        "TCGA-02-0002": [0, 1, 0],
        "Sample-XX-YY": [1, 1, 0],
        VARIANT_CLASSIFICATION_COLUMN: [
            "Missense_Mutation",
            "Nonsense_Mutation",
            "Frame_Shift_Del",
        ],
        FUNCOTATION_COLUMN: [
            "Gene:TP53|Variant_Classification:Missense_Mutation",
            "Gene:PIK3CA|Variant_Classification:Nonsense_Mutation",
            "Gene:BRCA1|Variant_Classification:Frame_Shift_Del",
        ],
    }
    return pd.DataFrame(data)
