import pandas as pd
import pytest

from src.pyMut.annotate.cosmic_cancer_annotate import (
    _read_file_auto,
    _create_synonyms_dict,
    _create_oncokb_synonyms_dict,
    _apply_synonyms_mapping,
    _annotate_with_pandas,
    _annotate_with_duckdb,
    CancerAnnotateMixin,
)

from .fixtures.cosmic_cancer_fixtures import (
    maf_df_min,
    cosmic_df,
    oncokb_df,
    tmp_tsv_files,
    make_tsv,
)


# ==================== IO ====================

def test_read_file_auto_gz_and_plain_equivalent(tmp_tsv_files):
    df_plain = _read_file_auto(tmp_tsv_files["cosmic_tsv"], sep="\t")
    df_gz = _read_file_auto(tmp_tsv_files["cosmic_gz"], sep="\t")
    pd.testing.assert_frame_equal(df_plain.sort_index(axis=1), df_gz.sort_index(axis=1))


# ==================== Synonyms ====================

def test_create_synonyms_dict_and_apply_basic(maf_df_min, cosmic_df):
    syn_map = _create_synonyms_dict(cosmic_df, synonyms_column="SYNONYMS")
    # From cosmic_df fixture: TP53 has P53 and TRP53
    assert syn_map.get("P53") == "TP53"
    assert syn_map.get("TRP53") == "TP53"

    df = _apply_synonyms_mapping(maf_df_min.copy(), maf_join_col="Hugo_Symbol", synonyms_dict=syn_map)
    assert "_mapped_gene_symbol" in df.columns
    # At least TP53 should remain/resolve correctly
    assert "TP53" in set(df["_mapped_gene_symbol"]) or "TP53" in set(df["Hugo_Symbol"].fillna(""))


def test_create_oncokb_synonyms_dict_basic(oncokb_df):
    syn_map = _create_oncokb_synonyms_dict(oncokb_df, synonyms_column="Gene Aliases")
    assert syn_map.get("TRP53") == "TP53"
    # EGFR has ERBB1 alias in fixture
    assert syn_map.get("ERBB1") == "EGFR"


# ==================== Pandas backend ====================

@pytest.mark.parametrize("with_oncokb", [True, False])
def test_pandas_annotate_happy_path_with_oncokb(maf_df_min, cosmic_df, oncokb_df, tmp_path, with_oncokb):
    cosmic_path = tmp_path / "cosmic.tsv"
    oncokb_path = tmp_path / "oncokb.tsv"
    cosmic_df.to_csv(cosmic_path, sep="\t", index=False)
    oncokb_df.to_csv(oncokb_path, sep="\t", index=False)

    res = _annotate_with_pandas(
        data=maf_df_min.copy(),
        annotation_table=cosmic_path,
        join_column="Hugo_Symbol",
        synonyms_column="SYNONYMS",
        oncokb_table=(oncokb_path if with_oncokb else None),
        oncokb_synonyms_column="Gene Aliases",
    )

    # Renamed columns present
    assert any(c.startswith("COSMIC_") for c in res.columns)
    if with_oncokb:
        assert any(c.startswith("OncoKB_") for c in res.columns)

    # Internal columns removed
    assert "GENE_SYMBOL" not in res.columns
    assert "Hugo Symbol" not in res.columns
    assert "_mapped_gene_symbol" not in res.columns

    # Fillna rule: COSMIC_TIER always ""
    if "COSMIC_TIER" in res.columns:
        assert res["COSMIC_TIER"].isna().sum() == 0
        assert "" in set(res["COSMIC_TIER"])  # at least one row filled with ""


def test_pandas_annotate_missing_join_column_raises(cosmic_df, tmp_path):
    cosmic_path = tmp_path / "cosmic.tsv"
    cosmic_df.to_csv(cosmic_path, sep="\t", index=False)

    bad_df = pd.DataFrame({"WrongGeneCol": ["TP53"]})
    with pytest.raises(ValueError):
        _annotate_with_pandas(
            data=bad_df,
            annotation_table=cosmic_path,
            join_column="Hugo_Symbol",
            synonyms_column="SYNONYMS",
            oncokb_table=None,
        )


def test_pandas_annotate_oncokb_missing_hugo_symbol_raises(maf_df_min, cosmic_df, tmp_path):
    cosmic_path = tmp_path / "cosmic.tsv"
    broken_oncokb_path = tmp_path / "oncokb.tsv"

    cosmic_df.to_csv(cosmic_path, sep="\t", index=False)
    # Missing 'Hugo Symbol'
    pd.DataFrame({"Gene Aliases": ["P53"], "Is Oncogene": ["True"]}).to_csv(
        broken_oncokb_path, sep="\t", index=False
    )

    with pytest.raises(ValueError):
        _annotate_with_pandas(
            data=maf_df_min,
            annotation_table=cosmic_path,
            join_column="Hugo_Symbol",
            synonyms_column="SYNONYMS",
            oncokb_table=broken_oncokb_path,
        )


# ==================== DuckDB backend ====================

def test_duckdb_annotate_happy_path_minimal(maf_df_min, cosmic_df, tmp_path):
    cosmic_path = tmp_path / "cosmic.tsv"
    cosmic_df.to_csv(cosmic_path, sep="\t", index=False)

    res = _annotate_with_duckdb(
        data=maf_df_min.copy(),
        annotation_table=cosmic_path,
        join_column="Hugo_Symbol",
        synonyms_column="SYNONYMS",
        oncokb_table=None,
    )

    assert any(c.startswith("COSMIC_") for c in res.columns)
    assert "_mapped_gene_symbol" not in res.columns
    if "COSMIC_TIER" in res.columns:
        assert res["COSMIC_TIER"].isna().sum() == 0


# ==================== CancerAnnotateMixin.knownCancer ====================

class DummyAnnotator(CancerAnnotateMixin):
    def __init__(self, df):
        self.data = df.copy()
        class Meta:
            notes = ""
        self.metadata = Meta()


def test_knownCancer_computes_Is_Oncogene_any(maf_df_min, cosmic_df, oncokb_df, tmp_path):
    cosmic_path = tmp_path / "cosmic.tsv"
    oncokb_path = tmp_path / "oncokb.tsv"
    cosmic_df.to_csv(cosmic_path, sep="\t", index=False)
    oncokb_df.to_csv(oncokb_path, sep="\t", index=False)

    ann = DummyAnnotator(maf_df_min)
    out = ann.knownCancer(
        annotation_table=cosmic_path,
        oncokb_table=oncokb_path,
        join_column="Hugo_Symbol",
        in_place=False,
    )

    assert "Is_Oncogene_any" in out.columns
    assert out["Is_Oncogene_any"].dtype == bool or str(out["Is_Oncogene_any"].dtype) == "boolean"
    assert out["Is_Oncogene_any"].any()


def test_knownCancer_in_place_updates_self_data(maf_df_min, cosmic_df, tmp_path):
    cosmic_path = tmp_path / "cosmic.tsv"
    cosmic_df.to_csv(cosmic_path, sep="\t", index=False)

    ann = DummyAnnotator(maf_df_min)
    ret = ann.knownCancer(
        annotation_table=cosmic_path,
        oncokb_table=None,
        join_column="Hugo_Symbol",
        in_place=True,
    )

    assert ret is None
    assert hasattr(ann, "data") and any(c.startswith("COSMIC_") or c == "Is_Oncogene_any" for c in ann.data.columns)


def test_knownCancer_writes_output_and_respects_compress_flag(maf_df_min, cosmic_df, tmp_path):
    cosmic_path = tmp_path / "cosmic.tsv"
    cosmic_df.to_csv(cosmic_path, sep="\t", index=False)

    out_path = tmp_path / "out.maf"
    ann = DummyAnnotator(maf_df_min)

    ann.knownCancer(
        annotation_table=cosmic_path,
        oncokb_table=None,
        join_column="Hugo_Symbol",
        output_path=out_path,
        compress_output=True,
        in_place=True,
    )

    # Must write .gz
    final = tmp_path / (out_path.name + ".gz")
    assert final.exists() and final.is_file()


def test_knownCancer_preserves_original_columns(maf_df_min, cosmic_df, tmp_path):
    cosmic_path = tmp_path / "cosmic.tsv"
    cosmic_df.to_csv(cosmic_path, sep="\t", index=False)

    ann = DummyAnnotator(maf_df_min)
    out = ann.knownCancer(
        annotation_table=cosmic_path,
        join_column="Hugo_Symbol",
        in_place=False,
    )

    for col in ["Hugo_Symbol", "OtherCol"]:
        assert col in out.columns
