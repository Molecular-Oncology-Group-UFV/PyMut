import pandas as pd
import numpy as np
import pytest

from src.pyMut.core import PyMutation, MutationMetadata
from src.pyMut.combination import combine_pymutations
from .fixtures.combination_fixtures import make_meta, make_pm, sort_variants


class TestCombinePyMutations:
    def test_assembly_mismatch_raises(self):
        pm1 = make_pm({"CHROM": ["1"], "POS": [1], "REF": ["A"], "ALT": ["G"]}, ["S1"], make_meta("a.vcf", assembly="37"))
        pm2 = make_pm({"CHROM": ["1"], "POS": [1], "REF": ["A"], "ALT": ["G"]}, ["S2"], make_meta("b.vcf", assembly="38"))
        with pytest.raises(ValueError):
            combine_pymutations(pm1, pm2)

    def test_merge_samples_and_no_duplicates(self):
        pm1 = make_pm({"CHROM": ["1"], "POS": [1], "REF": ["A"], "ALT": ["G"], "S1": [10]}, ["S1"], make_meta("a.vcf"))
        pm2 = make_pm({"CHROM": ["2"], "POS": [2], "REF": ["C"], "ALT": ["T"], "S2": [20]}, ["S2"], make_meta("b.vcf"))
        out = combine_pymutations(pm1, pm2)
        assert set(out.samples) == {"S1", "S2"}
        assert len(out.data) == 2

    def test_variant_deduplication_and_common_columns_priority(self):
        pm1 = make_pm({
            "CHROM": ["1"], "POS": [1], "REF": ["A"], "ALT": ["G"],
            "Gene": ["TP53"],
            "Info": ["from1"],
        }, [], make_meta("a.vcf"))
        pm2 = make_pm({
            "CHROM": ["1"], "POS": [1], "REF": ["A"], "ALT": ["G"],
            "Gene": ["TP53"],
            "Info": ["from2"],
        }, [], make_meta("b.vcf"))
        out = combine_pymutations(pm1, pm2)
        df = out.data
        assert len(df) == 1  # deduplicated
        assert df.iloc[0]["Info"] == "from1"  # priority to pymut1 when both non-null

    def test_null_handling_with_nan_and_none(self):
        pm1 = make_pm({"CHROM": ["1"], "POS": [1], "REF": ["A"], "ALT": ["G"], "Ann": [np.nan]}, [], make_meta("a.vcf"))
        pm2 = make_pm({"CHROM": ["1"], "POS": [1], "REF": ["A"], "ALT": ["G"], "Ann": ["val"]}, [], make_meta("b.vcf"))
        out = combine_pymutations(pm1, pm2)
        assert out.data.iloc[0]["Ann"] == "val"

    def test_unique_columns_preserved(self):
        pm1 = make_pm({"CHROM": ["1"], "POS": [1], "REF": ["A"], "ALT": ["G"], "Only1": [1]}, [], make_meta("a.vcf"))
        pm2 = make_pm({"CHROM": ["2"], "POS": [2], "REF": ["C"], "ALT": ["T"], "Only2": [2]}, [], make_meta("b.vcf"))
        out = combine_pymutations(pm1, pm2)
        df = sort_variants(out.data)
        assert {"Only1", "Only2"}.issubset(df.columns)
        assert pd.isna(df.loc[df["CHROM"] == "2", "Only1"]).all()
        assert pd.isna(df.loc[df["CHROM"] == "1", "Only2"]).all()

    def test_standard_columns_added_and_order(self):
        pm1 = make_pm({"CHROM": ["1"], "POS": [1], "REF": ["A"], "ALT": ["G"]}, [], make_meta("a.vcf"))
        pm2 = make_pm({"CHROM": ["2"], "POS": [2], "REF": ["C"], "ALT": ["T"]}, [], make_meta("b.vcf"))
        out = combine_pymutations(pm1, pm2)
        cols = list(out.data.columns)
        expected_prefix = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER"]
        assert cols[: len(expected_prefix)] == expected_prefix

    def test_metadata_fields_combined(self):
        pm1 = make_pm({"CHROM": ["1"], "POS": [1], "REF": ["A"], "ALT": ["G"]}, [], make_meta("a.vcf", fmt="VCF", assembly="38"))
        pm2 = make_pm({"CHROM": ["2"], "POS": [2], "REF": ["C"], "ALT": ["T"]}, [], make_meta("b.vcf", fmt="MAF", assembly="38"))
        out = combine_pymutations(pm1, pm2)
        m = out.metadata
        assert m.source_format == "COMBINED VCF + MAF"
        assert m.file_path == "a.vcf + b.vcf"
        assert m.filters == ["."]
        assert m.assembly == "38"
        assert "Combined from a.vcf and b.vcf" in m.notes

    def test_no_leaked_internal_columns(self):
        pm1 = make_pm({"CHROM": ["1"], "POS": [1], "REF": ["A"], "ALT": ["G"]}, [], make_meta("a.vcf"))
        pm2 = make_pm({"CHROM": ["2"], "POS": [2], "REF": ["C"], "ALT": ["T"]}, [], make_meta("b.vcf"))
        out = combine_pymutations(pm1, pm2)
        assert "_variant_id" not in out.data.columns

    def test_empty_and_non_empty_inputs(self):
        pm_empty = make_pm({"CHROM": [], "POS": [], "REF": [], "ALT": []}, [], make_meta("empty.vcf"))
        pm_full = make_pm({"CHROM": ["1"], "POS": [1], "REF": ["A"], "ALT": ["G"]}, ["S1"], make_meta("full.vcf"))
        out = combine_pymutations(pm_empty, pm_full)
        assert len(out.data) == 1
        assert set(out.samples) == {"S1"}
        # Ensure standard cols exist
        expected_prefix = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER"]
        assert list(out.data.columns)[: len(expected_prefix)] == expected_prefix

    def test_both_empty_inputs(self):
        pm1 = make_pm({"CHROM": [], "POS": [], "REF": [], "ALT": []}, [], make_meta("a.vcf"))
        pm2 = make_pm({"CHROM": [], "POS": [], "REF": [], "ALT": []}, [], make_meta("b.vcf"))
        out = combine_pymutations(pm1, pm2)
        assert len(out.data) == 0
        # Standard columns should still be present
        expected_prefix = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER"]
        assert list(out.data.columns)[: len(expected_prefix)] == expected_prefix

    def test_missing_sample_columns_do_not_appear(self):
        # samples list includes names that are not columns in data
        pm1 = make_pm({"CHROM": ["1"], "POS": [1], "REF": ["A"], "ALT": ["G"]}, ["S1"], make_meta("a.vcf"))
        pm2 = make_pm({"CHROM": ["2"], "POS": [2], "REF": ["C"], "ALT": ["T"], "S2": [5]}, ["S2", "S3"], make_meta("b.vcf"))
        out = combine_pymutations(pm1, pm2)
        # S1 and S3 are not real columns in data; only S2 exists as a column
        assert "S2" in out.data.columns
        assert "S1" not in out.data.columns
        assert "S3" not in out.data.columns
        # But the combined samples list still contains the union
        assert set(out.samples) == {"S1", "S2", "S3"}
