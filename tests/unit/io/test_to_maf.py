import io
from pathlib import Path

import pandas as pd
import pytest

from src.pyMut import output
from src.pyMut.output import required_columns_MAF
from .fixtures.output_fixtures import DummyPyMutation, DummyMeta


class TestToMAFBasicValidations:
    def test_to_maf_raises_if_missing_vcf_like_columns(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [123],
            "REF": ["A"],
            # Missing ALT
            "ID": ["rs1"],
            "S1": ["A|T"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        with pytest.raises(ValueError, match="Missing required VCF-style columns for MAF export"):
            pm.to_maf(tmp_path / "out.maf")

    def test_to_maf_raises_if_missing_sample_columns(self, minimal_df, dummy_meta, tmp_path):
        pm = DummyPyMutation(minimal_df, ["S1", "S2"], dummy_meta)
        with pytest.raises(ValueError, match="Missing sample columns for MAF export"):
            pm.to_maf(tmp_path / "out.maf")


class TestToMAFGenotypeProcessing:
    def test_to_maf_filters_ref_ref_genotypes(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1", "chr1"],
            "POS": [100, 200],
            "REF": ["A", "A"],
            "ALT": ["T", "T"],
            "ID": ["rs1", "rs2"],
            "S1": ["A|A", "A|T"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "maf.tsv"
        pm.to_maf(out)
        df_out = pd.read_csv(out, sep="\t")
        # Only the non REF|REF genotype should remain
        assert len(df_out) == 1
        assert df_out["Tumor_Sample_Barcode"].iloc[0] == "S1"

    def test_to_maf_splits_genotypes_with_pipe(self, minimal_df, dummy_meta, tmp_path):
        pm = DummyPyMutation(minimal_df, ["S1"], dummy_meta)
        out = tmp_path / "maf_pipe.tsv"
        pm.to_maf(out)
        df_out = pd.read_csv(out, sep="\t")
        assert df_out.loc[0, "Tumor_Seq_Allele1"] == "A"
        assert df_out.loc[0, "Tumor_Seq_Allele2"] == "T"

    def test_to_maf_splits_genotypes_with_slash(self, dummy_meta, tmp_path):
        # Ensure first row uses '/' to take that branch
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [123],
            "REF": ["A"],
            "ALT": ["T"],
            "ID": ["rs1"],
            "S1": ["A/T"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "maf_slash.tsv"
        pm.to_maf(out)
        df_out = pd.read_csv(out, sep="\t")
        assert df_out.loc[0, "Tumor_Seq_Allele1"] == "A"
        assert df_out.loc[0, "Tumor_Seq_Allele2"] == "T"

    def test_to_maf_fallback_when_no_separator(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [123],
            "REF": ["A"],
            "ALT": ["T"],
            "ID": ["rs1"],
            "S1": ["NA"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "maf_no_sep.tsv"
        pm.to_maf(out)
        df_out = pd.read_csv(out, sep="\t")
        assert df_out.loc[0, "Tumor_Seq_Allele1"] == "A"
        assert df_out.loc[0, "Tumor_Seq_Allele2"] == "T"


class TestToMAFEndPosition:
    def test_to_maf_end_position_for_snp(self, minimal_df, dummy_meta, tmp_path):
        pm = DummyPyMutation(minimal_df, ["S1"], dummy_meta)
        out = tmp_path / "maf_snp.tsv"
        pm.to_maf(out)
        df_out = pd.read_csv(out, sep="\t")
        assert int(df_out.loc[0, "End_Position"]) == int(df_out.loc[0, "Start_Position"])  # SNP

    def test_to_maf_end_position_for_deletion(self, dummy_meta, tmp_path):
        # REF longer than allele2
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [200],
            "REF": ["AT"],
            "ALT": ["A"],
            "ID": ["rs2"],
            "S1": ["AT|A"],  # not REF|REF, so it won't be filtered
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "maf_del.tsv"
        pm.to_maf(out)
        df_out = pd.read_csv(out, sep="\t")
        assert int(df_out.loc[0, "End_Position"]) == int(df_out.loc[0, "Start_Position"]) + len("AT") - 1

    def test_to_maf_end_position_for_insertion(self, dummy_meta, tmp_path):
        # Insertion-like: allele2 longer than REF. Per the MAF convention
        # (and matching the fix documented in output.py), an insertion's
        # End_Position flanks the inserted bases: End = Start + 1. The old
        # test expected End == Start, which silently mis-set insertions.
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [300],
            "REF": ["A"],
            "ALT": ["AT"],
            "ID": ["rs3"],
            "S1": ["A|AT"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "maf_ins.tsv"
        pm.to_maf(out)
        df_out = pd.read_csv(out, sep="\t")
        assert int(df_out.loc[0, "End_Position"]) == int(df_out.loc[0, "Start_Position"]) + 1


class TestToMAFColumnsAndMetadata:
    def test_to_maf_builds_base_columns_and_metadata(self, minimal_df, dummy_meta, tmp_path):
        pm = DummyPyMutation(minimal_df, ["S1"], dummy_meta)
        out = tmp_path / "maf_base.tsv"
        pm.to_maf(out)
        df_out = pd.read_csv(out, sep="\t")
        assert str(df_out.loc[0, "Chromosome"]) == "1"  # removed 'chr'
        assert int(df_out.loc[0, "Start_Position"]) == 123
        assert isinstance(df_out.loc[0, "Reference_Allele"], str)
        assert df_out.loc[0, "NCBI_Build"] == dummy_meta.assembly
        assert df_out.loc[0, "dbSNP_RS"] == str(minimal_df.loc[0, "ID"])  # cast to str

    def test_to_maf_includes_only_info_columns_not_vcf_or_samples(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [123],
            "REF": ["A"],
            "ALT": ["T"],
            "ID": ["rs1"],
            "S1": ["A|T"],
            "Gene": ["TP53"],
            "CSQ": ["missense"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "maf_info.tsv"
        pm.to_maf(out)
        df_out = pd.read_csv(out, sep="\t")
        # Gene/CSQ should be present (not part of VCF-like or sample columns)
        assert "Gene" in df_out.columns and "CSQ" in df_out.columns

    def test_to_maf_required_columns_added_if_missing(self, minimal_df, dummy_meta, tmp_path, monkeypatch):
        # Add a fake required column and ensure it's added with '.' by to_maf
        new_required = required_columns_MAF + ["DummyReq"]
        monkeypatch.setattr(output, "required_columns_MAF", new_required, raising=False)
        pm = DummyPyMutation(minimal_df, ["S1"], dummy_meta)
        out = tmp_path / "maf_req.tsv"
        pm.to_maf(out)
        df_out = pd.read_csv(out, sep="\t")
        assert "DummyReq" in df_out.columns
        # since data exists, rows are present and DummyReq should be '.'
        assert set(df_out["DummyReq"].unique()) == {"."}

    def test_to_maf_column_order_uses_preferred_order_when_available(self, minimal_df, dummy_meta, tmp_path, monkeypatch):
        # Force preferred order
        monkeypatch.setattr(output, "_load_maf_column_order", lambda: [
            "dbSNP_RS", "Chromosome", "Start_Position", "Tumor_Sample_Barcode"
        ], raising=False)
        pm = DummyPyMutation(minimal_df, ["S1"], dummy_meta)
        out = tmp_path / "maf_order.tsv"
        pm.to_maf(out)
        with open(out, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n")
        cols = header.split("\t")
        # The preferred ones that exist must come first in the given order
        expected_prefix = ["dbSNP_RS", "Chromosome", "Start_Position", "Tumor_Sample_Barcode"]
        assert cols[: len(expected_prefix)] == expected_prefix

    def test_to_maf_end_position_is_int(self, minimal_df, dummy_meta, tmp_path):
        pm = DummyPyMutation(minimal_df, ["S1"], dummy_meta)
        out = tmp_path / "maf_dtype.tsv"
        pm.to_maf(out)
        df_out = pd.read_csv(out, sep="\t")
        assert pd.api.types.is_integer_dtype(df_out["End_Position"]) or df_out["End_Position"].dtype == "int64"


class TestToMAFCommentsAndLogging:
    def test_to_maf_writes_only_info_comments_from_metadata_notes(self, minimal_df, tmp_path):
        notes = "\n".join([
            "Random comment",  # should be ignored
            "INFO=Something about fields",  # should be written as #INFO...
            "",  # ignored
            "##INFO=Defined field",  # written unchanged
            "Other text",  # ignored
        ])
        meta = DummyMeta(assembly="GRCh38", notes=notes)
        pm = DummyPyMutation(minimal_df, ["S1"], meta)
        out = tmp_path / "maf_notes.tsv"
        pm.to_maf(out)
        with open(out, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # First two lines should be the two INFO lines (prefixed with '#') then header
        info_lines = [l for l in lines if l.startswith("#INFO") or l.startswith("##INFO")]
        assert any(l.startswith("#INFO=") for l in info_lines)
        assert any(l.startswith("##INFO=") for l in info_lines)
        # Ensure non-INFO lines are not present prefixed as comments
        assert not any(l.startswith("#Random comment") for l in lines)

    def test_to_maf_handles_no_variants(self, dummy_meta, tmp_path, caplog):
        caplog.set_level("WARNING")
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [123],
            "REF": ["A"],
            "ALT": ["A"],
            "ID": ["rs1"],
            "S1": ["A|A"],  # filtered out
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "maf_empty.tsv"
        pm.to_maf(out)
        with open(out, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n")
            data = f.read().strip()
        assert header.split("\t") == required_columns_MAF
        assert data == ""  # no rows
        assert any("No variant data found to export" in r.message for r in caplog.records)

    def test_to_maf_logs_progress_and_summary(self, dummy_meta, tmp_path, caplog_info):
        df = pd.DataFrame({
            "CHROM": ["chr1", "chr1"],
            "POS": [123, 124],
            "REF": ["A", "A"],
            "ALT": ["T", "T"],
            "ID": ["rs1", "rs2"],
            "S1": ["A|T", "A|T"],
            "S2": ["A|T", "A|T"],
        })
        pm = DummyPyMutation(df, ["S1", "S2"], dummy_meta)
        out = tmp_path / "maf_logs.tsv"
        pm.to_maf(out)
        msgs = "\n".join(r.message for r in caplog_info.records)
        assert "Starting MAF export" in msgs
        assert "Processing sample" in msgs
        assert "Total variants found" in msgs
        assert "MAF export completed successfully" in msgs

    def test_to_maf_small_dataset_writes_without_chunking_and_no_trailing_newline(self, minimal_df, dummy_meta, tmp_path):
        pm = DummyPyMutation(minimal_df, ["S1"], dummy_meta)
        out = tmp_path / "maf_small.tsv"
        pm.to_maf(out)
        # The file should not end with an extra newline due to truncate
        with open(out, "rb") as f:
            content = f.read()
        assert len(content) > 0
        assert content[-1:] != b"\n"


class TestLoadMafColumnOrder:
    def setup_method(self):
        # Clear cache before each test
        output._MAF_COLUMN_ORDER_CACHE = None

    def test_load_maf_column_order_reads_csv_when_exists(self, monkeypatch, caplog):
        caplog.set_level("DEBUG")
        # Make Path.exists return True for the target file
        real_exists = output.Path.exists

        def fake_exists(p: Path):
            if str(p).endswith("MAF_COL_ORDER.csv"):
                return True
            return real_exists(p)

        monkeypatch.setattr(output, "pd", output.pd)  # ensure module reference
        monkeypatch.setattr(output, "Path", output.Path, raising=False)
        monkeypatch.setattr(output.Path, "exists", fake_exists, raising=False)

        # Mock read_csv to supply shuffled rows
        def fake_read_csv(_):
            return pd.DataFrame({"id": [2, 1], "nombre": ["B", "A"]})

        monkeypatch.setattr(output.pd, "read_csv", lambda p: fake_read_csv(p), raising=False)

        cols = output._load_maf_column_order()
        assert cols == ["A", "B"]
        assert output._MAF_COLUMN_ORDER_CACHE == ["A", "B"]
        assert any("Loaded 2 column names" in r.message for r in caplog.records)

    def test_load_maf_column_order_uses_cache_on_second_call(self, monkeypatch):
        output._MAF_COLUMN_ORDER_CACHE = ["A", "B"]
        calls = {"n": 0}

        def counting_read_csv(_):
            calls["n"] += 1
            return pd.DataFrame({"id": [1], "nombre": ["A"]})

        monkeypatch.setattr(output.pd, "read_csv", counting_read_csv, raising=False)
        c1 = output._load_maf_column_order()
        c2 = output._load_maf_column_order()
        assert c1 == ["A", "B"] and c2 == ["A", "B"]
        assert calls["n"] == 0  # Should not be called because cache used

    def test_load_maf_column_order_returns_empty_and_warns_if_file_missing(self, monkeypatch, caplog):
        caplog.set_level("WARNING")
        # Force exists() to False
        monkeypatch.setattr(output.Path, "exists", lambda self: False, raising=False)
        output._MAF_COLUMN_ORDER_CACHE = None
        cols = output._load_maf_column_order()
        assert cols == []
        assert any("not found" in r.message for r in caplog.records)

    def test_load_maf_column_order_handles_exception_and_warns(self, monkeypatch, caplog):
        caplog.set_level("WARNING")
        # Path exists but read_csv raises
        monkeypatch.setattr(output.Path, "exists", lambda self: True, raising=False)
        monkeypatch.setattr(output.pd, "read_csv", lambda p: (_ for _ in ()).throw(RuntimeError("boom")), raising=False)
        output._MAF_COLUMN_ORDER_CACHE = None
        cols = output._load_maf_column_order()
        assert cols == []
        assert any("Error loading" in r.message for r in caplog.records)
