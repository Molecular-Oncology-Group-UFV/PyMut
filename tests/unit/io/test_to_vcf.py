from pathlib import Path

import pandas as pd
import pytest

from src.pyMut import output
from .fixtures.output_fixtures import DummyPyMutation, DummyMeta


class TestToVCFValidations:
    def test_to_vcf_raises_if_missing_vcf_like_columns(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [123],
            "REF": ["A"],
            # Missing ALT
            "ID": ["rs1"],
            "S1": ["A|T"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        with pytest.raises(ValueError, match="Missing required VCF-style columns for VCF export"):
            pm.to_vcf(tmp_path / "out.vcf")

    def test_to_vcf_raises_if_missing_sample_columns(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [123],
            "REF": ["A"],
            "ALT": ["T"],
            "ID": ["rs1"],
            "S1": ["A|T"],
        })
        pm = DummyPyMutation(df, ["S1", "S2"], dummy_meta)
        with pytest.raises(ValueError, match="Missing sample columns for VCF export"):
            pm.to_vcf(tmp_path / "out.vcf")


class TestToVCFHeader:
    def test_to_vcf_header_includes_metadata_and_contigs(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1", "2"],
            "POS": [123, 456],
            "REF": ["A", "C"],
            "ALT": ["T", "G"],
            "ID": ["rs1", "rs2"],
            "S1": ["A|T", "C|G"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "out.vcf"
        pm.to_vcf(out)
        with open(out, "r", encoding="utf-8") as f:
            header_lines = []
            for line in f:
                if line.startswith("##"):
                    header_lines.append(line.strip())
                else:
                    break
        header = "\n".join(header_lines)
        assert "##fileformat=VCFv4.3" in header
        assert any(l.startswith("##fileDate=") for l in header_lines)
        assert f"##reference={dummy_meta.assembly}" in header
        assert "##FILTER=<ID=PASS" in header
        assert "##contig=<ID=1>" in header  # chr1 -> 1
        assert "##contig=<ID=2>" in header

    def test_to_vcf_format_and_info_definitions(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [123],
            "REF": ["A"],
            "ALT": ["T"],
            "ID": ["rs1"],
            "S1": ["A|T"],
            "Gene": ["TP53"],
            "Effect": ["missense"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "fmtinfo.vcf"
        pm.to_vcf(out)
        with open(out, "r", encoding="utf-8") as f:
            header = "\n".join([l.strip() for l in f.readlines()[:20]])
        assert "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Phased Genotype\">" in header
        assert "##INFO=<ID=PMUT,Number=.,Type=String,Description=\"Consequence annotations columns from PyMut. Format: Gene|Effect\">" in header

    def test_to_vcf_column_header_line(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [123],
            "REF": ["A"],
            "ALT": ["T"],
            "ID": ["rs1"],
            "S1": ["A|T"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "cols.vcf"
        pm.to_vcf(out)
        with open(out, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Find the header line that starts with #CHROM
        header_line = next(l for l in lines if l.startswith("#CHROM"))
        assert header_line.strip().split("\t") == ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "S1"]


class TestToVCFBody:
    def test_to_vcf_default_values_for_missing_columns(self, dummy_meta, tmp_path):
        # Omit QUAL, FILTER, FORMAT
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [123],
            "REF": ["A"],
            "ALT": ["T"],
            "ID": ["rs1"],
            "S1": ["A|T"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "defaults.vcf"
        pm.to_vcf(out)
        body = pd.read_csv(out, sep="\t", comment="#", header=None)
        # Columns in body are without '#'
        assert body.iloc[0, 5] == "."  # QUAL
        assert body.iloc[0, 6] == "PASS"  # FILTER
        assert body.iloc[0, 8] == "GT"  # FORMAT

    def test_to_vcf_info_field_populated_with_pmut(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [123],
            "REF": ["A"],
            "ALT": ["T"],
            "ID": ["rs1"],
            "S1": ["A|T"],
            "Gene": ["TP53"],
            "Effect": ["missense"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "info.vcf"
        pm.to_vcf(out)
        body = pd.read_csv(out, sep="\t", comment="#", header=None)
        assert body.iloc[0, 7] == "PMUT=TP53|missense"

    def test_to_vcf_chromosome_values_strip_chr_prefix(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1", "chrX", "2"],
            "POS": [1, 2, 3],
            "REF": ["A", "C", "G"],
            "ALT": ["T", "G", "A"],
            "ID": ["rs1", "rs2", "rs3"],
            "S1": ["A|T", "C|G", "G|A"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "chrom.vcf"
        pm.to_vcf(out)
        body = pd.read_csv(out, sep="\t", comment="#", header=None)
        assert list(body.iloc[:, 0]) == ["1", "X", "2"]

    def test_to_vcf_genotype_bases_replaced_by_indices_for_pipe_separator(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1", "chr1"],
            "POS": [1, 2],
            "REF": ["A", "C"],
            "ALT": ["C,G", "C"],
            "ID": ["rs1", "rs2"],
            "S1": ["A|G", "C|C"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "gt_idx.vcf"
        pm.to_vcf(out)
        body = pd.read_csv(out, sep="\t", comment="#", header=None)
        # First row: A|G -> 0|2 (ALT has C,G)
        assert body.iloc[0, 9] == "0|2"
        # Second row: C|C when REF=C, ALT=C => 1|1
        assert body.iloc[1, 9] == "1|1"

    def test_to_vcf_genotype_with_unknown_allele_is_added_to_alt(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [1],
            "REF": ["A"],
            "ALT": ["C"],
            "ID": ["rs1"],
            "S1": ["A|T"],  # T not in ALT -> should be added
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "unknown_alt.vcf"
        pm.to_vcf(out)
        body = pd.read_csv(out, sep="\t", comment="#", header=None)
        # ALT should now be C,T
        assert body.iloc[0, 4] == "C,T"
        # Genotype should be 0|2
        assert body.iloc[0, 9] == "0|2"

    def test_to_vcf_genotype_no_call_is_skipped(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [1],
            "REF": ["A"],
            "ALT": ["C"],
            "ID": ["rs1"],
            "S1": [".|."],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "nocall.vcf"
        pm.to_vcf(out)
        body = pd.read_csv(out, sep="\t", comment="#", header=None)
        # Should be unchanged
        assert body.iloc[0, 9] == ".|."

    def test_to_vcf_genotypes_with_slash_are_not_converted(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [1],
            "REF": ["A"],
            "ALT": ["C"],
            "ID": ["rs1"],
            "S1": ["A/T"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "slash.vcf"
        pm.to_vcf(out)
        body = pd.read_csv(out, sep="\t", comment="#", header=None)
        assert body.iloc[0, 9] == "A/T"

    def test_to_vcf_column_order_in_body(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [123],
            "REF": ["A"],
            "ALT": ["T"],
            "ID": ["rs1"],
            "S1": ["A|T"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "order.vcf"
        pm.to_vcf(out)
        with open(out, "r", encoding="utf-8") as f:
            lines = f.readlines()
        header_line = next(l for l in lines if l.startswith("#CHROM"))
        body_line = next(l for l in lines if not l.startswith("#"))
        header_cols = header_line.strip().split("\t")
        body_cols = body_line.strip().split("\t")
        assert len(header_cols) == len(body_cols)


class TestToVCFLoggingAndFileEnding:
    def test_to_vcf_logs_processing_and_completion(self, dummy_meta, tmp_path, caplog):
        caplog.set_level("INFO")
        df = pd.DataFrame({
            "CHROM": ["chr1", "chr1"],
            "POS": [1, 2],
            "REF": ["A", "A"],
            "ALT": ["T", "T"],
            "ID": ["rs1", "rs2"],
            "S1": ["A|T", "A|T"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "logs.vcf"
        pm.to_vcf(out)
        msgs = "\n".join(r.message for r in caplog.records)
        assert "Processing genotype data to replace bases with indices" in msgs
        assert "VCF export completed successfully" in msgs

    def test_to_vcf_small_dataset_writes_without_chunking_and_no_trailing_newline(self, dummy_meta, tmp_path):
        df = pd.DataFrame({
            "CHROM": ["chr1"],
            "POS": [123],
            "REF": ["A"],
            "ALT": ["T"],
            "ID": ["rs1"],
            "S1": ["A|T"],
        })
        pm = DummyPyMutation(df, ["S1"], dummy_meta)
        out = tmp_path / "end.vcf"
        pm.to_vcf(out)
        with open(out, "rb") as f:
            content = f.read()
        assert len(content) > 0
        assert content[-1:] != b"\n"
