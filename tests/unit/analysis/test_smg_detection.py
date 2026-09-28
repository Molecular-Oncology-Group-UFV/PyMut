import pandas as pd
import pytest

from src.pyMut.analysis import smg_detection
from src.pyMut.analysis.smg_detection import (
    SmgDetectionMixin,
    _calculate_optimal_permutations,
    _generate_transcript_regions,
    process_oncodriveclustl_results,
)
from src.pyMut.utils.format import reverse_format_chr


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

class _Meta:
    def __init__(self, assembly="37"):
        self.assembly = assembly


class DummyPM(SmgDetectionMixin):
    def __init__(self, df, samples=None, assembly="37", metadata=True):
        self.data = df
        self.samples = samples or []
        self.metadata = _Meta(assembly) if metadata else None


def _patch_regions(monkeypatch):
    """Avoid pyensembl entirely: return a minimal transcript table."""
    monkeypatch.setattr(
        smg_detection, "_generate_transcript_regions",
        lambda genome_version: pd.DataFrame({
            "CHROMOSOME": ["1"],
            "START": [1],
            "END": [1000],
            "ELEMENT": ["ENST000001"],
            "SYMBOL": ["GENE1"],
            "STRAND": ["+"],
        }),
    )


@pytest.fixture
def vcf_df():
    return pd.DataFrame({
        "CHROM": ["chr1", "chr1", "chr1"],
        "POS": [100, 200, 300],
        "REF": ["A", "C", "G"],
        "ALT": ["G", "T", "A"],
        "S1": ["0/1", "0/0", "1|1"],
        "S2": ["0/0", "0/1", "."],
    })


# ---------------------------------------------------------------------------
# _generate_transcript_regions
# ---------------------------------------------------------------------------

def test_generate_regions_requires_pyensembl(monkeypatch):
    monkeypatch.setattr(smg_detection, "EnsemblRelease", None)
    with pytest.raises(ImportError, match="pyensembl"):
        _generate_transcript_regions("GRCh37")


def test_generate_regions_rejects_unknown_version(monkeypatch):
    class _FakeEnsembl:
        def __init__(self, release):
            self.release = release

    monkeypatch.setattr(smg_detection, "EnsemblRelease", _FakeEnsembl)
    with pytest.raises(ValueError, match="Unsupported genome version"):
        _generate_transcript_regions("GRCh36")


# ---------------------------------------------------------------------------
# _validate_chromosome_format
# ---------------------------------------------------------------------------

def _write_regions(path, chromosomes):
    pd.DataFrame({
        "CHROMOSOME": chromosomes, "START": [1] * len(chromosomes),
        "END": [10] * len(chromosomes),
    }).to_csv(path, sep="\t", index=False, compression="gzip")


def test_validate_chromosome_format_consistent(tmp_path):
    mut = tmp_path / "mutations.tsv"
    mut.write_text("CHROMOSOME\tPOSITION\tREF\tALT\tSAMPLE\nchr1\t100\tA\tG\tS1\n")
    reg = tmp_path / "regions.gz"
    _write_regions(reg, ["chr1"])
    assert smg_detection._validate_chromosome_format(mut, reg) is True


def test_validate_chromosome_format_mismatch(tmp_path, caplog):
    mut = tmp_path / "mutations.tsv"
    mut.write_text("CHROMOSOME\tPOSITION\tREF\tALT\tSAMPLE\nchr1\t100\tA\tG\tS1\n")
    reg = tmp_path / "regions.gz"
    _write_regions(reg, ["1"])  # no chr prefix -> mismatch
    with caplog.at_level("WARNING"):
        assert smg_detection._validate_chromosome_format(mut, reg) is False
    assert any("mismatch" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# _calculate_optimal_permutations (pure function)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("snv_count,expected", [
    (0, 1000),
    (9999, 1000),
    (10000, 10000),
    (50000, 10000),
    (50001, 20000),
    (10 ** 6, 20000),
])
def test_optimal_permutations(snv_count, expected):
    assert _calculate_optimal_permutations(snv_count) == expected


# ---------------------------------------------------------------------------
# _run_oncodriveclustl (external binary -> exercise both paths with fakes)
# ---------------------------------------------------------------------------

def test_run_oncodriveclustl_missing_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(smg_detection.shutil, "which", lambda name: None)
    ok = smg_detection._run_oncodriveclustl(
        mutations_file=tmp_path / "m.tsv", regions_file=tmp_path / "r.gz",
        genome_build="hg19", output_dir=tmp_path / "out", n_permutations=1000,
    )
    assert ok is False


def test_run_oncodriveclustl_success(monkeypatch, tmp_path):
    monkeypatch.setattr(smg_detection.shutil, "which", lambda name: "/usr/bin/oncodriveclustl")

    class _FakeProc:
        returncode = 0  # the implementation reads proc.returncode after wait()

        def __init__(self, cmd, **kwargs):
            self.stdout = iter(["line1\n", "line2\n"])

        def wait(self):
            return 0

    monkeypatch.setattr(smg_detection.subprocess, "Popen", _FakeProc)
    out_dir = tmp_path / "out"
    ok = smg_detection._run_oncodriveclustl(
        mutations_file=tmp_path / "m.tsv", regions_file=tmp_path / "r.gz",
        genome_build="hg19", output_dir=out_dir, n_permutations=1000,
    )
    assert ok is True
    assert out_dir.exists()


# ---------------------------------------------------------------------------
# process_oncodriveclustl_results
# ---------------------------------------------------------------------------

def test_process_results_filters_and_sorts(tmp_path):
    res = tmp_path / "elements_results.txt"
    pd.DataFrame({
        "ELEMENT": ["G1", "G2", "G3"],
        "Q_EMPIRICAL": [0.5, 0.01, 0.05],
    }).to_csv(res, sep="\t", index=False)

    out = process_oncodriveclustl_results(tmp_path, threshold=0.10)
    assert list(out["ELEMENT"]) == ["G2", "G3"]
    assert list(out["Q_EMPIRICAL"]) == [0.01, 0.05]


def test_process_results_missing_file_returns_empty(tmp_path):
    out = process_oncodriveclustl_results(tmp_path, threshold=0.1)
    assert out.empty


def test_process_results_missing_column_returns_empty(tmp_path):
    res = tmp_path / "elements_results.txt"
    pd.DataFrame({"ELEMENT": ["G1"]}).to_csv(res, sep="\t", index=False)
    out = process_oncodriveclustl_results(tmp_path, threshold=0.1)
    assert out.empty


# ---------------------------------------------------------------------------
# SmgDetectionMixin.detect_smg_oncodriveclustl (run_analysis=False -> offline)
# ---------------------------------------------------------------------------

def test_detect_expands_vcf_genotypes(vcf_df, monkeypatch, tmp_path, caplog):
    _patch_regions(monkeypatch)
    monkeypatch.chdir(tmp_path)
    pm = DummyPM(vcf_df, samples=["S1", "S2"])

    with caplog.at_level("INFO"):
        mutations_df, significant = pm.detect_smg_oncodriveclustl(run_analysis=False)

    # 3 mutated (sample, variant) pairs: (row0,S1), (row1,S2), (row2,S1)
    assert len(mutations_df) == 3
    assert list(mutations_df.columns) == ["CHROMOSOME", "POSITION", "REF", "ALT", "SAMPLE"]
    assert set(mutations_df["SAMPLE"]) == {"S1", "S2"}
    # chr prefix handling goes through reverse_format_chr, whatever it normalizes to
    assert mutations_df["CHROMOSOME"].iloc[0] == reverse_format_chr("1")
    assert mutations_df["POSITION"].iloc[0] == 100
    # 0/0, 0|0 and "." genotypes are skipped
    assert not ((mutations_df["POSITION"] == 200) & (mutations_df["SAMPLE"] == "S1")).any()
    assert not ((mutations_df["POSITION"] == 300) & (mutations_df["SAMPLE"] == "S2")).any()
    # run_analysis=False -> no significant genes and a skip log
    assert significant.empty
    assert any("Skipping OncodriveCLUSTL" in r.message for r in caplog.records)


def test_detect_maf_format_uses_tumor_sample_barcode(monkeypatch, tmp_path):
    _patch_regions(monkeypatch)
    monkeypatch.chdir(tmp_path)
    maf = pd.DataFrame({
        "CHROM": ["chr2"],
        "Start_Position": [1000],
        "Reference_Allele": ["T"],
        "Tumor_Seq_Allele1": ["T"],
        "Tumor_Seq_Allele2": ["C"],
        "Tumor_Sample_Barcode": ["SAMPLE_A"],
    })
    pm = DummyPM(maf)
    mutations_df, _ = pm.detect_smg_oncodriveclustl(run_analysis=False)
    assert len(mutations_df) == 1
    assert mutations_df["SAMPLE"].iloc[0] == "SAMPLE_A"
    assert mutations_df["ALT"].iloc[0] == "C"


def test_detect_maf_format_falls_back_to_allele1(monkeypatch, tmp_path):
    _patch_regions(monkeypatch)
    monkeypatch.chdir(tmp_path)
    maf = pd.DataFrame({
        "CHROM": ["chr2"],
        "Start_Position": [1000],
        "Reference_Allele": ["T"],
        "Tumor_Seq_Allele1": ["G"],
        "Tumor_Seq_Allele2": [""],
        "Tumor_Sample_Barcode": ["SAMPLE_B"],
    })
    pm = DummyPM(maf)
    mutations_df, _ = pm.detect_smg_oncodriveclustl(run_analysis=False)
    assert mutations_df["ALT"].iloc[0] == "G"


def test_detect_requires_metadata(vcf_df, monkeypatch, tmp_path):
    _patch_regions(monkeypatch)
    monkeypatch.chdir(tmp_path)
    pm = DummyPM(vcf_df, samples=["S1", "S2"], metadata=False)
    with pytest.raises(ValueError, match="Metadata"):
        pm.detect_smg_oncodriveclustl(run_analysis=False)


def test_detect_rejects_unsupported_assembly(vcf_df, monkeypatch, tmp_path):
    _patch_regions(monkeypatch)
    monkeypatch.chdir(tmp_path)
    pm = DummyPM(vcf_df, samples=["S1", "S2"], assembly="19")
    with pytest.raises(ValueError, match="Unsupported assembly"):
        pm.detect_smg_oncodriveclustl(run_analysis=False)


def test_detect_rejects_unrecognized_format(monkeypatch, tmp_path):
    _patch_regions(monkeypatch)
    monkeypatch.chdir(tmp_path)
    pm = DummyPM(pd.DataFrame({"FOO": [1]}))
    with pytest.raises(ValueError, match="not recognized"):
        pm.detect_smg_oncodriveclustl(run_analysis=False)


def test_detect_raises_when_no_snps(monkeypatch, tmp_path):
    _patch_regions(monkeypatch)
    monkeypatch.chdir(tmp_path)
    indels = pd.DataFrame({
        "CHROM": ["chr1"], "POS": [100], "REF": ["AT"], "ALT": ["A"],
        "S1": ["0/1"],
    })
    pm = DummyPM(indels, samples=["S1"])
    with pytest.raises(ValueError, match="No SNP variants"):
        pm.detect_smg_oncodriveclustl(run_analysis=False)
