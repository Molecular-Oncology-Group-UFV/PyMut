import io
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.pyMut.core import MutationMetadata, PyMutation
from src.pyMut.utils.constants import (
    DEFAULT_PLOT_FIGSIZE,
    DEFAULT_SUMMARY_FIGSIZE,
    DEFAULT_TOP_GENES_COUNT,
    VARIANT_CLASSIFICATION_COLUMN,
    VARIANT_TYPE_COLUMN,
    FUNCOTATION_COLUMN,
    GENE_COLUMN,
    SAMPLE_COLUMN,
    MODE_VARIANTS,
)


@pytest.mark.unit
class TestMutationMetadata:
    def test_mutationmetadata_init_sets_fields_and_loaded_at(self):
        md = MutationMetadata(
            source_format="VCF",
            file_path="/path/to/file.vcf",
            filters=["PASS"],
            assembly="GRCh38",
        )
        assert md.source_format == "VCF"
        assert md.file_path == "/path/to/file.vcf"
        assert isinstance(md.loaded_at, datetime)
        assert datetime.now() - md.loaded_at < timedelta(seconds=5)
        assert md.filters == ["PASS"]
        assert md.notes is None
        assert md.assembly == "GRCh38"


@pytest.mark.unit
class TestPyMutationBasics:
    def test_pymutation_init_defaults_samples_and_metadata(self, df_minimo):
        pm = PyMutation(df_minimo)
        assert pm.data.equals(df_minimo)
        assert pm.samples == []
        assert pm.metadata is None

        md = MutationMetadata("MAF", "file", [], "GRCh38")
        pm2 = PyMutation(df_minimo, metadata=md, samples=["S1"]) 
        assert pm2.samples == ["S1"]
        assert pm2.metadata is md

    def test_head_returns_dataframe_head(self, df_minimo):
        pm = PyMutation(df_minimo)
        assert pm.head().equals(df_minimo.head(5))
        assert pm.head(1).equals(df_minimo.head(1))
        assert pm.head(10).equals(df_minimo.head(10))

    def test_info_delegates_to_pandas_info_and_prints(self, df_minimo, capsys):
        pm = PyMutation(df_minimo)
        ret = pm.info()
        assert ret is None  # pandas.DataFrame.info returns None by default
        captured = capsys.readouterr().out
        # Check that column names appear in the output
        for col in df_minimo.columns:
            assert col in captured

    def test_save_figure_calls_savefig_with_quality_defaults_and_prints(self, capsys):
        # Create a dummy figure and spy on savefig
        fig = plt.figure()
        calls = {}

        def fake_savefig(fname, dpi=None, bbox_inches=None, **kwargs):
            calls["args"] = (fname,)
            calls["dpi"] = dpi
            calls["bbox_inches"] = bbox_inches
            calls["kwargs"] = kwargs

        # Monkeypatch the method on this instance only
        original = fig.savefig
        fig.savefig = fake_savefig  # type: ignore
        try:
            pm = PyMutation(pd.DataFrame())
            pm.save_figure(fig, "out.png")
            out = capsys.readouterr().out
            assert "📁 Figure saved: out.png" in out
            assert calls["args"] == ("out.png",)
            assert calls["dpi"] == 300
            assert calls["bbox_inches"] == "tight"

            # Pass extra kwargs
            pm.save_figure(fig, "out.pdf", format="pdf")
            assert calls["kwargs"].get("format") == "pdf"
        finally:
            fig.savefig = original  # restore
            plt.close(fig)

    def test_configure_high_quality_plots_sets_rcparams(self):
        import matplotlib as mpl

        # Reset some rcParams to non-expected values to ensure they are set
        mpl.rcParams["figure.dpi"] = 100
        mpl.rcParams["savefig.dpi"] = 100
        mpl.rcParams["savefig.bbox"] = None
        mpl.rcParams["savefig.format"] = "jpg"
        mpl.rcParams["savefig.transparent"] = True
        mpl.rcParams["savefig.facecolor"] = "black"
        mpl.rcParams["savefig.edgecolor"] = "black"

        PyMutation.configure_high_quality_plots()
        # Idempotent call
        PyMutation.configure_high_quality_plots()

        assert mpl.rcParams["figure.dpi"] == 300
        assert mpl.rcParams["savefig.dpi"] == 300
        assert mpl.rcParams["savefig.bbox"] == "tight"
        assert mpl.rcParams["savefig.format"] == "png"
        assert mpl.rcParams["savefig.transparent"] is False
        assert mpl.rcParams["savefig.facecolor"] == "white"
        assert mpl.rcParams["savefig.edgecolor"] == "none"


@pytest.mark.unit
class TestSummaryPlots:
    def test_summary_plot_calls_extractors_and_helper_and_closes_figure(self, monkeypatch, df_minimo):
        pm = PyMutation(df_minimo.copy())

        called = {"extract_cls": 0, "extract_type": 0, "helper": 0}

        def fake_extract_cls(df, variant_column, funcotation_column):
            called["extract_cls"] += 1
            assert variant_column == VARIANT_CLASSIFICATION_COLUMN
            assert funcotation_column == FUNCOTATION_COLUMN
            return df

        def fake_extract_type(df, variant_column, funcotation_column):
            called["extract_type"] += 1
            assert variant_column == VARIANT_TYPE_COLUMN
            assert funcotation_column == FUNCOTATION_COLUMN
            return df

        def fake_helper(self_obj, figsize, title, max_samples, top_genes_count):
            called["helper"] += 1
            assert figsize == DEFAULT_SUMMARY_FIGSIZE
            assert title is not None
            assert top_genes_count == DEFAULT_TOP_GENES_COUNT
            # Keep None value for max_samples when passed
            assert max_samples in (200, None)
            return plt.figure()

        monkeypatch.setattr(
            "src.pyMut.utils.data_processing.extract_variant_classifications",
            fake_extract_cls,
        )
        monkeypatch.setattr(
            "src.pyMut.utils.data_processing.extract_variant_types",
            fake_extract_type,
        )
        monkeypatch.setattr(
            "src.pyMut.visualizations.summary._create_summary_plot",
            fake_helper,
        )

        fig = pm.summary_plot()
        assert isinstance(fig, plt.Figure)
        assert called["extract_cls"] == 1 and called["extract_type"] == 1 and called["helper"] == 1
        assert not plt.fignum_exists(fig.number)

        # With None for max_samples
        fig2 = pm.summary_plot(max_samples=None)
        assert not plt.fignum_exists(fig2.number)

    def test_variant_classification_plot_extracts_and_sets_title(self, monkeypatch, df_minimo):
        pm = PyMutation(df_minimo.copy())

        def fake_extract(df, variant_column, funcotation_column):
            assert variant_column == "Variant_Classification"
            assert funcotation_column == "FUNCOTATION"
            return df

        created = {}

        def fake_helper(self_obj, ax, set_title):
            created["ax"] = ax
            assert set_title is False
            return ax

        monkeypatch.setattr(
            "src.pyMut.utils.data_processing.extract_variant_classifications",
            fake_extract,
        )
        monkeypatch.setattr(
            "src.pyMut.visualizations.summary._create_variant_classification_plot",
            fake_helper,
        )

        fig = pm.variant_classification_plot(title="Variant Classification")
        assert isinstance(fig, plt.Figure)
        assert fig._suptitle.get_text() == "Variant Classification"
        assert not plt.fignum_exists(fig.number)

        # Without title
        fig2 = pm.variant_classification_plot(title="")
        assert getattr(fig2, "_suptitle", None) is None
        assert not plt.fignum_exists(fig2.number)

    def test_variant_type_plot_extracts_and_sets_title(self, monkeypatch, df_minimo):
        pm = PyMutation(df_minimo.copy())

        def fake_extract(df, variant_column, funcotation_column):
            assert variant_column == "Variant_Type"
            assert funcotation_column == "FUNCOTATION"
            return df

        def fake_helper(self_obj, ax, set_title):
            assert set_title is False
            return ax

        monkeypatch.setattr(
            "src.pyMut.utils.data_processing.extract_variant_types",
            fake_extract,
        )
        monkeypatch.setattr(
            "src.pyMut.visualizations.summary._create_variant_type_plot",
            fake_helper,
        )

        fig = pm.variant_type_plot(title="Variant Type")
        assert isinstance(fig, plt.Figure)
        assert fig._suptitle.get_text() == "Variant Type"
        assert not plt.fignum_exists(fig.number)

    def test_snv_class_plot_calls_helper_with_columns_and_title(self, monkeypatch, df_minimo):
        pm = PyMutation(df_minimo.copy())

        seen = {}

        def fake_helper(self_obj, ref_column, alt_column, ax, set_title):
            seen["ref"] = ref_column
            seen["alt"] = alt_column
            assert set_title is False
            return ax

        monkeypatch.setattr(
            "src.pyMut.visualizations.summary._create_snv_class_plot",
            fake_helper,
        )

        fig = pm.snv_class_plot(title="SNV Class", ref_column="REF", alt_column="ALT")
        assert isinstance(fig, plt.Figure)
        assert seen["ref"] == "REF" and seen["alt"] == "ALT"
        assert fig._suptitle.get_text() == "SNV Class"
        assert not plt.fignum_exists(fig.number)

        # Custom columns
        fig2 = pm.snv_class_plot(title="My SNV", ref_column="RefCol", alt_column="AltCol")
        assert not plt.fignum_exists(fig2.number)

    def test_variants_per_sample_plot_normalizes_variant_column_and_title_y_when_custom(
        self, monkeypatch
    ):
        # lower-case variant column to test normalization
        df = pd.DataFrame({
            "variant_classification": ["A", "B"],
            SAMPLE_COLUMN: ["S1", "S2"],
            FUNCOTATION_COLUMN: ["", ""],
        })
        pm = PyMutation(df)

        seen = {}

        def fake_extract(df, variant_column, funcotation_column):
            # Should receive the normalized actual column name present in df
            assert variant_column == "variant_classification"
            assert funcotation_column == "FUNCOTATION"
            return df

        def fake_helper(self_obj, variant_column, sample_column, ax, set_title, max_samples):
            seen["variant_column"] = variant_column
            seen["sample_column"] = sample_column
            seen["max_samples"] = max_samples
            return ax

        monkeypatch.setattr(
            "src.pyMut.utils.data_processing.extract_variant_classifications",
            fake_extract,
        )
        monkeypatch.setattr(
            "src.pyMut.visualizations.summary._create_variants_per_sample_plot",
            fake_helper,
        )

        fig = pm.variants_per_sample_plot(title="Custom Title", max_samples=3)
        assert isinstance(fig, plt.Figure)
        assert seen["variant_column"] == "variant_classification"
        assert seen["sample_column"] == SAMPLE_COLUMN
        assert seen["max_samples"] == 3
        # Custom title should be positioned with a slightly higher y, we just check it exists
        assert fig._suptitle.get_text() == "Custom Title"
        assert not plt.fignum_exists(fig.number)

        # Title starting with default
        fig2 = pm.variants_per_sample_plot(title="Variants per Sample (median: 1.2)", max_samples=None)
        assert seen["max_samples"] is None
        assert fig2._suptitle.get_text().startswith("Variants per Sample")
        assert not plt.fignum_exists(fig2.number)

    def test_variant_classification_summary_plot_detects_wide_format_and_calls_helper(
        self, monkeypatch, df_wide_format, capsys
    ):
        pm = PyMutation(df_wide_format.copy())

        def fake_extract(df, variant_column, funcotation_column):
            assert variant_column == VARIANT_CLASSIFICATION_COLUMN
            assert funcotation_column == FUNCOTATION_COLUMN
            return df

        seen = {}

        def fake_helper(self_obj, variant_column, sample_column, ax, show_labels, set_title):
            seen["variant_column"] = variant_column
            seen["sample_column"] = sample_column
            seen["show_labels"] = show_labels
            seen["set_title"] = set_title
            return ax

        monkeypatch.setattr(
            "src.pyMut.utils.data_processing.extract_variant_classifications",
            fake_extract,
        )
        monkeypatch.setattr(
            "src.pyMut.visualizations.summary._create_variant_classification_summary_plot",
            fake_helper,
        )

        fig = pm.variant_classification_summary_plot()
        out = capsys.readouterr().out
        assert "Detected wide format with" in out
        assert seen["variant_column"] == VARIANT_CLASSIFICATION_COLUMN
        assert seen["sample_column"] == SAMPLE_COLUMN
        assert seen["show_labels"] is True and seen["set_title"] is False
        assert not plt.fignum_exists(fig.number)


@pytest.mark.unit
class TestTopMutatedGenesPlot:
    def test_top_mutated_genes_plot_validates_count_and_mode(self, df_minimo):
        pm = PyMutation(df_minimo.copy())
        with pytest.raises(ValueError) as e:
            pm.top_mutated_genes_plot(count=1.5)
        assert "The 'count' parameter must be an integer" in str(e.value)

        with pytest.raises(ValueError) as e2:
            pm.top_mutated_genes_plot(count=0)
        assert "positive integer" in str(e2.value)

        with pytest.raises(ValueError) as e3:
            pm.top_mutated_genes_plot(mode="invalid")
        assert "Allowed values" in str(e3.value)

    def test_top_mutated_genes_plot_normalizes_columns_and_sets_expected_title(
        self, monkeypatch
    ):
        # Build a DF with lowercase variant and gene columns for normalization
        df = pd.DataFrame({
            "hugo_symbol": ["TP53", "TP53"],
            "variant_classification": ["Missense_Mutation", "Nonsense_Mutation"],
            SAMPLE_COLUMN: ["S1", "S2"],
            FUNCOTATION_COLUMN: ["", ""],
        })
        pm = PyMutation(df)

        seen = {}

        def fake_extract(df, variant_column, funcotation_column):
            # Expect normalized column names passed to extractor
            assert variant_column == "variant_classification"
            assert funcotation_column == FUNCOTATION_COLUMN
            return df

        def fake_helper(self_obj, mode, variant_column, gene_column, sample_column, count, ax, set_title):
            seen.update(
                mode=mode,
                variant_column=variant_column,
                gene_column=gene_column,
                sample_column=sample_column,
                count=count,
                set_title=set_title,
            )
            return ax

        monkeypatch.setattr(
            "src.pyMut.utils.data_processing.extract_variant_classifications",
            fake_extract,
        )
        monkeypatch.setattr(
            "src.pyMut.visualizations.summary._create_top_mutated_genes_plot",
            fake_helper,
        )

        fig = pm.top_mutated_genes_plot()
        assert seen["mode"] == MODE_VARIANTS
        assert seen["variant_column"] == "variant_classification"
        assert seen["gene_column"] == "hugo_symbol"
        assert seen["sample_column"] == SAMPLE_COLUMN
        assert seen["count"] == DEFAULT_TOP_GENES_COUNT
        assert seen["set_title"] is False
        assert fig._suptitle.get_text() == "Top mutated genes (variants)"
        assert not plt.fignum_exists(fig.number)

        # With samples mode and custom title
        fig2 = pm.top_mutated_genes_plot(mode="samples", title="Custom")
        assert fig2._suptitle.get_text() == "Custom"
        assert not plt.fignum_exists(fig2.number)
