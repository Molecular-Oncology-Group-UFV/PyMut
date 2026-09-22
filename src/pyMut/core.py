"""
PyMutation core module.

Este módulo contiene la clase principal PyMutation que sirve como API principal
para la librería pyMut. Proporciona métodos para generar todos los tipos de
visualizaciones a partir de datos de mutación.
"""

import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from .analysis.mutation_burden import MutationBurdenMixin
from .analysis.mutational_signature import MutationalSignatureMixin
from .analysis.pfam_annotation import PfamAnnotationMixin
from .analysis.smg_detection import SmgDetectionMixin
from .annotate.actionable_mutation import ActionableMutationMixin
from .annotate.cosmic_cancer_annotate import CancerAnnotateMixin
from .filters.chrom_sample_filter import ChromSampleFilterMixin
from .filters.genomic_range import GenomicRangeMixin
from .filters.pass_filter import PassFilterMixin
from .filters.tissue_expression import TissueExpressionMixin
from .output import OutputMixin
from .utils.constants import (
    ALT_COLUMN,
    DEFAULT_ONCOPLOT_FIGSIZE,
    DEFAULT_ONCOPLOT_MAX_SAMPLES,
    DEFAULT_ONCOPLOT_TOP_GENES,
    DEFAULT_PLOT_FIGSIZE,
    DEFAULT_PLOT_TITLE,
    DEFAULT_SUMMARY_FIGSIZE,
    DEFAULT_TCGA_COMPARE_FIGSIZE,
    DEFAULT_TOP_GENES_COUNT,
    FUNCOTATION_COLUMN,
    GENE_COLUMN,
    MODE_VARIANTS,
    REF_COLUMN,
    SAMPLE_COLUMN,
    VALID_PLOT_MODES,
    VARIANT_CLASSIFICATION_COLUMN,
    VARIANT_TYPE_COLUMN,
)

logger = logging.getLogger(__name__)

# Bundled SBS mutational signature catalogs shipped with pyMut
# (data/examples/COSMIC_catalogue-signatures_SBS96_v3.4/), used as the
# default cosmic_path for signature_bar_chart(), cosine_similarity_heatmap()
# and mutational_signature_analysis() when the user doesn't supply their own.
_SIGNATURE_CATALOG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "examples", "COSMIC_catalogue-signatures_SBS96_v3.4",
)
_BUNDLED_COSMIC_CATALOGS = {
    "official": os.path.join(_SIGNATURE_CATALOG_DIR, "COSMIC_v3.4_SBS_GRCh37.txt"),
    "maftools": os.path.join(_SIGNATURE_CATALOG_DIR, "maftools_SBS_catalog.tsv"),
}


def _resolve_cosmic_catalog_path(cosmic_path):
    """
    Resolve a cosmic_path argument to an actual COSMIC catalog file path.

    - None (default): use the official bundled COSMIC v3.4 SBS (GRCh37)
      catalog (86 signatures) - no need for the user to supply anything.
    - "maftools": use the older, 67-signature catalog bundled with maftools
      instead, e.g. to reproduce results obtained with maftools itself.
    - False: explicitly disable COSMIC alignment (only meaningful for
      signature_bar_chart(), which can also run without matching COSMIC).
    - Any other string: used as-is, as a path to a user-supplied catalog.
    """
    if cosmic_path is False:
        return False
    if cosmic_path is None:
        return _BUNDLED_COSMIC_CATALOGS["official"]
    if isinstance(cosmic_path, str) and cosmic_path.lower() == "maftools":
        return _BUNDLED_COSMIC_CATALOGS["maftools"]
    return cosmic_path


_BUNDLED_AETIOLOGY_CATALOG = os.path.join(_SIGNATURE_CATALOG_DIR, "maftools_SBS_aetiology.tsv")


def _resolve_aetiology_catalog_path(aetiology_path):
    """
    Resolve an aetiology_path argument to an actual aetiology TSV file path.

    - None (default): no aetiology table is used - panel titles show just
      "SBSx-like (cos=..)" with no aetiology line, unchanged from before.
    - "maftools": use the bundled maftools SBS aetiology table.
    - Any other string: used as-is, as a path to a user-supplied TSV.
    """
    if aetiology_path is None:
        return None
    if isinstance(aetiology_path, str) and aetiology_path.lower() == "maftools":
        return _BUNDLED_AETIOLOGY_CATALOG
    return aetiology_path

class MutationMetadata:
    """
    Clase para almacenar metadatos de mutaciones.

    Atributos:
        source_format (str): Formato de origen (VCF, MAF, etc.).
        file_path (str): Ruta del archivo de origen.
        loaded_at (datetime): Fecha y hora de carga.
        filters (List[str]): Filtros aplicados al archivo.
        assembly (str): Versión del genoma (37 o 38).
        notes (Optional[str]): Notas adicionales.
    """

    def __init__(self, source_format: str, file_path: str, filters: List[str], assembly: str,
                 notes: Optional[str] = None):
        self.source_format = source_format
        self.file_path = file_path
        self.loaded_at = datetime.now()
        self.filters = filters
        self.notes = notes
        self.assembly = assembly


class PyMutation(CancerAnnotateMixin, ActionableMutationMixin, MutationBurdenMixin, MutationalSignatureMixin, PfamAnnotationMixin, SmgDetectionMixin, OutputMixin, ChromSampleFilterMixin, GenomicRangeMixin, PassFilterMixin, TissueExpressionMixin):
    def __init__(self, data: pd.DataFrame, metadata: Optional[MutationMetadata] = None,
                 samples: Optional[List[str]] = None):
        self.data = data
        self.samples = samples if samples is not None else []
        self.metadata = metadata

    def head(self, n: int = 5):
        """
        Return the first n rows of the mutation data.
        This method delegates to the pandas DataFrame head() method.
        """
        return self.data.head(n)

    def info(self):
        """
        Print a concise summary of the mutation data.
        This method delegates to the pandas DataFrame info() method
        """
        return self.data.info()

    def save_figure(self, figure: Figure, filename: str, dpi: int = 300, bbox_inches: str = 'tight',
                    **kwargs) -> None:
        """
        Save a figure with high-quality configuration by default.

        This method centralizes figure saving to ensure all visualizations
        are saved with the best possible quality.

        Args:
            figure: The matplotlib figure to save.
            filename: Filename where to save the figure.
            dpi: Resolution in dots per inch (300 = high quality).
            bbox_inches: Margin adjustment ('tight' = no unnecessary spaces).
            **kwargs: Additional parameters for matplotlib.savefig().

        Examples:
            >>> py_mut = PyMutation(data)
            >>> fig = py_mut.summary_plot()
            >>> py_mut.save_figure(fig, 'my_summary.png')  # Automatic high quality
            >>> py_mut.save_figure(fig, 'my_summary.pdf', dpi=600)  # Very high quality
        """
        figure.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        print(f"📁 Figure saved: {filename} (DPI: {dpi}, margins: {bbox_inches})")

    @staticmethod
    def configure_high_quality_plots():
        """
        Configure matplotlib to generate high-quality plots by default.

        This function modifies matplotlib's global configuration so that
        ALL figures are automatically saved with high quality, without
        needing to specify parameters each time.

        Applied configurations:
        - DPI: 300 (high resolution)
        - bbox_inches: 'tight' (optimized margins)
        - Format: PNG with optimized compression

        Examples:
            >>> PyMutation.configure_high_quality_plots()  # Configure once
            >>> py_mut = PyMutation(data)
            >>> fig = py_mut.summary_plot()
            >>> fig.savefig('plot.png')  # Automatically high quality!

        Note:
            This configuration affects ALL matplotlib figures in the session.
            It's recommended to call this function at the beginning of the script.
        """
        import matplotlib as mpl

        # Configure default DPI for high resolution
        mpl.rcParams['figure.dpi'] = 300
        mpl.rcParams['savefig.dpi'] = 300

        # Configure automatic margins
        mpl.rcParams['savefig.bbox'] = 'tight'

        # Configure format and compression
        mpl.rcParams['savefig.format'] = 'png'
        mpl.rcParams['savefig.transparent'] = False

        # Improve text quality
        mpl.rcParams['savefig.facecolor'] = 'white'
        mpl.rcParams['savefig.edgecolor'] = 'none'

    def summary_plot(self, figsize: Tuple[int, int] = DEFAULT_SUMMARY_FIGSIZE, title: str = DEFAULT_PLOT_TITLE,
                     max_samples: Optional[int] = 200, top_genes_count: int = DEFAULT_TOP_GENES_COUNT,
                     include_silent: bool = False, non_syn_classifications: Optional[set] = None) -> Figure:
        """
        Generate a comprehensive summary plot with general mutation statistics.

        This visualization includes six subplots:
        - Variant Classification: Distribution of variant classifications
        - Variant Type: Distribution of variant types (SNP, INS, DEL, etc.)
        - SNV Class: Distribution of single nucleotide variant classes (A>G, C>T, etc.)
        - Variants per Sample: Distribution of variants per sample with median (TMB)
        - Variant Classification Summary: Boxplot showing variant count variability across samples
        - Top Mutated Genes: Most frequently mutated genes in the cohort

        Args:
            figsize: Figure size as (width, height) in inches.
            title: Main plot title.
            max_samples: Maximum number of samples to show in the variants per sample plot.
                        If None, all samples are shown.
            top_genes_count: Number of top genes to show in the top mutated genes plot.
                        If there are fewer genes than this number, all will be shown.
            include_silent: If False (default, matches maftools' default vc_nonSyn
                        whitelist), silent/synonymous and non-coding variants are
                        excluded from every subplot except SNV Class (which always
                        shows all SNPs, like maftools' TiTv view). Applies uniformly
                        to all six panels, including Top Mutated Genes.
            non_syn_classifications: Custom whitelist of "non-synonymous" variant
                        classifications, mirroring maftools' vc_nonSyn argument.
                        If None, uses maftools' own default whitelist. Ignored
                        when include_silent=True.

        Returns:
            Matplotlib Figure object with the summary plot.
            
        Examples:
            >>> py_mut = PyMutation(data)
            >>> fig = py_mut.summary_plot(max_samples=100, top_genes_count=15)
            >>> py_mut.save_figure(fig, 'summary.png')

            >>> # Include silent mutations in every panel
            >>> fig = py_mut.summary_plot(include_silent=True)
        """
        start_time = time.time()
        logger.info("Generating summary plot...")
        
        from .utils.data_processing import (
            extract_variant_classifications,
            extract_variant_types,
        )
        from .visualizations.summary_plot import _create_summary_plot

        self.data = extract_variant_classifications(self.data, variant_column=VARIANT_CLASSIFICATION_COLUMN,
                                                    funcotation_column=FUNCOTATION_COLUMN)

        self.data = extract_variant_types(self.data, variant_column=VARIANT_TYPE_COLUMN,
                                          funcotation_column=FUNCOTATION_COLUMN)

        fig = _create_summary_plot(self, figsize, title, max_samples, top_genes_count,
                                   include_silent=include_silent,
                                   non_syn_classifications=non_syn_classifications)

        plt.close(fig)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Summary plot generated in {elapsed_time:.2f} seconds")
        
        return fig

    def variant_classification_plot(
        self,
        figsize: Tuple[int, int] = DEFAULT_PLOT_FIGSIZE,
        title: str = "Variant Classification",
        include_silent: bool = False,
        non_syn_classifications: Optional[set] = None,
    ) -> Figure:
        """
        Generate a horizontal bar plot showing the distribution of variant classifications.

        This plot displays the frequency of each variant classification type
        (e.g., Missense_Mutation, Nonsense_Mutation, Frame_Shift_Del) in the dataset.
        By default, excludes silent/synonymous mutations.

        Args:
            figsize: Figure size as (width, height) in inches.
            title: Plot title.
            include_silent: Whether to include silent/synonymous and non-coding variants.
                           Default is False.
            non_syn_classifications: Custom whitelist of "non-synonymous" variant
                           classifications, mirroring maftools' vc_nonSyn argument.
                           If None, uses maftools' own default whitelist. Ignored
                           when include_silent=True.

        Returns:
            Matplotlib Figure object with the variant classification plot.

        Examples:
            >>> py_mut = PyMutation(data)
            >>> fig = py_mut.variant_classification_plot()
            >>> py_mut.save_figure(fig, 'variant_classification.png')
        """
        start_time = time.time()
        logger.info("Generating variant classification plot...")

        from .utils.data_processing import extract_variant_classifications
        from .visualizations.summary_plot import _create_variant_classification_plot

        self.data = extract_variant_classifications(
            self.data,
            variant_column="Variant_Classification",
            funcotation_column="FUNCOTATION",
        )

        fig, ax = plt.subplots(figsize=figsize)
        _create_variant_classification_plot(
            self, ax=ax, set_title=False, include_silent=include_silent,
            non_syn_classifications=non_syn_classifications,
        )

        if title:
            fig.suptitle(title, fontsize=16, fontweight="bold")

        plt.tight_layout()
        plt.close(fig)

        elapsed_time = time.time() - start_time
        logger.info(
            f"Variant classification plot generated in {elapsed_time:.2f} seconds"
        )

        return fig

    def variant_type_plot(
        self,
        figsize: Tuple[int, int] = DEFAULT_PLOT_FIGSIZE,
        title: str = "Variant Type",
        include_silent: bool = False,
        non_syn_classifications: Optional[set] = None,
    ) -> Figure:
        """
        Generate a horizontal bar plot showing the distribution of variant types.

        This plot displays the frequency of each variant type (SNP, INS, DEL, DNP, TNP, ONP)
        in the dataset. By default, excludes silent/synonymous mutations.

        Args:
            figsize: Figure size as (width, height) in inches.
            title: Plot title.
            include_silent: Whether to include silent/synonymous and non-coding variants.
                           Default is False.
            non_syn_classifications: Custom whitelist of "non-synonymous" variant
                           classifications, mirroring maftools' vc_nonSyn argument.
                           If None, uses maftools' own default whitelist. Ignored
                           when include_silent=True.

        Returns:
            Matplotlib Figure object with the variant types plot.

        Examples:
            >>> py_mut = PyMutation(data)
            >>> fig = py_mut.variant_type_plot()
            >>> py_mut.save_figure(fig, 'variant_type.png')
        """
        start_time = time.time()
        logger.info("Generating variant type plot...")
        
        from .utils.data_processing import extract_variant_types
        from .visualizations.summary_plot import _create_variant_type_plot

        self.data = extract_variant_types(self.data, variant_column="Variant_Type", funcotation_column="FUNCOTATION")

        fig, ax = plt.subplots(figsize=figsize)
        _create_variant_type_plot(
            self, ax=ax, set_title=False, include_silent=include_silent,
            non_syn_classifications=non_syn_classifications,
        )

        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.close(fig)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Variant type plot generated in {elapsed_time:.2f} seconds")
        
        return fig

    def snv_class_plot(self, figsize: Tuple[int, int] = DEFAULT_PLOT_FIGSIZE, title: str = "SNV Class",
                       ref_column: str = "REF", alt_column: str = "ALT", 
                       normalize_pyrimidine: bool = True) -> Figure:
        """
        Generate a horizontal bar plot showing the distribution of SNV classes.

        SNV classes represent single nucleotide substitutions (e.g., C>T, G>A). By default, mutations are
        normalized to pyrimidine bases (C or T) as reference, showing 6 canonical classes instead of 12.
        This follows mutational signature analysis conventions.

        Args:
            figsize: Figure size as (width, height) in inches.
            title: Plot title.
            ref_column: Name of the column containing the reference allele.
            alt_column: Name of the column containing the alternative allele.
            normalize_pyrimidine: Whether to normalize to pyrimidine bases (C/T as reference).
                                 Default True (6 canonical classes). Set False for all 12 substitution types.

        Returns:
            Matplotlib Figure object with the SNV classes plot.
            
        Examples:
            >>> py_mut = PyMutation(data)
            >>> fig = py_mut.snv_class_plot()
            >>> py_mut.save_figure(fig, 'snv_class.png')
            
            >>> # Show all 12 substitution types without normalization
            >>> fig = py_mut.snv_class_plot(normalize_pyrimidine=False)
        """
        start_time = time.time()
        logger.info("Generating SNV class plot...")
        
        from .visualizations.summary_plot import _create_snv_class_plot

        fig, ax = plt.subplots(figsize=figsize)
        _create_snv_class_plot(self, ref_column=ref_column, alt_column=alt_column, ax=ax, 
                              set_title=False, normalize_pyrimidine=normalize_pyrimidine)

        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.close(fig)
        
        elapsed_time = time.time() - start_time
        logger.info(f"SNV class plot generated in {elapsed_time:.2f} seconds")
        
        return fig

    def variants_per_sample_plot(
        self,
        figsize: Tuple[int, int] = DEFAULT_PLOT_FIGSIZE,
        title: str = "Variants per Sample",
        variant_column: str = "Variant_Classification",
        sample_column: str = "Tumor_Sample_Barcode",
        max_samples: Optional[int] = 200,
        include_silent: bool = False,
        non_syn_classifications: Optional[set] = None,
    ) -> Figure:
        """
        Generate a stacked bar plot showing the number of variants per sample (TMB).

        This plot displays the tumor mutation burden (TMB) for each sample, with bars
        stacked by variant classification type. A red dashed line indicates the median TMB.
        By default, excludes silent/synonymous mutations.

        Args:
            figsize: Figure size as (width, height) in inches.
            title: Plot title.
            variant_column: Name of the column containing the variant classification.
            sample_column: Name of the column containing the sample identifier.
            max_samples: Maximum number of samples to show. If None, all are shown.
                        Samples are sorted by TMB and the top samples are displayed.
            include_silent: Whether to include silent/synonymous and non-coding variants.
                           Default is False.
            non_syn_classifications: Custom whitelist of "non-synonymous" variant
                           classifications, mirroring maftools' vc_nonSyn argument.
                           If None, uses maftools' own default whitelist. Ignored
                           when include_silent=True.

        Returns:
            Matplotlib Figure object with the variants per sample plot.

        Examples:
            >>> py_mut = PyMutation(data)
            >>> fig = py_mut.variants_per_sample_plot(max_samples=100)
            >>> py_mut.save_figure(fig, 'tmb.png')
        """
        start_time = time.time()
        logger.info("Generating variants per sample plot...")
        
        from .utils.data_processing import extract_variant_classifications
        from .visualizations.summary_plot import _create_variants_per_sample_plot

        if variant_column not in self.data.columns:
            column_lower = variant_column.lower()
            for col in self.data.columns:
                if col.lower() == column_lower:
                    variant_column = col
                    break

        self.data = extract_variant_classifications(self.data, variant_column=variant_column,
                                                    funcotation_column="FUNCOTATION")

        fig, ax = plt.subplots(figsize=figsize)
        _create_variants_per_sample_plot(
            self,
            variant_column=variant_column,
            sample_column=sample_column,
            ax=ax,
            set_title=False,
            max_samples=max_samples,
            include_silent=include_silent,
            non_syn_classifications=non_syn_classifications,
        )

        if title and not title.startswith("Variants per Sample"):
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        elif title:
            fig.suptitle(title, fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.close(fig)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Variants per sample plot generated in {elapsed_time:.2f} seconds")
        
        return fig

    def variant_classification_summary_plot(
        self,
        figsize: Tuple[int, int] = DEFAULT_PLOT_FIGSIZE,
        title: str = "Variant Classification Summary",
        variant_column: str = "Variant_Classification",
        sample_column: str = "Tumor_Sample_Barcode",
        include_silent: bool = False,
        non_syn_classifications: Optional[set] = None,
    ) -> Figure:
        """
        Generate a boxplot showing the distribution of variant counts per sample for each variant classification.

        This plot visualizes the variability between samples for each type of variant classification,
        allowing identification of which variant types show more differences across samples.
        By default, excludes silent/synonymous mutations.

        Args:
            figsize: Figure size as (width, height) in inches.
            title: Plot title.
            variant_column: Name of the column containing the variant classification.
            sample_column: Name of the column containing the sample identifier.
                           If it doesn't exist, samples are assumed to be columns (wide format).
            include_silent: Whether to include silent/synonymous and non-coding variants.
                           Default is False.
            non_syn_classifications: Custom whitelist of "non-synonymous" variant
                           classifications, mirroring maftools' vc_nonSyn argument.
                           If None, uses maftools' own default whitelist. Ignored
                           when include_silent=True.

        Returns:
            Matplotlib Figure object with the boxplot.

        Examples:
            >>> py_mut = PyMutation(data)
            >>> fig = py_mut.variant_classification_summary_plot()
            >>> py_mut.save_figure(fig, 'variant_summary.png')
        """
        start_time = time.time()
        logger.info("Generating variant classification summary plot...")
        
        from .utils.data_processing import extract_variant_classifications
        from .visualizations.summary_plot import (
            _create_variant_classification_summary_plot,
        )

        self.data = extract_variant_classifications(self.data, variant_column=variant_column,
                                                    funcotation_column="FUNCOTATION")

        is_wide_format = sample_column not in self.data.columns
        if is_wide_format:
            sample_cols = [col for col in self.data.columns if
                           col.startswith('TCGA-') or (isinstance(col, str) and col.count('-') >= 2)]
            if sample_cols:
                logger.debug(f"Detected wide format with {len(sample_cols)} possible sample columns.")

        fig, ax = plt.subplots(figsize=figsize)
        _create_variant_classification_summary_plot(
            self,
            variant_column=variant_column,
            sample_column=sample_column,
            ax=ax,
            set_title=False,
            include_silent=include_silent,
            non_syn_classifications=non_syn_classifications,
        )

        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.close(fig)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Variant classification summary plot generated in {elapsed_time:.2f} seconds")
        
        return fig

    def top_mutated_genes_plot(self, figsize: Tuple[int, int] = DEFAULT_PLOT_FIGSIZE, title: str = "Top Mutated Genes",
                               mode: str = MODE_VARIANTS, variant_column: str = VARIANT_CLASSIFICATION_COLUMN,
                               gene_column: str = GENE_COLUMN, sample_column: str = SAMPLE_COLUMN,
                               count: int = DEFAULT_TOP_GENES_COUNT, include_silent: bool = False,
                               non_syn_classifications: Optional[set] = None) -> Figure:
        """
        Generate a horizontal bar plot showing the most mutated genes.

        This plot displays the top mutated genes with bars stacked by variant classification type.
        Two counting modes are available: total variant count or affected sample count.
        By default, excludes silent/synonymous mutations (matching maftools' default
        vc_nonSyn whitelist), consistent with every other plot in this module.

        NOTE: the default for include_silent used to be True in earlier versions of
        this method, which was inconsistent with every other plot here (and with
        maftools). It is now False by default. Pass include_silent=True explicitly
        if you relied on the old behavior of counting every variant classification.

        Args:
            figsize: Figure size as (width, height) in inches.
            title: Plot title.
            mode: Mutation counting mode: "variants" (counts total number of variants)
                  or "samples" (counts number of affected samples).
            variant_column: Name of the column containing the variant classification.
            gene_column: Name of the column containing the gene symbol.
            sample_column: Name of the column containing the sample identifier.
            count: Number of top genes to show.
            include_silent: Whether to include silent/synonymous and non-coding variants.
                           Default is False.
            non_syn_classifications: Custom whitelist of "non-synonymous" variant
                           classifications, mirroring maftools' vc_nonSyn argument.
                           If None, uses maftools' own default whitelist. Ignored
                           when include_silent=True.

        Returns:
            Matplotlib Figure object with the top mutated genes plot.

        Raises:
            ValueError: If 'count' is not a positive integer or 'mode' is not valid.
            
        Examples:
            >>> py_mut = PyMutation(data)
            >>> fig = py_mut.top_mutated_genes_plot(count=20, mode="samples")
            >>> py_mut.save_figure(fig, 'top_genes.png')
            
            >>> # Include silent mutations too
            >>> fig = py_mut.top_mutated_genes_plot(count=10, include_silent=True)
        """
        start_time = time.time()
        logger.info(f"Generating top mutated genes plot (mode={mode}, count={count})...")
        
        from .utils.data_processing import extract_variant_classifications
        from .visualizations.summary_plot import _create_top_mutated_genes_plot

        if not isinstance(count, int):
            raise ValueError(f"The 'count' parameter must be an integer, received: {count}")
        if count <= 0:
            raise ValueError(f"The 'count' parameter must be a positive integer, received: {count}")

        if mode not in VALID_PLOT_MODES:
            raise ValueError(f"Mode '{mode}' is not valid. Allowed values are: {', '.join(VALID_PLOT_MODES)}")

        if variant_column not in self.data.columns:
            column_lower = variant_column.lower()
            for col in self.data.columns:
                if col.lower() == column_lower:
                    variant_column = col
                    break

        if gene_column not in self.data.columns:
            column_lower = gene_column.lower()
            for col in self.data.columns:
                if col.lower() == column_lower:
                    gene_column = col
                    break

        self.data = extract_variant_classifications(self.data, variant_column=variant_column,
                                                    funcotation_column=FUNCOTATION_COLUMN)

        fig, ax = plt.subplots(figsize=figsize)
        _create_top_mutated_genes_plot(self, mode=mode, variant_column=variant_column, gene_column=gene_column,
                                       sample_column=sample_column, count=count, ax=ax, set_title=False,
                                       include_silent=include_silent,
                                       non_syn_classifications=non_syn_classifications)

        if title:
            if mode == "variants" and title == "Top Mutated Genes":
                fig.suptitle("Top mutated genes (variants)", fontsize=16, fontweight='bold', y=0.98)
            elif mode == "samples" and title == "Top Mutated Genes":
                fig.suptitle("Top mutated genes (samples)", fontsize=16, fontweight='bold', y=0.98)
            else:
                fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout(pad=1.2)
        plt.subplots_adjust(left=0.15, right=0.9)
        plt.close(fig)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Top mutated genes plot generated in {elapsed_time:.2f} seconds")
        
        return fig

    def oncoplot(self, figsize: Optional[Tuple[int, int]] = None, title: str = "Oncoplot",
                 gene_column: str = GENE_COLUMN, variant_column: str = VARIANT_CLASSIFICATION_COLUMN,
                 ref_column: str = REF_COLUMN, alt_column: str = ALT_COLUMN, top_genes_count: int = None,
                 max_samples: int = None, include_silent: bool = False,
                 non_syn_classifications: Optional[set] = None,
                 sample_prefix: Optional[str] = None) -> Figure:
        """
        Generates an oncoplot showing mutation patterns in a heatmap.
        
        The oncoplot is a fundamental visualization in cancer genomics that shows
        mutation patterns across samples and genes in heatmap format.
        
        Features:
        - Automatic detection of sample columns (TCGA and .GT format)
        - Optional custom sample-column prefix
        - Support for multiple genotype formats (A|G, A/G, etc.)
        - Multi_Hit detection for samples with multiple mutations
        - Standard color schemes for mutation types
        - Smart gene ordering by mutation frequency
        - Sample ordering by mutational burden
        
        Args:
            figsize: Figure size (width, height) in inches.
                    If None, uses DEFAULT_ONCOPLOT_FIGSIZE.
            title: Title for the visualization.
            gene_column: Name of the column containing gene symbols.
            variant_column: Name of the column containing variant classifications.
            ref_column: Name of the column containing reference alleles.
            alt_column: Name of the column containing alternative alleles.
            top_genes_count: Number of top mutated genes to show.
                           If None, uses DEFAULT_ONCOPLOT_TOP_GENES.
            max_samples: Maximum number of samples to show.
                        If None, uses DEFAULT_ONCOPLOT_MAX_SAMPLES.
            include_silent: If False (default, matches maftools' default
                           behavior exactly), only variant classifications in
                           `non_syn_classifications` count as mutations --
                           this excludes "Silent" but also anything else not
                           in that whitelist (e.g. "Intron", "3'UTR", "IGR").
                           Set to True to count every variant classification
                           as a mutation (no filtering at all).
            non_syn_classifications: Custom whitelist of "non-synonymous"
                           variant classifications to treat as mutations,
                           mirroring maftools' `vc_nonSyn` argument in
                           read.maf(). If None, uses maftools' own default
                           9-classification whitelist (Frame_Shift_Del,
                           Frame_Shift_Ins, Splice_Site,
                           Translation_Start_Site, Nonsense_Mutation,
                           Nonstop_Mutation, In_Frame_Del, In_Frame_Ins,
                           Missense_Mutation). Ignored when
                           include_silent=True.
            sample_prefix: Optional prefix used to identify custom sample
                           columns, for example ``"C3[A::Z]-"``.
            
        Returns:
            plt.Figure: matplotlib Figure object with the oncoplot.
            
        Raises:
            ValueError: If required columns are missing, no mutation data,
                       or problems with data format.
            
        Examples:
            Basic usage:
            >>> py_mut = PyMutation(data)
            >>> fig = py_mut.oncoplot()
            >>> fig.savefig('oncoplot.png')
            
            With custom parameters:
            >>> fig = py_mut.oncoplot(
            ...     title="TCGA Samples Oncoplot",
            ...     top_genes_count=20,
            ...     max_samples=100
            ... )

            Including silent mutations in the counts/percentages:
            >>> fig = py_mut.oncoplot(include_silent=True)

            Using a custom non-synonymous whitelist (like maftools' vc_nonSyn):
            >>> fig = py_mut.oncoplot(
            ...     non_syn_classifications={"Missense_Mutation", "Nonsense_Mutation"}
            ... )
            
        Note:
            - The method automatically detects sample columns using common
              patterns like 'TCGA-*' and '*.GT'
            - Genes are ordered by mutation frequency (most mutated at top)
            - Samples are ordered by total mutational burden
            - Colors follow cancer genomics standards
            - Multi_Hit detection is handled automatically
        """
        # Validate input parameters
        if top_genes_count is None:
            top_genes_count = DEFAULT_ONCOPLOT_TOP_GENES
        if max_samples is None:
            max_samples = DEFAULT_ONCOPLOT_MAX_SAMPLES
        if figsize is None:
            figsize = DEFAULT_ONCOPLOT_FIGSIZE

        # Parameter validation
        if top_genes_count <= 0:
            raise ValueError("top_genes_count must be a positive integer")
        if max_samples <= 0:
            raise ValueError("max_samples must be a positive integer")
        if len(figsize) != 2 or any(x <= 0 for x in figsize):
            raise ValueError("figsize must be a tuple of two positive numbers")

        # Validate required columns
        required_columns = [gene_column, variant_column, ref_column, alt_column]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        try:
            from .visualizations.oncoplot import _create_oncoplot_plot
            # Generate the oncoplot
            fig = _create_oncoplot_plot(py_mut=self, gene_column=gene_column, variant_column=variant_column,
                                        ref_column=ref_column, alt_column=alt_column, top_genes_count=top_genes_count,
                                        max_samples=max_samples,
                                        figsize=figsize, title=title, include_silent=include_silent,
                                        non_syn_classifications=non_syn_classifications,
                                        sample_prefix=sample_prefix)
            plt.close(fig)
            return fig

        except Exception as e:
            raise ValueError(f"Error generating oncoplot: {str(e)}")

    def mutational_signature_plot(self,
                                fasta_file: str,
                                n_signatures: int = 3,
                                sample_column: str = SAMPLE_COLUMN,
                                ref_column: str = REF_COLUMN,
                                alt_column: str = ALT_COLUMN,
                                context_column: Optional[str] = None,
                                cosmic_signatures: Optional[pd.DataFrame] = None,
                                figsize: Tuple[int, int] = (18, 22),
                                title: str = "Mutational Signature Analysis",
                                show_interactive: bool = False) -> Figure:
        """
        Generate a comprehensive mutational signature analysis visualization.
        
        This analysis extracts mutational signatures from the data using Non-negative
        Matrix Factorization (NMF) and creates multiple visualizations:
        
        A. Signature profiles - Bar charts showing the 96 trinucleotide contexts
        B. Cosine similarity - Heatmap comparing with COSMIC signatures
        C. Sample contributions - Heatmap showing signature contributions per sample
        D. Relative contributions - Stacked bar plot of signature proportions
        E. Overall proportions - Donut plot of total signature contributions
        
        Args:
            fasta_file: Path to the reference genome FASTA file.
                        This is required to generate trinucleotide contexts if not present.
            n_signatures: Number of signatures to extract (default: 3)
            sample_column: Column containing sample identifiers
            ref_column: Column containing reference alleles
            alt_column: Column containing alternative alleles
            context_column: Column containing trinucleotide context (optional)
                          If None, will try to auto-detect context column
            cosmic_signatures: DataFrame with COSMIC reference signatures (optional)
                             If None, will use synthetic signatures for demonstration
            figsize: Overall figure size (default: (20, 24))
            title: Main title for the analysis
            show_interactive: If True, display the plot interactively
            
        Returns:
            matplotlib.figure.Figure: Complete mutational signature analysis figure
            
        Example:
            >>> # Basic usage
            >>> fig = py_mut.mutational_signature_plot(n_signatures=3)
            >>> fig.savefig('mutational_signatures.png')
            
            >>> # With COSMIC comparison
            >>> cosmic_df = pd.read_csv('COSMIC_signatures.tsv', sep='\t', index_col=0)
            >>> fig = py_mut.mutational_signature_plot(
            ...     n_signatures=4,
            ...     cosmic_signatures=cosmic_df,
            ...     figsize=(24, 28)
            ... )
        """
        logger.warning(
            "mutational_signature_plot is currently deprecated. "
            "Please use individual signature visualization methods:\n"
            "- signature_bar_chart() for signature profiles\n"
            "- cosine_similarity_heatmap() for COSMIC comparison\n"
            "- signature_contribution_heatmap() for sample contributions\n"
            "- signature_stacked_bar_chart() for stacked bar visualization\n"
            "- signature_donut_plot() for global proportions"
        )
        
        # For now, return the signature bar chart as the main visualization
        return self.signature_bar_chart(
            ref_genome=fasta_file,
            n_signatures=n_signatures,
            sample_column=sample_column,
            ref_column=ref_column,
            alt_column=alt_column,
            context_column=context_column,
            figsize=figsize,
            title=title
        )
    
    def signature_bar_chart(self,
                           ref_genome: str,
                           n_signatures: int = 3,
                           cosmic_path: Optional[Union[str, bool]] = None,
                           figsize: Optional[Tuple[int, int]] = None,
                           title: str = "Mutational Signature Profiles",
                           update_data: bool = False,
                           ignoreChr: Optional[list] = None,
                           pConstant: Optional[float] = None,
                           exclude_artifacts: bool = True,
                           aetiology_path: Optional[str] = None) -> Figure:
        """
        Generate a bar chart visualization showing mutational signature profiles.
        
        This creates signature profile bar charts showing the 96 trinucleotide contexts
        for each identified mutational signature, with colors representing different
        substitution types (C>A, C>G, C>T, T>A, T>C, T>G).

        The visualization follows COSMIC standards with:
        - One panel per signature (stacked vertically)
        - 96 trinucleotide context bars per panel
        - Color-coded by substitution type
        - Percentage-based Y-axis
        - Trinucleotide labels on the bottom panel

        Args:
            ref_genome: Path to the reference genome FASTA file.
                        Required to generate trinucleotide contexts if not present.
            n_signatures: Number of signatures to extract (default: 3)
            cosmic_path: Path to a COSMIC-format SBS catalog file. If None
                        (default), uses the official bundled COSMIC v3.4 SBS
                        (GRCh37) catalog automatically - no need to supply a
                        file. Pass "maftools" to use the older, 67-signature
                        catalog bundled with maftools instead, a custom path
                        to use your own catalog, or False to skip COSMIC
                        alignment entirely and show raw NMF signature shapes.
            figsize: Figure size (width, height). If None, automatically calculated as (12, 3 * n_signatures)
            title: Plot title
            update_data: If True, permanently replaces self.data with the
                        filtered SNV-only subset used to compute the trinucleotide
                        matrix (indels/multiallelic/unresolved-context variants
                        dropped). Off by default — only enable this if you
                        specifically want to keep working with that reduced,
                        signature-ready subset for the rest of your session.
                        See trinucleotideMatrix()'s update_data docs for details.
            ignoreChr: Chromosomes to exclude before building the context matrix
                        (e.g. ['X']). Passed straight through to trinucleotideMatrix().
            pConstant: Small positive pseudo-count added to the context matrix
                        before NMF. Passed straight through to extractSignatures().
            exclude_artifacts: If True (default), known artifact/sequencing-noise
                        COSMIC signatures (SBS27, SBS43, SBS45-60, SBS95, and any
                        signature ending in 'c' or containing 'artefact') are
                        excluded when matching against cosmic_path. Set to False
                        to allow matches against those too — e.g. to reproduce
                        maftools' plotSignatures(), which doesn't exclude them.
            aetiology_path: Path to a two-column TSV (Signature, Aetiology), e.g.
                        data/examples/maftools_SBS_aetiology.tsv. When given,
                        each panel's title gets a second line with the proposed
                        aetiology of its matched signature, matching maftools'
                        plotSignatures() (which always shows it).

        Returns:
            matplotlib.figure.Figure: Figure with signature bar charts

        Raises:
            ValueError: If no mutations found or ref_genome not provided when needed

        Example:
            >>> # Display in notebook
            >>> py_mut.signature_bar_chart(ref_genome="genome.fa", n_signatures=3)
            >>> 
            >>> # Get figure for saving or further customization
            >>> fig = py_mut.signature_bar_chart(ref_genome="genome.fa", n_signatures=3)
            >>> fig.savefig('signature_profiles.pdf', dpi=300)
            >>> 
            >>> # Suppress display in notebook by assigning to variable with semicolon
            >>> fig = py_mut.signature_bar_chart(ref_genome="genome.fa", n_signatures=3);
        
        Notes:
            Uses trinucleotideMatrix() to generate context matrix and
            extractSignatures() to extract mutational signatures using NMF.
        """
        from .analysis.mutational_signature import (
            align_signatures_to_cosmic,
            extractSignatures,
        )
        from .visualizations.mutational_signature_analysis import (
            create_signature_bar_chart,
        )
        
        start_time = time.time()
        logger.info("Generating signature bar chart...")
        
        # Generate trinucleotide matrix from mutation data. By default this does
        # NOT touch self.data (see update_data docstring above) — pass
        # update_data=True explicitly if you want to permanently keep the
        # filtered/enriched SNV-only subset as self.data going forward.
        contexts_df, _ = self.trinucleotideMatrix(
            ref_genome=ref_genome, update_data=update_data, ignoreChr=ignoreChr
        )

        # Extract mutational signatures using NMF
        try:
            result = extractSignatures(contexts_df, n=n_signatures, pConstant=pConstant)
        except Exception as e:
            raise ValueError(f"Signature extraction failed: {e}") from e
        
        # Align with COSMIC catalog unless explicitly disabled (cosmic_path=False)
        cosmic_path = _resolve_cosmic_catalog_path(cosmic_path)
        aetiology_path = _resolve_aetiology_catalog_path(aetiology_path)
        if cosmic_path is not False:
            W_aligned, _, signature_names = align_signatures_to_cosmic(
                W=result['signatures'],
                H=result['contributions'],
                cosmic_path=cosmic_path,
                min_cosine=0.5,
                exclude_artifacts=exclude_artifacts,
                aetiology_path=aetiology_path
            )
            signatures = W_aligned.values
        else:
            signatures = result['signatures'].values
            signature_names = [f"Signature {i+1}" for i in range(n_signatures)]
        
        # Panel titles grow to two lines when aetiology_path is set ("SBSx-like
        # (cos=..)" + "Aetiology: ..."), so they need extra vertical room —
        # otherwise the top panel's title collides with the figure suptitle.
        has_aetiology_line = aetiology_path is not None
        if figsize is None and has_aetiology_line:
            figsize = (12, 3.6 * n_signatures)

        # Create visualization
        fig, _ = create_signature_bar_chart(
            signatures=signatures,
            signature_names=signature_names,
            figsize=figsize
        )

        # Add main title with proper spacing
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
            top_margin = 0.90 if has_aetiology_line else 0.93
            plt.subplots_adjust(top=top_margin, hspace=0.55 if has_aetiology_line else 0.4)
        
        # Close to prevent automatic display in notebooks
        plt.close(fig)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Signature bar chart generated in {elapsed_time:.2f}s")
        
        return fig

    def cosine_similarity_heatmap(
        self,
        ref_genome: str,
        cosmic_path: Optional[str] = None,
        n_signatures: int = 3,
        prefix: Optional[str] = None,
        add: bool = True,
        ignoreChr: Optional[list] = None,
        useSyn: bool = True,
        signature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 3),
        title: str = "Cosine Similarity: Extracted vs COSMIC Signatures",
        update_data: bool = False,
        pConstant: Optional[float] = None,
    ) -> Figure:
        """
        Generate cosine similarity heatmap comparing extracted signatures with COSMIC catalog.

        This method creates a heatmap visualization showing the cosine similarity between
        mutational signatures extracted from your data and the COSMIC reference catalog.
        Darker blue indicates higher similarity, helping identify which known cancer
        signatures are present in your dataset.

        The workflow:
        1. Extract trinucleotide context matrix from mutations
        2. Perform NMF decomposition to extract signatures
        3. Load COSMIC reference signatures
        4. Calculate cosine similarity between extracted and COSMIC signatures
        5. Visualize as a k * m heatmap (k extracted * m COSMIC signatures)

        Parameters
        ----------
        ref_genome : str
            Path to reference genome FASTA file (e.g., "hg38.fa").
            Used to extract trinucleotide contexts around each mutation.
        cosmic_path : Optional[str], default None
            Path to COSMIC signature catalog TSV file. If None (default),
            uses the official bundled COSMIC v3.4 SBS (GRCh37) catalog
            automatically. Pass "maftools" to use the bundled maftools
            catalog instead, or a custom path to use your own.
            Must have 96 rows (trinucleotide contexts) and columns for each signature.
        n_signatures : int, default 3
            Number of mutational signatures to extract from the data.
            Choose based on estimateSignatures() results or domain knowledge.
        prefix : Optional[str], default None
            Prefix to add to chromosomes when accessing reference genome.
            Use "chr" if genome uses "chr1, chr2" format.
        add : bool, default True
            If True, adds trinucleotide context as new column to the data.
        ignoreChr : Optional[list], default None
            Chromosomes to ignore (e.g., ['chrM', 'chrY']).
        useSyn : bool, default True
            If True, includes synonymous mutations in the analysis.
        signature_names : Optional[List[str]], default None
            Custom names for extracted signatures (e.g., ["Sig A", "Sig B"]).
            If None, uses "Signature 1", "Signature 2", etc.
        figsize : Tuple[int, int], default (14, 3)
            Figure size as (width, height) in inches.
        title : str, default "Cosine Similarity: Extracted vs COSMIC Signatures"
            Title for the plot. Set to empty string "" to hide the title.
        update_data : bool, default False
            If True, permanently replaces self.data with the filtered SNV-only
            subset used to compute the trinucleotide matrix. Off by default,
            matching maftools' behavior of never mutating your mutation table
            as a side effect of a plotting call. See trinucleotideMatrix()'s
            update_data docs for details.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the cosine similarity heatmap.
            Can be displayed with plt.show() or saved with fig.savefig().

        Raises
        ------
        ValueError
            If no mutations found, invalid parameters, or COSMIC file invalid
        FileNotFoundError
            If ref_genome or cosmic_path files don't exist

        Notes
        -----
        - The trinucleotide contexts in your data and COSMIC catalog must match.
          This is handled automatically by using the standard 96-context order.
        - Cosine similarity ranges from 0 (completely different) to 1 (identical).
          Values > 0.6 typically indicate meaningful similarity.
        - Artifact signatures (e.g., SBS27, SBS43-60, SBS95) are automatically
          filtered from the COSMIC catalog.

        Examples
        --------
        >>> # Basic usage with 3 signatures
        >>> fig = py_mut.cosine_similarity_heatmap(
        ...     ref_genome="data/hg38.fa",
        ...     cosmic_path="data/COSMIC_v3.4_SBS_GRCh38.txt",
        ...     n_signatures=3
        ... )
        >>> plt.show()

        >>> # Advanced: custom signature names and figure size
        >>> fig = py_mut.cosine_similarity_heatmap(
        ...     ref_genome="data/hg38.fa",
        ...     cosmic_path="data/COSMIC_v3.4_SBS_GRCh38.txt",
        ...     n_signatures=4,
        ...     signature_names=["UV", "Smoking", "APOBEC", "Unknown"],
        ...     figsize=(18, 5),
        ...     prefix="chr",
        ...     ignoreChr=["chrM", "chrY"]
        ... )
        >>> py_mut.save_figure(fig, "cosine_similarity.png", dpi=300)

        See Also
        --------
        trinucleotideMatrix : Extract trinucleotide context matrix
        extractSignatures : Perform NMF to extract signatures
        estimateSignatures : Determine optimal number of signatures
        signature_bar_chart : Visualize individual signature profiles
        """
        import time

        from .analysis.mutational_signature import extractSignatures
        from .visualizations.mutational_signature_analysis import (
            create_cosine_similarity_heatmap,
        )
        start_time = time.time()
        
        logger.info("Generating cosine similarity heatmap...")
        
        contexts_df, _ = self.trinucleotideMatrix(
            ref_genome=ref_genome,
            prefix=prefix,
            add=add,
            ignoreChr=ignoreChr,
            useSyn=useSyn,
            update_data=update_data
        )
        
        if contexts_df.sum().sum() == 0:
            raise ValueError(
                "No mutations found in trinucleotide context matrix. "
                "Check your data and reference genome."
            )
        
        result = extractSignatures(
            contexts_df=contexts_df,
            n=n_signatures,
            parallel=4,
            pConstant=pConstant
        )

        W = result['signatures'].values
        
        if signature_names is None:
            signature_names = [f"Signature {i+1}" for i in range(n_signatures)]
        
        fig, _ = create_cosine_similarity_heatmap(
            W=W,
            cosmic_path=_resolve_cosmic_catalog_path(cosmic_path),
            signature_names=signature_names,
            figsize=figsize,
            title=title
        )
        
        # Close to prevent display in notebooks
        plt.close(fig)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Cosine similarity heatmap generated in {elapsed_time:.2f}s")
        
        return fig

    def signature_contribution_heatmap(self,
                                      ref_genome: str,
                                      n_signatures: int = 3,
                                      sample_column: str = SAMPLE_COLUMN,
                                      prefix: Optional[str] = None,
                                      add: bool = True,
                                      ignoreChr: Optional[list] = None,
                                      useSyn: bool = True,
                                      signature_names: Optional[List[str]] = None,
                                      cosmic_path: Optional[Union[str, bool]] = None,
                                      figsize: Optional[Tuple[float, float]] = None,
                                      cmap: str = 'Blues',
                                      show_values: bool = False,
                                      show_labels: bool = True,
                                      title: str = "Signature Contributions per Sample",
                                      update_data: bool = False,
                                      exclude_artifacts: bool = True,
                                      aetiology_path: Optional[str] = None) -> Figure:
        """
        Generate heatmap showing relative signature contributions per sample (Panel C).
        
        This visualization shows the intensity of each mutational signature's contribution
        to each sample using color-coded cells. Darker colors indicate higher contributions.
        Each sample (column) is normalized independently to sum to 1.0, allowing fair
        comparison across samples with different total mutation counts.
        
        The workflow:
        1. Extract trinucleotide context matrix from mutations
        2. Perform NMF decomposition to extract signatures
        3. Optionally align with COSMIC catalog for signature naming
        4. Normalize contributions per sample (columns sum to 1)
        5. Visualize as k * n heatmap (k signatures * n samples)
        
        Parameters
        ----------
        ref_genome : str
            Path to reference genome FASTA file (e.g., "hg38.fa").
            Used to extract trinucleotide contexts around each mutation.
        n_signatures : int, default 3
            Number of mutational signatures to extract from the data.
        sample_column : str, default "Tumor_Sample_Barcode"
            Column name containing sample identifiers.
        prefix : Optional[str], default None
            Prefix to add to chromosomes when accessing reference genome.
            Use "chr" if genome uses "chr1, chr2" format.
        add : bool, default True
            If True, adds trinucleotide context as new column to the data.
        ignoreChr : Optional[list], default None
            Chromosomes to ignore (e.g., ['chrM', 'chrY']).
        useSyn : bool, default True
            If True, includes synonymous mutations in the analysis.
        signature_names : Optional[List[str]], default None
            Custom names for signatures (e.g., ["SBS1-like", "SBS5-like"]).
            If None, uses generic names like "Signature 1", "Signature 2", etc.
        cosmic_path : Optional[Union[str, bool]], default None
            Path to a COSMIC-format SBS catalog file. If None (default), uses
            the official bundled COSMIC v3.4 SBS (GRCh37) catalog automatically
            to align and rename signatures (e.g., "SBS1-like (cos=0.95)").
            Pass "maftools" to use the bundled maftools catalog instead, a
            custom path to use your own, or False to skip COSMIC alignment.
        figsize : Optional[Tuple[float, float]], default None
            Figure size as (width, height) in inches.
            If None, automatically calculated based on number of samples.
        cmap : str, default 'Blues'
            Colormap for the heatmap (e.g., 'Blues', 'Reds', 'viridis').
        show_values : bool, default False
            Whether to annotate cells with numerical values (0.00-1.00).
            Only recommended for small datasets (<20 samples).
        show_labels : bool, default True
            Whether to show sample labels on X-axis. Set to False for large datasets
            or composite plots to reduce clutter.
        title : str, default "Signature Contributions per Sample"
            Title for the plot.
        update_data : bool, default False
            If True, permanently replaces self.data with the filtered SNV-only
            subset used to compute the trinucleotide matrix. Off by default,
            matching maftools' behavior of never mutating your mutation table
            as a side effect of a plotting call. See trinucleotideMatrix()'s
            update_data docs for details.
        
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the signature contribution heatmap.
        
        Raises
        ------
        ValueError
            If no mutations found or invalid parameters
        FileNotFoundError
            If ref_genome file doesn't exist
        
        Notes
        -----
        - Each column (sample) is normalized independently to sum to 1.0
        - Samples with zero mutations are displayed as white/NaN
        - Color intensity represents relative contribution (0 = none, 1 = 100%)
        - X-axis labels are hidden if more than 50 samples for readability
        
        Examples
        --------
        >>> # Basic usage
        >>> fig = py_mut.signature_contribution_heatmap(
        ...     ref_genome="data/hg38.fa",
        ...     n_signatures=3
        ... )
        >>> plt.show()
        
        >>> # With COSMIC alignment
        >>> fig = py_mut.signature_contribution_heatmap(
        ...     ref_genome="data/hg38.fa",
        ...     cosmic_path="data/COSMIC_v3.4_SBS_GRCh38.txt",
        ...     n_signatures=3,
        ...     figsize=(16, 4),
        ...     cmap='viridis'
        ... )
        >>> py_mut.save_figure(fig, "contributions.png", dpi=300)
        
        See Also
        --------
        trinucleotideMatrix : Extract trinucleotide context matrix
        extractSignatures : Perform NMF to extract signatures
        signature_bar_chart : Visualize signature profiles
        cosine_similarity_heatmap : Compare with COSMIC signatures
        """
        import time
        
        from .analysis.mutational_signature import (
            align_signatures_to_cosmic,
            extractSignatures,
        )
        from .visualizations.mutational_signature_analysis import (
            create_signature_contribution_heatmap,
        )
        
        start_time = time.time()
        
        logger.info("Generating signature contribution heatmap...")

        # Get MAF sample order (alphabetically sorted)
        maf_sample_order = sorted(pd.unique(self.data[SAMPLE_COLUMN]).tolist())
        
        # Generate trinucleotide context matrix
        contexts_df, _ = self.trinucleotideMatrix(
            ref_genome=ref_genome,
            prefix=prefix,
            add=add,
            ignoreChr=ignoreChr,
            useSyn=useSyn,
            update_data=update_data
        )
        
        if contexts_df.sum().sum() == 0:
            raise ValueError(
                "No mutations found in trinucleotide context matrix. "
                "Check your data and reference genome."
            )
        
        result = extractSignatures(
            contexts_df=contexts_df,
            n=n_signatures,
            parallel=4
        )
        
        # Get contributions DataFrame and reorder to match MAF order
        contributions_df = result['contributions']
        common_samples = [s for s in maf_sample_order if s in contributions_df.columns]
        contributions_df = contributions_df[common_samples]
        
        # Optionally align with COSMIC for signature naming (unless explicitly disabled)
        cosmic_path = _resolve_cosmic_catalog_path(cosmic_path)
        aetiology_path = _resolve_aetiology_catalog_path(aetiology_path)
        if cosmic_path is not False:
            _, H_aligned, signature_names_aligned = align_signatures_to_cosmic(
                W=result['signatures'],
                H=contributions_df,
                cosmic_path=cosmic_path,
                min_cosine=0.5,
                exclude_artifacts=exclude_artifacts,
                aetiology_path=aetiology_path
            )
            
            contributions_df = H_aligned
            if signature_names is None:
                signature_names = signature_names_aligned
        
        # Create contribution heatmap
        fig, ax = create_signature_contribution_heatmap(
            contributions=contributions_df,
            signature_names=signature_names,
            figsize=figsize,
            cmap=cmap,
            show_values=show_values,
            show_labels=show_labels
        )
        
        # Add title with extra padding to avoid overlap
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
            plt.subplots_adjust(top=0.92)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Signature contribution heatmap generated in {elapsed_time:.2f}s")
        
        plt.close(fig)
        
        return fig

    def signature_contribution_barplot(self,
                                      ref_genome: str,
                                      n_signatures: int = 3,
                                      sample_column: str = SAMPLE_COLUMN,
                                      prefix: Optional[str] = None,
                                      add: bool = True,
                                      ignoreChr: Optional[list] = None,
                                      useSyn: bool = True,
                                      signature_names: Optional[List[str]] = None,
                                      figsize: Tuple[int, int] = (10, 8),
                                      title: str = "Relative Signature Contributions",
                                      update_data: bool = False) -> Figure:
        """
        Generate a bar plot showing relative signature contributions across the cohort.
        
        This method is currently deprecated in favor of signature_donut_plot which provides
        a better visualization of global signature proportions.
        
        Args:
            ref_genome: Path to reference genome FASTA file
            n_signatures: Number of signatures to extract
            sample_column: Column containing sample identifiers
            prefix: Optional prefix for chromosome names
            add: If True, add trinucleotide context to data
            ignoreChr: List of chromosomes to exclude
            useSyn: If True, include synonymous mutations
            signature_names: Optional custom signature names
            figsize: Figure size
            title: Plot title
            update_data: If True, permanently replaces self.data with the
                filtered SNV-only subset used internally. Off by default.
            
        Returns:
            matplotlib.figure.Figure: Figure with bar plot
        """
        logger.warning(
            "signature_contribution_barplot is deprecated. "
            "Use signature_donut_plot for better visualization of global signature proportions."
        )
        
        # Redirect to donut plot for now
        return self.signature_donut_plot(
            ref_genome=ref_genome,
            n_signatures=n_signatures,
            sample_column=sample_column,
            prefix=prefix,
            add=add,
            ignoreChr=ignoreChr,
            useSyn=useSyn,
            signature_names=signature_names,
            figsize=figsize,
            title=title,
            update_data=update_data
        )

    def signature_donut_plot(
        self,
        ref_genome: str,
        n_signatures: int = 3,
        prefix: Optional[str] = None,
        add: bool = True,
        ignoreChr: Optional[list] = None,
        useSyn: bool = True,
        signature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (8, 8),
        title: str = "Relative Contribution of Mutational Signatures",
        update_data: bool = False,
    ) -> Figure:
        """
        Generate a donut plot showing the global proportion of each signature across the cohort.

        This visualization shows the relative contribution of each signature based on
        absolute mutation counts summed across all samples. It provides a cohort-level
        overview of mutational processes.

        The donut plot uses absolute mutation counts to calculate global contributions:
        - Each signature's total mutations across all samples are summed
        - Final proportions show which signatures explain the most mutations cohort-wide

        Args:
            ref_genome: Path to reference genome FASTA file for trinucleotide context extraction
            n_signatures: Number of signatures to extract via NMF
            prefix: Optional prefix for chromosome names (e.g., 'chr')
            add: If True, add trinucleotide context to data
            ignoreChr: List of chromosomes to exclude from analysis
            useSyn: If True, include synonymous mutations
            signature_names: Optional custom names for signatures. If None, uses generic names.
            figsize: Figure size (width, height) in inches
            title: Plot title
            update_data: If True, permanently replaces self.data with the
                filtered SNV-only subset used to compute the trinucleotide
                matrix. Off by default, matching maftools' behavior of never
                mutating your mutation table as a side effect of a plotting
                call. See trinucleotideMatrix()'s update_data docs for details.

        Returns:
            matplotlib.figure.Figure: Figure with donut plot

        Raises:
            ValueError: If no mutations found or invalid parameters
            FileNotFoundError: If reference genome file not found

        Examples:
            >>> # Basic usage
            >>> fig = py_mut.signature_donut_plot(
            ...     ref_genome="path/to/genome.fa",
            ...     n_signatures=3
            ... )

            >>> # With custom signature names
            >>> fig = py_mut.signature_donut_plot(
            ...     ref_genome="path/to/genome.fa",
            ...     n_signatures=3,
            ...     signature_names=["Age-related", "UV damage", "Smoking"]
            ... )
        """
        import time

        from .analysis.mutational_signature import extractSignatures
        from .visualizations.mutational_signature_analysis import (
            create_signature_donut_plot,
        )

        start_time = time.time()

        logger.info("Generating signature donut plot...")

        # Generate trinucleotide matrix
        contexts_df, _ = self.trinucleotideMatrix(
            ref_genome=ref_genome,
            prefix=prefix,
            add=add,
            ignoreChr=ignoreChr,
            useSyn=useSyn,
            update_data=update_data,
        )

        if contexts_df.sum().sum() == 0:
            raise ValueError("No mutations found in trinucleotide matrix")

        # Extract signatures via NMF
        result = extractSignatures(contexts_df, n=n_signatures)

        # Get absolute contributions
        contributions_abs = result["contributions_abs"].values

        # Use provided signature names or defaults
        if signature_names is None:
            signature_names = result["contributions_abs"].index.tolist()

        # Create donut plot
        fig, ax = create_signature_donut_plot(
            contributions_abs=contributions_abs,
            signature_names=signature_names,
            figsize=figsize,
            title=title,
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Signature donut plot generated in {elapsed_time:.2f} seconds")

        plt.close(fig)
        return fig

    def signature_stacked_bar_chart(
        self,
        ref_genome: str,
        n_signatures: int = 3,
        sample_column: str = SAMPLE_COLUMN,
        prefix: Optional[str] = None,
        add: bool = True,
        ignoreChr: Optional[list] = None,
        useSyn: bool = True,
        signature_names: Optional[List[str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: str = "Signature Contributions per Sample",
        sort_samples: bool = False,
        max_samples: Optional[int] = None,
        show_labels: bool = True,
        update_data: bool = False,
    ) -> Figure:
        """
        Generate a stacked bar chart showing signature contributions per sample.

        Each bar represents one sample, with stacked colors showing the relative
        contribution of each mutational signature. This visualization helps identify
        which signatures are dominant in each sample.

        Args:
            ref_genome: Path to reference genome FASTA file for trinucleotide context extraction
            n_signatures: Number of signatures to extract via NMF
            sample_column: Column name containing sample identifiers
            prefix: Optional prefix for chromosome names (e.g., 'chr')
            add: If True, add trinucleotide context to data
            ignoreChr: List of chromosomes to exclude from analysis
            useSyn: If True, include synonymous mutations
            signature_names: Optional custom names for signatures. If None, uses generic names.
            figsize: Figure size (width, height) in inches. Auto-calculated if None.
            title: Plot title
            sort_samples: If True, sort samples by dominant signature. Default is False (original order).
            max_samples: Maximum number of samples to display. If None, shows all.
            show_labels: If True, show sample labels on X-axis. Default is True.
            update_data: If True, permanently replaces self.data with the
                filtered SNV-only subset used to compute the trinucleotide
                matrix. Off by default, matching maftools' behavior of never
                mutating your mutation table as a side effect of a plotting
                call. See trinucleotideMatrix()'s update_data docs for details.

        Returns:
            matplotlib.figure.Figure: Figure with stacked bar chart

        Examples:
            >>> fig = py_mut.signature_stacked_bar_chart(
            ...     ref_genome="path/to/genome.fa",
            ...     n_signatures=3,
            ...     max_samples=50
            ... )
        """
        import time

        from .analysis.mutational_signature import extractSignatures
        from .visualizations.mutational_signature_analysis import (
            create_signature_stacked_bar_chart,
        )
        
        start_time = time.time()
        
        logger.info("Generating signature stacked bar chart...")
        
        # Get MAF sample order (alphabetically sorted for consistency)
        maf_sample_order = sorted(pd.unique(self.data[SAMPLE_COLUMN]).tolist())
        
        # Generate trinucleotide matrix
        contexts_df, _ = self.trinucleotideMatrix(
            ref_genome=ref_genome,
            prefix=prefix,
            add=add,
            ignoreChr=ignoreChr,
            useSyn=useSyn,
            update_data=update_data
        )
        
        if contexts_df.sum().sum() == 0:
            raise ValueError("No mutations found in trinucleotide matrix")
        
        # Extract signatures via NMF
        result = extractSignatures(contexts_df, n=n_signatures)
        
        # Get contributions and reorder to match MAF sample order
        contributions_df = result['contributions']
        common_samples = [s for s in maf_sample_order if s in contributions_df.columns]
        contributions_df = contributions_df[common_samples]
        
        H = contributions_df.values
        sample_names = contributions_df.columns.tolist()
        
        # Use provided signature names or defaults
        if signature_names is None:
            signature_names = contributions_df.index.tolist()
        
        # Create stacked bar chart
        fig, ax = create_signature_stacked_bar_chart(
            H=H,
            sample_names=sample_names,
            signature_names=signature_names,
            figsize=figsize,
            title=title,
            sort_samples=sort_samples,
            max_samples=max_samples,
            show_labels=show_labels
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Signature stacked bar chart generated in {elapsed_time:.2f} seconds")
        
        plt.close(fig)
        return fig

    def mutational_signature_analysis(
        self,
        ref_genome: str,
        cosmic_path: Optional[str] = None,
        n_signatures: int = 3,
        sample_column: str = SAMPLE_COLUMN,
        prefix: Optional[str] = None,
        add: bool = True,
        ignoreChr: Optional[list] = None,
        useSyn: bool = True,
        signature_names: Optional[List[str]] = None,
        selected_signatures: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (24, 12),
        title: str = "Mutational Signature Analysis",
        update_data: bool = False,
        pConstant: Optional[float] = None,
    ) -> Figure:
        """
        Create a comprehensive mutational signature analysis visualization.

        This method combines all five signature visualizations into a single figure:
        A. Signature Bar Charts (96-context profiles for each signature)
        B. Cosine Similarity Heatmap (comparison with COSMIC signatures)
        C. Contribution Heatmap (absolute contributions per sample)
        D. Stacked Bar Chart (relative contributions per sample)
        E. Donut Plot (overall signature distribution)

        This is the main visualization method that provides a complete overview
        of mutational signature analysis results, ideal for publications and
        comprehensive reports.

        Parameters
        ----------
        ref_genome : str
            Reference genome FASTA file path for trinucleotide context extraction.
        cosmic_path : Optional[str], default None
            Path to COSMIC signature catalog file (TSV format). If None
            (default), uses the official bundled COSMIC v3.4 SBS (GRCh37)
            catalog automatically. Pass "maftools" to use the bundled
            maftools catalog instead, or a custom path to use your own.
        n_signatures : int, default 3
            Number of signatures to extract via NMF decomposition.
        sample_column : str, default "Tumor_Sample_Barcode"
            Column name containing sample identifiers.
        prefix : Optional[str], default None
            Path prefix for intermediate file storage. If None, uses current directory.
        add : bool, default True
            Whether to include additional mutation types beyond SNVs.
        ignoreChr : Optional[list], default None
            List of chromosomes to exclude from analysis (e.g., ['chrX', 'chrY', 'chrM']).
        useSyn : bool, default True
            Whether to include synonymous mutations in the analysis.
        signature_names : Optional[List[str]], default None
            Custom names for each signature. If None, uses generic names like
            "Signature 1", "Signature 2", etc. Must match n_signatures length.
        selected_signatures : Optional[List[str]], default None
            Subset of signatures to plot (e.g., ["Signature 1", "Signature 3"]).
            If None, all signatures are plotted. This allows filtering to focus
            on specific signatures of interest. Signature names must exist in
            signature_names or be within the range of extracted signatures.
        figsize : Tuple[int, int], default (24, 12)
            Figure size as (width, height) in inches. Optimized for 2-column layout
            with equal width columns (1/2 left, 1/2 right).
        title : str, default "Mutational Signature Analysis"
            Main title for the entire figure.
        update_data : bool, default False
            If True, permanently replaces self.data with the filtered SNV-only
            subset used to compute the trinucleotide matrix (indels, multiallelic
            variants, and unresolved-context SNVs dropped; synonymous variants
            too if useSyn=False). Off by default, matching maftools' behavior
            of never mutating your mutation table as a side effect of a
            plotting/analysis call. See trinucleotideMatrix()'s update_data
            docs for details.

        Returns
        -------
        Figure
            Matplotlib figure containing all five visualization panels.

        Raises
        ------
        ValueError
            If signature_names length doesn't match n_signatures.
            If COSMIC catalog file is invalid or doesn't exist.
            If mutation data is insufficient for analysis.
        FileNotFoundError
            If ref_genome or cosmic_path files don't exist.

        Notes
        -----
        Layout structure:
        - Top section (A): Signature bar charts, one per signature
        - Middle (B): Cosine similarity heatmap spanning full width
        - Middle (C): Contribution heatmap spanning full width
        - Bottom-left (D): Stacked bar chart
        - Bottom-right (E): Donut plot

        The method automatically handles:
        - Trinucleotide context matrix generation
        - NMF signature extraction
        - Signature normalization
        - COSMIC catalog loading and validation
        - Color consistency across all panels
        - Proper spacing and layout

        Examples
        --------
        >>> # Basic usage with default parameters
        >>> py_mut = PyMutation(maf_data)
        >>> fig = py_mut.mutational_signature_analysis(
        ...     ref_genome='path/to/genome.fa',
        ...     cosmic_path='path/to/COSMIC_v3.4_SBS_GRCh38.txt',
        ...     n_signatures=3
        ... )
        >>> py_mut.save_figure(fig, 'complete_signature_analysis.png', dpi=300)

        >>> # With custom signature names
        >>> fig = py_mut.mutational_signature_analysis(
        ...     ref_genome='path/to/genome.fa',
        ...     cosmic_path='path/to/COSMIC_v3.4_SBS_GRCh38.txt',
        ...     n_signatures=3,
        ...     signature_names=['Aging', 'UV-exposure', 'APOBEC']
        ... )

        >>> # Excluding sex chromosomes and mitochondrial DNA
        >>> fig = py_mut.mutational_signature_analysis(
        ...     ref_genome='path/to/genome.fa',
        ...     cosmic_path='path/to/COSMIC_v3.4_SBS_GRCh38.txt',
        ...     n_signatures=4,
        ...     ignoreChr=['chrX', 'chrY', 'chrM']
        ... )

        >>> # Filtering to show only selected signatures
        >>> fig = py_mut.mutational_signature_analysis(
        ...     ref_genome='path/to/genome.fa',
        ...     cosmic_path='path/to/COSMIC_v3.4_SBS_GRCh38.txt',
        ...     n_signatures=5,
        ...     signature_names=['Sig1', 'Sig2', 'Sig3', 'Sig4', 'Sig5'],
        ...     selected_signatures=['Sig1', 'Sig3', 'Sig5']
        ... )

        See Also
        --------
        mutational_signature_plot : Individual signature profiles only
        signature_bar_chart : 96-context bar charts only
        cosine_similarity_heatmap : COSMIC comparison only
        signature_contribution_heatmap : Contribution heatmap only
        signature_stacked_bar_chart : Stacked bar chart only
        signature_donut_plot : Donut plot only
        """
        import time

        from .analysis.mutational_signature import extractSignatures
        from .visualizations.mutational_signature_analysis import (
            create_complete_signature_analysis,
        )
        start_time = time.time()
        
        logger.info("Generating complete mutational signature analysis...")
        
        # Get MAF sample order - ALPHABETICALLY SORTED for consistency with individual plots
        maf_sample_order = sorted(pd.unique(self.data[SAMPLE_COLUMN]).tolist())
        
        # Generate trinucleotide context matrix
        contexts_df, _ = self.trinucleotideMatrix(
            ref_genome=ref_genome,
            prefix=prefix,
            add=add,
            ignoreChr=ignoreChr,
            useSyn=useSyn,
            update_data=update_data
        )
        
        if contexts_df.empty:
            raise ValueError("No mutations available for signature analysis")
        
        # Extract signatures using NMF
        signature_result = extractSignatures(
            contexts_df,
            n=n_signatures,
            pConstant=pConstant
        )
        
        W = signature_result['signatures'].values  # 96 x n_signatures (numpy array)
        contributions_df = signature_result['contributions']  # n_signatures x n_samples (DataFrame, normalized)
        contributions_abs_df = signature_result['contributions_abs']  # n_signatures x n_samples (DataFrame, absolute counts)
        H = contributions_df.values  # Convert to numpy array (normalized)
        contributions_abs = contributions_abs_df.values  # Convert to numpy array (absolute)
        
        # Get sample names from contributions DataFrame
        sample_names = contributions_df.columns.tolist()
        
        # Reorder samples to match MAF order
        reorder_indices = []
        reordered_sample_names = []
        for maf_sample in maf_sample_order:
            if maf_sample in sample_names:
                idx = sample_names.index(maf_sample)
                reorder_indices.append(idx)
                reordered_sample_names.append(maf_sample)
        
        if len(reorder_indices) != len(sample_names):
            logger.warning(
                "Sample count mismatch: MAF has %d samples, but only %d found in signature matrix",
                len(maf_sample_order), len(reorder_indices)
            )
        
        # Apply reordering to matrices
        H = H[:, reorder_indices]
        contributions_abs = contributions_abs[:, reorder_indices]
        sample_names = reordered_sample_names
        
        # Create complete visualization
        fig, _ = create_complete_signature_analysis(
            W=W,
            H=H,
            sample_names=sample_names,
            cosmic_path=_resolve_cosmic_catalog_path(cosmic_path),
            contributions_abs=contributions_abs,
            signature_names=signature_names,
            selected_signatures=selected_signatures,
            figsize=figsize,
            title=title
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Complete mutational signature analysis generated in {elapsed_time:.2f}s")
        
        plt.close(fig)
        return fig

    def lollipop_plot(self,
                     gene: str,
                     aa_col: str = "HGVSp_Short",
                     transcript_id: Optional[str] = None,
                     protein_id: Optional[str] = None,
                     domains_source: Optional[str] = "pfam",
                     custom_domains: Optional[List[Dict]] = None,
                     count_by: str = "mutations",
                     label_top_n: int = 20,
                     label_pos: Optional[Union[int, List[int], str]] = None,
                     show_lollipops: bool = True,
                     figsize: Tuple[int, int] = (16, 9),
                     title: Optional[str] = None) -> Figure:
        """
        Create a lollipop plot for a specific gene showing mutation distribution along the protein.

        This visualization displays:
        - Protein domain architecture (PFAM domains by default)
        - Mutation distribution along the protein sequence (if show_lollipops=True)
        - Mutation hotspots with size proportional to frequency
        - Labeled top mutated positions

        Lollipop plots are essential for identifying:
        - Mutation hotspots (frequently mutated positions)
        - Distribution patterns along functional domains
        - Comparison of mutation types at specific residues

        Args:
            gene: Gene symbol to visualize (e.g., "FLT3", "TP53").
            aa_col: Column name containing protein change annotation.
                   Common values: "HGVSp_Short", "Protein_Change", "AAChange".
                   Must contain HGVS notation like "p.D835Y".
            transcript_id: Specific transcript ID to use for isoform selection (optional).
            protein_id: Specific UniProt protein ID to use (optional).
            domains_source: Source for domain annotations:
                          - "pfam": Use PFAM domains from internal database
                          - None: Show full-length protein without domains
            custom_domains: Custom domain definitions to override automatic detection.
                          Format: [{"start": 100, "end": 200, "name": "Kinase"}, ...]
            count_by: How to count mutations:
                     - "mutations": Count all mutation events (default)
                     - "samples": Count unique samples (deduplicate by sample)
            label_top_n: Number of top mutated positions to label (default: 20).
                        Ignored when label_pos is provided.
            label_pos: Mirrors the `labelPos` argument of maftools' lollipopPlot():
                      - None (default): label the `label_top_n` most mutated positions.
                      - "all": label every mutated position.
                      - int or list of int: label only those specific protein
                        positions (e.g. 882 or [132, 882]).
            show_lollipops: If True (default), show mutation lollipops; if False, show only protein domains.
            figsize: Figure size as (width, height) in inches. Default: (16, 9).
            title: Custom plot title. If None, auto-generated from gene name and counts.

        Returns:
            matplotlib Figure object with the lollipop plot.

        Raises:
            ValueError: If gene not found, no valid mutations, or missing required columns.

        Examples:
            >>> # Basic usage with default PFAM domains
            >>> fig = py_mut.lollipop_plot(gene="FLT3")
            >>> py_mut.save_figure(fig, "FLT3_lollipop.png")

            >>> # Count by unique samples instead of all mutations
            >>> fig = py_mut.lollipop_plot(gene="TP53", count_by="samples", label_top_n=15)

            >>> # Show only protein domains without mutations
            >>> fig = py_mut.lollipop_plot(gene="TP53", show_lollipops=False)

            >>> # Use custom domains
            >>> domains = [
            ...     {"start": 100, "end": 300, "name": "DNA Binding"},
            ...     {"start": 320, "end": 355, "name": "Tetramerization"}
            ... ]
            >>> fig = py_mut.lollipop_plot(gene="TP53", custom_domains=domains)

            >>> # Use alternative protein change column
            >>> fig = py_mut.lollipop_plot(gene="NRAS", aa_col="Protein_Change")

            >>> # Label only specific hotspot positions (like maftools' labelPos)
            >>> fig = py_mut.lollipop_plot(gene="DNMT3A", aa_col="Protein_Change", label_pos=882)
            >>> fig = py_mut.lollipop_plot(gene="TP53", label_pos=[175, 248, 273])

            >>> # Label every mutated position
            >>> fig = py_mut.lollipop_plot(gene="TP53", label_pos="all")

        Notes:
            - The plot uses a fixed color palette for reproducibility
            - Circle size represents mutation count at each position
            - Domains are validated and clipped to protein length
        """
        from .visualizations.lollipop_plot import _create_lollipop_plot

        fig = _create_lollipop_plot(
            py_mut=self,
            gene=gene,
            aa_col=aa_col,
            transcript_id=transcript_id,
            protein_id=protein_id,
            domains_source=domains_source,
            custom_domains=custom_domains,
            count_by=count_by,
            label_top_n=label_top_n,
            label_pos=label_pos,
            show_lollipops=show_lollipops,
            figsize=figsize,
            title=title
        )

        plt.close(fig)
        return fig

    def tcga_compare(self,
                     cohort_name: str = "Input",
                     capture_size: Optional[float] = 50.0,
                     tcga_capture_size: float = 35.8,
                     tcga_cohorts: Optional[List[str]] = None,
                     primary_site: bool = False,
                     logscale: bool = True,
                     rm_hyper: bool = False,
                     rm_zero: bool = True,
                     decreasing: bool = False,
                     figsize: Tuple[int, int] = DEFAULT_TCGA_COMPARE_FIGSIZE,
                     title: Optional[str] = None) -> Figure:
        """
        Compare this cohort's tumor mutation burden (TMB) against the ~33
        TCGA cohorts bundled with maftools, reproducing maftools'
        ``tcgaCompare()`` plot.

        Each TCGA cohort (plus this one, labeled `cohort_name`) is drawn as
        a column of dots - one per sample, sorted by TMB - with a red median
        line, ordered left to right by ascending median TMB (or descending,
        if `decreasing=True`). This cohort's dots are drawn in black, every
        TCGA cohort's in gray, matching maftools' default `col`.

        Non-synonymous mutation counts reuse `calculate_tmb_analysis`, so
        results are directly comparable with the rest of this module's TMB
        outputs. For the underlying tables (median TMB per cohort, per-sample
        TMB, pairwise t-tests) without plotting, see `calculate_tcga_compare`.

        Args:
            cohort_name: Label for this cohort in the comparison (maftools'
                        `cohortName`).
            capture_size: Interrogated region size in Mb used to normalize
                        this cohort's TMB (maftools' `capture_size`). If
                        None, raw non-synonymous counts are compared instead
                        (maftools' own default) - only meaningful when the
                        TCGA cohorts were sequenced with a comparable
                        capture size.
            tcga_capture_size: Interrogated region size in Mb used to
                        normalize the TCGA cohorts (maftools'
                        `tcga_capture_size`; default 35.8, the typical TCGA
                        whole-exome capture size). Ignored if `capture_size`
                        is None.
            tcga_cohorts: Restrict the comparison to these TCGA cohort codes
                        (e.g. `["LAML", "PAAD"]`). If None, all ~33 bundled
                        cohorts are used.
            primary_site: Group TCGA samples by tumor primary site instead
                        of by TCGA project code.
            logscale: Plot TMB on a log10 y-axis (maftools' default).
            rm_hyper: Remove per-cohort outliers (Tukey's boxplot.stats
                        rule) before comparing, matching maftools'
                        `rm_hyper`.
            rm_zero: Drop samples with zero non-synonymous mutations from
                        this cohort before comparing, matching maftools'
                        `rm_zero`.
            decreasing: Order cohorts by descending median TMB instead of
                        ascending.
            figsize: Figure size as (width, height) in inches.
            title: Custom plot title. If None, auto-generated from
                        `cohort_name`.

        Returns:
            matplotlib Figure object with the TCGA comparison plot.

        Raises:
            ValueError: If no samples remain for this cohort, or
                        `tcga_cohorts` matches no bundled TCGA cohort.

        Examples:
            >>> fig = py_mut.tcga_compare(cohort_name="LAML-TB", capture_size=50)
            >>> py_mut.save_figure(fig, "tcga_compare.png")

            >>> # Restrict to a handful of TCGA cohorts, raw mutation counts
            >>> fig = py_mut.tcga_compare(cohort_name="PAAD-TB", capture_size=None,
            ...                           tcga_cohorts=["PAAD", "LAML", "SKCM"])
        """
        from .visualizations.tcga_compare_plot import _create_tcga_compare_plot

        fig = _create_tcga_compare_plot(
            py_mut=self,
            cohort_name=cohort_name,
            capture_size=capture_size,
            tcga_capture_size=tcga_capture_size,
            tcga_cohorts=tcga_cohorts,
            primary_site=primary_site,
            logscale=logscale,
            rm_hyper=rm_hyper,
            rm_zero=rm_zero,
            decreasing=decreasing,
            figsize=figsize,
            title=title,
        )

        plt.close(fig)
        return fig

    def somatic_interactions(self,
                            top_genes: int = 25,
                            gene_column: str = GENE_COLUMN,
                            sample_column: str = SAMPLE_COLUMN,
                            figsize: Tuple[int, int] = (12, 10),
                            title: Optional[str] = None,
                            vmin: float = -3.0,
                            vmax: float = 3.0,
                            pvalue: Tuple[float, float] = (0.05, 0.1),
                            show_counts: bool = True,
                            fdr_correction: bool = False,
                            fdr_method: str = "fdr_bh") -> Figure:
        """
        Create somatic interactions heatmap showing co-occurrence and mutual exclusivity.

        This visualization identifies gene pairs that are:
        - Co-occurring: Mutated together more often than expected by chance
        - Mutually exclusive: Rarely mutated together in the same samples

        Uses Fisher's exact test to determine statistical significance of interactions.
        The heatmap displays signed -log10(p-value) where:
        - Positive values (brown/orange): Co-occurrence (odds ratio > 1)
        - Negative values (blue/green): Mutual exclusivity (odds ratio < 1)
        - Asterisks (*): Highly significant interactions (p < 0.05)
        - Dots (·): Moderately significant interactions (0.05 ≤ p < 0.1)

        This analysis is essential for:
        - Understanding functional relationships between genes
        - Identifying potential synthetic lethal interactions
        - Discovering pathway-level mutation patterns
        - Guiding combination therapy strategies

        Args:
            top_genes: Number of most frequently mutated genes to include (default: 25).
                Genes are ranked by number of samples with mutations.
            gene_column: Column name containing gene symbols (default: "Hugo_Symbol").
            sample_column: Column name containing sample identifiers
                (default: "Tumor_Sample_Barcode").
            figsize: Figure size as (width, height) in inches (default: (12, 10)).
            title: Custom plot title. If None, auto-generated with top_genes count
                (and an "[FDR-corrected]" suffix when `fdr_correction=True`).
            vmin: Minimum value for color scale (default: -3.0).
                Controls saturation for mutual exclusivity.
            vmax: Maximum value for color scale (default: 3.0).
                Controls saturation for co-occurrence.
            pvalue: Tuple of p-value thresholds for significance markers.
                Default: (0.05, 0.1) where 0.05 shows asterisks and 0.1 shows dots.
                Applied to whichever p-value is displayed (see `fdr_correction`).
            show_counts: Whether to display sample counts next to gene names (default: True).
                Format: "GENE [count]" (e.g., "TP53 [52]").
            fdr_correction: If False (default), the heatmap colors and significance
                markers are based on the RAW Fisher's exact test p-value, exactly
                as before. If True, both are based on the p-value after False
                Discovery Rate correction (see `fdr_method`), computed across the
                C(top_genes, 2) pairwise tests performed on this gene set. The
                colorbar label and legend automatically indicate which p-value
                is being shown.
            fdr_method: Multiple-testing correction method forwarded to
                `statsmodels.stats.multitest.multipletests` (default: "fdr_bh",
                Benjamini-Hochberg). Only used when `fdr_correction=True`.

        Returns:
            matplotlib Figure object with the somatic interactions heatmap.

        Raises:
            ValueError: If fewer than 2 genes are mutated in the dataset.

        Examples:
            >>> # Basic usage with top 25 genes
            >>> fig = py_mut.somatic_interactions()
            >>> py_mut.save_figure(fig, "somatic_interactions.png")

            >>> # Analyze more genes with custom color range
            >>> fig = py_mut.somatic_interactions(top_genes=50, vmin=-5, vmax=5)

            >>> # Hide sample counts and use stricter significance threshold
            >>> fig = py_mut.somatic_interactions(
            ...     show_counts=False,
            ...     pvalue=(0.01, 0.001)
            ... )

            >>> # Smaller figure for presentations
            >>> fig = py_mut.somatic_interactions(top_genes=15, figsize=(8, 7))

            >>> # Color/mark significance using the FDR-adjusted p-value instead
            >>> fig = py_mut.somatic_interactions(top_genes=25, fdr_correction=True)

        Notes:
            - Only the upper-left triangle is displayed (avoids redundant pairs)
            - Diagonal is excluded (gene vs itself)
            - Fisher's exact test is used for statistical testing
            - P-values < 1e-10 are clipped to avoid log(0)
            - Results are deterministic and reproducible
            - The FDR correction (when enabled) is recomputed for the exact
              `top_genes` requested, so it always reflects C(top_genes, 2) tests

        See Also:
            - oncoplot(): Visualize mutation patterns across samples
            - top_mutated_genes_plot(): Identify most frequently mutated genes
            - somatic_interactions_table(): Get the underlying pairwise results
              (raw + FDR-adjusted p-values) as a printable DataFrame
        """
        from .visualizations.somatic_interactions import (
            _create_somatic_interactions_plot,
        )

        fig = _create_somatic_interactions_plot(
            py_mut=self,
            top_genes=top_genes,
            gene_column=gene_column,
            sample_column=sample_column,
            figsize=figsize,
            title=title,
            vmin=vmin,
            vmax=vmax,
            pvalue=pvalue,
            show_counts=show_counts,
            fdr_correction=fdr_correction,
            fdr_method=fdr_method,
        )

        plt.close(fig)

        return fig

    def somatic_interactions_table(self,
                                  top_genes: int = 25,
                                  gene_column: str = GENE_COLUMN,
                                  sample_column: str = SAMPLE_COLUMN,
                                  fdr_method: str = "fdr_bh") -> pd.DataFrame:
        """
        Compute the somatic interactions pairwise table (co-occurrence /
        mutual exclusivity) for the top mutated genes, including False
        Discovery Rate (FDR) corrected p-values.

        This is the tabular counterpart of `somatic_interactions()`: it uses
        the exact same variant filtering and top-k gene selection, so the
        table is always consistent with what the heatmap displays for the
        same `top_genes`. The FDR correction is recomputed over the
        C(top_genes, 2) pairwise tests actually performed, so it
        automatically adapts to whatever `top_genes` value is selected.

        Args:
            top_genes: Number of most frequently mutated genes to include (default: 25).
                Genes are ranked by number of samples with mutations.
            gene_column: Column name containing gene symbols (default: "Hugo_Symbol").
            sample_column: Column name containing sample identifiers
                (default: "Tumor_Sample_Barcode").
            fdr_method: Multiple-testing correction method forwarded to
                `statsmodels.stats.multitest.multipletests` (default: "fdr_bh",
                Benjamini-Hochberg). Other options include "bonferroni", "holm", etc.

        Returns:
            pandas DataFrame with one row per gene pair, sorted by FDR-adjusted
            p-value ascending, with columns:
                - Gene 1, Gene 2
                - Altered samples gene 1, Altered samples gene 2
                - Both altered, Only gene 1 altered, Only gene 2 altered, Neither altered
                - Odds ratio
                - P-value (raw Fisher's exact test p-value, clipped at 1e-10)
                - P-value adjusted (FDR)
                - Significant (FDR < 0.05)
                - Association ("co-occurrence" / "mutual_exclusivity" / "neutral")

        Raises:
            ValueError: If fewer than 2 genes are mutated in the dataset.

        Examples:
            >>> # Get the table and print it directly
            >>> table = py_mut.somatic_interactions_table(top_genes=25)
            >>> print(table)

            >>> # Export it, same as any other DataFrame
            >>> table.to_csv("somatic_interactions.csv", index=False)
            >>> table.to_excel("somatic_interactions.xlsx", index=False)

            >>> # Inspect specific gene pairs
            >>> table[table["Gene 1"].isin(["NPM1", "FLT3"]) &
            ...       table["Gene 2"].isin(["NPM1", "FLT3"])]

        See Also:
            - somatic_interactions(): Heatmap visualization of the same analysis
        """
        from .visualizations.somatic_interactions import (
            _get_somatic_interactions_table,
        )

        return _get_somatic_interactions_table(
            py_mut=self,
            top_genes=top_genes,
            gene_column=gene_column,
            sample_column=sample_column,
            fdr_method=fdr_method,
        )

    def comut_mutation_burden(self,
                            sample_column: str = SAMPLE_COLUMN,
                            variant_column: str = VARIANT_CLASSIFICATION_COLUMN,
                            territory_bp: int = 60456963,
                            figsize: Tuple[float, float] = (14, 1.8),
                            title: Optional[str] = None,
                            hypermutator_threshold: Optional[float] = None,
                            sample_ids: Optional[List[str]] = None,
                            max_samples: Optional[int] = 50,
                            somatic_only: bool = True,
                            pass_only: bool = True) -> Figure:
        """
        Create CoMut Panel A: Mutation Burden visualization.
        
        Displays a stacked bar chart showing synonymous and non-synonymous mutation
        rates (Muts/Mb) for each sample. Samples are ordered by non-synonymous TMB.
        
        Args:
            sample_column: Column name for sample IDs (default: 'Tumor_Sample_Barcode')
            variant_column: Column name for variant classification (default: 'Variant_Classification')
            territory_bp: Territory size in bp for normalization (default: 60,456,963 bp = exome)
            figsize: Figure size (width, height) in inches (default: (14, 1.8))
            title: Plot title (default: "Mutation Burden")
            hypermutator_threshold: TMB threshold (Muts/Mb) for hypermutator line (None = don't show)
            sample_ids: Optional list of specific sample IDs to include in exact order.
                       If provided, max_samples is ignored. Useful for multi-panel coordination.
            max_samples: Maximum samples to display (default: 50, None = all).
                        Samples are selected by highest non-synonymous TMB.
                        Ignored if sample_ids is provided.
            somatic_only: If True, filter to Variant_Status == 'Somatic' (default: True)
            pass_only: If True, filter to FILTER in ['PASS', '.'] (default: True)
            
        Returns:
            matplotlib.figure.Figure: The mutation burden bar chart
            
        Example:
            >>> # Show top 50 samples (default behavior)
            >>> fig = py_mut.comut_mutation_burden()
            >>> 
            >>> # Use specific sample list (for multi-panel coordination)
            >>> sample_order = load_sample_order_from_file("samples.txt")
            >>> fig = py_mut.comut_mutation_burden(sample_ids=sample_order)
        """
        from .visualizations.comut_plot import _create_comut_mutation_burden_plot
        
        return _create_comut_mutation_burden_plot(
            self,
            sample_column=sample_column,
            variant_column=variant_column,
            territory_bp=territory_bp,
            figsize=figsize,
            title=title,
            hypermutator_threshold=hypermutator_threshold,
            sample_ids=sample_ids,
            max_samples=max_samples,
            somatic_only=somatic_only,
            pass_only=pass_only
        )


    def comut_mutation_signatures_plot(self,
                                      signatures_df: Optional[pd.DataFrame] = None,
                                      signatures_tsv: Optional[str] = None,
                                      sample_order: Optional[List[str]] = None,
                                      signature_labels: Optional[List[str]] = None,
                                      normalize: bool = True,
                                      figsize: Tuple[float, float] = (12, 2.0),
                                      colors: Optional[List[str]] = None,
                                      title: Optional[str] = None) -> Figure:
        """
        Create CoMut Panel B: Mutational Signatures visualization.
        
        See _create_comut_mutation_signatures_plot() for detailed documentation.
        
        Args:
            signatures_df: Pre-computed signature contributions DataFrame
            signatures_tsv: Path to TSV file with signature contributions
            sample_order: List of sample IDs in desired order (for alignment with Panel A)
            signature_labels: List of signature names in desired display order
            normalize: If True, normalize each row to sum to 1.0
            figsize: Figure size (width, height) in inches
            colors: List of colors for signatures
            title: Plot title
            
        Returns:
            matplotlib.figure.Figure: The mutational signatures plot
            
        Example:
            >>> fig = py_mut.comut_mutation_signatures_plot(
            ...     signatures_tsv="sig_contribution.tsv",
            ...     sample_order=sample_order
            ... )
        """
        from .visualizations.comut_plot import _create_comut_mutation_signatures_plot
        
        fig, _ = _create_comut_mutation_signatures_plot(
            self,
            signatures_df=signatures_df,
            signatures_tsv=signatures_tsv,
            sample_order=sample_order,
            signature_labels=signature_labels,
            normalize=normalize,
            figsize=figsize,
            colors=colors,
            title=title
        )
        
        return fig

    def comut_purity_plot(self,
                         purity_df: Optional[pd.DataFrame] = None,
                         purity_tsv: Optional[str] = None,
                         sample_order: Optional[List[str]] = None,
                         figsize: Tuple[float, float] = (12, 1.5),
                         title: Optional[str] = None) -> Figure:
        """
        Create CoMut Panel C: Purity heatmap visualization.
        
        Displays a 1-row heatmap showing tumor purity values (0-1) for each sample.
        This panel should be aligned with Panels A (mutation burden) and B (signatures)
        using the same sample_order.
        
        Args:
            purity_df: Pre-computed purity DataFrame or Series.
                      DataFrame should have columns: sample ID and purity value.
                      Series should be indexed by sample IDs with purity values.
            purity_tsv: Path to TSV file with purity data.
                       Expected columns: sample, value (or purity/tumor_purity).
                       Values can be 0-1 or 0-100 (auto-detected and converted).
            sample_order: List of sample IDs in desired order (for alignment with Panels A/B).
                         Samples not in purity data will show as white (NaN).
                         If None, uses all samples in the order from purity_df.
            figsize: Figure size (width, height) in inches (default: (12, 1.5)).
            title: Plot title (default: None, will show "Purity" as Y-label instead).
            
        Returns:
            matplotlib.figure.Figure: The purity heatmap
            
        Example:
            >>> # Align with Panels A and B
            >>> sample_order = ['SAMPLE1', 'SAMPLE2', ...]  # from Panel A
            >>> fig = py_mut.comut_purity_plot(
            ...     purity_tsv="purity.tsv",
            ...     sample_order=sample_order
            ... )
            >>> fig.savefig("comut_panel_c_purity.png")
        """
        from .visualizations.comut_plot import _create_comut_purity_plot
        
        fig, _ = _create_comut_purity_plot(
            self,
            purity_df=purity_df,
            purity_tsv=purity_tsv,
            sample_order=sample_order,
            figsize=figsize,
            title=title
        )
        
        return fig

    def comut_mutation_type_plot(self,
                                mutation_data_df: Optional[pd.DataFrame] = None,
                                mutation_data_tsv: Optional[str] = None,
                                sample_order: Optional[List[str]] = None,
                                gene_order: Optional[List[str]] = None,
                                figsize: Tuple[float, float] = (12, 4.0),
                                title: Optional[str] = None) -> Figure:
        """
        Create CoMut Panel D: Mutation Type (Oncoprint) visualization.
        
        Displays a gene x sample matrix showing the predominant mutation type for each
        gene-sample pair. This panel should be aligned with Panels A-C using the same
        sample_order.
        
        This visualization follows oncoprint conventions:
        - 5 mutation categories: Missense, Nonsense, Frameshift indel, Silent, Multiple
        - Genes displayed from top to bottom (typically: MAP3K1, CDH1, TP53, PIK3CA)
        - Samples in same order as Panels A-C for perfect alignment
        - White cells for genes with no mutations in a sample
        
        Args:
            mutation_data_df: Pre-computed mutation type DataFrame.
                             Expected columns: sample (or sample_id), gene (or category),
                             mutation_type (or value).
            mutation_data_tsv: Path to TSV file with mutation type data.
                              Expected columns: sample, category (gene), value (type).
                              Mutation types: Missense, Nonsense, Frameshift indel,
                              In frame indel, Silent, Multiple.
            sample_order: List of sample IDs in desired order (for alignment with Panels A-C).
                         Should match the order used in mutation burden plot.
                         If None, uses all samples in data.
            gene_order: List of genes to display (top to bottom order).
                       Example: ["MAP3K1", "CDH1", "TP53", "PIK3CA"]
                       If None, uses all genes ordered by mutation frequency.
            figsize: Figure size (width, height) in inches.
                    Default: (12, 4.0) for ~4 genes. Adjust height for more/fewer genes.
                    Cells are rectangular (taller than wide) for better visualization.
            title: Plot title (default: "D  Mutation Type").
            
        Returns:
            matplotlib.figure.Figure: The mutation type oncoprint
            
        Example:
            >>> # Align with Panels A-C
            >>> sample_order = ['SAMPLE1', 'SAMPLE2', ...]  # from Panel A
            >>> gene_order = ["MAP3K1", "CDH1", "TP53", "PIK3CA"]
            >>> fig = py_mut.comut_mutation_type_plot(
            ...     mutation_data_tsv="mutation_data.tsv",
            ...     sample_order=sample_order,
            ...     gene_order=gene_order
            ... )
            >>> fig.savefig("comut_panel_d_mutation_type.png")
        """
        from .visualizations.comut_plot import _create_comut_mutation_type_plot
        
        fig, _ = _create_comut_mutation_type_plot(
            self,
            mutation_data_df=mutation_data_df,
            mutation_data_tsv=mutation_data_tsv,
            sample_order=sample_order,
            gene_order=gene_order,
            figsize=figsize,
            title=title
        )
        
        return fig

    def comut_cna_plot(self,
                      cna_df: Optional[pd.DataFrame] = None,
                      cna_tsv: Optional[str] = None,
                      sample_order: Optional[List[str]] = None,
                      gene_order: Optional[List[str]] = None,
                      figsize: Tuple[float, float] = (12, 2.0),
                      title: Optional[str] = None) -> Figure:
        """
        Create CoMut Panel E: Copy Number Alteration (CNA) heatmap.
        
        Displays a gene × sample heatmap showing the allelic CNA state for each
        gene-sample pair. This panel should be aligned with other CoMut panels using
        the same sample_order.
        
        This visualization shows CNA states:
        - aCN = 0: Homozygous deletion (complete loss)
        - Allelic deletion: Heterozygous deletion (partial loss)
        - Allelic amplification: Copy number gain
        - Baseline: Diploid/neutral state (no alteration)
        
        For duplicate (sample, gene) entries, the most severe state is kept using
        this priority: aCN = 0 > Allelic deletion > Allelic amplification > Baseline
        
        Args:
            cna_df: Pre-loaded DataFrame with CNA data.
                   Expected columns: sample, category (gene), value (CNA state)
            cna_tsv: Path to TSV file with CNA data.
                    Expected columns: sample, category, value
                    CNA states: "Baseline", "Allelic amplification", 
                               "Allelic deletion", "aCN = 0"
            sample_order: List of sample IDs in desired order (for alignment with Panels A-D).
                         Should match the order used in mutation burden plot.
                         If None, uses all samples found in data.
            gene_order: List of genes to display (top to bottom order).
                       Example: ["ERBB2", "CDKN2A", "MYC"]
                       If None, uses all genes ordered by alteration frequency.
            figsize: Figure size (width, height) in inches.
                    Default: (12, 2.0) for ~3 genes. Adjust height for more/fewer genes.
            title: Plot title (default: "E  Copy Number Alteration").
            
        Returns:
            matplotlib.figure.Figure: The CNA heatmap
            
        Example:
            >>> # Align with other CoMut panels
            >>> sample_order = ['SAMPLE1', 'SAMPLE2', ...]  # from Panel A
            >>> gene_order = ["ERBB2", "CDKN2A", "MYC"]
            >>> fig = py_mut.comut_cna_plot(
            ...     cna_tsv="cna.tsv",
            ...     sample_order=sample_order,
            ...     gene_order=gene_order
            ... )
            >>> fig.savefig("comut_panel_e_cna.png", dpi=300, bbox_inches='tight')
        """
        from .visualizations.comut_plot import _create_comut_cna_plot
        
        fig, _ = _create_comut_cna_plot(
            self,
            cna_df=cna_df,
            cna_tsv=cna_tsv,
            sample_order=sample_order,
            gene_order=gene_order,
            figsize=figsize,
            title=title
        )
        
        return fig

    def comut_wgd_plot(self,
                      wgd_df: Optional[pd.DataFrame] = None,
                      wgd_tsv: Optional[str] = None,
                      sample_order: Optional[List[str]] = None,
                      figsize: Tuple[float, float] = (12, 1.2),
                      title: Optional[str] = None) -> Figure:
        """
        Create CoMut Panel F: Whole Genome Doubling (WGD) track.
        
        Displays a single-row track with gray rectangles for samples with WGD=Yes
        and blank (white) for samples with WGD=No or missing data. This panel should
        be aligned with other CoMut panels using the same sample_order.
        
        This visualization shows WGD display:
        - Gray (#8b8b8b) rectangle: WGD = Yes (genome doubling detected)
        - Blank (white): WGD = No or missing data
        - Small gaps between rectangles for visual clarity
        
        WGD data interpretation (flexible column naming and value encoding):
        - Sample column: sample, sample_id, tumor_sample_barcode (case-insensitive)
        - WGD column: wgd, status, value (case-insensitive)
        - WGD values (normalized to boolean):
          * Yes/1/True/Y/T → True (WGD present)
          * No/0/False/N/F → False (no WGD)
          * Missing/NaN → NaN (drawn as blank)
        
        For duplicate (sample) entries, groups by sample and uses any() → True if
        any row indicates WGD.
        
        Args:
            wgd_df: Pre-loaded DataFrame with WGD data.
                   Expected columns: sample, category/wgd/status, value
            wgd_tsv: Path to TSV file with WGD data.
                    Expected columns: sample, category, value
                    Values: "Yes"/"No" or similar boolean encodings
            sample_order: List of sample IDs in desired order (for alignment with Panels A-E).
                         Should match the order used in mutation burden plot.
                         If None, uses all samples found in data.
            figsize: Figure size (width, height) in inches.
                    Default: (12, 1.2) for single-row track with rectangular cells.
            title: Plot title (default: "F  Whole Genome Doubling").
            
        Returns:
            matplotlib.figure.Figure: The WGD track figure
            
        Example:
            >>> # Align with other CoMut panels
            >>> sample_order = ['SAMPLE1', 'SAMPLE2', ...]  # from Panel A
            >>> fig = py_mut.comut_wgd_plot(
            ...     wgd_tsv="wgd.tsv",
            ...     sample_order=sample_order
            ... )
            >>> fig.savefig("comut_panel_f_wgd.png", dpi=300, bbox_inches='tight')
        """
        from .visualizations.comut_plot import _create_comut_wgd_plot
        
        fig, _ = _create_comut_wgd_plot(
            self,
            wgd_df=wgd_df,
            wgd_tsv=wgd_tsv,
            sample_order=sample_order,
            figsize=figsize,
            title=title
        )
        
        return fig

    def comut_same_patient_plot(self,
                               sp_df: Optional[pd.DataFrame] = None,
                               sp_tsv: Optional[str] = None,
                               sample_order: Optional[List[str]] = None,
                               figsize: Tuple[float, float] = (12, 1.2),
                               title: Optional[str] = None) -> Figure:
        """
        Create Panel G: Same Patient track.
        
        Shows which samples belong to the same patient by drawing:
        - A black dot for each sample (single horizontal track)
        - Horizontal black lines connecting dots of samples from the same patient
        
        Args:
            sp_df: DataFrame with columns ['sample', 'group'] where 'group' identifies the patient.
                  If provided, sp_tsv is ignored.
            sp_tsv: Path to TSV file with columns: sample, group (patient ID)
            sample_order: List of sample IDs in desired order (x-axis).
                         If None, uses all samples from sp data.
            figsize: Figure size (width, height)
            title: Panel title (default: "G  Same Patient")
            
        Returns:
            matplotlib.figure.Figure: Panel G figure
            
        Example:
            >>> fig = py_mut.comut_same_patient_plot(
            ...     sp_tsv="sp.tsv",
            ...     sample_order=sample_order,
            ...     figsize=(12, 1.2)
            ... )
            >>> fig.savefig("comut_panel_g_same_patient.png", dpi=300, bbox_inches='tight')
        """
        from .visualizations.comut_plot import _create_comut_same_patient_plot
        
        fig, _ = _create_comut_same_patient_plot(
            self,
            sp_df=sp_df,
            sp_tsv=sp_tsv,
            sample_order=sample_order,
            figsize=figsize,
            title=title
        )
        
        return fig

    def create_comut_plot(self,
                         sample_order: Optional[List[str]] = None,
                         gene_order: Optional[List[str]] = None,
                         cna_gene_order: Optional[List[str]] = None,
                         signatures_tsv: Optional[str] = None,
                         purity_tsv: Optional[str] = None,
                         mutation_data_tsv: Optional[str] = None,
                         cna_tsv: Optional[str] = None,
                         wgd_tsv: Optional[str] = None,
                         sp_tsv: Optional[str] = None,
                         territory_bp: int = 60456963,
                         max_samples: Optional[int] = 50,
                         somatic_only: bool = True,
                         pass_only: bool = True,
                         signature_labels: Optional[List[str]] = None,
                         figsize: Optional[Tuple[float, float]] = None) -> Figure:
        """
        Create complete CoMut plot with all panels (A-G) stacked vertically.
        
        This creates a comprehensive multi-panel visualization combining:
        - Panel A: Mutation burden (synonymous/non-synonymous)
        - Panel B: Mutational signatures
        - Panel C: Purity heatmap
        - Panel D: Mutation type (oncoprint)
        - Panel E: Copy Number Alteration (CNA)
        - Panel F: Whole Genome Doubling (WGD)
        - Panel G: Same Patient (samples from same patient connected)
        
        All panels are perfectly aligned by sample order.
        
        Args:
            sample_order: List of sample IDs in desired order (for all panels).
                         If None, will be calculated from mutation burden.
            gene_order: List of genes to display in Panel D (top to bottom).
                       Example: ["MAP3K1", "CDH1", "TP53", "PIK3CA"]
            cna_gene_order: List of genes to display in Panel E (top to bottom).
                           Example: ["ERBB2", "CDKN2A", "MYC"]
                           If None, uses all genes from CNA data.
            signatures_tsv: Path to TSV file with signature contributions (Panel B).
            purity_tsv: Path to TSV file with purity data (Panel C).
            mutation_data_tsv: Path to TSV file with mutation type data (Panel D).
            cna_tsv: Path to TSV file with CNA data (Panel E).
            wgd_tsv: Path to TSV file with WGD data (Panel F).
            sp_tsv: Path to TSV file with same patient data (Panel G).
                   Format: 'sample' and 'group' columns (group = patient ID).
            territory_bp: Territory size in bp for TMB normalization (Panel A).
            max_samples: Maximum number of samples to display (if sample_order=None).
            somatic_only: If True, filter to somatic variants (Panel A).
            pass_only: If True, filter to PASS variants (Panel A).
            signature_labels: Labels for signatures in Panel B.
            figsize: Overall figure size (width, height). If None, auto-calculated.
            
        Returns:
            matplotlib.figure.Figure: Complete CoMut plot with all panels stacked
            
        Example:
            >>> fig = py_mut.create_comut_plot(
            ...     sample_order=sample_order,
            ...     gene_order=["MAP3K1", "CDH1", "TP53", "PIK3CA"],
            ...     cna_gene_order=["ERBB2", "CDKN2A", "MYC"],
            ...     signatures_tsv="sig_contribution.tsv",
            ...     purity_tsv="purity.tsv",
            ...     mutation_data_tsv="mutation_data.tsv",
            ...     cna_tsv="cna.tsv",
            ...     wgd_tsv="wgd.tsv",
            ...     sp_tsv="sp.tsv",
            ...     signature_labels=["Sig 1", "Sig 2", "Sig 3", "Sig 4"]
            ... )
            >>> fig.savefig("comut_plot_complete.png", dpi=300, bbox_inches='tight')
        """
        from .visualizations.comut_plot import create_comut_plot
        
        fig = create_comut_plot(
            self,
            sample_order=sample_order,
            gene_order=gene_order,
            cna_gene_order=cna_gene_order,
            signatures_tsv=signatures_tsv,
            purity_tsv=purity_tsv,
            mutation_data_tsv=mutation_data_tsv,
            cna_tsv=cna_tsv,
            wgd_tsv=wgd_tsv,
            sp_tsv=sp_tsv,
            territory_bp=territory_bp,
            max_samples=max_samples,
            somatic_only=somatic_only,
            pass_only=pass_only,
            signature_labels=signature_labels,
            figsize=figsize
        )
        
        plt.close(fig)
        
        return fig


# PyMutation.region = region
# PyMutation.gen_region = gen_region
# PyMutation.pass_filter = pass_filter
# PyMutation.filter_by_chrom_sample = filter_by_chrom_sample
# PyMutation.filter_by_tissue_expression = filter_by_tissue_expression
# PyMutation.annotate_pfam = annotate_pfam
# PyMutation.pfam_domains = pfam_domains
# PyMutation.knownCancer = knownCancer
