import logging
import os
import re
from typing import Optional, Dict

import numpy as np
import pandas as pd

# Logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Change to DEBUG for more verbosity
if not logger.handlers:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )


class MutationBurdenMixin:
    """
    Mixin class providing mutation burden analysis functionality for PyMutation objects.
    
    This mixin adds TMB (Tumor Mutation Burden) analysis capabilities to PyMutation,
    following the same architectural pattern as other mixins in the project.
    """

    def calculate_tmb_analysis(self,
                               variant_classification_column: Optional[str] = None,
                               genome_size_bp: int = 50_000_000,
                               output_dir: str = ".",
                               save_files: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Calculate Tumor Mutation Burden (TMB) analysis for each sample in a PyMutation object.

        This method analyzes mutation data from PyMutation objects to calculate
        per-sample TMB, reproducing maftools' ``tmb()`` output exactly: same
        9-class "non-silent" Variant_Classification set, and
        ``total_perMB = non_synonymous_count / captureSize_Mb``. With the default
        ``genome_size_bp`` (50,000,000 bp = 50 Mb, maftools' own default
        ``captureSize``), ``total_perMB`` here reproduces maftools' ``total_perMB``
        column exactly for the same MAF. To compare against a maftools/pyMutR run
        that used a different ``captureSize`` (Mb), pass
        ``genome_size_bp = captureSize * 1_000_000``.

        Parameters
        ----------
        variant_classification_column : str, optional
            Name of the column containing variant classification information.
            If None, will automatically detect variant classification columns.
        genome_size_bp : int, default 50_000_000
            Size of the interrogated region in base pairs used to normalize TMB
            (maftools' ``captureSize``, expressed in bp instead of Mb). Default is
            50,000,000 bp (50 Mb), matching maftools' own default ``captureSize = 50``.
            Use the exact capture kit size (bp) for a real WES/panel, or
            ~3,000,000,000 bp for Whole Genome Sequencing (WGS).
        output_dir : str, default "."
            Directory where output files will be saved.
        save_files : bool, default True
            Whether to save the results to TSV files.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing:
            - 'analysis': Per-sample TMB DataFrame, with exactly maftools' ``tmb()``
              columns: ``Tumor_Sample_Barcode``, ``total`` (non-silent mutation
              count), ``total_perMB``, ``total_perMB_log`` (log10 of
              ``total_perMB``). Ordered ascending by ``total_perMB``, like maftools.
            - 'statistics': Global TMB statistics DataFrame (``total`` /
              ``total_perMB`` summarized: mean, median, quartiles, etc.)

        Notes
        -----
        Non-synonymous mutations are defined using maftools' default
        ``vc_nonSyn`` ("non-silent") Variant_Classification set (see
        ``maftools:::validateMaf``):
        - Missense_Mutation, Nonsense_Mutation, Frame_Shift_Del, Frame_Shift_Ins
        - Nonstop_Mutation, Translation_Start_Site, Splice_Site
        - In_Frame_Del, In_Frame_Ins

        Everything else — including Silent, Intron, 3'UTR/5'UTR, IGR, Flank, RNA,
        Splice_Region, and (perhaps counter-intuitively) De_novo_Start_InFrame,
        De_novo_Start_OutOfFrame, Start_Codon_SNP/Ins, Stop_Codon_Del — is treated
        by maftools as "silent" and excluded from TMB, so it is excluded here too.

        The method generates two output files:
        - TMB_analysis.tsv: Per-sample analysis with mutation counts and normalized TMB
        - TMB_statistics.tsv: Global statistics (mean, median, quartiles, etc.)
        """

        # Non-synonymous ("non-silent") mutation types, exactly matching maftools'
        # default vc_nonSyn list used inside read.maf()/validateMaf() to build the
        # "total" column that tmb() normalizes. Do NOT add extra classes here:
        # maftools explicitly classifies De_novo_Start_In/OutOfFrame, Start_Codon_*
        # and Stop_Codon_Del as SILENT, not non-synonymous.
        non_synonymous_types = {
            'MISSENSE_MUTATION', 'NONSENSE_MUTATION', 'FRAME_SHIFT_DEL', 'FRAME_SHIFT_INS',
            'NONSTOP_MUTATION', 'TRANSLATION_START_SITE', 'SPLICE_SITE', 'IN_FRAME_DEL',
            'IN_FRAME_INS'
        }

        # Validate PyMutation structure
        if not hasattr(self, 'data') or not hasattr(self, 'samples'):
            raise ValueError("Invalid PyMutation object: missing 'data' or 'samples' attributes")

        if self.data.empty:
            raise ValueError("PyMutation data is empty")

        if not self.samples:
            raise ValueError("No samples found in PyMutation object")

        # Check for required columns
        required_cols = ['REF', 'ALT']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in PyMutation data: {missing_cols}")

        # Auto-detect variant classification column if not provided
        if variant_classification_column is None:
            # Look for variant classification columns using regex pattern
            pattern = re.compile(r'^(gencode_\d+_)?variant[_]?classification$', flags=re.IGNORECASE)
            variant_cols = [col for col in self.data.columns if pattern.match(col)]

            if variant_cols:
                variant_classification_column = variant_cols[0]
                logger.info(f"Auto-detected variant classification column: {variant_classification_column}")
            else:
                logger.warning("No variant classification column found. Non-synonymous counts will be 0.")
                variant_classification_column = None

        # Validate variant classification column if provided
        if variant_classification_column and variant_classification_column not in self.data.columns:
            raise ValueError(f"Column provide '{variant_classification_column}' not found in data")

        # Precompute the non-synonymous flag once for the whole table (vectorized),
        # instead of recomputing it per sample.
        if variant_classification_column:
            classifications = self.data[variant_classification_column].astype(str).str.upper()
            is_non_syn = classifications.isin(non_synonymous_types) & self.data[variant_classification_column].notna()
        else:
            is_non_syn = pd.Series(False, index=self.data.index)

        ref_alleles = self.data['REF'].astype(str)

        # Initialize results
        results = []

        # Process each sample
        for sample in self.samples:
            if sample not in self.data.columns:
                logger.warning(f"Sample '{sample}' not found in data columns. Skipping.")
                continue

            # Find mutations for this sample, when the genotype is not REF|REF
            # (or REF/REF). Compute validity on the original series to avoid
            # turning NaN into the string 'nan'.
            original_genotypes = self.data[sample]
            valid_genotypes = original_genotypes.notna() & (original_genotypes != '') & (original_genotypes != '.')

            # Normalize the '/' phasing separator to '|' so both are handled the
            # same way, then split into (at most) two alleles - vectorized, no
            # per-row Python loop.
            genotypes = original_genotypes.astype(str).str.replace('/', '|', regex=False)
            alleles = genotypes.str.split('|', expand=True)

            # A mutation exists if any allele differs from REF (and isn't missing).
            has_mutation = pd.Series(False, index=self.data.index)
            for col in alleles.columns:
                allele = alleles[col].str.strip()
                has_mutation |= allele.notna() & (allele != '') & (allele != '.') & (allele != ref_alleles)

            mutation_mask = valid_genotypes & has_mutation

            # Count non-synonymous ("non-silent") mutations for this sample -
            # this is maftools' "total" (see getSampleSummary()).
            total = int((mutation_mask & is_non_syn).sum())

            # total_perMB = total / captureSize_Mb, same formula as maftools' tmb().
            total_perMB = (total / genome_size_bp) * 1_000_000 if genome_size_bp > 0 else 0

            # Store results (keep full precision until final output). Columns match
            # maftools::tmb() exactly: Tumor_Sample_Barcode / total / total_perMB.
            results.append({
                'Tumor_Sample_Barcode': sample,
                'total': total,
                'total_perMB': total_perMB,
            })

        # Create analysis DataFrame
        if not results:
            raise ValueError("No valid samples found for TMB analysis")

        analysis_df = pd.DataFrame(results)

        # total_perMB_log mirrors maftools' tmb()$total_perMB_log (log10 of
        # total_perMB; 0 mutations -> -inf, same as R's log10(0)).
        with np.errstate(divide='ignore'):
            analysis_df['total_perMB_log'] = np.log10(analysis_df['total_perMB'])

        # maftools' tmb() returns the table ordered ascending by total_perMB.
        analysis_df = analysis_df.sort_values('total_perMB', ascending=True).reset_index(drop=True)

        # Calculate global statistics
        stats_data = []
        metrics = ['total', 'total_perMB']

        for metric in metrics:
            if metric in analysis_df.columns:
                values = analysis_df[metric]
                stats_data.append({
                    'Metric': metric,
                    'Count': len(values),
                    'Mean': values.mean(),
                    'Median': values.median(),
                    'Min': values.min(),
                    'Max': values.max(),
                    'Q1': values.quantile(0.25),
                    'Q3': values.quantile(0.75),
                    'Std': values.std()
                })

        statistics_df = pd.DataFrame(stats_data)

        # Create output directory if it doesn't exist
        if save_files:
            os.makedirs(output_dir, exist_ok=True)

            analysis_path = os.path.join(output_dir, "TMB_analysis.tsv")
            statistics_path = os.path.join(output_dir, "TMB_statistics.tsv")

            # Save files with proper formatting (maintain precision)
            analysis_df.to_csv(analysis_path, sep='\t', index=False, float_format='%.6f')
            statistics_df.to_csv(statistics_path, sep='\t', index=False, float_format='%.6f')

            logger.info(f"TMB analysis saved to: {analysis_path}")
            logger.info(f"TMB statistics saved to: {statistics_path}")
            logger.info(f"Analyzed {len(analysis_df)} samples with {len(self.data)} total mutations")

        log_tmb_summary(analysis_df)

        return {
            'analysis': analysis_df,
            'statistics': statistics_df
        }

    def calculate_tcga_compare(self,
                                cohort_name: str = "Input",
                                capture_size: Optional[float] = 50.0,
                                tcga_capture_size: float = 35.8,
                                tcga_cohorts: Optional[list] = None,
                                primary_site: bool = False,
                                rm_hyper: bool = False,
                                rm_zero: bool = True,
                                decreasing: bool = False,
                                output_dir: str = ".",
                                save_files: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Place this cohort's per-sample TMB in the context of the ~33 TCGA
        cohorts bundled with maftools, reproducing the data behind
        maftools' ``tcgaCompare()`` (without the plot - see
        ``PyMutation.tcga_compare()`` for the plotting counterpart).

        Non-synonymous mutation counts are computed with
        :meth:`calculate_tmb_analysis`, so the "total" per sample matches it
        exactly. Those counts are combined with the bundled TCGA reference
        table (``pyMut/data/tcga_cohort.txt.gz``, maftools' own
        ``tcga_cohort.txt.gz``) and, optionally, normalized to
        mutations/Mb using ``capture_size``/``tcga_capture_size``.

        Parameters
        ----------
        cohort_name : str, default "Input"
            Label used for this cohort in the comparison (equivalent to
            maftools' ``cohortName``).
        capture_size : float, optional, default 50.0
            Interrogated region size in Mb used to normalize this cohort's
            TMB (maftools' ``capture_size``). If None, raw non-synonymous
            counts are compared instead (maftools' own default), which is
            only meaningful when TCGA cohorts were also sequenced with a
            comparable capture size.
        tcga_capture_size : float, default 35.8
            Interrogated region size in Mb used to normalize the TCGA
            cohorts (maftools' ``tcga_capture_size`` default: the typical
            TCGA whole-exome capture size). Ignored if ``capture_size`` is
            None.
        tcga_cohorts : list of str, optional
            Restrict the comparison to these TCGA cohort codes (e.g.
            ``["LAML", "PAAD"]``). If None, all ~33 bundled cohorts are used.
        primary_site : bool, default False
            Group TCGA samples by tumor primary site instead of by TCGA
            project code.
        rm_hyper : bool, default False
            Remove per-cohort outliers (Tukey's boxplot.stats rule) before
            comparing, matching maftools' ``rm_hyper``.
        rm_zero : bool, default True
            Drop samples with zero non-synonymous mutations from this
            cohort before comparing, matching maftools' ``rm_zero``.
        decreasing : bool, default False
            Sort cohorts by descending median TMB instead of ascending.
        output_dir : str, default "."
            Directory where output files are saved when ``save_files=True``.
        save_files : bool, default True
            Whether to save the results to TSV files.

        Returns
        -------
        Dict[str, pd.DataFrame]
            - 'median_mutation_burden': one row per cohort (this one plus
              every TCGA cohort compared against), with ``Cohort``,
              ``Cohort_Size`` and ``Median_Mutations``, sorted by
              ``Median_Mutations`` (maftools' ``median_mutation_burden``).
            - 'mutation_burden_perSample': per-sample ``total`` (and
              ``total_perMB`` when ``capture_size`` is given) for every
              sample in every compared cohort (maftools'
              ``mutation_burden_perSample``).
            - 'pairwise_t_test': pooled-variance pairwise t-tests between
              every pair of cohorts (Benjamini-Hochberg/FDR adjusted),
              equivalent to maftools' ``pairwise_t_test``.
        """
        from ..visualizations.tcga_compare_plot import compute_tcga_compare

        result = compute_tcga_compare(
            self,
            cohort_name=cohort_name,
            capture_size=capture_size,
            tcga_capture_size=tcga_capture_size,
            tcga_cohorts=tcga_cohorts,
            primary_site=primary_site,
            rm_hyper=rm_hyper,
            rm_zero=rm_zero,
            decreasing=decreasing,
        )

        public_result = {
            'median_mutation_burden': result['median_mutation_burden'],
            'mutation_burden_perSample': result['mutation_burden_perSample'],
            'pairwise_t_test': result['pairwise_t_test'],
        }

        if save_files:
            os.makedirs(output_dir, exist_ok=True)

            median_path = os.path.join(output_dir, "tcga_compare_median_mutation_burden.tsv")
            persample_path = os.path.join(output_dir, "tcga_compare_mutation_burden_perSample.tsv")
            pairwise_path = os.path.join(output_dir, "tcga_compare_pairwise_t_test.tsv")

            public_result['median_mutation_burden'].to_csv(median_path, sep='\t', index=False, float_format='%.6f')
            public_result['mutation_burden_perSample'].to_csv(persample_path, sep='\t', index=False, float_format='%.6f')
            public_result['pairwise_t_test'].to_csv(pairwise_path, sep='\t', index=False, float_format='%.6g')

            logger.info(f"TCGA compare median mutation burden saved to: {median_path}")
            logger.info(f"TCGA compare per-sample mutation burden saved to: {persample_path}")
            logger.info(f"TCGA compare pairwise t-test saved to: {pairwise_path}")

        return public_result


def log_tmb_summary(analysis_df: pd.DataFrame) -> None:
    """
    Log a comprehensive summary of TMB analysis results.

    This function provides a detailed summary of the TMB analysis using proper logging
    instead of print statements. It displays key insights about the mutation burden
    across all samples in the analysis.

    Parameters
    ----------
    analysis_df : pd.DataFrame
        The analysis DataFrame returned by calculate_tmb_analysis containing
        per-sample TMB metrics.
    """
    if analysis_df.empty:
        logger.warning("No analysis data provided for summary")
        return

    logger.info("TMB ANALYSIS SUMMARY")

    # Basic statistics
    total_samples = len(analysis_df)
    avg_total = analysis_df['total'].mean()
    avg_tmb = analysis_df['total_perMB'].mean()
    median_tmb = analysis_df['total_perMB'].median()

    logger.info(f"• Total samples analyzed: {total_samples}")
    logger.info(f"• Average non-synonymous mutations per sample: {avg_total:.1f}")
    logger.info(f"• Average TMB: {avg_tmb:.6f} mutations/Mb")
    logger.info(f"• Median TMB: {median_tmb:.6f} mutations/Mb")

    # Extreme values
    max_tmb_idx = analysis_df['total_perMB'].idxmax()
    min_tmb_idx = analysis_df['total_perMB'].idxmin()

    max_tmb_sample = analysis_df.loc[max_tmb_idx, 'Tumor_Sample_Barcode']
    max_tmb_value = analysis_df['total_perMB'].max()

    min_tmb_sample = analysis_df.loc[min_tmb_idx, 'Tumor_Sample_Barcode']
    min_tmb_value = analysis_df['total_perMB'].min()

    logger.info(f"• Sample with highest TMB: {max_tmb_sample}")
    logger.info(f"  - TMB value: {max_tmb_value:.6f} mutations/Mb")
    logger.info(f"• Sample with lowest TMB: {min_tmb_sample}")
    logger.info(f"  - TMB value: {min_tmb_value:.6f} mutations/Mb")
