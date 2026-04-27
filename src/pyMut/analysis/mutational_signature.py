"""
Mutational signature analysis module for PyMutation.

This module provides functionality for analyzing trinucleotide contexts
and generating mutational signature matrices.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

# Define the 96 trinucleotide contexts in standard order
TRINUCLEOTIDE_CONTEXTS = [
    # C>A mutations
    "A[C>A]A", "A[C>A]C", "A[C>A]G", "A[C>A]T",
    "C[C>A]A", "C[C>A]C", "C[C>A]G", "C[C>A]T",
    "G[C>A]A", "G[C>A]C", "G[C>A]G", "G[C>A]T",
    "T[C>A]A", "T[C>A]C", "T[C>A]G", "T[C>A]T",

    # C>G mutations
    "A[C>G]A", "A[C>G]C", "A[C>G]G", "A[C>G]T",
    "C[C>G]A", "C[C>G]C", "C[C>G]G", "C[C>G]T",
    "G[C>G]A", "G[C>G]C", "G[C>G]G", "G[C>G]T",
    "T[C>G]A", "T[C>G]C", "T[C>G]G", "T[C>G]T",

    # C>T mutations
    "A[C>T]A", "A[C>T]C", "A[C>T]G", "A[C>T]T",
    "C[C>T]A", "C[C>T]C", "C[C>T]G", "C[C>T]T",
    "G[C>T]A", "G[C>T]C", "G[C>T]G", "G[C>T]T",
    "T[C>T]A", "T[C>T]C", "T[C>T]G", "T[C>T]T",

    # T>A mutations
    "A[T>A]A", "A[T>A]C", "A[T>A]G", "A[T>A]T",
    "C[T>A]A", "C[T>A]C", "C[T>A]G", "C[T>A]T",
    "G[T>A]A", "G[T>A]C", "G[T>A]G", "G[T>A]T",
    "T[T>A]A", "T[T>A]C", "T[T>A]G", "T[T>A]T",

    # T>C mutations
    "A[T>C]A", "A[T>C]C", "A[T>C]G", "A[T>C]T",
    "C[T>C]A", "C[T>C]C", "C[T>C]G", "C[T>C]T",
    "G[T>C]A", "G[T>C]C", "G[T>C]G", "G[T>C]T",
    "T[T>C]A", "T[T>C]C", "T[T>C]G", "T[T>C]T",

    # T>G mutations
    "A[T>G]A", "A[T>G]C", "A[T>G]G", "A[T>G]T",
    "C[T>G]A", "C[T>G]C", "C[T>G]G", "C[T>G]T",
    "G[T>G]A", "G[T>G]C", "G[T>G]G", "G[T>G]T",
    "T[T>G]A", "T[T>G]C", "T[T>G]G", "T[T>G]T"
]

CONTEXT_TO_INDEX = {context: idx for idx, context in enumerate(TRINUCLEOTIDE_CONTEXTS)}
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}


def _get_reverse_complement(sequence: str) -> str:
    """Get reverse complement of a DNA sequence."""
    return ''.join(COMPLEMENT[base] for base in reversed(sequence))


def _normalize_to_pyrimidine(ref: str, alt: str, trinuc: str) -> Tuple[str, str, str]:
    """
    Normalize mutation to pyrimidine context (C or T as reference).

    Parameters
    ----------
    ref : str
        Reference allele
    alt : str
        Alternative allele
    trinuc : str
        Trinucleotide context

    Returns
    -------
    Tuple[str, str, str]
        Normalized (ref, alt, trinuc)
    """
    if ref in ['C', 'T']:
        return ref, alt, trinuc
    else:
        # Convert to reverse complement for pyrimidine context
        new_ref = COMPLEMENT[ref]
        new_alt = COMPLEMENT[alt]
        new_trinuc = _get_reverse_complement(trinuc)
        return new_ref, new_alt, new_trinuc


def _get_trinucleotide_context(fasta, chrom: str, pos: int) -> Optional[str]:
    """
    Extract trinucleotide context from FASTA file.

    Parameters
    ----------
    fasta : pyfaidx.Fasta
        FASTA file object
    chrom : str
        Chromosome name
    pos : int
        Position (1-based)

    Returns
    -------
    Optional[str]
        Trinucleotide context or None if not available
    """
    try:
        # Handle chromosome name variations
        chrom_key = chrom
        if chrom_key not in fasta.keys():
            if not chrom_key.startswith('chr'):
                chrom_key = f'chr{chrom}'
            elif chrom_key.startswith('chr'):
                chrom_key = chrom_key[3:]

        if chrom_key not in fasta.keys():
            logger.warning(f"Chromosome {chrom} not found in FASTA file")
            return None

        # Extract trinucleotide (1-based coordinates)
        trinuc = fasta[chrom_key][pos - 2:pos + 1].seq.upper()

        if len(trinuc) != 3 or 'N' in trinuc:
            return None

        return trinuc

    except Exception as e:
        logger.warning(f"Error extracting trinucleotide for {chrom}:{pos}: {e}")
        return None


def _create_context_label(ref: str, alt: str, trinuc: str) -> str:
    """
    Create the 96-context label.

    Parameters
    ----------
    ref : str
        Reference allele
    alt : str
        Alternative allele
    trinuc : str
        Trinucleotide context

    Returns
    -------
    str
        Context label in format "X[REF>ALT]Z"
    """
    return f"{trinuc[0]}[{ref}>{alt}]{trinuc[2]}"


class MutationalSignatureMixin:
    """
    Mixin class providing mutational signature analysis functionality for PyMutation objects.
    
    This mixin adds trinucleotide context matrix generation and mutational signature
    analysis capabilities to PyMutation, following the same architectural pattern 
    as other mixins in the project.
    """

    def trinucleotideMatrix(
        self,
        ref_genome: str,
        prefix: Optional[str] = None,
        add: bool = True,
        ignoreChr: Optional[list] = None,
        useSyn: bool = True,
        fn: Optional[str] = None,
        apobec_window: int = 20,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate the 96xN trinucleotide context matrix and return the enriched SNV table.

        Parameters
        ----------
        ref_genome : str
            Path to reference FASTA (e.g., hs37d5.fa or hs37d5.fa.gz)
        prefix : str, optional
            Prefix to add or remove from chromosome names (e.g., 'chr').
        add : bool
            If True, add `prefix` to chromosome names in the MAF; if False, remove it.
        ignoreChr : list of str, optional
            Chromosomes to ignore from analysis (e.g., ['chrM']).
        useSyn : bool
            Include synonymous variants. If False, they are excluded.
        fn : str, optional
            If provided, writes APOBEC results to an output file with basename `fn`.
        apobec_window : int
            Window size (+/-) for background TCW motif estimation (approximation here).

        Returns
        -------
        contexts_df : pd.DataFrame
            96 x N matrix (index = 96 standard contexts, columns = samples)
        enriched_df : pd.DataFrame
            Valid SNVs with added columns: 'trinuc', 'class96', 'idx96', 'norm_alt'
        """
        try:
            import pyfaidx
        except ImportError:
            raise ImportError("pyfaidx is required. Install with: pip install pyfaidx")

        if not hasattr(self, 'data'):
            raise AttributeError("Object has no attribute 'data'. Expected a PyMutation-like object.")

        df = self.data.copy()

        # Resolve aliases to canonical column names using fields.py
        try:
            from pyMut.utils.fields import find_alias
        except Exception:
            find_alias = None  # Fallback if utils not available

        def ensure_canonical(df_in: pd.DataFrame, canonical: str, required: bool = False) -> None:
            """Ensure df has a canonical column by copying from an alias if present."""
            if canonical in df_in.columns:
                return
            alias = None
            if find_alias is not None:
                try:
                    alias = find_alias(df_in.columns, canonical)
                except Exception:
                    alias = None
            if alias is not None:
                df_in[canonical] = df_in[alias]
                return
            if required:
                raise ValueError(f"Missing required column '{canonical}' (no alias found) in PyMutation.data")

        # Essential fields
        ensure_canonical(df, 'Chromosome', required=True)
        ensure_canonical(df, 'Start_Position', required=True)
        ensure_canonical(df, 'Reference_Allele', required=True)
        ensure_canonical(df, 'Tumor_Seq_Allele2', required=True)
        # Optional but commonly used fields
        ensure_canonical(df, 'End_Position', required=False)
        ensure_canonical(df, 'Tumor_Sample_Barcode', required=False)
        ensure_canonical(df, 'Variant_Classification', required=False)
        ensure_canonical(df, 'Variant_Type', required=False)
        ensure_canonical(df, 'Hugo_Symbol', required=False)
        ensure_canonical(df, 'Tumor_Seq_Allele1', required=False)

        # If End_Position is still missing, set it equal to Start_Position for SNPs
        if 'End_Position' not in df.columns:
            df['End_Position'] = df['Start_Position']

        # Optionally exclude synonymous if useSyn = False
        if not useSyn and 'Variant_Classification' in df.columns:
            df = df[~df['Variant_Classification'].isin({
                'Silent', 'Synonymous_SNV', 'Synonymous', 'IGR_Synonymous'
            })]

        # Keep only valid SNVs A/C/G/T and REF != ALT
        bases = {'A', 'C', 'G', 'T'}
        snv_mask = (
            df['Reference_Allele'].isin(bases) &
            df['Tumor_Seq_Allele2'].isin(bases) &
            (df['Reference_Allele'] != df['Tumor_Seq_Allele2'])
        )
        df = df[snv_mask].copy()
        if df.empty:
            raise ValueError('Zero SNPs to analyze!')

        # Apply prefix/add
        if prefix is not None:
            if add:
                df['Chromosome'] = df['Chromosome'].astype(str).apply(
                    lambda c: c if c.startswith(prefix) else f"{prefix}{c}"
                )
            else:
                df['Chromosome'] = df['Chromosome'].astype(str).str.replace(
                    prefix, '', regex=False
                )

        # Ignore chromosomes
        if ignoreChr is not None:
            df = df[~df['Chromosome'].isin(set(ignoreChr))]

        # Drop NA positions
        na_pos = df['Start_Position'].isna().sum()
        if na_pos:
            logger.info(f"-Removed {na_pos} loci with NAs in Start_Position")
            df = df[df['Start_Position'].notna()]
        if df.empty:
            raise ValueError('Zero SNPs to analyze!')

        # Open FASTA
        try:
            fasta = pyfaidx.Fasta(ref_genome)
        except Exception as e:
            raise ValueError(f"Could not open FASTA {ref_genome}: {e}")

        # Resolve chromosome names against FASTA keys
        fasta_keys = set(fasta.keys())

        def _resolve_chrom(ch: str) -> Optional[str]:
            if ch in fasta_keys:
                return ch
            if not ch.startswith('chr') and f'chr{ch}' in fasta_keys:
                return f'chr{ch}'
            if ch.startswith('chr') and ch[3:] in fasta_keys:
                return ch[3:]
            return None

        resolved = df['Chromosome'].astype(str).map(_resolve_chrom)
        missing = resolved.isna()
        miss_count = int(missing.sum())
        if miss_count:
            miss_chr = df.loc[missing, 'Chromosome'].astype(str).unique()
            logger.warning(
                "Chromosome names in MAF must match reference. "
                f"Ignoring {miss_count} SNVs from chromosomes: {', '.join(map(str, miss_chr[:10]))}" +
                (" ..." if len(miss_chr) > 10 else "")
            )
            df = df[~missing].copy()
            resolved = resolved[~missing]
        if df.empty:
            raise ValueError('Zero SNPs to analyze! Maybe add or remove prefix?')

        # Extract trinucleotide context and normalize to pyrimidine
        trinuc_list, class96_list, idx96_list, norm_alt_list = [], [], [], []

        for i, row in df.iterrows():
            chrom = str(row['Chromosome'])
            pos = int(row['Start_Position'])
            ref = str(row['Reference_Allele']).upper()
            alt = str(row['Tumor_Seq_Allele2']).upper()

            trinuc = _get_trinucleotide_context(fasta, chrom, pos)
            if trinuc is None:
                trinuc_list.append(None)
                class96_list.append(None)
                idx96_list.append(None)
                norm_alt_list.append(None)
                continue

            nref, nalt, ntrinuc = _normalize_to_pyrimidine(ref, alt, trinuc)
            label = _create_context_label(nref, nalt, ntrinuc)
            cidx = CONTEXT_TO_INDEX.get(label)

            trinuc_list.append(ntrinuc)
            class96_list.append(label)
            idx96_list.append(cidx)
            norm_alt_list.append(nalt)

        df['trinuc'] = trinuc_list
        df['class96'] = class96_list
        df['idx96'] = idx96_list
        df['norm_alt'] = norm_alt_list

        valid = df[df['idx96'].notna()].copy()
        if valid.empty:
            raise ValueError('No SNVs with valid trinucleotide contexts found')

        # Build 96 x samples matrix
        if 'Tumor_Sample_Barcode' in valid.columns:
            # Long format
            samples = valid['Tumor_Sample_Barcode'].astype(str).unique().tolist()
            M = np.zeros((96, len(samples)), dtype=int)
            for j, s in enumerate(samples):
                vc = valid.loc[valid['Tumor_Sample_Barcode'].astype(str) == s, 'idx96'].value_counts()
                for k, cnt in vc.items():
                    M[int(k), j] = int(cnt)
        else:
            # Wide format (heuristic). If none detected, global counts under 'ALL'
            standard_cols = {
                'Chromosome','Start_Position','End_Position','Reference_Allele','Tumor_Seq_Allele2',
                'Variant_Classification','Variant_Type','Hugo_Symbol','Tumor_Seq_Allele1'
            }
            samples = [c for c in valid.columns if c not in standard_cols and '|' in ''.join(valid[c].astype(str).head(10))]
            if not samples:
                samples = ['ALL']
                M = np.zeros((96, 1), dtype=int)
                vc = valid['idx96'].value_counts()
                for k, cnt in vc.items():
                    M[int(k), 0] = int(cnt)
            else:
                M = np.zeros((96, len(samples)), dtype=int)
                for _, row in valid.iterrows():
                    k = int(row['idx96'])
                    for j, s in enumerate(samples):
                        g = str(row[s])
                        if '|' in g:
                            a1, a2 = g.split('|')
                            alt_norm = str(row.get('norm_alt', row['Tumor_Seq_Allele2'])).upper()
                            cnt = int(a1.upper() == alt_norm) + int(a2.upper() == alt_norm)
                            M[k, j] += cnt

        contexts_df = pd.DataFrame(M, index=TRINUCLEOTIDE_CONTEXTS, columns=samples)

        # Optional APOBEC enrichment (approximate background)
        def _is_tcw(tri: str) -> bool:
            # tri in pyrimidine-normalized orientation
            return tri in {'TCA', 'TCT'}

        if fn is not None:
            from collections import defaultdict
            per_sample = defaultdict(lambda: dict(n_tcw=0, n_C=0, background_c=0, background_tcw=0))

            for _, row in valid.iterrows():
                sample = row['Tumor_Sample_Barcode'] if 'Tumor_Sample_Barcode' in valid.columns else samples[0]
                tri = row['trinuc']
                if pd.notna(row['class96']) and '[C>' in row['class96']:
                    per_sample[sample]['n_C'] += 1
                    if tri is not None and _is_tcw(tri):
                        per_sample[sample]['n_tcw'] += 1
                if tri is not None:
                    if len(tri) == 3 and tri[1] == 'C':
                        per_sample[sample]['background_c'] += 1
                    if _is_tcw(tri):
                        per_sample[sample]['background_tcw'] += 1

            try:
                from scipy.stats import fisher_exact
            except Exception:
                fisher_exact = None

            rows = []
            for s, d in per_sample.items():
                n_tcw = d['n_tcw']
                n_C = d['n_C'] if d['n_C'] > 0 else 1
                background_c = d['background_c'] if d['background_c'] > 0 else 1
                background_tcw = d['background_tcw'] if d['background_tcw'] > 0 else 1
                E = (n_tcw * background_c) / (n_C * background_tcw)
                pval = np.nan
                if fisher_exact is not None:
                    table = [[n_tcw, max(n_C - n_tcw, 0)], [background_tcw, max(background_c - background_tcw, 0)]]
                    try:
                        _, pval = fisher_exact(table, alternative='greater')
                    except Exception:
                        pval = np.nan
                rows.append({
                    'Tumor_Sample_Barcode': s,
                    'n_tcw': d['n_tcw'],
                    'n_C': d['n_C'],
                    'background_tcw': d['background_tcw'],
                    'background_c': d['background_c'],
                    'E': E,
                    'pvalue': pval,
                })
            apobec_df = pd.DataFrame(rows).sort_values('Tumor_Sample_Barcode')
            try:
                apobec_df.to_csv(f"{fn}.apobec_enrichment.tsv", sep='\t', index=False)
            except Exception as e:
                logger.warning(f"Could not write {fn}.apobec_enrichment.tsv: {e}")

        return contexts_df, valid




def estimateSignatures(contexts_df: pd.DataFrame, nMin: int = 2, nTry: int = 6,
                       nrun: int = 5, parallel: int = 4, pConstant: Optional[float] = None) -> Dict:
    """
    Estimate optimal number of mutational signatures using NMF decomposition.

    This method normalizes the input matrix to frequencies, performs NMF decomposition
    for different numbers of signatures (k), and calculates stability metrics to
    identify the optimal number of signatures.

    Parameters
    ----------
    contexts_df : pd.DataFrame
        96 x samples matrix with trinucleotide context counts (from trinucleotideMatrix)
    nMin : int, default 2
        Minimum number of signatures to test
    nTry : int, default 6
        Maximum number of signatures to test
    nrun : int, default 5
        Number of NMF runs per k value for stability assessment
    parallel : int, default 4
        Number of CPU cores to use for parallel processing
    pConstant : float, optional
        Small positive constant to add to matrix if NMF fails due to zeros

    Returns
    -------
    Dict
        Dictionary containing:
        - 'metrics': DataFrame with stability metrics for each k
        - 'models': List of NMF models for each k and run
        - 'optimal_k': Suggested optimal number of signatures

    Raises
    ------
    ImportError
        If required packages (scikit-learn, scipy) are not installed
    ValueError
        If input matrix is invalid or NMF consistently fails
    """
    import pandas as pd  # Local import to ensure availability
    try:
        import concurrent.futures

        from scipy.cluster.hierarchy import cophenet, linkage
        from scipy.spatial.distance import pdist
        from sklearn.decomposition import NMF
        from sklearn.metrics import mean_squared_error
    except ImportError as e:
        missing_pkg = str(e).split("'")[1] if "'" in str(e) else "required package"
        raise ImportError(f"{missing_pkg} is required for signature estimation. "
                          f"Install with: pip install scikit-learn scipy")

    # Validate input parameters
    if not isinstance(contexts_df, pd.DataFrame):
        raise ValueError("contexts_df must be a pandas DataFrame")

    if contexts_df.shape[0] != 96:
        raise ValueError("contexts_df must have 96 rows (trinucleotide contexts)")

    if nMin < 1 or nTry < nMin:
        raise ValueError("nMin must be >= 1 and nTry must be >= nMin")

    logger.info(f"Starting signature estimation for k={nMin} to k={nTry} with {nrun} runs each")

    # Normalize matrix to frequencies
    matrix = contexts_df.values.astype(float)
    col_sums = matrix.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    normalized_matrix = matrix / col_sums

    logger.info(f"Normalized matrix shape: {normalized_matrix.shape}")

    # Apply pseudocount if matrix has too many zeros
    zero_fraction = (normalized_matrix == 0).sum() / normalized_matrix.size
    if zero_fraction > 0.8 and pConstant is not None:
        logger.warning(f"Matrix has {zero_fraction:.2%} zeros, adding pConstant={pConstant}")
        normalized_matrix += pConstant
        col_sums = normalized_matrix.sum(axis=0, keepdims=True)
        normalized_matrix = normalized_matrix / col_sums

    def _run_nmf_single(k, run_idx, matrix):
        """Run a single NMF decomposition."""
        try:
            nmf = NMF(n_components=k, init='random', random_state=run_idx,
                      max_iter=1000, tol=1e-4)
            W = nmf.fit_transform(matrix)
            H = nmf.components_

            reconstructed = np.dot(W, H)
            rss = mean_squared_error(matrix, reconstructed) * matrix.size

            return {
                'k': k,
                'run': run_idx,
                'model': nmf,
                'W': W,
                'H': H,
                'rss': rss,
                'success': True
            }
        except Exception as e:
            logger.warning(f"NMF failed for k={k}, run={run_idx}: {e}")
            return {
                'k': k,
                'run': run_idx,
                'model': None,
                'W': None,
                'H': None,
                'rss': np.inf,
                'success': False
            }

    # Execute NMF runs in parallel and collect results
    all_results = []
    k_values = range(nMin, nTry + 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        for k in k_values:
            for run_idx in range(nrun):
                future = executor.submit(_run_nmf_single, k, run_idx, normalized_matrix)
                futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            all_results.append(result)

    results_by_k = {}
    for result in all_results:
        k = result['k']
        if k not in results_by_k:
            results_by_k[k] = []
        results_by_k[k].append(result)

    # Calculate metrics for each k
    metrics_data = []
    models_data = []

    for k in k_values:
        k_results = results_by_k[k]
        successful_results = [r for r in k_results if r['success']]

        if len(successful_results) == 0:
            logger.warning(f"All NMF runs failed for k={k}")
            continue

        # Calculate stability metrics
        rss_values = [r['rss'] for r in successful_results]
        mean_rss = np.mean(rss_values)
        std_rss = np.std(rss_values)

        # Calculate cophenetic correlation
        if len(successful_results) >= 2:
            H_matrices = [r['H'] for r in successful_results]
            n_samples = H_matrices[0].shape[1]

            consensus_matrix = np.zeros((n_samples, n_samples))

            for i in range(len(H_matrices)):
                for j in range(i + 1, len(H_matrices)):
                    H1, H2 = H_matrices[i], H_matrices[j]
                    corr_matrix = np.corrcoef(H1.T, H2.T)[:n_samples, n_samples:]
                    consensus_matrix += np.abs(corr_matrix)

            consensus_matrix /= (len(H_matrices) * (len(H_matrices) - 1) / 2)

            try:
                distance_matrix = 1 - consensus_matrix
                condensed_dist = pdist(distance_matrix)
                linkage_matrix = linkage(condensed_dist, method='average')
                cophenetic_corr, _ = cophenet(linkage_matrix, condensed_dist)
            except Exception:
                cophenetic_corr = np.nan
        else:
            cophenetic_corr = np.nan

        # Calculate dispersion (coefficient of variation of RSS)
        dispersion = std_rss / mean_rss if mean_rss > 0 else np.inf

        metrics_data.append({
            'k': k,
            'mean_rss': mean_rss,
            'std_rss': std_rss,
            'cophenetic_corr': cophenetic_corr,
            'dispersion': dispersion,
            'successful_runs': len(successful_results),
            'total_runs': len(k_results)
        })

        models_data.extend(successful_results)

    if not metrics_data:
        raise ValueError("All NMF decompositions failed. Try adjusting pConstant or input data.")

    metrics_df = pd.DataFrame(metrics_data)

    # Determine optimal number of signatures
    optimal_k = nMin
    if len(metrics_df) > 1:
        valid_coph = metrics_df.dropna(subset=['cophenetic_corr'])
        if len(valid_coph) > 1:
            coph_diff = valid_coph['cophenetic_corr'].diff().abs()
            if not coph_diff.isna().all():
                max_drop_idx = coph_diff.idxmax()
                optimal_k = valid_coph.loc[max_drop_idx, 'k']

    logger.info(f"Signature estimation completed. Suggested optimal k: {optimal_k}")

    return {
        'metrics': metrics_df,
        'models': models_data,
        'optimal_k': optimal_k,
        'normalized_matrix': normalized_matrix,
        'original_matrix': matrix
    }





def compare_signatures(W: np.ndarray, cosmic_path: str, min_cosine: float = 0.6,
                       return_matrix: bool = False) -> Dict:
    """
    Compare extracted signatures with COSMIC catalog using cosine similarity.

    This function loads the COSMIC catalog, validates context compatibility,
    calculates cosine similarity between each signature in W and each COSMIC signature,
    and returns a summary of best matches along with optionally the full similarity matrix.

    Parameters
    ----------
    W : numpy.ndarray
        96 * k matrix from extract_signatures with normalized signature profiles
        (each column sums to 1)
    cosmic_path : str
        Path to COSMIC catalog file (e.g., "COSMIC_v3.4_SBS_GRCh38.txt")
    min_cosine : float, default 0.6
        Minimum cosine similarity threshold for considering a match
    return_matrix : bool, default False
        If True, also return the full cosine similarity matrix

    Returns
    -------
    Dict
        Dictionary containing:
        - 'summary_df': pandas.DataFrame with columns ['Signature_W', 'Best_COSMIC', 'Cosine', 'Aetiology']
        - 'cosine_matrix': numpy.ndarray (k * N) - Only if return_matrix=True

    Raises
    ------
    FileNotFoundError
        If cosmic_path file does not exist
    ValueError
        If contexts don't match between W and COSMIC catalog
    ImportError
        If required packages are not installed

    Notes
    -----
    The function expects the COSMIC catalog to have 96 rows corresponding to 
    trinucleotide contexts and columns for different signatures. The first column
    should contain context labels, and there should be an 'aetiology' column or
    similar metadata.
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        missing_pkg = str(e).split("'")[1] if "'" in str(e) else "scikit-learn"
        raise ImportError(f"{missing_pkg} is required for cosine similarity calculation. "
                          f"Install with: pip install scikit-learn")

    # Validate input and load COSMIC catalog
    if not isinstance(W, np.ndarray):
        raise ValueError("W must be a numpy array")

    if W.shape[0] != 96:
        raise ValueError("W must have 96 rows (trinucleotide contexts)")

    if len(W.shape) != 2:
        raise ValueError("W must be a 2D array")

    logger.info(f"Comparing {W.shape[1]} signatures with COSMIC catalog")

    try:
        cosmic_df = pd.read_csv(cosmic_path, sep='\t')
    except FileNotFoundError:
        raise FileNotFoundError(f"COSMIC catalog file not found: {cosmic_path}")
    except Exception as e:
        raise ValueError(f"Error reading COSMIC catalog: {e}")

    logger.info(f"Loaded COSMIC catalog with shape {cosmic_df.shape}")

    if cosmic_df.shape[0] != 96:
        raise ValueError(f"COSMIC catalog must have 96 rows (trinucleotide contexts), got {cosmic_df.shape[0]}")

    # Process COSMIC catalog structure
    context_column = cosmic_df.columns[0]  # First column should be 'Type'
    contexts = cosmic_df[context_column].values  # All rows are data (no header row to skip)
    cosmic_df = cosmic_df.set_index(context_column)

    # Normalize W matrix and align with COSMIC catalog
    W_sums = W.sum(axis=0)
    if not np.allclose(W_sums, 1.0, rtol=1e-5):
        logger.warning("W matrix columns do not sum to 1, renormalizing...")
        W = W / W_sums.reshape(1, -1)
        logger.info("W matrix renormalized")

    logger.info("Aligning COSMIC catalog with standard trinucleotide context order...")

    missing_contexts = set(TRINUCLEOTIDE_CONTEXTS) - set(contexts)
    if missing_contexts:
        raise ValueError(f"COSMIC catalog is missing required contexts: {missing_contexts}")

    try:
        cosmic_df = cosmic_df.reindex(TRINUCLEOTIDE_CONTEXTS)
        if cosmic_df.isnull().any().any():
            raise ValueError("Some standard trinucleotide contexts are missing in COSMIC catalog")
        logger.info("COSMIC catalog successfully aligned to standard context order")
    except Exception as e:
        raise ValueError(f"Failed to align COSMIC catalog contexts: {e}")

    signature_columns = list(cosmic_df.columns)

    if len(signature_columns) == 0:
        raise ValueError("No signature columns found in COSMIC catalog")

    # Filter out artifact signatures
    artifact_signatures = {
        'SBS27', 'SBS43', 'SBS45', 'SBS46', 'SBS47', 'SBS48', 'SBS49', 'SBS50',
        'SBS51', 'SBS52', 'SBS53', 'SBS54', 'SBS55', 'SBS56', 'SBS57', 'SBS58',
        'SBS59', 'SBS60', 'SBS95'
    }

    filtered_columns = []
    for col in signature_columns:
        if col.endswith('c'):
            logger.info(f"Removing artifact signature {col} (ends with 'c')")
            continue
        if 'artefact' in col.lower():
            logger.info(f"Removing artifact signature {col} (contains 'artefact')")
            continue
        if col in artifact_signatures:
            logger.info(f"Removing artifact signature {col} (specified artifact)")
            continue
        filtered_columns.append(col)

    signature_columns = filtered_columns
    cosmic_df = cosmic_df[signature_columns]

    cosmic_matrix = cosmic_df.values.astype(float)

    column_sums = cosmic_matrix.sum(axis=0)
    zero_sum_mask = column_sums == 0
    if zero_sum_mask.any():
        zero_sum_sigs = [signature_columns[i] for i in range(len(signature_columns)) if zero_sum_mask[i]]
        logger.info(f"Removing signatures with zero sum: {zero_sum_sigs}")
        non_zero_mask = ~zero_sum_mask
        cosmic_matrix = cosmic_matrix[:, non_zero_mask]
        signature_columns = [sig for i, sig in enumerate(signature_columns) if non_zero_mask[i]]

    logger.info(f"Found {len(signature_columns)} valid COSMIC signatures after filtering")

    # Validate alignment and calculate cosine similarity
    if len(TRINUCLEOTIDE_CONTEXTS) != 96:
        raise ValueError(f"Expected 96 trinucleotide contexts in standard order, got {len(TRINUCLEOTIDE_CONTEXTS)}")

    if cosmic_matrix.shape[0] != 96:
        raise ValueError(f"COSMIC matrix must have 96 rows, got {cosmic_matrix.shape[0]}")

    current_contexts = cosmic_df.index.tolist()
    if current_contexts != TRINUCLEOTIDE_CONTEXTS:
        raise ValueError("COSMIC catalog contexts are not properly aligned with standard order")

    cosmic_normalized = cosmic_matrix / cosmic_matrix.sum(axis=0, keepdims=True)

    logger.info("COSMIC signatures normalized")

    cosine_matrix = cosine_similarity(W.T, cosmic_normalized.T)

    logger.info(f"Calculated cosine similarity matrix: {cosine_matrix.shape}")

    # Create comparison summary
    summary_data = []

    for i in range(W.shape[1]):
        similarities = cosine_matrix[i, :]
        best_idx = np.argmax(similarities)
        best_cosine = similarities[best_idx]
        best_cosmic = signature_columns[best_idx]

        if best_cosine >= min_cosine:
            match_status = best_cosmic
        else:
            match_status = "No match"

        aetiology = "Unknown"

        summary_data.append({
            'Signature_W': f'Signature {i + 1}',
            'Best_COSMIC': match_status,
            'Cosine': best_cosine,
            'Aetiology': aetiology
        })

    summary_df = pd.DataFrame(summary_data)

    logger.info(f"Created summary with {len(summary_data)} signature comparisons")

    result = {'summary_df': summary_df}

    if return_matrix:
        result['cosine_matrix'] = cosine_matrix

    return result



def align_signatures_to_cosmic(W: pd.DataFrame, H: pd.DataFrame, cosmic_path: str,
                               min_cosine: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Align extracted signatures with COSMIC catalog and reorder by best match.
    
    This function solves the NMF permutation problem by:
    1. Calculating cosine similarity between each extracted signature and all COSMIC signatures
    2. Finding the best COSMIC match for each signature
    3. Reordering signatures by their best match (e.g., SBS1-like, SBS5-like, SBS13-like)
    4. Renaming signatures with their COSMIC match and cosine similarity
    
    Parameters
    ----------
    W : pd.DataFrame
        Signature matrix (96 * k) from extractSignatures
    H : pd.DataFrame
        Contribution matrix (k * samples) from extractSignatures
    cosmic_path : str
        Path to COSMIC catalog file (e.g., "COSMIC_v3.4_SBS_GRCh37.txt")
    min_cosine : float, default 0.5
        Minimum cosine similarity to report a match
    
    Returns
    -------
    W_aligned : pd.DataFrame
        Reordered signature matrix with new column names
    H_aligned : pd.DataFrame
        Reordered contribution matrix with new row names
    signature_names : List[str]
        New signature names (e.g., ["SBS1-like (cos=0.95)", "SBS5-like (cos=0.89)"])
    
    Notes
    -----
    - This resolves the NMF permutation ambiguity
    - Signatures are ordered by their COSMIC match number (SBS1, SBS2, ...)
    - If no good match is found (cosine < min_cosine), signature keeps generic name
    
    Examples
    --------
    >>> result = py_mut.extractSignatures(contexts_df, n=3)
    >>> W_aligned, H_aligned, names = align_signatures_to_cosmic(
    ...     result['signatures'], 
    ...     result['contributions'],
    ...     "COSMIC_v3.4_SBS_GRCh37.txt"
    ... )
    """
    logger.info("Aligning extracted signatures with COSMIC catalog...")
    
    # Use compare_signatures to get best matches
    comparison = compare_signatures(
        W.values, 
        cosmic_path, 
        min_cosine=min_cosine, 
        return_matrix=False  # We only need the summary
    )
    
    summary_df = comparison['summary_df']
    
    # Create mapping: signature index -> (COSMIC name, cosine, sort key)
    signature_mapping = []
    for i in range(W.shape[1]):
        sig_name = W.columns[i]
        row = summary_df[summary_df['Signature_W'] == sig_name].iloc[0]
        best_cosmic = row['Best_COSMIC']
        cosine_val = row['Cosine']
        
        # Extract SBS number for sorting (e.g., "SBS1" -> 1, "SBS13" -> 13)
        try:
            sbs_num = int(best_cosmic.replace('SBS', ''))
        except (ValueError, AttributeError):
            sbs_num = 9999  # Put unmatched at the end
        
        # Create new name
        if cosine_val >= min_cosine:
            new_name = f"{best_cosmic}-like (cos={cosine_val:.2f})"
        else:
            new_name = f"Signature_{i+1} (no match)"
        
        signature_mapping.append({
            'original_idx': i,
            'original_name': sig_name,
            'new_name': new_name,
            'cosmic_match': best_cosmic,
            'cosine': cosine_val,
            'sort_key': sbs_num
        })
    
    # Sort by SBS number (ascending)
    signature_mapping_sorted = sorted(signature_mapping, key=lambda x: x['sort_key'])
    
    # Reorder W (columns) and H (rows)
    new_order = [x['original_idx'] for x in signature_mapping_sorted]
    new_names = [x['new_name'] for x in signature_mapping_sorted]
    
    W_aligned = W.iloc[:, new_order].copy()
    W_aligned.columns = new_names
    
    H_aligned = H.iloc[new_order, :].copy()
    H_aligned = H_aligned.set_axis(new_names, axis=0)
    
    logger.info(f"Signatures aligned and reordered: {new_names}")
    
    return W_aligned, H_aligned, new_names


def extractSignatures(contexts_df: pd.DataFrame, n: int, parallel: int = 4,
                      pConstant: Optional[float] = None) -> Dict:
    """
    Extract mutational signatures from trinucleotide context matrix.

    This function decomposing a 96 x N trinucleotide-context matrix into n
    signatures using Non-negative Matrix Factorization (NMF).

    Parameters
    ----------
    contexts_df : pd.DataFrame
        Input matrix of dimension 96 x N generated by trinucleotideMatrix.
    n : int
        Number of signatures to extract.
    parallel : int, default 4
        Number of cores to use
    pConstant : float, optional
        A small positive value to add to the matrix. Use it ONLY if the
        function throws numerical errors.

    Returns
    -------
    Dict
        Dictionary containing:
        - 'signatures': pd.DataFrame - Scaled signature matrix (96 x n)
        - 'contributions': pd.DataFrame - Scaled contribution matrix (n x samples)
        - 'nmfObj': NMF model object
        - 'contributions_abs': pd.DataFrame - Absolute contribution matrix (n x samples)

    Raises
    ------
    ImportError
        If scikit-learn is not available
    ValueError
        If input parameters are invalid
    """
    try:
        from sklearn.decomposition import NMF
    except ImportError:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

    import time

    start_time = time.time()

    # Validate input
    if not isinstance(contexts_df, pd.DataFrame):
        raise ValueError("contexts_df must be a pandas DataFrame")

    if contexts_df.shape[0] != 96:
        raise ValueError("contexts_df must have 96 rows (trinucleotide contexts)")

    if n < 1:
        raise ValueError("n must be >= 1")

    # Transpose matrix: we expect 96 x samples; NMF expects samples x 96 here
    matrix = contexts_df.T.values.astype(float)

    # Check for zero mutation classes
    zero_mut_mask = matrix.sum(axis=0) == 0
    zero_mut_classes = contexts_df.index[zero_mut_mask].tolist()

    if len(zero_mut_classes) > 0:
        logger.info('-Found zero mutations for conversions:')
        for temp in zero_mut_classes:
            logger.info(f"  {temp}")

    # Auto pConstant: if any sample has total mutations = 0, use a small constant by default
    sample_sums = matrix.sum(axis=1)
    zero_sum_samples_mask = sample_sums == 0
    if np.any(zero_sum_samples_mask):
        zero_samples = contexts_df.columns[zero_sum_samples_mask].tolist()
        logger.info(f"-Found {len(zero_samples)} samples with total mutations = 0; applying pConstant=0.01")
        if pConstant is None:
            pConstant = 0.01
    
    # Add pConstant if specified to avoid numerical issues
    if pConstant is not None:
        if pConstant <= 0:
            raise ValueError("pConstant must be > 0")
        matrix = matrix + pConstant
        logger.debug(f"Added pConstant={pConstant} to avoid numerical issues")

    # Run NMF decomposition
    logger.debug(f"Running NMF for factorization rank: {n}")

    # Use deterministic initialization for reproducibility
    # init='random' with fixed random_state=0 ensures consistent results
    nmf_model = NMF(n_components=n, init='random', random_state=0,
                    solver='cd', beta_loss='frobenius', max_iter=500, tol=1e-4)

    try:
        W = nmf_model.fit_transform(matrix)  # samples x n (contributions)
        H = nmf_model.components_            # n x 96 (signatures)
    except Exception as e:
        raise ValueError(f"NMF decomposition failed: {e}")

    # Scale signatures (basis) - each signature should sum to 1
    # H is n x 96, we want each row (signature) to sum to 1
    signatures_scaled = H / H.sum(axis=1, keepdims=True)
    signatures_df = pd.DataFrame(
        signatures_scaled.T,  # Transpose to get 96 x n
        index=contexts_df.index,
        columns=[f'Signature {i+1}' for i in range(n)]
    )

    # Handle contributions
    # W is samples x n, we want n x samples for output
    contributions_abs = W.T  # n x samples
    contributions_abs_df = pd.DataFrame(
        contributions_abs,
        index=[f'Signature {i+1}' for i in range(n)],
        columns=contexts_df.columns
    )

    # Scale contributions - each sample should sum to 1
    if n == 1:
        # For single signature, contribution will be 100% per sample
        contributions_scaled = contributions_abs / contributions_abs
        contributions_scaled[~np.isfinite(contributions_scaled)] = 1.0
    else:
        # Scale so each column (sample) sums to 1
        col_sums = contributions_abs.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        contributions_scaled = contributions_abs / col_sums

    contributions_df = pd.DataFrame(
        contributions_scaled,
        index=[f'Signature {i+1}' for i in range(n)],
        columns=contexts_df.columns
    )

    elapsed_time = time.time() - start_time
    logger.info(f"NMF signature extraction completed in {elapsed_time:.2f} seconds")

    return {
        'signatures': signatures_df,
        'contributions': contributions_df,
        'nmfObj': nmf_model,
        'contributions_abs': contributions_abs_df
    }