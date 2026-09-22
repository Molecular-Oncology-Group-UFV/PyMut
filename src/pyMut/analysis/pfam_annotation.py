import logging
import re
from typing import Optional, Dict, Tuple

import duckdb
import pandas as pd

from ..utils.database import (
    PfamAnnotationError,
    connect_db
)
from ..utils.fields import col, find_alias

# Configure logger
logger = logging.getLogger(__name__)


class PfamAnnotationMixin:
    """
    Mixin class providing PFAM annotation functionality for PyMutation objects.
    
    This mixin adds PFAM domain annotation capabilities to PyMutation,
    following the same architectural pattern as other mixins in the project.
    """

    def resolve_uniprot_identifiers(self, df: pd.DataFrame, uniprot_column: str, db_conn: duckdb.DuckDBPyConnection) -> Tuple[
        pd.DataFrame, Dict[str, int]]:
        """
        Resolve UniProt identifiers to canonical accessions.
        
        Handles three types of values:
        1. Accession (format "P31946") - use as is
        2. Short name (format "1433B_HUMAN") - resolve via short_name column
        3. External identifiers (NP_*, ENSP_*, etc.) - resolve via prot_id column
        
        Args:
            df: DataFrame with UniProt identifiers
            uniprot_column: Name of column containing UniProt identifiers
            db_conn: DuckDB connection
            
        Returns:
            Tuple of (DataFrame with uniprot_resolved column, resolution statistics)
        """
        logger.debug("Resolving UniProt identifiers to canonical accessions...")

        # Initialize statistics
        stats = {
            'total': 0,
            'direct_accession': 0,
            'via_short_name': 0,
            'via_external_id': 0,
            'unresolved': 0
        }

        df_work = df.copy()
        df_work['uniprot_resolved'] = None
        df_work['resolution_method'] = None

        # Process each unique identifier
        unique_ids = df_work[uniprot_column].dropna().unique()
        resolution_cache = {}

        for uniprot_id in unique_ids:
            if pd.isna(uniprot_id) or str(uniprot_id).strip() == '':
                continue

            uniprot_str = str(uniprot_id).strip()
            stats['total'] += 1

            # Check if already a valid accession (starts with letter, contains digits)
            if re.match(r'^[A-Z][0-9A-Z]{5}$', uniprot_str) or re.match(r'^[OPQ][0-9][A-Z0-9]{3}[0-9]$', uniprot_str):
                resolution_cache[uniprot_id] = (uniprot_str, 'direct_accession')
                stats['direct_accession'] += 1
                continue

            # Try to resolve via short_name column
            try:
                result = db_conn.execute(
                    "SELECT uniprot FROM xref WHERE short_name = ? LIMIT 1",
                    [uniprot_str]
                ).fetchone()

                if result:
                    resolved_accession = result[0]
                    resolution_cache[uniprot_id] = (resolved_accession, 'via_short_name')
                    stats['via_short_name'] += 1
                    continue
            except Exception as e:
                logger.warning(f"Error querying short_name for {uniprot_str}: {e}")

            # Try to resolve via prot_id column (external identifiers)
            try:
                result = db_conn.execute(
                    "SELECT uniprot FROM xref WHERE prot_id = ? LIMIT 1",
                    [uniprot_str]
                ).fetchone()

                if result:
                    resolved_accession = result[0]
                    resolution_cache[uniprot_id] = (resolved_accession, 'via_external_id')
                    stats['via_external_id'] += 1
                    continue
            except Exception as e:
                logger.warning(f"Error querying prot_id for {uniprot_str}: {e}")

            # Mark as unresolved
            resolution_cache[uniprot_id] = (None, 'unresolved')
            stats['unresolved'] += 1

        # Apply resolutions to DataFrame
        for idx, row in df_work.iterrows():
            uniprot_id = row[uniprot_column]
            if uniprot_id in resolution_cache:
                resolved_accession, method = resolution_cache[uniprot_id]
                df_work.loc[idx, 'uniprot_resolved'] = resolved_accession
                df_work.loc[idx, 'resolution_method'] = method

        logger.info("UniProt resolution summary:")
        logger.info(f"   Total identifiers processed: {stats['total']:,}")
        logger.info(f"   Direct accessions: {stats['direct_accession']:,}")
        logger.info(f"   Resolved via short_name: {stats['via_short_name']:,}")
        logger.info(f"   Resolved via external ID: {stats['via_external_id']:,}")
        logger.info(f"   Unresolved: {stats['unresolved']:,}")

        return df_work, stats

    def _annotate_pfam_sql(self, df: pd.DataFrame, db_conn: duckdb.DuckDBPyConnection, aa_column: str,
                           uniprot_alias: str) -> pd.DataFrame:
        """Annotate PFAM domains using SQL for larger datasets."""
        logger.debug("Using SQL for PFAM annotation...")

        db_conn.register('variants_temp', df)

        # SQL query for range join - includes seq_start and seq_end coordinates,
        # plus pfam_description (real domain description, populated from the
        # official Pfam-A.clans.tsv.gz file when the database was built).
        query = f"""
        SELECT v.*, p.pfam_id, p.pfam_name, p.pfam_description, p.seq_start, p.seq_end
        FROM variants_temp v
        LEFT JOIN pfam p ON v.{uniprot_alias} = p.uniprot 
                        AND v.{aa_column} BETWEEN p.seq_start AND p.seq_end
        """

        try:
            result_df = db_conn.execute(query).df()
        except duckdb.Error:
            # Older database schema, built before pfam_description was added.
            # Fall back gracefully; suggest rebuilding for real domain names.
            logger.debug(
                "'pfam_description' column not found in the database (older schema). "
                "Rebuild the database (connect_db(force_rebuild=True)) to get real "
                "Pfam domain names/descriptions from Pfam-A.clans.tsv.gz."
            )
            query = f"""
            SELECT v.*, p.pfam_id, p.pfam_name, p.seq_start, p.seq_end
            FROM variants_temp v
            LEFT JOIN pfam p ON v.{uniprot_alias} = p.uniprot 
                            AND v.{aa_column} BETWEEN p.seq_start AND p.seq_end
            """
            result_df = db_conn.execute(query).df()
            result_df['pfam_description'] = None

        db_conn.unregister('variants_temp')

        return result_df

    def _annotate_with_database(self, df, db_conn, aa_column, uniprot_alias):
        """Use database for precise PFAM annotation with enhanced UniProt resolution"""
        # Create a working copy with extracted columns
        df_work = df.copy()

        # VEP commonly stores positions as strings such as ``237`` or
        # ``237/1104``. DuckDB needs a numeric value for the range join.
        df_work[aa_column] = pd.to_numeric(
            df_work[aa_column].astype(str).str.extract(r'(\d+)')[0],
            errors='coerce',
        )

        # Extract uniprot and aa_pos if they don't exist
        if 'uniprot' not in df_work.columns:
            df_work['uniprot'] = df_work[uniprot_alias].apply(
                lambda x: str(x).split('.')[0] if pd.notna(x) and x != '' else None
            )

        # Resolve UniProt identifiers to canonical accessions
        df_work, resolution_stats = self.resolve_uniprot_identifiers(df_work, 'uniprot', db_conn)

        # Filter for valid data (must have resolved accession and amino acid position)
        df_valid = df_work.dropna(subset=['uniprot_resolved', aa_column])
        df_valid = df_valid[df_valid['uniprot_resolved'] != '']

        if len(df_valid) == 0:
            logger.warning("No variants with resolved UniProt accessions and amino acid positions")
            # Add empty PFAM columns
            df_work['pfam_id'] = None
            df_work['pfam_name'] = None
            df_work['pfam_description'] = None
            df_work['seq_start'] = None
            df_work['seq_end'] = None
            # Store resolution statistics even if no valid rows to annotate
            df_work.attrs['resolution_stats'] = resolution_stats
            return df_work

        logger.debug(f"Annotating {len(df_valid)} variants with PFAM domains...")

        # Get all variants with PFAM annotations from SQL query using resolved accessions
        result_df = self._annotate_pfam_sql(df_valid, db_conn, aa_column, 'uniprot_resolved')

        # Count successful annotations
        pfam_annotated_count = result_df['pfam_id'].notna().sum()
        logger.info(f"Variants annotated with PFAM: {pfam_annotated_count}/{len(result_df)}")

        # Add PFAM columns to the working dataframe
        df_work['pfam_id'] = None
        df_work['pfam_name'] = None
        df_work['pfam_description'] = None
        df_work['seq_start'] = None
        df_work['seq_end'] = None

        # Update working dataframe with PFAM annotations
        for idx in result_df.index:
            if pd.notna(result_df.loc[idx, 'pfam_id']):
                df_work.loc[idx, 'pfam_id'] = result_df.loc[idx, 'pfam_id']
                df_work.loc[idx, 'pfam_name'] = result_df.loc[idx, 'pfam_name']
                df_work.loc[idx, 'pfam_description'] = result_df.loc[idx, 'pfam_description']
                df_work.loc[idx, 'seq_start'] = result_df.loc[idx, 'seq_start']
                df_work.loc[idx, 'seq_end'] = result_df.loc[idx, 'seq_end']

        # Store resolution statistics for reporting
        df_work.attrs['resolution_stats'] = resolution_stats

        return df_work

    def _annotate_with_vep_domains(self, df, db_conn=None):
        """Parse PFAM from VEP_DOMAINS as fallback"""
        logger.debug("Extracting PFAM domains from VEP_DOMAINS column...")

        # Check if VEP_DOMAINS column exists
        domains_series = col(df, 'Domains')
        if domains_series is None:
            logger.warning("VEP_DOMAINS column not found")
            result_data = df.copy()
            for pfam_col in ['pfam_id', 'pfam_name', 'pfam_description', 'seq_start', 'seq_end']:
                result_data[pfam_col] = None
            return result_data

        # Build a pfam_id -> (name, description) lookup so that even fallback
        # annotations coming from VEP_DOMAINS (which only gives us the Pfam
        # accession, not its name) can show the real domain name/description
        # instead of repeating the accession, when a db connection is available.
        pfam_name_lookup: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        if db_conn is not None:
            try:
                lookup_df = db_conn.execute(
                    "SELECT DISTINCT pfam_id, pfam_name, pfam_description FROM pfam"
                ).df()
                pfam_name_lookup = {
                    row.pfam_id: (row.pfam_name, row.pfam_description)
                    for row in lookup_df.itertuples()
                }
            except duckdb.Error:
                logger.debug("Could not build pfam_id -> name lookup for the VEP_DOMAINS fallback.")

        # Extract UniProt IDs and PFAM domains from VEP columns
        result_rows = []

        for idx, row in df.iterrows():
            new_row = row.to_dict()

            # Extract UniProt ID if missing
            if 'uniprot' not in new_row or pd.isna(new_row.get('uniprot')):
                uniprot_id = None
                uniprot_series = col(pd.DataFrame([row]), 'UNIPROT')
                if uniprot_series is not None and pd.notna(uniprot_series.iloc[0]) and uniprot_series.iloc[0] != '':
                    uniprot_id = str(uniprot_series.iloc[0]).split('.')[0]  # Remove version
                new_row['uniprot'] = uniprot_id

            # Extract amino acid position if missing
            if 'aa_pos' not in new_row or pd.isna(new_row.get('aa_pos')):
                aa_pos = None
                protein_change_series = col(pd.DataFrame([row]), 'Protein_Change')
                if protein_change_series is not None and pd.notna(protein_change_series.iloc[0]):
                    match = re.search(r'p\.[A-Za-z]*?(\d+)', str(protein_change_series.iloc[0]))
                    if match:
                        aa_pos = int(match.group(1))
                new_row['aa_pos'] = aa_pos

            # Extract PFAM domains from VEP_DOMAINS
            pfam_domains = []
            domains_series = col(pd.DataFrame([row]), 'Domains')
            if domains_series is not None and pd.notna(domains_series.iloc[0]) and domains_series.iloc[0] != '':
                domains_str = str(domains_series.iloc[0])
                if 'Pfam:' in domains_str:
                    pfam_matches = re.findall(r'Pfam:([^,;\s]+)', domains_str)
                    pfam_domains.extend(pfam_matches)

            # Set PFAM information
            if pfam_domains:
                pfam_id_found = pfam_domains[0]  # Take first domain
                resolved_name, resolved_description = pfam_name_lookup.get(
                    pfam_id_found, (pfam_id_found, None)
                )
                new_row['pfam_id'] = pfam_id_found
                new_row['pfam_name'] = resolved_name
                new_row['pfam_description'] = resolved_description
                new_row['seq_start'] = None
                new_row['seq_end'] = None
            else:
                new_row['pfam_id'] = None
                new_row['pfam_name'] = None
                new_row['pfam_description'] = None
                new_row['seq_start'] = None
                new_row['seq_end'] = None

            result_rows.append(new_row)

        result_df = pd.DataFrame(result_rows)

        # Show summary
        total_variants = len(result_df)
        with_uniprot = result_df['uniprot'].notna().sum()
        with_aa_pos = result_df['aa_pos'].notna().sum()
        with_pfam = result_df['pfam_id'].notna().sum()

        logger.debug("Processing summary:")
        logger.debug(f"   Total variants: {total_variants:,}")
        logger.debug(f"   With UniProt ID: {with_uniprot:,}")
        logger.debug(f"   With amino acid position: {with_aa_pos:,}")
        logger.debug(f"   With PFAM domains: {with_pfam:,}")

        return result_df

    def _annotate_with_text_domains(self, df, db_conn=None):
        """Parse a loose PFAM accession from domain-related text columns."""
        result_df = df.copy()
        result_df['pfam_id'] = None
        result_df['pfam_name'] = None
        result_df['pfam_description'] = None
        result_df['seq_start'] = None
        result_df['seq_end'] = None

        lookup = {}
        if db_conn is not None:
            try:
                lookup_df = db_conn.execute(
                    "SELECT DISTINCT pfam_id, pfam_name, pfam_description FROM pfam"
                ).df()
                lookup = {
                    row.pfam_id: (row.pfam_name, row.pfam_description)
                    for row in lookup_df.itertuples()
                }
            except duckdb.Error:
                logger.debug("Could not build PFAM lookup for loose text parsing.")

        text_columns = [
            column for column in ('VEP_DOMAINS', 'DOMAINS', 'Domains',
                                  'Protein_domains', 'Pfam_domain',
                                  'Pfam_ID', 'Pfam_Name')
            if column in result_df.columns
        ]
        if not text_columns:
            return result_df

        accession_re = re.compile(r'PF\d{5}', re.IGNORECASE)
        for idx, values in result_df[text_columns].fillna('').astype(str).iterrows():
            text = '|'.join(values.tolist())
            match = accession_re.search(text)
            if match:
                pfam_id = match.group(0).upper()
                name, description = lookup.get(pfam_id, (pfam_id, None))
                result_df.loc[idx, 'pfam_id'] = pfam_id
                result_df.loc[idx, 'pfam_name'] = name
                result_df.loc[idx, 'pfam_description'] = description

        return result_df

    def _annotate_pfam_basic(self, df: pd.DataFrame, db_conn: duckdb.DuckDBPyConnection):
        """
        strategy='basic': maftools-equivalent PFAM annotation, and nothing
        else.

        This is a thin wrapper around the exact same helpers used for the
        gene-symbol mapping (_resolve_hugo_column,
        _resolve_aachange_column, _annotate_with_gene_position) - no
        annotation logic is duplicated here. This strategy is unconditional: it always attempts the gene-symbol mapping and never
        falls back to anything else, so any variant it cannot resolve simply
        stays unmapped. That mirrors maftools' pfamDomains() itself, which
        also has no fallback.

        Raises ValueError if the required columns or the bundled gene->Pfam
        table are unavailable, since silently returning an all-empty
        annotation would be more confusing than a clear error for a strategy
        whose entire point is "do this one specific thing".
        """
        from ..core import PyMutation

        hugo_alias = self._resolve_hugo_column(df)
        aachange_alias = self._resolve_aachange_column(df)
        if hugo_alias is None or aachange_alias is None:
            raise ValueError(
                "strategy='basic' requires a gene-symbol column (e.g. Hugo_Symbol) "
                "and an HGVS-style protein-change column (e.g. HGVSp_Short / "
                "Protein_Change / AAChange), the same way maftools' pfamDomains() "
                "does. Neither could be found by name in this data."
            )

        result_df = self._annotate_with_gene_position(
            df, db_conn, hugo_alias, aachange_alias, 'aa_pos'
        )
        if result_df is None:
            raise ValueError(
                "strategy='basic' requires the bundled gene->Pfam domain table "
                "(data/pfam_domains.csv), which was not found in this pyMut "
                "installation."
            )

        total = len(result_df)
        mapped = int(result_df['pfam_id'].notna().sum())
        logger.info(
            f"strategy='basic': {mapped:,}/{total:,} coding-relevant variants "
            f"({100 * mapped / total:.2f}%) mapped to a Pfam domain via "
            f"gene symbol + protein change."
        )

        new_pymut = PyMutation(result_df, metadata=self.metadata, samples=self.samples)
        if new_pymut.metadata is not None:
            new_pymut.metadata.pfam_columns = {
                'strategy': 'basic',
                'hugo_column': hugo_alias,
                'aachange_column': aachange_alias,
                'aa_column': 'aa_pos',
            }
        return new_pymut

    def _annotate_pfam_extended(self, df, db_conn):
        """
        strategy='extended': layered PFAM annotation with an explicit,
        countable breakdown by category.

        Order of operations:
          1a  gene symbol + protein change   (same call as strategy='basic')
          1b  VEP-derived UniProt + position (requires a VEP-annotated MAF)

              Both are computed over the FULL dataset (not sequentially).
              Per variant: if only one of the two resolved it, that value is
              kept; if BOTH resolved it, the 1b (UniProt) value wins, because
              its coordinates are isoform-specific rather than a gene-level
              approximation - but the 1a value is kept alongside in
              pfam_id_gene/pfam_name_gene for auditing even when it "loses".

          2   explicit VEP domain fields   (only variants still unmapped)
          3   fallback text parsing        (only variants still unmapped)

        Steps 2 and 3 are a genuine per-variant cascade: each receives only
        the variants the previous steps left unresolved, and never overwrites
        an already-mapped variant. That guarantees the per-category counts
        logged at the end partition the mapped set exactly (they sum to the
        total number of mapped variants, with no double counting).
        """
        from ..core import PyMutation

        vep_uniprot = next(
            (column for column in ('VEP_SWISSPROT', 'VEP_TREMBL', 'VEP_UNIPROT')
             if column in df.columns),
            None,
        )
        vep_position = next(
            (column for column in ('VEP_Protein_position', 'Protein_position')
             if column in df.columns),
            None,
        )
        if vep_uniprot is None or vep_position is None:
            missing = [
                name for name, value in (
                    ('VEP_SWISSPROT', vep_uniprot),
                    ('VEP_Protein_position', vep_position),
                ) if value is None
            ]
            raise ValueError(
                "strategy='extended' requires VEP-annotated data; missing "
                + ', '.join(missing)
                + ". Run VEP annotation first, or use strategy='basic'."
            )

        result_df = df.copy()
        for column in ('pfam_id', 'pfam_name', 'pfam_description',
                       'seq_start', 'seq_end', 'pfam_id_gene',
                       'pfam_name_gene', 'pfam_id_uniprot',
                       'pfam_name_uniprot', 'pfam_mapping_source'):
            result_df[column] = None

        hugo_alias = self._resolve_hugo_column(df)
        aachange_alias = self._resolve_aachange_column(df)
        gene_result = None
        if hugo_alias is not None and aachange_alias is not None:
            gene_result = self._annotate_with_gene_position(
                df, db_conn, hugo_alias, aachange_alias, 'aa_pos'
            )

        uniprot_result = self._annotate_with_database(
            df, db_conn, vep_position, vep_uniprot
        )

        if gene_result is not None:
            gene_mask = gene_result['pfam_id'].notna()
            result_df.loc[gene_mask, 'pfam_id_gene'] = gene_result.loc[gene_mask, 'pfam_id']
            result_df.loc[gene_mask, 'pfam_name_gene'] = gene_result.loc[gene_mask, 'pfam_name']

        uniprot_mask = uniprot_result['pfam_id'].notna()
        result_df.loc[uniprot_mask, 'pfam_id_uniprot'] = uniprot_result.loc[uniprot_mask, 'pfam_id']
        result_df.loc[uniprot_mask, 'pfam_name_uniprot'] = uniprot_result.loc[uniprot_mask, 'pfam_name']

        gene_mask = result_df['pfam_id_gene'].notna()
        uniprot_mask = result_df['pfam_id_uniprot'].notna()
        both_mask = gene_mask & uniprot_mask
        gene_only = gene_mask & ~uniprot_mask
        uniprot_only = uniprot_mask & ~gene_mask

        result_df.loc[gene_only, 'pfam_id'] = result_df.loc[gene_only, 'pfam_id_gene']
        result_df.loc[gene_only, 'pfam_name'] = result_df.loc[gene_only, 'pfam_name_gene']
        result_df.loc[gene_only, 'pfam_mapping_source'] = 'gene_symbol'
        result_df.loc[uniprot_only, 'pfam_id'] = result_df.loc[uniprot_only, 'pfam_id_uniprot']
        result_df.loc[uniprot_only, 'pfam_name'] = result_df.loc[uniprot_only, 'pfam_name_uniprot']
        result_df.loc[uniprot_only, 'pfam_mapping_source'] = 'uniprot_db'
        result_df.loc[both_mask, 'pfam_id'] = result_df.loc[both_mask, 'pfam_id_uniprot']
        result_df.loc[both_mask, 'pfam_name'] = result_df.loc[both_mask, 'pfam_name_uniprot']
        result_df.loc[both_mask, 'pfam_mapping_source'] = 'both_conflict_resolved_to_uniprot'

        pending = result_df['pfam_mapping_source'].isna()
        if pending.any():
            vep_pending = result_df.loc[pending].copy()
            vep_pending['__combined_row_id__'] = vep_pending.index
            vep_result = self._annotate_with_vep_domains(vep_pending, db_conn)
            vep_mask = vep_result['pfam_id'].notna()
            vep_mapped = vep_result.loc[vep_mask].set_index('__combined_row_id__')
            result_df.loc[vep_mapped.index, 'pfam_id'] = vep_mapped['pfam_id']
            result_df.loc[vep_mapped.index, 'pfam_name'] = vep_mapped['pfam_name']
            result_df.loc[vep_mapped.index, 'pfam_mapping_source'] = 'vep_domains'

        pending = result_df['pfam_mapping_source'].isna()
        if pending.any():
            text_pending = result_df.loc[pending].copy()
            text_pending['__combined_row_id__'] = text_pending.index
            text_result = self._annotate_with_text_domains(text_pending, db_conn)
            text_mask = text_result['pfam_id'].notna()
            text_mapped = text_result.loc[text_mask].set_index('__combined_row_id__')
            result_df.loc[text_mapped.index, 'pfam_id'] = text_mapped['pfam_id']
            result_df.loc[text_mapped.index, 'pfam_name'] = text_mapped['pfam_name']
            result_df.loc[text_mapped.index, 'pfam_mapping_source'] = 'text_parse'

        result_df = result_df.drop(columns=['__combined_row_id__'], errors='ignore')

        # --- Per-category breakdown, logged so callers can see exactly how
        # coverage was achieved without having to inspect pfam_mapping_source
        # themselves. Categories partition the mapped set exactly (see the
        # cascade invariant in the docstring above), so they sum to
        # total_mapped with no double counting. ---
        category_labels = {
            'gene_symbol': '1a only (gene symbol + protein change)',
            'uniprot_db': '1b only (VEP-derived UniProt + position)',
            'both_conflict_resolved_to_uniprot': '1a + 1b agree/conflict -> resolved to UniProt (1b)',
            'vep_domains': '2 (explicit VEP domain fields)',
            'text_parse': '3 (fallback text parsing)',
        }
        total_variants = len(result_df)
        category_counts = result_df['pfam_mapping_source'].value_counts(dropna=True).to_dict()
        total_mapped = int(sum(category_counts.values()))

        logger.info(
            f"strategy='extended': {total_mapped:,}/{total_variants:,} variants "
            f"({100 * total_mapped / total_variants:.2f}%) mapped to a Pfam domain."
        )
        for key, label in category_labels.items():
            n = int(category_counts.get(key, 0))
            pct_of_mapped = 100 * n / total_mapped if total_mapped else 0.0
            pct_of_total = 100 * n / total_variants if total_variants else 0.0
            logger.info(
                f"   {label}: {n:,} variants "
                f"({pct_of_mapped:.2f}% of mapped, {pct_of_total:.2f}% of all variants)"
            )

        new_pymut = PyMutation(result_df, metadata=self.metadata, samples=self.samples)
        if new_pymut.metadata is not None:
            new_pymut.metadata.pfam_columns = {
                'strategy': 'extended',
                'dual_columns': {'gene': 'pfam_id_gene', 'uniprot': 'pfam_id_uniprot'},
                'conflict_policy': 'prefer_uniprot',
                'uniprot_column': vep_uniprot,
                'aa_column': vep_position,
                'category_counts': {k: int(v) for k, v in category_counts.items()},
            }
        return new_pymut

    # Default column-name candidates, tried in order, before falling back to
    # pattern-based detection. Only used by pfam_domains()'s own aa_column
    # resolution fallback now that annotate_pfam() itself no longer has a
    # UniProt/transcript-column-based strategy.
    _AA_POS_DEFAULT_CANDIDATES = [
        'aa_pos', 'aapos', 'AApos', 'AA_pos', 'UniProt_AApos',
        'Protein_position', 'AA_position',
    ]

    # Gene symbol column candidates. Hugo_Symbol is part of the MAF spec, so
    # this is virtually always present, unlike UniProt/transcript columns.
    _HUGO_DEFAULT_CANDIDATES = [
        'Hugo_Symbol', 'hugo_symbol', 'Gene_Symbol', 'gene_symbol', 'gene',
    ]

    # HGVS-style protein change column candidates, tried in the same order
    # maftools' pfamDomains() uses for its AACol default
    # (c("HGVSp_Short", "Protein_Change", "AAChange")). Like Hugo_Symbol,
    # one of these is present in essentially every real-world MAF, which is
    # what lets maftools annotate Pfam domains for any cohort without
    # needing a UniProt/transcript column at all.
    _AACHANGE_DEFAULT_CANDIDATES = [
        'HGVSp_Short', 'Protein_Change', 'AAChange', 'HGVSp',
    ]

    # Columns that look numeric but are NOT amino-acid positions (genomic
    # coordinates, chromosome numbers, etc.). Used to avoid false positives
    # during pattern-based detection of the aa position column.
    _AA_POS_EXCLUDE_KEYWORDS = [
        'start_position', 'end_position', 'chromosome', 'chrom',
        'genomic', 'pos_hg', 'hg19', 'hg38',
        # ID-like MAF columns that happen to be numeric and can fall in a
        # plausible aa-position range by coincidence (e.g. Entrez_Gene_Id,
        # NCBI_Build) but aren't amino-acid positions.
        'entrez', 'gene_id', 'ncbi_build', 'build',
    ]

    def _resolve_aa_column(self, df: pd.DataFrame, aa_column: Optional[str]) -> Tuple[Optional[str], str]:
        """
        Resolve which column holds amino-acid positions.

        Priority: 1) explicit argument, 2) known default names,
        3) pattern-based detection on the column values.

        Returns (column_name_or_None, source) where source is one of
        'explicit', 'default', 'pattern'.
        """
        if aa_column is not None:
            if aa_column not in df.columns:
                raise PfamAnnotationError(
                    f"Column '{aa_column}' specified in aa_column "
                    f"does not exist in the data."
                )
            return aa_column, 'explicit'

        for candidate in self._AA_POS_DEFAULT_CANDIDATES:
            if candidate in df.columns:
                return candidate, 'default'

        detected = self._detect_aa_position_column(df)
        if detected:
            return detected, 'pattern'

        return None, 'none'

    def _resolve_hugo_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the Hugo gene symbol column, by known default name only."""
        for candidate in self._HUGO_DEFAULT_CANDIDATES:
            if candidate in df.columns:
                return candidate
        return None

    def _resolve_aachange_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find an HGVS-style protein change column (e.g. 'p.R882H'), by known
        default name only, in the same priority order maftools' pfamDomains()
        uses for its AACol default.
        """
        for candidate in self._AACHANGE_DEFAULT_CANDIDATES:
            if candidate in df.columns:
                return candidate
        return None

    @staticmethod
    def _parse_hgvs_aa_position(value) -> Optional[int]:
        """
        Extract the amino acid position from an HGVS-style protein change
        string (e.g. 'p.R882H', 'ENST00000369535.4:p.Arg882His',
        'p.700_704del'), mirroring maftools' own parsing in pfamDomains():

        1. Keep only the text after the last '.' (drops any transcript/
           prefix, e.g. 'NM_022552.4:p.R882H' -> 'p' is dropped too, only
           'R882H' is kept... actually keeps whatever follows the LAST dot).
        2. Strip every letter (amino acid codes), keeping just digits/
           separators.
        3. Drop a trailing/leading '*' (stop codon) and anything after an
           internal '*'.
        4. For a range like '700_704', keep only the first number.

        Returns None if no numeric position could be extracted.
        """
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None

        text = str(value).strip()
        if not text:
            return None

        conv = text.split('.')[-1]
        pos_str = re.sub(r'[A-Za-z]', '', conv)
        pos_str = re.sub(r'\*.*', '', pos_str)
        pos_str = pos_str.strip()
        if not pos_str:
            return None

        if '_' in pos_str:
            pos_str = pos_str.split('_')[0]

        pos_str = pos_str.strip()
        # Keep a leading '-' rather than stripping it: maftools parses this
        # the same way (as.numeric on the cleaned string), so a malformed
        # change like 'p.-169fs' becomes a genuinely negative position.
        # That's not a bug to paper over here - it just means the position
        # can never fall inside a domain range (Start/End are always >= 1),
        # so it naturally ends up unmatched, exactly like in R.
        if not re.fullmatch(r'-?\d+', pos_str):
            return None

        return int(pos_str)

    def _load_gene_domain_table(self) -> Optional[pd.DataFrame]:
        """
        Load the maftools-equivalent gene -> Pfam domain table bundled with
        pyMut (data/pfam_domains.csv, the same 'protein_domains' resource
        maftools ships with the package). Cached on the class after first
        load since it's ~93k rows and doesn't change at runtime.

        Columns used: HGNC (gene symbol), Start/End (amino acid range),
        Label (short domain name), pfam (Pfam/CDD accession), Description.
        """
        cached = getattr(PfamAnnotationMixin, '_GENE_DOMAIN_TABLE_CACHE', None)
        if cached is not None:
            return cached

        import os
        module_dir = os.path.dirname(os.path.abspath(__file__))
        domains_file = os.path.join(module_dir, '..', 'data', 'pfam_domains.csv')

        if not os.path.exists(domains_file):
            logger.debug(f"Gene-level Pfam domain table not found: {domains_file}")
            return None

        table = pd.read_csv(domains_file, low_memory=False)
        table = table[['HGNC', 'Start', 'End', 'Label', 'pfam', 'Description']].dropna(
            subset=['HGNC', 'Start', 'End']
        )
        PfamAnnotationMixin._GENE_DOMAIN_TABLE_CACHE = table
        return table

    def _annotate_with_gene_position(self, df: pd.DataFrame, db_conn: duckdb.DuckDBPyConnection,
                                     hugo_alias: str, aachange_alias: str, aa_column: str) -> Optional[pd.DataFrame]:
        """
        maftools-equivalent PFAM annotation: match each variant's gene symbol
        (Hugo_Symbol) and amino acid position (parsed from an HGVS protein
        change column) against pyMut's bundled gene -> Pfam domain table
        (data/pfam_domains.csv), the same resource maftools' pfamDomains()
        uses internally.

        Unlike the UniProt-based path, this needs no UniProt/transcript
        column resolution at all: Hugo_Symbol and a Protein_Change-style
        column are present in essentially every MAF, so this is the strategy
        that "always works across cohorts", matching the R implementation.

        Returns None if the bundled domain table isn't available.
        """
        domain_table = self._load_gene_domain_table()
        if domain_table is None:
            return None

        df_work = df.copy()
        df_work[aa_column] = df_work[aachange_alias].apply(self._parse_hgvs_aa_position)

        n_with_pos = df_work[aa_column].notna().sum()
        logger.debug(
            f"Parsed amino acid position for {n_with_pos:,}/{len(df_work):,} variants "
            f"from '{aachange_alias}'"
        )

        df_work['__pfam_row_id__'] = range(len(df_work))

        db_conn.register('variants_temp', df_work)
        db_conn.register('gene_domains_temp', domain_table)

        query = f"""
        SELECT v.* EXCLUDE (__pfam_row_id__),
               d.pfam AS pfam_id, d."Label" AS pfam_name,
               d."Description" AS pfam_description,
               d."Start" AS seq_start, d."End" AS seq_end
        FROM variants_temp v
        LEFT JOIN gene_domains_temp d
            ON v.{hugo_alias} = d."HGNC"
            AND v.{aa_column} BETWEEN d."Start" AND d."End"
        QUALIFY row_number() OVER (
            PARTITION BY v.__pfam_row_id__
            ORDER BY d."Start" NULLS LAST, d."End" NULLS LAST
        ) = 1
        """
        try:
            result_df = db_conn.execute(query).df()
        finally:
            db_conn.unregister('variants_temp')
            db_conn.unregister('gene_domains_temp')

        n_mapped = result_df['pfam_id'].notna().sum()
        logger.info(
            f"Gene symbol + position (maftools-style): {n_mapped:,}/{len(result_df):,} "
            f"variants mapped to a Pfam domain"
        )

        return result_df

    def annotate_pfam(self,
                      db_conn: Optional[duckdb.DuckDBPyConnection] = None,
                      *, strategy: str = 'basic'):
        """
        Annotate PyMutation data with PFAM domains.

        Args:
            db_conn: DuckDB connection (if None, will create one)
            strategy: Annotation strategy.
                ``'basic'`` (default): maftools-equivalent only - gene symbol
                    + protein change against the bundled gene->Pfam table,
                    nothing else. Any variant this can't resolve stays
                    unmapped; there is no fallback of any kind, by design.
                    Logs how many variants it mapped. Needs no VEP
                    annotation: Hugo_Symbol and a Protein_Change/HGVSp_Short
                    -style column are part of the MAF spec, so this works on
                    essentially any cohort.
                ``'extended'``: gene-symbol (1a) and VEP-derived UniProt (1b)
                    mappings computed together over every variant, with 1b
                    winning wherever both resolve the same variant, followed
                    by a real per-variant cascade into explicit VEP domain
                    fields (2) and loose text parsing (3) for whatever is
                    still unmapped. Requires a VEP-annotated MAF for 1b/2/3 to
                    contribute anything - raises ValueError otherwise. Logs a
                    full per-category breakdown (how many variants each of
                    1a/1b/2/3 resolved).

        Returns:
            PyMutation: New PyMutation object with PFAM domain annotations
        """
        if db_conn is None:
            db_conn = connect_db()
            close_conn = True
        else:
            close_conn = False

        try:
            df = self.data.copy()

            if strategy not in {'basic', 'extended'}:
                raise ValueError("strategy must be 'basic' or 'extended'")
            if strategy == 'basic':
                return self._annotate_pfam_basic(df, db_conn)
            return self._annotate_pfam_extended(df, db_conn)

        finally:
            if close_conn:
                db_conn.close()

    def pfam_domains(self, *, aa_column: Optional[str] = None, summarize_by: str = 'PfamDomain',
                     top_n: int = 10, include_synonymous: bool = False) -> pd.DataFrame:
        """
        Summarize PFAM domain annotations similar to maftools pfamDomains function.

        Args:
            aa_column: Column name containing amino acid positions. If None,
                it is looked up in this order: 1) the column recorded by a
                previous annotate_pfam() call (stored in metadata), 2) common
                default names, 3) pattern-based detection on the data.
            summarize_by: 'PfamDomain' or 'AAPos' - how to group results
            top_n: Number of top results to return
            include_synonymous: Whether to include synonymous/silent variants
                (matched case-insensitively against 'Silent', 'Synonymous_Variant', etc.)

        Returns:
            DataFrame with, for summarize_by='PfamDomain':
                pfam_id            Pfam-A accession (e.g. 'PF00096')
                pfam_name          Real short domain name if the database was
                                    built with Pfam-A.clans.tsv.gz (e.g. 'zf-C2H2'),
                                    otherwise falls back to the accession.
                pfam_description   Human-readable description (e.g. 'Zinc finger,
                                    C2H2 type'), only present if the database has it.
                display_name       Convenience column: pfam_name, or pfam_id if no
                                    real name was found.
                n_genes            Number of unique genes with a variant in this domain
                n_variants         Number of variants annotated to this domain
                pct_of_mapped      % of all Pfam-mapped variants that fall in this domain
                pct_of_considered  % of all variants considered (after the
                                    synonymous/silent filter) that fall in this domain
            For summarize_by='AAPos', the same plus the uniprot/aa position columns,
            grouped at residue level instead of domain level.
        """
        logger.debug(f"Summarizing PFAM domains (summarize_by={summarize_by}, top_n={top_n})")

        # Use self.data instead of df parameter
        df = self.data

        # Resolve aa_column: explicit arg → column used by a previous
        # annotate_pfam() call (stored in metadata) → default names → pattern detection
        stored = getattr(self.metadata, 'pfam_columns', None) if self.metadata is not None else None

        if aa_column is not None:
            if aa_column not in df.columns:
                raise PfamAnnotationError(
                    f"Column '{aa_column}' specified in aa_column does not exist in the data."
                )
        else:
            if stored and stored.get('aa_column') in df.columns:
                aa_column = stored['aa_column']
                logger.info(f"Amino acid position column (from previous annotate_pfam): '{aa_column}'")
            else:
                aa_column, aa_source = self._resolve_aa_column(df, None)
                if aa_column is not None:
                    logger.info(f"Amino acid position column: '{aa_column}' ({aa_source})")
                    print(df[[aa_column]].head())

        if aa_column is None or aa_column not in df.columns:
            raise PfamAnnotationError(
                "Could not determine the amino acid position column (aa_column). "
                "Run annotate_pfam() first, or specify aa_column explicitly."
            )

        # Detect the correct pfam_id / pfam_name / pfam_description columns
        pfam_id_col = None
        pfam_name_col = None
        pfam_description_col = None

        if 'pfam_id' in df.columns:
            pfam_id_col = 'pfam_id'
            pfam_name_col = 'pfam_name'
            if 'pfam_description' in df.columns:
                pfam_description_col = 'pfam_description'
        else:
            raise PfamAnnotationError("No PFAM columns found. Run annotate_pfam() first.")

        logger.debug(f"Using PFAM columns: {pfam_id_col}, {pfam_name_col}, {pfam_description_col}")

        # Filter data if needed
        df_work = df.copy()

        if not include_synonymous:
            # Filter out synonymous/silent variants if a Variant_Classification-like
            # column exists. Compared case-insensitively (and against a small set of
            # common synonyms: 'Silent', 'SILENT', 'Synonymous_Variant', ...) since
            # MAF-style files are inconsistent about casing/naming for this class.
            variant_class_candidates = ['Variant_Classification', 'variant_classification', 'Mutation_Type']
            variant_class_col = None
            for candidate in variant_class_candidates:
                if candidate in df_work.columns:
                    variant_class_col = candidate
                    break

            if variant_class_col is not None:
                synonymous_labels = {'silent', 'synonymous_variant', 'synonymous', 'synonymous snv'}
                is_synonymous = df_work[variant_class_col].astype(str).str.strip().str.lower().isin(synonymous_labels)
                n_removed = int(is_synonymous.sum())
                df_work = df_work[~is_synonymous]
                logger.debug(f"Excluded {n_removed:,} synonymous/silent variants (include_synonymous=False)")

        total_considered = len(df_work)

        # Filter for variants with PFAM annotations. Use pfam_name_col (not
        # pfam_id_col) as the "matched" indicator: pfam_name/pfam_id are
        # always populated together in the UniProt-database path, but the
        # gene-symbol (maftools-style) path can have a real domain match
        # (pfam_name set) with pfam_id left NaN - the bundled gene-domain
        # table itself has some domain entries with no clean accession, only
        # a name/description. Filtering on pfam_id_col here would silently
        # drop those genuine matches.
        df_pfam = df_work.dropna(subset=[pfam_name_col])
        total_mapped = len(df_pfam)

        if total_considered > 0:
            logger.info(
                f"{total_mapped:,}/{total_considered:,} variants "
                f"({total_mapped / total_considered:.1%}) mapped to a Pfam domain"
            )

        if len(df_pfam) == 0:
            logger.warning("No variants with PFAM domain annotations found")
            empty_cols = ['pfam_id', 'pfam_name']
            if pfam_description_col:
                empty_cols.append('pfam_description')
            if summarize_by == 'PfamDomain':
                return pd.DataFrame(columns=empty_cols + ['n_genes', 'n_variants', 'pct_of_mapped', 'pct_of_considered'])
            elif summarize_by == 'AAPos':
                uniprot_candidates = ['uniprot', 'UniProt', 'UNIPROT', 'uniprot_id']
                uniprot_alias = next((c for c in uniprot_candidates if c in df.columns), 'uniprot')
                return pd.DataFrame(columns=[uniprot_alias, aa_column] + empty_cols +
                                             ['n_variants', 'n_genes', 'pct_of_mapped', 'pct_of_considered'])
            else:
                return pd.DataFrame()

        logger.debug(f"Found {len(df_pfam)} variants with PFAM domain annotations")

        if summarize_by == 'PfamDomain':
            # Group by PFAM domain
            hugo_candidates = ['Hugo_Symbol', 'hugo_symbol', 'Gene_Symbol', 'gene_symbol', 'gene']
            hugo_alias = None
            for candidate in hugo_candidates:
                if candidate in df_pfam.columns:
                    hugo_alias = candidate
                    break

            if hugo_alias is None:
                raise PfamAnnotationError("Hugo_Symbol column not found (no alias found)")

            # Group by domain NAME (pfam_name / 'Label'), not by (pfam_id,
            # pfam_name) together. This matches maftools' pfamDomains(),
            # which groups purely `by = DomainLabel`: the bundled gene-domain
            # table (data/pfam_domains.csv, used by the gene-symbol strategy)
            # isn't a clean 1:1 accession<->name mapping - the same domain
            # name can show up with a real Pfam/CDD accession in one row and
            # a garbled description string standing in for the accession in
            # another (a quirk of the underlying maftools resource itself).
            # Grouping by pfam_id too would silently split what maftools
            # counts as a single domain into several rows. pfam_id/
            # pfam_description are still shown, taken from whichever row of
            # that domain happens to come first (purely cosmetic - the counts
            # below don't depend on it).
            agg_spec = {pfam_id_col: 'first'}
            if pfam_description_col:
                agg_spec[pfam_description_col] = 'first'
            agg_spec[hugo_alias] = 'nunique'  # Number of unique genes
            agg_spec[aa_column] = 'count'  # Number of variants

            summary = df_pfam.groupby(pfam_name_col, dropna=False).agg(agg_spec).reset_index()

            result_cols = ['pfam_name', 'pfam_id'] + (['pfam_description'] if pfam_description_col else [])
            summary.columns = result_cols + ['n_genes', 'n_variants']
            summary = summary[['pfam_id', 'pfam_name'] + (['pfam_description'] if pfam_description_col else [])
                               + ['n_genes', 'n_variants']]

            # Under the gene-symbol (maftools-style) strategy, the pfam_id/
            # pfam_description picked above via 'first' reflect whichever
            # matched *variant* happened to come first in this dataset -
            # that's not what maftools shows. maftools always displays the
            # pfam/Description of the first row (by HGNC, Start, End - the
            # bundled table's own sort key) carrying that Label *anywhere in
            # the whole domain table*, regardless of which gene/variant in
            # this dataset triggered the match (`gff[!duplicated(gff$Label)]`
            # after `setkey(gff, HGNC, Start, End)`). Overlay that here so a
            # domain like 'PTPc' shows the same representative accession/
            # description as R, instead of an arbitrary alternate one.
            if stored and stored.get('strategy') == 'gene_symbol':
                domain_table = self._load_gene_domain_table()
                if domain_table is not None:
                    canonical = (
                        domain_table.sort_values(['HGNC', 'Start', 'End'])
                                    .drop_duplicates(subset='Label', keep='first')
                                    [['Label', 'pfam', 'Description']]
                                    .rename(columns={
                                        'Label': 'pfam_name',
                                        'pfam': '_canonical_pfam_id',
                                        'Description': '_canonical_pfam_description',
                                    })
                    )
                    canonical_labels = set(canonical['pfam_name'])
                    summary = summary.merge(canonical, on='pfam_name', how='left')
                    # Use the canonical value whenever this Label exists in
                    # the bundled table at all - even when its canonical
                    # pfam/Description is itself null, since that's exactly
                    # what R would show too. Only keep the old 'first'-over-
                    # matched-variants value for a Label the table doesn't
                    # have (shouldn't normally happen here, but keeps this
                    # safe as a fallback).
                    has_canonical = summary['pfam_name'].isin(canonical_labels)
                    summary['pfam_id'] = summary['_canonical_pfam_id'].where(
                        has_canonical, summary['pfam_id']
                    )
                    if pfam_description_col:
                        summary['pfam_description'] = summary['_canonical_pfam_description'].where(
                            has_canonical, summary['pfam_description']
                        )
                    summary = summary.drop(columns=['_canonical_pfam_id', '_canonical_pfam_description'])

            # Tie-break alphabetically by domain name for a deterministic,
            # reproducible row order (R's own tie order falls out of
            # data.table internals tied to gene first-appearance order in the
            # MAF, which isn't worth reproducing exactly - the domain set and
            # every count already match exactly regardless of row order).
            summary = summary.sort_values(
                ['n_variants', 'pfam_name'], ascending=[False, True], kind='mergesort'
            )

            # Percentages make the counts interpretable at a glance: how much of
            # this specific domain's contribution is relative to all annotated
            # variants (pct_of_mapped) and to all variants considered (pct_of_considered).
            summary['pct_of_mapped'] = (summary['n_variants'] / total_mapped * 100).round(2)
            summary['pct_of_considered'] = (summary['n_variants'] / total_considered * 100).round(2)

            # display_name: real domain name when available, otherwise falls back
            # to the accession (e.g. if the database predates the Pfam-A.clans.tsv.gz
            # enrichment, or this particular family is missing from that file).
            summary['display_name'] = summary.apply(
                lambda r: r['pfam_name'] if r['pfam_name'] != r['pfam_id'] else r['pfam_id'],
                axis=1
            )

            # Per-protein breakdown: a domain can be shared by several genes
            # (e.g. a common structural domain repeated across a gene family),
            # so knowing "n_genes" isn't enough - we also want to know *which*
            # gene(s) and in what proportion. This computes, for every domain,
            # how many (and what %) of its variants come from each gene.
            gene_breakdown = (
                df_pfam.groupby([pfam_name_col, hugo_alias], dropna=False)
                       .size()
                       .reset_index(name='n_variants')
                       .rename(columns={hugo_alias: 'Hugo_Symbol'})
            )
            # Compute each domain's total directly from gene_breakdown itself
            # (rather than from summary['n_variants'], which is a `.count()`
            # over aa_column and can differ - even be 0 - if aa_column has
            # NaNs for some rows of that domain, causing a division by zero).
            # This is also vectorized instead of a per-row .apply().
            domain_totals_vec = gene_breakdown.groupby(pfam_name_col)['n_variants'].transform('sum')
            gene_breakdown['pct_of_domain'] = (
                gene_breakdown['n_variants'] / domain_totals_vec * 100
            ).round(2)
            gene_breakdown = gene_breakdown.sort_values(
                [pfam_name_col, 'n_variants'], ascending=[True, False]
            ).reset_index(drop=True)

            # Carry the domain's representative pfam_id along, so callers
            # that look a domain up by its accession (e.g.
            # pfam_domain_protein_breakdown('PF00096')) can still find it
            # here even though the grouping key is now pfam_name (see note
            # above on why pfam_id can't be the group key).
            gene_breakdown = gene_breakdown.merge(
                summary[['pfam_name', 'pfam_id']], on='pfam_name', how='left'
            )

            def _breakdown_string(pfam_name_value, max_genes=3):
                subset = gene_breakdown[gene_breakdown[pfam_name_col] == pfam_name_value]
                parts = [f"{r.Hugo_Symbol} {r.pct_of_domain:.0f}%" for r in subset.head(max_genes).itertuples()]
                remaining = len(subset) - max_genes
                if remaining > 0:
                    parts.append(f"+{remaining} more")
                return ", ".join(parts)

            # Compact, human-readable summary of which protein(s) this domain
            # belongs to, e.g. "TP53 65%, KRAS 20%, +2 more".
            summary['protein_breakdown'] = summary['pfam_name'].apply(_breakdown_string)

        elif summarize_by == 'AAPos':
            # Group by amino acid position
            uniprot_candidates = ['uniprot', 'UniProt', 'UNIPROT', 'uniprot_id']
            uniprot_alias = None
            for candidate in uniprot_candidates:
                if candidate in df_pfam.columns:
                    uniprot_alias = candidate
                    break

            hugo_candidates = ['Hugo_Symbol', 'hugo_symbol', 'Gene_Symbol', 'gene_symbol', 'gene']
            hugo_alias = None
            for candidate in hugo_candidates:
                if candidate in df_pfam.columns:
                    hugo_alias = candidate
                    break

            if uniprot_alias is not None and hugo_alias is not None:
                group_cols = [uniprot_alias, aa_column, pfam_id_col, pfam_name_col] + (
                    [pfam_description_col] if pfam_description_col else []
                )
                summary = df_pfam.groupby(group_cols, dropna=False).size().reset_index(name='n_variants')
                gene_counts = df_pfam.groupby(group_cols, dropna=False)[hugo_alias].nunique().reset_index(name='n_genes')
                summary = summary.merge(gene_counts, on=group_cols)
                summary = summary.sort_values('n_variants', ascending=False)

                summary['pct_of_mapped'] = (summary['n_variants'] / total_mapped * 100).round(2)
                summary['pct_of_considered'] = (summary['n_variants'] / total_considered * 100).round(2)
                summary['display_name'] = summary.apply(
                    lambda r: r[pfam_name_col] if r[pfam_name_col] != r[pfam_id_col] else r[pfam_id_col],
                    axis=1
                )
            else:
                logger.warning("'UNIPROT' or 'Hugo_Symbol' column not found, cannot group by amino acid position")
                uniprot_candidates = ['uniprot', 'UniProt', 'UNIPROT', 'uniprot_id']
                uniprot_alias = next((c for c in uniprot_candidates if c in df_pfam.columns or c in df.columns), 'uniprot')
                empty_cols = [uniprot_alias, aa_column, pfam_id_col, pfam_name_col]
                if pfam_description_col:
                    empty_cols.append(pfam_description_col)
                return pd.DataFrame(columns=empty_cols + ['n_variants', 'n_genes', 'pct_of_mapped', 'pct_of_considered'])

        else:
            raise ValueError(f"Invalid summarize_by value: {summarize_by}. Must be 'PfamDomain' or 'AAPos'")

        # Return top N results
        result = summary.head(top_n).reset_index(drop=True)
        # Attach overall coverage counts as attrs so plot_pfam_domains() (and any
        # other caller) can report "X/Y variants mapped" without recomputing it.
        result.attrs['total_considered'] = total_considered
        result.attrs['total_mapped'] = total_mapped
        if summarize_by == 'PfamDomain':
            result.attrs['gene_breakdown'] = gene_breakdown

        return result

    def pfam_domain_protein_breakdown(self, pfam_id: str, *, aa_column: Optional[str] = None,
                                      include_synonymous: bool = False) -> pd.DataFrame:
        """
        Zoom into a single Pfam domain and break it down by protein: how many
        variants (and what % of that domain's total) come from each gene that
        carries the domain.

        Useful once pfam_domains() has flagged a domain as heavily mutated and
        you want to know whether that comes from a single protein or is spread
        across several genes that happen to share the same domain (e.g. a
        common structural/repeat domain found across a gene family).

        Args:
            pfam_id: Domain identifier to zoom into - either its pfam_id
                (e.g. 'PF00096') or its pfam_name/display_name (e.g.
                'zf-C2H2'), exactly as shown in pfam_domains()'s output.
                Matched against whichever of the two the value looks like,
                since with the gene-symbol (maftools-style) strategy
                pfam_id isn't always a clean, unique accession.
            aa_column, include_synonymous: same meaning as in pfam_domains().

        Returns:
            DataFrame with columns: pfam_id, Hugo_Symbol, n_variants,
            pct_of_domain - sorted by n_variants descending. Empty (with the
            right columns) if the pfam_id has no annotated variants.
        """
        full_summary = self.pfam_domains(
            aa_column=aa_column, summarize_by='PfamDomain',
            top_n=10 ** 9,  # effectively "all domains" so pfam_id is never cut off
            include_synonymous=include_synonymous,
        )

        gene_breakdown = full_summary.attrs.get('gene_breakdown')
        empty = pd.DataFrame(columns=['pfam_id', 'Hugo_Symbol', 'n_variants', 'pct_of_domain'])

        if gene_breakdown is None or len(gene_breakdown) == 0:
            return empty

        # Match against pfam_id first, falling back to pfam_name: pfam_id can
        # be NaN or non-unique for some domains in the gene-symbol strategy
        # (see the grouping note in pfam_domains()), so pfam_name is the
        # reliable key, but callers commonly pass a clean accession too.
        is_match = pd.Series(False, index=gene_breakdown.index)
        if 'pfam_id' in gene_breakdown.columns:
            is_match |= gene_breakdown['pfam_id'] == pfam_id
        if 'pfam_name' in gene_breakdown.columns:
            is_match |= gene_breakdown['pfam_name'] == pfam_id
        result = gene_breakdown[is_match].copy()

        if len(result) == 0:
            logger.warning(
                f"No annotated variants found for Pfam domain '{pfam_id}'. "
                "Check pfam_domains() output for valid Pfam IDs."
            )
            return empty

        return result.sort_values('n_variants', ascending=False).reset_index(drop=True)

    def plot_pfam_domains(self, *, aa_column: Optional[str] = None, summarize_by: str = 'PfamDomain',
                          top_n: int = 15, include_synonymous: bool = False,
                          color: Optional[str] = None, ax=None, figsize: Optional[Tuple[float, float]] = None,
                          save_path: Optional[str] = None, base_fontsize: float = 10.0,
                          title_fontsize: float = 14.0, subtitle_fontsize: Optional[float] = None):
        """
        Plot the top Pfam domains (or amino-acid positions) hit by variants,
        as a horizontal bar chart, sized automatically for whatever dataset
        you throw at it.

        The figure size adapts to two things so it stays readable both for a
        small gene panel and for a whole-exome dataset:
          - height grows with the number of bars (top_n / rows actually returned)
          - width grows with the length of the domain name labels

        Args:
            aa_column, summarize_by, top_n, include_synonymous: same as pfam_domains()
            color: bar color (default: a single muted blue)
            ax: existing matplotlib Axes to draw into (creates a new figure if None)
            save_path: if given, also saves the figure to this path (e.g. 'top_domains.png')

        Returns:
            (fig, ax) matplotlib objects, so you can further customize or save the plot.
        """
        # Delegate plotting to visualizations.pfam_plots to keep plotting
        # logic separate from data/annotation logic.
        try:
            from ..visualizations import pfam_plots
        except Exception:
            raise PfamAnnotationError("Unable to import pfam_plots visualization helpers")

        summary = self.pfam_domains(
            aa_column=aa_column, summarize_by=summarize_by, top_n=top_n,
            include_synonymous=include_synonymous,
        )

        if len(summary) == 0:
            raise PfamAnnotationError(
                "No PFAM-annotated variants to plot. Run annotate_pfam() first, or check that top_n / filters aren't excluding everything."
            )

        return pfam_plots.plot_pfam_domains(
            summary, color=color, figsize=figsize, ax=ax, save_path=save_path,
            base_fontsize=base_fontsize, title_fontsize=title_fontsize,
            subtitle_fontsize=subtitle_fontsize,
        )

    def plot_pfam_domain_breakdown(self, pfam_id: str, *, aa_column: Optional[str] = None,
                                   include_synonymous: bool = False, top_n: int = 10,
                                   kind: str = 'bar', ax=None, figsize: Optional[Tuple[float, float]] = None,
                                   save_path: Optional[str] = None, base_fontsize: float = 10.0,
                                   title_fontsize: float = 14.0, donut_width: float = 0.42):
        """
        Plot how a single Pfam domain's variants are distributed across the
        different proteins/genes that carry it - i.e. in what proportion this
        domain is mutated in each protein.

        Pairs naturally with plot_pfam_domains(): once that chart shows a
        domain is frequently hit, call this one with that domain's pfam_id to
        see whether the mutations concentrate in one protein or spread across
        several genes that share the domain.

        Args:
            pfam_id: Pfam accession to zoom into, e.g. 'PF00096'.
            aa_column, include_synonymous: same meaning as in pfam_domains().
            top_n: genes beyond this are collapsed into a single 'Other' slice/bar
                so the chart stays readable.
            kind: 'bar' (default) for a horizontal bar chart, or 'donut' for a
                donut chart. Both show variant count and % of the domain per gene.
            ax: existing matplotlib Axes to draw into (creates a new figure if None).
            save_path: if given, also saves the figure to this path.

        Returns:
            (fig, ax) matplotlib objects.
        """
        # Delegate plotting to visualizations.pfam_plots for rendering
        try:
            from ..visualizations import pfam_plots
        except Exception:
            raise PfamAnnotationError("Unable to import pfam_plots visualization helpers")

        breakdown = self.pfam_domain_protein_breakdown(
            pfam_id, aa_column=aa_column, include_synonymous=include_synonymous,
        )

        if len(breakdown) == 0:
            raise PfamAnnotationError(
                f"No PFAM-annotated variants found for domain '{pfam_id}'. Check pfam_domains() output for valid Pfam IDs."
            )

        # Collapse long tails into a single 'Other' slice/bar so the chart
        # doesn't turn into an unreadable wall of tiny genes.
        if len(breakdown) > top_n:
            head = breakdown.iloc[:top_n - 1].copy()
            tail = breakdown.iloc[top_n - 1:]
            other_row = pd.DataFrame([{
                'pfam_id': pfam_id,
                'Hugo_Symbol': f'Other ({len(tail)} genes)',
                'n_variants': tail['n_variants'].sum(),
                'pct_of_domain': round(tail['pct_of_domain'].sum(), 2),
            }])
            breakdown = pd.concat([head, other_row], ignore_index=True)

        return pfam_plots.plot_pfam_domain_breakdown(
            breakdown, pfam_id=pfam_id, kind=kind, figsize=figsize, ax=ax, save_path=save_path,
            base_fontsize=base_fontsize, title_fontsize=title_fontsize, donut_width=donut_width,
        )


# Legacy function imports for backward compatibility (deprecated)
def annotate_pfam(self, *args, **kwargs):
    """Deprecated: Use PyMutation.annotate_pfam() method instead."""
    import warnings
    warnings.warn("Direct function import is deprecated. Use PyMutation.annotate_pfam() method instead.", 
                  DeprecationWarning, stacklevel=2)
    return self.annotate_pfam(*args, **kwargs)


def pfam_domains(self, *args, **kwargs):
    """Deprecated: Use PyMutation.pfam_domains() method instead."""
    import warnings
    warnings.warn("Direct function import is deprecated. Use PyMutation.pfam_domains() method instead.", 
                  DeprecationWarning, stacklevel=2)
    return self.pfam_domains(*args, **kwargs)