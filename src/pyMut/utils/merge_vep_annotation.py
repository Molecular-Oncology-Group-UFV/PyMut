import gzip
import logging
from pathlib import Path
from typing import Optional, Dict

import duckdb
import pandas as pd

from .format import format_chr

logger = logging.getLogger(__name__)


def _normalize_maf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not hasattr(df, "columns"):
        return df

    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    aliases = {
        "hugo_symbol": "Hugo_Symbol",
        "entrez_gene_id": "Entrez_Gene_Id",
        "chromosome": "Chromosome",
        "start_position": "Start_Position",
        "end_position": "End_Position",
        "strand": "Strand",
        "variant_classification": "Variant_Classification",
        "variant_type": "Variant_Type",
        "reference_allele": "Reference_Allele",
        "tumor_seq_allele1": "Tumor_Seq_Allele1",
        "tumor_seq_allele2": "Tumor_Seq_Allele2",
        "tumor_sample_barcode": "Tumor_Sample_Barcode",
        "matched_norm_sample_barcode": "Matched_Norm_Sample_Barcode",
        "dbsnp_rs": "dbSNP_RS",
        "dbsnp_val_status": "dbSNP_Val_Status",
    }

    rename_map = {}
    for lower_name, canonical in aliases.items():
        if lower_name in lower_to_original and canonical not in df.columns:
            rename_map[lower_to_original[lower_name]] = canonical

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _region_key_from_location(location_value: object) -> str | None:
    if pd.isna(location_value):
        return None

    value = str(location_value).strip()
    if not value or value == "nan":
        return None

    if value.lower().startswith("chr"):
        value = value[3:]

    if ":" not in value:
        return value

    chrom, pos = value.split(":", 1)
    if "-" in pos:
        start, end = pos.split("-", 1)
    else:
        start = end = pos

    return f"{chrom}:{start}-{end}"


def _parse_vep_extra_column(extra_str: str) -> Dict[str, str]:
    """
    Parse the VEP Extra column which contains semicolon-separated key-value pairs.

    Parameters
    ----------
    extra_str : str
        The Extra column content from VEP output

    Returns
    -------
    Dict[str, str]
        Dictionary with parsed key-value pairs
    """
    if pd.isna(extra_str) or extra_str == "-" or extra_str == "":
        return {}

    result = {}
    pairs = extra_str.split(';')

    for pair in pairs:
        if '=' in pair:
            key, value = pair.split('=', 1)
            result[key] = value

    return result


def _get_field(row: pd.Series, name: str):
    """Fetch a field from a MAF row regardless of the casing used for its column name."""
    if name in row.index:
        return row[name]
    lower_to_original = {str(c).strip().lower(): c for c in row.index}
    original = lower_to_original.get(name.lower())
    if original is None:
        raise KeyError(name)
    return row[original]


def _create_region_key_from_maf(row: pd.Series) -> str:
    """
    Create a region key from MAF row that matches the VEP Uploaded_variation format.
    
    Generates the same key that VEP uses by:
    - Using format_chr for proper chromosome formatting (23→X, 24→Y)
    - Using the actual End_Position for indels
    - Using proper reference/alternative allele logic

    Parameters
    ----------
    row : pd.Series
        MAF row with Chromosome, Start_Position, End_Position, Reference_Allele, 
        Tumor_Seq_Allele1, and Tumor_Seq_Allele2 (any casing of these column names)

    Returns
    -------
    str
        Region key in format chr:start-end:len(ref)/alt
    """
    chrom = format_chr(str(_get_field(row, 'Chromosome')))  # "chr1…chrX, chrY"
    start = int(_get_field(row, 'Start_Position'))
    end = int(_get_field(row, 'End_Position'))  # Use the real range
    ref = str(_get_field(row, 'Reference_Allele'))
    alt = str(_get_field(row, 'Tumor_Seq_Allele2') or _get_field(row, 'Tumor_Seq_Allele1') or "-")

    return f"{chrom}:{start}-{end}:{len(ref)}/{alt}"


def merge_maf_with_vep_annotations(
        maf_file: str | Path,
        vep_file: str | Path,
        output_file: Optional[str | Path] = None,
        compress: bool = False
) -> tuple[pd.DataFrame, Path]:
    """
    Merge MAF file with VEP annotations using pandas and DuckDB for optimization.

    Parameters
    ----------
    maf_file : str | Path
        Path to the original MAF file (.maf or .maf.gz)
    vep_file : str | Path
        Path to the VEP annotation file (.txt)
    output_file : str | Path, optional
        Output file path. If None, creates filename with "_annotated" suffix
    compress : bool, optional
        Whether to compress the output file with gzip (default: False)

    Returns
    -------
    tuple[pd.DataFrame, Path]
        A tuple containing:
        - Merged DataFrame with MAF data and VEP annotations
        - Path to the output file that was created
    """
    maf_file = Path(maf_file)
    vep_file = Path(vep_file)

    if output_file is None:
        # Create output filename with _annotated suffix
        if maf_file.suffix == '.gz':
            stem = maf_file.stem.replace('.maf', '')
            base_name = f"{stem}_VEP_annotated.maf"
        else:
            stem = maf_file.stem
            base_name = f"{stem}_VEP_annotated{maf_file.suffix}"

        # Add .gz extension if compression is requested
        if compress:
            output_file = maf_file.parent / f"{base_name}.gz"
        else:
            output_file = maf_file.parent / base_name
    else:
        output_file = Path(output_file)
        # If compression is requested but output file doesn't end with .gz, add it
        if compress and not str(output_file).endswith('.gz'):
            output_file = output_file.with_suffix(output_file.suffix + '.gz')

    logger.info(f"Reading MAF file: {maf_file}")

    if maf_file.suffix == '.gz':
        with gzip.open(maf_file, 'rt') as f:
            maf_df = pd.read_csv(f, sep='\t', comment='#', low_memory=False)
    else:
        maf_df = pd.read_csv(maf_file, sep='\t', comment='#', low_memory=False)

    maf_df = _normalize_maf_columns(maf_df)
    logger.info(f"MAF file loaded: {maf_df.shape[0]} rows, {maf_df.shape[1]} columns")

    logger.info(f"Reading VEP file: {vep_file}")

    # Find the header line (starts with #Uploaded_variation)
    header_line = None
    with open(vep_file, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('#Uploaded_variation'):
                header_line = i
                break

    if header_line is None:
        raise ValueError("Could not find VEP header line starting with #Uploaded_variation")

    # Read the file starting from the header line
    vep_df = pd.read_csv(vep_file, sep='\t', skiprows=header_line, low_memory=False)
    if vep_df.columns[0].startswith('#'):
        vep_df.columns = [vep_df.columns[0][1:]] + list(vep_df.columns[1:])

    logger.info(f"VEP file loaded: {vep_df.shape[0]} rows, {vep_df.shape[1]} columns")

    logger.info("Creating region keys for MAF data...")
    maf_df['region_key'] = maf_df.apply(_create_region_key_from_maf, axis=1)

    # Use VEP Uploaded_variation as the key (after removing # prefix)
    vep_df['region_key'] = vep_df['Uploaded_variation']

    logger.info("Parsing VEP Extra column...")
    vep_extra_parsed = vep_df['Extra'].apply(_parse_vep_extra_column)
    extra_df = pd.json_normalize(vep_extra_parsed)

    # Combine VEP data with parsed extra columns
    vep_with_extra = pd.concat([vep_df, extra_df], axis=1)

    # Remove rows without meaningful annotations (only IMPACT=MODIFIER)
    meaningful_annotations = vep_with_extra[
        ~((vep_with_extra['Extra'] == 'IMPACT=MODIFIER') |
          (vep_with_extra['Extra'].isna()) |
          (vep_with_extra['Gene'] == '-'))
    ].copy()

    logger.info(f"Filtered to {meaningful_annotations.shape[0]} meaningful annotations")

    logger.info("Removing VEP duplicates...")
    original_vep_count = len(meaningful_annotations)
    meaningful_annotations = meaningful_annotations.drop_duplicates("region_key", keep="first")
    logger.info(f"Removed {original_vep_count - len(meaningful_annotations)} duplicate VEP entries")

    logger.info("Performing optimized merge with DuckDB...")

    conn = duckdb.connect()
    # Register DataFrames with DuckDB
    conn.register('maf_data', maf_df)
    conn.register('vep_data', meaningful_annotations)

    # Dynamic SQL query
    vep_columns_mapping = {
        'Gene': 'VEP_Gene',
        'Feature': 'VEP_Feature',
        'Feature_type': 'VEP_Feature_type',
        'Consequence': 'VEP_Consequence',
        'cDNA_position': 'VEP_cDNA_position',
        'CDS_position': 'VEP_CDS_position',
        'Protein_position': 'VEP_Protein_position',
        'Amino_acids': 'VEP_Amino_acids',
        'Codons': 'VEP_Codons',
        'Existing_variation': 'VEP_Existing_variation',
        'SYMBOL': 'VEP_SYMBOL',
        'SYMBOL_SOURCE': 'VEP_SYMBOL_SOURCE',
        'HGNC_ID': 'VEP_HGNC_ID',
        'ENSP': 'VEP_ENSP',
        'SWISSPROT': 'VEP_SWISSPROT',
        'TREMBL': 'VEP_TREMBL',
        'UNIPARC': 'VEP_UNIPARC',
        'UNIPROT_ISOFORM': 'VEP_UNIPROT_ISOFORM',
        'DOMAINS': 'VEP_DOMAINS',
        'IMPACT': 'VEP_IMPACT',
        'STRAND': 'VEP_STRAND',
        'DISTANCE': 'VEP_DISTANCE'
    }

    # Select columns that exist in the VEP data
    available_vep_columns = meaningful_annotations.columns.tolist()
    vep_select_clauses = []

    for vep_col, alias in vep_columns_mapping.items():
        if vep_col in available_vep_columns:
            vep_select_clauses.append(f"v.{vep_col} as {alias}")

    vep_select_str = ",\n        ".join(vep_select_clauses)

    # Perform the merge using SQL
    merge_query = f"""
    SELECT 
        m.*,
        {vep_select_str}
    FROM maf_data m
    LEFT JOIN vep_data v ON m.region_key = v.region_key
    """

    result_df = conn.execute(merge_query).df()

    # Remove the temporary region_key column
    result_df = result_df.drop('region_key', axis=1)

    # Remove duplicate information - if VEP_SYMBOL is the same as Hugo_Symbol, don't duplicate
    if 'Hugo_Symbol' in result_df.columns and 'VEP_SYMBOL' in result_df.columns:
        mask = result_df['Hugo_Symbol'] == result_df['VEP_SYMBOL']
        result_df.loc[mask, 'VEP_SYMBOL'] = None

    # Replace missing VEP annotations with empty strings instead of "-" or NaN
    vep_columns = [col for col in result_df.columns if col.startswith('VEP_')]
    for col in vep_columns:
        result_df[col] = result_df[col].fillna("")
        result_df[col] = result_df[col].replace("-", "")

    logger.info(f"Merge completed: {result_df.shape[0]} rows, {result_df.shape[1]} columns")

    logger.info(f"Saving annotated file to: {output_file}")

    # Save file with or without compression
    if compress or str(output_file).endswith('.gz'):
        with gzip.open(output_file, 'wt') as f:
            result_df.to_csv(f, sep='\t', index=False)
    else:
        result_df.to_csv(output_file, sep='\t', index=False)

    conn.close()

    return result_df, output_file


# ---------------------------------------------------------------------------
# Alternative, lighter-weight merge based on the VEP "Location" column
# ---------------------------------------------------------------------------
# This path does not rely on DuckDB nor on reconstructing VEP's
# Uploaded_variation format from the MAF; instead it normalizes both sides to
# a simple "chrom:start-end" key derived from VEP's own Location column.

def _build_maf_region_key(df):
    df = _normalize_maf_columns(df)
    required = ["Chromosome", "Start_Position", "End_Position"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required MAF columns for region merge: {missing}")

    df = df.copy()
    df["_maf_region_key"] = (
        df["Chromosome"].astype(str).str.replace("chr", "", regex=False).str.strip()
        + ":"
        + df["Start_Position"].astype(str).str.strip()
        + "-"
        + df["End_Position"].astype(str).str.strip()
    )
    return df


def merge_vep_annotation(maf_file, vep_file, output_file=None, **kwargs):
    """
    Lightweight alternative to ``merge_maf_with_vep_annotations`` that joins
    on a ``chrom:start-end`` key derived directly from VEP's ``Location``
    column, instead of reconstructing VEP's ``Uploaded_variation`` format.

    Parameters
    ----------
    maf_file : str | Path
        Path to the MAF file (tab-separated, ``#`` comment lines allowed).
    vep_file : str | Path
        Path to the VEP annotation file (tab-separated, ``#`` comment lines
        allowed), expected to contain a ``Location`` column.
    output_file : str | Path, optional
        If provided, the merged result is written there as TSV.
    **kwargs
        Reserved for future options; currently unused.

    Returns
    -------
    pd.DataFrame
        The merged MAF + VEP annotation table.
    """
    maf_df = pd.read_csv(maf_file, sep="\t", comment="#", low_memory=False)
    maf_df = _normalize_maf_columns(maf_df)
    maf_df = _build_maf_region_key(maf_df)

    vep_df = pd.read_csv(vep_file, sep="\t", comment="#", low_memory=False)
    location_col = next((c for c in vep_df.columns if str(c).lower() == "location"), None)
    if location_col is None:
        raise KeyError("VEP annotation file is missing a 'Location' column required for merge.")

    vep_df = vep_df.copy()
    vep_df["_vep_region_key"] = vep_df[location_col].map(_region_key_from_location)

    merged = maf_df.merge(
        vep_df[[c for c in vep_df.columns if c != "_vep_region_key"] + ["_vep_region_key"]],
        left_on="_maf_region_key",
        right_on="_vep_region_key",
        how="left",
        suffixes=("_maf", "_vep"),
    )

    if output_file is not None:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, sep="\t", index=False)

    return merged
