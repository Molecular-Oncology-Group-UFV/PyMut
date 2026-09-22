import gzip
import hashlib
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

import shutil
import urllib.request


class PfamAnnotationError(Exception):
    """Custom exception for Pfam annotation errors."""
    pass


def get_resources_path() -> Path:
    """Get the path to the resources directory."""
    return Path(__file__).parent.parent / "data" / "resources"


def get_db_path() -> Path:
    """Get the path to the DuckDB database file."""
    return get_resources_path() / "data.duckdb"
    
def ensure_resource_directories() -> tuple[Path, Path, Path]:
    resources_path = get_resources_path()
    pfam_dir = resources_path / "pfam"
    mappings_dir = resources_path / "mappings"

    resources_path.mkdir(parents=True, exist_ok=True)
    pfam_dir.mkdir(parents=True, exist_ok=True)
    mappings_dir.mkdir(parents=True, exist_ok=True)

    return resources_path, pfam_dir, mappings_dir


def _download_progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Report hook for urlretrieve: prints download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        print(f"\r  {pct:5.1f}% ({downloaded / 1e6:.0f} MB / {total_size / 1e6:.0f} MB)", end="", flush=True)
    else:
        print(f"\r  {downloaded / 1e6:.0f} MB downloaded", end="", flush=True)


def download_file(url: str, destination: Path, label: str) -> None:
    print(f"{label} not found. Downloading from:")
    print(f"{url}")

    tmp_destination = destination.with_suffix(destination.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp_destination, reporthook=_download_progress_hook)
    print()  # newline after the progress bar
    shutil.move(tmp_destination, destination)

    print(f"Downloaded {label} to: {destination}")


def is_valid_gzip(filepath: Path) -> bool:
    """Check that a .gz file can be opened and read without integrity errors."""
    try:
        with gzip.open(filepath, 'rb') as f:
            while f.read(1024 * 1024):
                pass
        return True
    except (gzip.BadGzipFile, OSError, EOFError):
        return False


def _download_missing_resources(resources: list) -> None:
    """
    Shared download loop: given a list of (file_path, url, label) tuples,
    drop any corrupted partial downloads, then download whatever is still
    missing. Used by both ensure_resource_files() (full/extended pipeline)
    and ensure_pfam_only_files() (lite pipeline).
    """
    for file_path, _, _ in resources:
        if file_path.exists() and not is_valid_gzip(file_path):
            print(f"⚠️  {file_path} looks corrupted. It will be re-downloaded.")
            file_path.unlink()

    missing = [r for r in resources if not r[0].exists()]
    if missing:
        print(f"Setting up the PFAM database: {len(missing)} file(s) need to be downloaded.\n")

    download_count = 0
    for file_path, url, label in resources:
        if file_path.exists():
            print(f"✅ {label} already present: {file_path}")
        else:
            download_count += 1
            print(f"[{download_count}/{len(missing)}] {label}")
            download_file(url, file_path, label)


def ensure_resource_files() -> tuple[Path, Path, Path]:
    """
    Ensure all three files needed by the EXTENDED annotation pipeline are
    present locally, downloading whatever is missing: Pfam domain
    coordinates, Pfam family names/descriptions, and the (large) UniProt
    cross-reference mapping file.
    """
    _, pfam_dir, mappings_dir = ensure_resource_directories()

    pfam_file = pfam_dir / "Pfam-A.regions.tsv.gz"
    mapping_file = mappings_dir / "HUMAN_9606_idmapping_selected.tab.gz"
    clans_file = pfam_dir / "Pfam-A.clans.tsv.gz"

    pfam_url = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.regions.tsv.gz"
    mapping_url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping_selected.tab.gz"
    # Official Pfam family names/descriptions file. Pfam-A.regions.tsv.gz only
    # has coordinates (pfamA_acc, seq_start, seq_end) but no human-readable
    # name, so we need this companion file to populate pfam_name/pfam_description.
    clans_url = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.clans.tsv.gz"

    resources = [
        (pfam_file, pfam_url, "Pfam domain annotations file (Pfam-A.regions.tsv.gz)"),
        (mapping_file, mapping_url, "UniProt human mapping file (HUMAN_9606_idmapping_selected.tab.gz)"),
        (clans_file, clans_url, "Pfam family names & descriptions file (Pfam-A.clans.tsv.gz)"),
    ]
    _download_missing_resources(resources)

    return pfam_file, mapping_file, clans_file


def ensure_pfam_only_files() -> tuple[Path, Path]:
    """
    Ensure only the two files needed for domain coordinates/names are
    present locally: Pfam-A.regions.tsv.gz and Pfam-A.clans.tsv.gz.

    Used by the LITE pipeline, which deliberately skips the (much larger)
    UniProt cross-reference mapping file downloaded by ensure_resource_files()
    - lite annotation doesn't resolve arbitrary protein/transcript IDs, so
    that file isn't needed at all.
    """
    _, pfam_dir, _ = ensure_resource_directories()

    pfam_file = pfam_dir / "Pfam-A.regions.tsv.gz"
    clans_file = pfam_dir / "Pfam-A.clans.tsv.gz"

    pfam_url = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.regions.tsv.gz"
    clans_url = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.clans.tsv.gz"

    resources = [
        (pfam_file, pfam_url, "Pfam domain annotations file (Pfam-A.regions.tsv.gz)"),
        (clans_file, clans_url, "Pfam family names & descriptions file (Pfam-A.clans.tsv.gz)"),
    ]
    _download_missing_resources(resources)

    return pfam_file, clans_file


def ensure_gene_mapping_file() -> Path:
    """
    Ensure the HGNC complete gene set file is present locally, downloading
    it if missing. This is the small file the LITE pipeline uses to build a
    Gene Symbol -> canonical UniProt accession lookup (the same "one gene,
    one canonical protein" simplification maftools' pfamDomains() makes),
    instead of the large per-variant UniProt cross-reference database used
    by the extended pipeline.
    """
    _, _, mappings_dir = ensure_resource_directories()
    gene_mapping_file = mappings_dir / "hgnc_complete_set.txt"
    gene_mapping_url = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"

    resources = [
        (gene_mapping_file, gene_mapping_url, "HGNC gene symbol -> UniProt mapping file (hgnc_complete_set.txt)"),
    ]

    # This file isn't gzipped, so skip the gzip integrity check used for the
    # other resources and just check whether it exists / is empty.
    if gene_mapping_file.exists() and gene_mapping_file.stat().st_size < 1000:
        print(f"⚠️  {gene_mapping_file} looks corrupted or empty. It will be re-downloaded.")
        gene_mapping_file.unlink()

    if gene_mapping_file.exists():
        print(f"✅ {resources[0][2]} already present: {gene_mapping_file}")
    else:
        print("Setting up the lite PFAM database: 1 additional file needs to be downloaded.\n")
        download_file(gene_mapping_url, gene_mapping_file, resources[0][2])

    return gene_mapping_file


def load_pfam_clans_mapping(clans_file: Path) -> pd.DataFrame:
    """
    Parse the official Pfam-A.clans.tsv(.gz) file, which provides the
    human-readable short name and description for each Pfam-A family
    (complementary to Pfam-A.regions.tsv.gz, which only has coordinates).

    The file has no header. Columns, in order:
        1. pfam_id     e.g. 'PF00096'
        2. clan_acc    e.g. 'CL0361' (empty if the family belongs to no clan)
        3. clan_id     short clan name (empty if no clan)
        4. short_name  e.g. 'zf-C2H2'
        5. description e.g. 'Zinc finger, C2H2 type'

    Args:
        clans_file: Path to Pfam-A.clans.tsv.gz

    Returns:
        DataFrame with columns: pfam_id, clan_acc, clan_id, short_name, description
    """
    print(f"Parsing Pfam family names/descriptions file: {clans_file}")
    clans_df = pd.read_csv(
        clans_file,
        sep='\t',
        header=None,
        names=['pfam_id', 'clan_acc', 'clan_id', 'short_name', 'description'],
        dtype=str,
        keep_default_na=False,
    )
    print(f"Loaded {len(clans_df):,} Pfam-A family descriptions")
    return clans_df


def parse_hgnc_gene_to_uniprot(gene_mapping_file: Path) -> pd.DataFrame:
    """
    Parse the HGNC complete gene set file into a Hugo_Symbol -> canonical
    UniProt accession table for the LITE annotation pipeline.

    HGNC assigns at most one reviewed UniProt accession per approved human
    gene symbol via its 'uniprot_ids' column. This is exactly the "one
    gene, one canonical protein" simplification maftools' pfamDomains()
    relies on - see annotate_pfam_lite() in pfam_annotation.py for the
    matching trade-off this implies (it assumes a single canonical isoform
    per gene rather than the exact isoform used elsewhere in your data).

    Args:
        gene_mapping_file: Path to hgnc_complete_set.txt

    Returns:
        DataFrame with columns: hugo_symbol, uniprot
    """
    print(f"Parsing gene symbol -> UniProt mapping file: {gene_mapping_file}")
    hgnc_df = pd.read_csv(gene_mapping_file, sep='\t', dtype=str, low_memory=False)

    if 'symbol' not in hgnc_df.columns or 'uniprot_ids' not in hgnc_df.columns:
        raise PfamAnnotationError(
            "Unexpected format in hgnc_complete_set.txt: missing the 'symbol' "
            "or 'uniprot_ids' column. The HGNC file format may have changed; "
            "manual review of the downloaded file is recommended."
        )

    hgnc_df = hgnc_df.dropna(subset=['symbol', 'uniprot_ids'])
    hgnc_df = hgnc_df[hgnc_df['uniprot_ids'].str.strip() != '']

    # A gene occasionally has more than one UniProt ID listed (multiple
    # reviewed isoforms/entries); HGNC lists the primary one first, so we
    # keep only that one to get a single canonical protein per gene.
    hgnc_df = hgnc_df.assign(uniprot=hgnc_df['uniprot_ids'].str.split('|').str[0].str.strip())
    result = hgnc_df[['symbol', 'uniprot']].rename(columns={'symbol': 'hugo_symbol'})
    result = result.drop_duplicates(subset=['hugo_symbol']).reset_index(drop=True)

    print(f"Loaded {len(result):,} gene symbol -> canonical UniProt mappings")
    return result


def calculate_file_hash(filepath: str) -> str:
    """Calculate SHA-256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def parse_idmapping_selected(mapping_file: Path, chunk_size: int = int(1e6)) -> pd.DataFrame:
    """
    Parse UniProt ID mapping file with column-based extraction.

    This function reads the UniProt ID mapping file and extracts mappings
    for Ensembl proteins (ENSP), Ensembl transcripts (ENST), RefSeq proteins
    (NP_) and RefSeq mRNAs/transcripts (NM_) to UniProt IDs, as well as short
    names from UniProt. These are used by the EXTENDED annotation pipeline
    (annotate_pfam_extended() in pfam_annotation.py) to resolve a variant's
    protein/transcript identifier to a UniProt accession.

    Expected format (idmapping_selected.tab, 0-indexed columns):
    - Column 0:  UniProt accession
    - Column 1:  Short name (e.g., 1433B_HUMAN)
    - Column 3:  RefSeq IDs (mixes NM_ mRNA and NP_ protein, semicolon-separated)
    - Column 19: Ensembl_TRS - Ensembl transcript IDs (ENST..., semicolon-separated)
    - Column 20: Ensembl_PRO - Ensembl protein IDs (ENSP..., semicolon-separated)

    Args:
        mapping_file: Path to the ID mapping file
        chunk_size: Size of chunks to process at a time

    Returns:
        DataFrame with columns: prot_id, uniprot, short_name
    """
    print(f"Parsing UniProt ID mapping file: {mapping_file}")

    all_mappings = []
    total_lines = 0
    kept_lines = 0

    def strip_version(identifier: str) -> str:
        return identifier.split('.')[0]

    try:
        with gzip.open(mapping_file, 'rt') as f:
            chunk_data = []

            for line_num, line in enumerate(f, 1):
                total_lines += 1

                parts = line.strip().split('\t')
                if len(parts) >= 21:  # Need at least 21 columns
                    uniprot_id = parts[0]
                    short_name = parts[1] if len(parts) > 1 and parts[1] != "-" else ""

                    # RefSeq IDs from column 4 (index 3): mixes NM_ (mRNA/transcript)
                    # and NP_ (protein) accessions together
                    refseq_ids = parts[3] if len(parts) > 3 else ""
                    if refseq_ids and refseq_ids != "-":
                        for refseq_id in refseq_ids.split(';'):
                            refseq_id = strip_version(refseq_id.strip())
                            if refseq_id.startswith('NP_'):
                                chunk_data.append({
                                    'prot_id': refseq_id,
                                    'uniprot': uniprot_id,
                                    'short_name': short_name
                                })
                                kept_lines += 1
                            elif refseq_id.startswith('NM_'):
                                # Transcript-level mapping, used only as an
                                # approximate fallback (see docstring above)
                                chunk_data.append({
                                    'prot_id': refseq_id,
                                    'uniprot': uniprot_id,
                                    'short_name': short_name
                                })
                                kept_lines += 1

                    # Ensembl transcript IDs from column 20 (index 19)
                    ensembl_trs_ids = parts[19] if len(parts) > 19 else ""
                    if ensembl_trs_ids and ensembl_trs_ids != "-":
                        for enst_id in ensembl_trs_ids.split(';'):
                            enst_id = strip_version(enst_id.strip())
                            if enst_id.startswith('ENST'):
                                # Transcript-level mapping, used only as an
                                # approximate fallback (see docstring above)
                                chunk_data.append({
                                    'prot_id': enst_id,
                                    'uniprot': uniprot_id,
                                    'short_name': short_name
                                })
                                kept_lines += 1

                    # Ensembl protein IDs from column 21 (index 20)
                    ensembl_ids = parts[20] if len(parts) > 20 else ""
                    if ensembl_ids and ensembl_ids != "-":
                        for ensembl_id in ensembl_ids.split(';'):
                            ensembl_id = strip_version(ensembl_id.strip())
                            if ensembl_id.startswith('ENSP'):
                                chunk_data.append({
                                    'prot_id': ensembl_id,
                                    'uniprot': uniprot_id,
                                    'short_name': short_name
                                })
                                kept_lines += 1

                    # Add row for short_name association if short_name exists
                    if short_name:
                        chunk_data.append({
                            'prot_id': short_name,
                            'uniprot': uniprot_id,
                            'short_name': short_name
                        })
                        kept_lines += 1

                if len(chunk_data) >= chunk_size:
                    all_mappings.extend(chunk_data)
                    chunk_data = []

                    if line_num % (chunk_size * 10) == 0:
                        print(f"  Processed {line_num:,} lines, kept {kept_lines:,} mappings...")

            if chunk_data:
                all_mappings.extend(chunk_data)

        print(f"Processed {total_lines:,} total lines")
        print(f"Kept {kept_lines:,} protein/transcript mappings")

        if all_mappings:
            df = pd.DataFrame(all_mappings)
            df = df.drop_duplicates()
            print(f"Final unique mappings: {len(df):,}")
            return df
        else:
            print("⚠️  No relevant mappings found")
            return pd.DataFrame(columns=['prot_id', 'uniprot', 'short_name'])

    except Exception as e:
        print(f"❌ Error parsing mapping file: {e}")
        return pd.DataFrame(columns=['prot_id', 'uniprot', 'short_name'])


def check_mapping_coverage(conn: duckdb.DuckDBPyConnection) -> bool:
    """
    Test if the xref table has reasonable coverage for common protein/transcript ID types.

    Args:
        conn: DuckDB connection

    Returns:
        True if coverage looks reasonable, False otherwise
    """
    try:
        ensp_count = conn.execute("SELECT COUNT(*) FROM xref WHERE prot_id LIKE 'ENSP%'").fetchone()[0]
        np_count = conn.execute("SELECT COUNT(*) FROM xref WHERE prot_id LIKE 'NP\\_%' ESCAPE '\\'").fetchone()[0]
        enst_count = conn.execute("SELECT COUNT(*) FROM xref WHERE prot_id LIKE 'ENST%'").fetchone()[0]
        nm_count = conn.execute("SELECT COUNT(*) FROM xref WHERE prot_id LIKE 'NM\\_%' ESCAPE '\\'").fetchone()[0]
        total_count = conn.execute("SELECT COUNT(*) FROM xref").fetchone()[0]

        print("Mapping coverage:")
        print(f"    Total mappings: {total_count:,}")
        print(f"    ENSP mappings (protein): {ensp_count:,}")
        print(f"    NP_ mappings (protein): {np_count:,}")
        print(f"    ENST mappings (transcript, approximate fallback): {enst_count:,}")
        print(f"    NM_ mappings (transcript, approximate fallback): {nm_count:,}")

        min_threshold = 1000  # Minimum expected mappings
        ensp_ok = ensp_count >= min_threshold
        np_ok = np_count >= min_threshold

        if ensp_ok and np_ok:
            print("✅ Mapping coverage looks good")
            return True
        else:
            print("⚠️  Low mapping coverage detected:")
            if not ensp_ok:
                print(f"    ENSP count {ensp_count:,} < {min_threshold:,}")
            if not np_ok:
                print(f"    NP_ count {np_count:,} < {min_threshold:,}")
            return False

    except Exception as e:
        print(f"❌ Error during mapping coverage test: {e}")
        return False


def _load_pfam_table(conn: duckdb.DuckDBPyConnection, pfam_file: Path, clans_file: Path) -> None:
    """
    Create and populate the `pfam` table (domain coordinates per UniProt
    accession) from Pfam-A.regions.tsv.gz + Pfam-A.clans.tsv.gz, and index
    it. Shared by both build_embedded_db() (extended pipeline) and
    build_lite_db() (lite pipeline), since domain coordinates are exactly
    the same data either way - only how a variant gets matched to a
    UniProt accession differs between the two pipelines.
    """
    # Load the official Pfam family names/descriptions (Pfam-A.clans.tsv.gz)
    # so we can populate real pfam_name / pfam_description values instead of
    # using the accession as a placeholder.
    clans_df = load_pfam_clans_mapping(clans_file)
    pfam_name_map = clans_df.set_index('pfam_id')['short_name'].to_dict()
    pfam_description_map = clans_df.set_index('pfam_id')['description'].to_dict()

    print("Loading Pfam data...")

    chunk_size = int(1e6)
    total_rows = 0
    names_resolved = 0
    names_missing = 0

    conn.execute("DROP TABLE IF EXISTS pfam")
    conn.execute("""
                 CREATE TABLE pfam
                 (
                     uniprot          VARCHAR,
                     seq_start        INTEGER,
                     seq_end          INTEGER,
                     pfam_id          VARCHAR,
                     pfam_name        VARCHAR,
                     pfam_description VARCHAR
                 )
                 """)

    with gzip.open(pfam_file, 'rt') as f:
        header = f.readline().strip().split('\t')
        print(f"Pfam file columns: {header}")

        f.seek(0)

        chunk_count = 0
        for chunk in pd.read_csv(f, sep='\t', chunksize=chunk_size):
            chunk_count += 1

            # Based on file structure: ['pfamseq_acc', 'seq_version', 'crc64', 'md5', 'pfamA_acc', 'seq_start', 'seq_end', 'ali_start', 'ali_end']
            column_mapping = {
                'pfamseq_acc': 'uniprot',
                'pfamA_acc': 'pfam_id',
                'seq_start': 'seq_start',
                'seq_end': 'seq_end'
            }

            chunk = chunk.rename(columns=column_mapping)

            # Look up the real short name / description for each pfam_id from
            # Pfam-A.clans.tsv.gz. Fall back to the accession itself only for
            # families that are somehow missing from the clans file (should be
            # rare/none in practice, since it's the official companion file).
            resolved_names = chunk['pfam_id'].map(pfam_name_map)
            names_resolved += resolved_names.notna().sum()
            names_missing += resolved_names.isna().sum()
            chunk['pfam_name'] = resolved_names.fillna(chunk['pfam_id'])
            chunk['pfam_description'] = chunk['pfam_id'].map(pfam_description_map).fillna('')

            required_cols = ['uniprot', 'seq_start', 'seq_end', 'pfam_id', 'pfam_name', 'pfam_description']
            available_cols = [col for col in required_cols if col in chunk.columns]

            if len(available_cols) == len(required_cols):
                chunk_selected = chunk[available_cols].copy()

                conn.register(f'pfam_chunk_{chunk_count}', chunk_selected)
                conn.execute(f"INSERT INTO pfam SELECT * FROM pfam_chunk_{chunk_count}")
                conn.unregister(f'pfam_chunk_{chunk_count}')

                total_rows += len(chunk_selected)

                del chunk_selected
                del chunk
            else:
                print(f"Warning: Missing columns. Available: {available_cols}, Required: {required_cols}")

            if chunk_count % 10 == 0:
                print(f"  Processed {chunk_count} chunks...")

    print(f"Loaded {total_rows:,} Pfam domain annotations")
    print(f"  Family names resolved from Pfam-A.clans.tsv.gz: {names_resolved:,}")
    if names_missing:
        print(f"  ⚠️  Family names NOT found in clans file (used accession as fallback): {names_missing:,}")

    conn.execute("CREATE INDEX IF NOT EXISTS ix_pfam ON pfam(uniprot, seq_start, seq_end)")


def _load_gene_canonical_table(conn: duckdb.DuckDBPyConnection, gene_mapping_file: Path) -> None:
    """
    Create and populate the `gene_canonical` table (Hugo Symbol -> canonical
    UniProt accession) used by the LITE pipeline, and index it.
    """
    gene_df = parse_hgnc_gene_to_uniprot(gene_mapping_file)

    conn.execute("DROP TABLE IF EXISTS gene_canonical")
    conn.execute("""
                 CREATE TABLE gene_canonical
                 (
                     hugo_symbol VARCHAR,
                     uniprot     VARCHAR
                 )
                 """)
    conn.register('gene_canonical_temp', gene_df)
    conn.execute("INSERT INTO gene_canonical SELECT * FROM gene_canonical_temp")
    conn.unregister('gene_canonical_temp')
    conn.execute("CREATE INDEX IF NOT EXISTS ix_gene_canonical_symbol ON gene_canonical(hugo_symbol)")


def build_embedded_db(force_rebuild: bool = False) -> str:
    """
    Build embedded DuckDB database with Pfam and mapping data (EXTENDED
    pipeline: exact per-variant UniProt resolution via a full cross-reference
    database). For the lite, maftools-style pipeline, see build_lite_db().

    Supported files:
    - idmapping_selected.tab.gz (global) or HUMAN_9606_idmapping_selected.tab.gz (organism-specific)
    - Detection by content prefix (ENSP/NP_), not fixed column positions
    - Pfam-A.regions.tsv.gz for domain annotations

    Args:
        force_rebuild: If True, rebuild the database even if it exists

    Returns:
        Path to the created database file
    """
    db_path = get_db_path()

    # First, check whether the database already exists and is valid.
    # Only touch/download the source files (Pfam-A.regions.tsv.gz and
    # HUMAN_9606_idmapping_selected.tab.gz), which can be several GB, if a
    # (re)build is actually needed.
    if db_path.exists() and not force_rebuild:
        try:
            conn = duckdb.connect(str(db_path))
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]
            if 'pfam' in table_names and 'xref' in table_names and 'meta' in table_names:
                pfam_cols = [row[1] for row in conn.execute("PRAGMA table_info('pfam')").fetchall()]
                if 'pfam_description' in pfam_cols:
                    print(f"✅ Database already exists at {db_path}")
                    conn.close()
                    return str(db_path)
                else:
                    print("⚠️  Existing database uses an older schema (missing 'pfam_description'). Rebuilding...")
            conn.close()
        except Exception:
            pass

    resources_path, _, _ = ensure_resource_directories()
    pfam_file, mapping_file, clans_file = ensure_resource_files()

    print("🔨 Building embedded DuckDB database...")

    conn = duckdb.connect(str(db_path))

    # 1. Build the `pfam` table (domain coordinates), shared with the lite pipeline
    _load_pfam_table(conn, pfam_file, clans_file)
    if not mapping_file.exists():
        print(f"Mapping file not found: {mapping_file}")
        print("Creating empty xref table.")
        print("📝  Note: Without UniProt mappings, Pfam annotation will be limited to variants")
        print("    that already have UniProt protein IDs in the input data.")
        conn.execute("DROP TABLE IF EXISTS xref")
        conn.execute("""
                     CREATE TABLE xref
                     (
                         prot_id    VARCHAR,
                         uniprot    VARCHAR,
                         short_name VARCHAR
                     )
                     """)
    else:
        file_size = mapping_file.stat().st_size
        if file_size < 1000:  # Less than 1KB suggests empty or corrupted file
            print(f"⚠️  Mapping file appears to be empty or corrupted (size: {file_size} bytes)")
            print(f"⚠️  Expected file location: {mapping_file}")
            print("⚠️  Creating empty xref table.")
            print("📝  Note: To enable full Pfam annotation functionality, please download")
            print("    the UniProt ID mapping file from:")
            print(
                "    https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/idmapping.dat.gz")
            print("    and place it at the expected location.")
            conn.execute("DROP TABLE IF EXISTS xref")
            conn.execute("""
                         CREATE TABLE xref
                         (
                             prot_id    VARCHAR,
                             uniprot    VARCHAR,
                             short_name VARCHAR
                         )
                         """)
        else:
            try:
                xref_df = parse_idmapping_selected(mapping_file, chunk_size)

                if len(xref_df) > 0:
                    conn.execute("DROP TABLE IF EXISTS xref")
                    conn.execute("""
                                 CREATE TABLE xref
                                 (
                                     prot_id    VARCHAR,
                                     uniprot    VARCHAR,
                                     short_name VARCHAR
                                 )
                                 """)

                    conn.register('xref_temp', xref_df)
                    conn.execute("INSERT INTO xref SELECT * FROM xref_temp")
                    conn.unregister('xref_temp')

                    check_mapping_coverage(conn)
                else:
                    print("⚠️  No mapping data found after filtering")
                    print("⚠️  Creating empty xref table. Pfam annotation may be limited.")
                    conn.execute("DROP TABLE IF EXISTS xref")
                    conn.execute("""
                                 CREATE TABLE xref
                                 (
                                     prot_id    VARCHAR,
                                     uniprot    VARCHAR,
                                     short_name VARCHAR
                                 )
                                 """)
            except Exception as e:
                print(f"⚠️  Error reading mapping file: {e}")
                print("⚠️  Creating empty xref table. Pfam annotation may be limited.")
                conn.execute("DROP TABLE IF EXISTS xref")
                conn.execute("""
                             CREATE TABLE xref
                             (
                                 prot_id    VARCHAR,
                                 uniprot    VARCHAR,
                                 short_name VARCHAR
                             )
                             """)

    # 3. Create remaining indices (ix_pfam was already created in _load_pfam_table)
    print("🔍 Creating database indices...")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_xref_prot_id ON xref(prot_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_xref_uniprot ON xref(uniprot)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_xref_short_name ON xref(short_name)")

    # 4. Create metadata table
    print("📝 Creating metadata table...")
    conn.execute("DROP TABLE IF EXISTS meta")
    conn.execute("""
                 CREATE TABLE meta
                 (
                     resource     VARCHAR,
                     file_path    VARCHAR,
                     release_date VARCHAR,
                     sha256_hash  VARCHAR,
                     created_at   TIMESTAMP
                 )
                 """)

    metadata_entries = [
        {
            'resource': 'pfam',
            'file_path': str(pfam_file),
            'release_date': 'unknown',
            'sha256_hash': calculate_file_hash(str(pfam_file)),
            'created_at': datetime.now()
        },
        {
            'resource': 'pfam_clans',
            'file_path': str(clans_file),
            'release_date': 'unknown',
            'sha256_hash': calculate_file_hash(str(clans_file)),
            'created_at': datetime.now()
        },
        {
            'resource': 'uniprot_mapping',
            'file_path': str(mapping_file),
            'release_date': 'unknown',
            'sha256_hash': calculate_file_hash(str(mapping_file)),
            'created_at': datetime.now()
        }
    ]

    meta_df = pd.DataFrame(metadata_entries)
    conn.register('meta_temp', meta_df)
    conn.execute("INSERT INTO meta SELECT * FROM meta_temp")
    conn.unregister('meta_temp')

    conn.close()

    print(f"✅ Database created successfully at {db_path}")
    return str(db_path)


def connect_db() -> duckdb.DuckDBPyConnection:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        print("Database not found. Building it now...")
        build_embedded_db()
    else:
        print(f"Using existing database: {db_path}")

    return duckdb.connect(str(db_path))


def build_lite_db(force_rebuild: bool = False) -> str:
    """
    Build (or reuse) the LITE, maftools-style annotation database: Pfam
    domain coordinates (`pfam` table, shared with the extended pipeline) +
    a small Gene Symbol -> canonical UniProt accession lookup
    (`gene_canonical` table, built from the HGNC gene set file).

    Unlike build_embedded_db(), this never downloads or parses the large
    UniProt cross-reference mapping file (HUMAN_9606_idmapping_selected.tab.gz):
    lite annotation doesn't resolve arbitrary protein/transcript IDs, it only
    needs one canonical UniProt accession per gene symbol, which comes from
    the much smaller HGNC gene set file instead.

    If build_embedded_db() has already been run in this environment, the
    `pfam` table is reused as-is (domain coordinates are identical either
    way) and only the small `gene_canonical` table is added on top - no
    re-download of the large Pfam file.

    Args:
        force_rebuild: If True, rebuild both tables even if they exist.

    Returns:
        Path to the database file (the same file used by build_embedded_db()).
    """
    db_path = get_db_path()

    # First, check what's already there. Only download/build what's missing.
    has_pfam, has_gene_canonical = False, False
    if db_path.exists() and not force_rebuild:
        try:
            conn = duckdb.connect(str(db_path))
            tables = {t[0] for t in conn.execute("SHOW TABLES").fetchall()}
            has_pfam = 'pfam' in tables
            has_gene_canonical = 'gene_canonical' in tables
            conn.close()
        except Exception:
            pass

    if has_pfam and has_gene_canonical:
        print(f"✅ Lite database already exists at {db_path}")
        return str(db_path)

    ensure_resource_directories()
    conn = duckdb.connect(str(db_path))

    if has_pfam:
        print("✅ 'pfam' table already present (built by the extended pipeline or a previous "
              "lite build) - reusing it, no download needed.")
    else:
        print("🔨 Building 'pfam' table (domain coordinates)...")
        pfam_file, clans_file = ensure_pfam_only_files()
        _load_pfam_table(conn, pfam_file, clans_file)

    print("🔨 Building 'gene_canonical' table (gene symbol -> canonical UniProt)...")
    gene_mapping_file = ensure_gene_mapping_file()
    _load_gene_canonical_table(conn, gene_mapping_file)

    conn.close()
    print(f"✅ Lite database ready at {db_path}")
    return str(db_path)


def connect_lite_db() -> duckdb.DuckDBPyConnection:
    """
    Same check-then-build pattern as connect_db(), but for the LITE
    (maftools-style) database: check whether the required tables already
    exist, build/download only what's missing otherwise.
    """
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    needs_build = True
    if db_path.exists():
        try:
            conn = duckdb.connect(str(db_path))
            tables = {t[0] for t in conn.execute("SHOW TABLES").fetchall()}
            needs_build = not ({'pfam', 'gene_canonical'} <= tables)
            conn.close()
        except Exception:
            needs_build = True

    if needs_build:
        print("Lite database not found (or incomplete). Building it now...")
        build_lite_db()
    else:
        print(f"Using existing lite database: {db_path}")

    return duckdb.connect(str(db_path))
