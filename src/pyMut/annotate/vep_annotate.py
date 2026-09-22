import csv
import gzip
import logging
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, Tuple

from ..utils.format import format_chr
from ..utils.merge_vep_annotation import merge_maf_with_vep_annotations

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VEP cache auto-resolution / auto-download configuration
# ---------------------------------------------------------------------------
# These defaults are only used when the caller does not provide an explicit
# ``cache_dir`` and no matching cache can be found automatically. They define
# which Ensembl VEP cache version/assembly will be offered for download.
DEFAULT_VEP_CACHE_VERSION = "114"
DEFAULT_VEP_CACHE_ASSEMBLY = "GRCh38"

# Ensembl publishes indexed VEP caches at a predictable URL pattern.
VEP_CACHE_DOWNLOAD_URL_TEMPLATE = (
    "https://ftp.ensembl.org/pub/release-{version}/variation/indexed_vep_cache/"
    "homo_sapiens_vep_{version}_{assembly}.tar.gz"
)

# Rough size (in GB) of a full indexed VEP cache. Used only for warnings.
ESTIMATED_VEP_CACHE_SIZE_GB = 25

# Pattern used to recognize VEP cache directories following the convention
# "homo_sapiens_vep_{version}_{assembly}" (e.g. "homo_sapiens_vep_114_GRCh38").
_CACHE_DIR_NAME_PATTERN = re.compile(r'^homo_sapiens_vep_(\d+)_([A-Za-z0-9\.]+)$')

# Pattern used to recognize a downloaded-but-not-yet-extracted VEP cache
# archive (e.g. "homo_sapiens_vep_114_GRCh38.tar.gz").
_CACHE_ARCHIVE_NAME_PATTERN = re.compile(r'^homo_sapiens_vep_(\d+)_([A-Za-z0-9\.]+)\.tar\.gz$')


def _get_default_cache_root() -> Path:
    """
    Return the default directory where pyMut expects to find the VEP cache.

    This file lives inside the installed package at
    ``<package_root>/<subpackage>/vep_annotate.py``, so walking up two levels
    from ``__file__`` yields the package root (e.g.
    ``.../site-packages/pyMut``). The expected default cache location is
    therefore ``<package_root>/data/resources/vep``.

    Returns:
        Path: Default VEP cache root directory (may or may not exist yet).
    """
    package_root = Path(__file__).resolve().parent.parent
    return package_root / "data" / "resources" / "vep"


def _is_interactive_session() -> bool:
    """
    Determine whether the current process is attached to an interactive
    terminal. Used to avoid blocking on ``input()`` in non-interactive
    contexts such as CI pipelines or scheduled jobs.

    Returns:
        bool: True if stdin appears to be an interactive TTY, False otherwise.
    """
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _prompt_yes_no(question: str) -> bool:
    """
    Ask the user a yes/no question on the console.

    Args:
        question: The question text to display (without the "[y/N]" suffix).

    Returns:
        bool: True if the user answered affirmatively ("y"/"yes"),
              False otherwise (including empty input, EOF, or any other input).
    """
    try:
        answer = input(f"{question} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _find_existing_cache(cache_root: Path) -> Optional[Path]:
    """
    Look for an already-installed, structurally valid VEP cache under
    ``cache_root``.

    A valid cache is expected to be a subdirectory named
    ``homo_sapiens_vep_{version}_{assembly}`` that additionally contains the
    standard VEP internal layout ``homo_sapiens/{version}_{assembly}/``
    (this is the structure produced when extracting Ensembl's indexed cache
    archive into that directory).

    If multiple valid caches are found, the one with the highest numeric
    version is selected and a warning is logged.

    Args:
        cache_root: Directory to scan for VEP cache subdirectories.

    Returns:
        Optional[Path]: Path to the selected cache directory, or None if no
                         valid cache was found.
    """
    if not cache_root.exists() or not cache_root.is_dir():
        return None

    candidates = []
    for entry in cache_root.iterdir():
        if not entry.is_dir():
            continue

        match = _CACHE_DIR_NAME_PATTERN.match(entry.name)
        if not match:
            continue

        version_str, assembly_str = match.group(1), match.group(2)
        inner_dir = entry / "homo_sapiens" / f"{version_str}_{assembly_str}"

        if inner_dir.exists() and inner_dir.is_dir():
            candidates.append((int(version_str), entry))
        else:
            logger.warning(
                f"Found a cache-like directory '{entry}' but its internal "
                f"structure looks incomplete (missing '{inner_dir}'); ignoring it."
            )

    if not candidates:
        return None

    # Prefer the most recent (highest version number) cache if several exist.
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    if len(candidates) > 1:
        logger.warning(
            "Multiple VEP caches found in the default location; "
            f"using the most recent one: {candidates[0][1]}"
        )

    return candidates[0][1]


def _find_existing_archive(cache_root: Path,
                           version: Optional[str] = None,
                           assembly: Optional[str] = None) -> Optional[Tuple[Path, str, str]]:
    """
    Look for a VEP cache archive (``.tar.gz``) that has already been
    downloaded but not yet extracted.

    This covers two common scenarios:
      - The user manually downloaded the archive and placed it directly
        inside the default cache root (``cache_root``).
      - A previous automatic download was interrupted before the extraction
        step, leaving the archive inside its target subdirectory
        (``cache_root/homo_sapiens_vep_{version}_{assembly}/``).

    Args:
        cache_root: Default VEP cache root directory to scan.
        version: If provided, only match archives for this exact version.
        assembly: If provided, only match archives for this exact assembly.

    Returns:
        Optional[Tuple[Path, str, str]]: A tuple of
        (archive_path, detected_version, detected_assembly) for the best
        match found, or None if no matching archive exists. If multiple
        archives match, the one with the highest version is preferred.
    """
    if not cache_root.exists() or not cache_root.is_dir():
        return None

    def _match_archive_name(name: str) -> Optional[Tuple[str, str]]:
        match = _CACHE_ARCHIVE_NAME_PATTERN.match(name)
        if not match:
            return None
        found_version, found_assembly = match.group(1), match.group(2)
        if version is not None and found_version != str(version):
            return None
        if assembly is not None and found_assembly != str(assembly):
            return None
        return found_version, found_assembly

    candidates = []

    # 1) Archives placed directly under the cache root.
    for entry in cache_root.iterdir():
        if entry.is_file():
            result = _match_archive_name(entry.name)
            if result is not None:
                candidates.append((entry, result[0], result[1]))

    # 2) Archives left inside a target-style subdirectory (e.g. an
    #    interrupted automatic download that stopped before extraction).
    for sub_entry in cache_root.iterdir():
        if not sub_entry.is_dir():
            continue
        for entry in sub_entry.iterdir():
            if entry.is_file():
                result = _match_archive_name(entry.name)
                if result is not None:
                    candidates.append((entry, result[0], result[1]))

    if not candidates:
        return None

    # Prefer the highest version if more than one archive is present.
    candidates.sort(key=lambda triple: int(triple[1]), reverse=True)
    return candidates[0]


def _build_archive_found_instructions(archive_path: Path, target_dir: Path,
                                      version: str, assembly: str) -> str:
    """
    Build a human-readable message informing the user that a compressed VEP
    cache archive was found but still needs to be extracted, including the
    exact command and destination path.

    Args:
        archive_path: Path to the ``.tar.gz`` archive that was found.
        target_dir: Directory into which the archive must be extracted.
        version: VEP cache version detected from the archive name.
        assembly: Genome assembly detected from the archive name.

    Returns:
        str: Multi-line instructional message.
    """
    expected_inner_dir = target_dir / "homo_sapiens" / f"{version}_{assembly}"
    return (
        "A VEP cache archive was found, but it has not been extracted yet.\n"
        f"  Archive found at : {archive_path}\n"
        f"  Must be extracted into: {target_dir}\n"
        "\n"
        "You can extract it manually, for example by running:\n"
        f'  mkdir -p "{target_dir}"\n'
        f'  tar -xzf "{archive_path}" -C "{target_dir}"\n'
        "\n"
        "After extraction, the following path must exist:\n"
        f"  {expected_inner_dir}\n"
    )


def _download_with_progress(url: str, destination: Path) -> None:
    """
    Download a file from ``url`` to ``destination``, logging periodic
    progress (every 5%) since VEP cache archives are very large (tens of GB).

    Args:
        url: Source URL of the file to download.
        destination: Local path where the downloaded file will be saved.

    Raises:
        Exception: Any exception raised by urllib during the download is
                   propagated to the caller so it can clean up partial files.
    """
    last_reported_percent = -1

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        nonlocal last_reported_percent
        if total_size <= 0:
            return  # Server did not report a content length; skip progress logging.

        downloaded = block_num * block_size
        percent = min(int(downloaded * 100 / total_size), 100)

        if percent != last_reported_percent and percent % 5 == 0:
            downloaded_gb = downloaded / (1024 ** 3)
            total_gb = total_size / (1024 ** 3)
            logger.info(
                f"Download progress: {percent}% "
                f"({downloaded_gb:.1f} / {total_gb:.1f} GB)"
            )
            last_reported_percent = percent

    urllib.request.urlretrieve(url, filename=str(destination), reporthook=_reporthook)


def _download_and_extract_vep_cache(target_dir: Path,
                                     version: str,
                                     assembly: str,
                                     download_url: str) -> None:
    """
    Download the Ensembl indexed VEP cache archive for the given
    version/assembly and extract it into ``target_dir``.

    The downloaded ``.tar.gz`` archive is removed after a successful
    extraction to avoid keeping a redundant ~25 GB copy on disk.

    Args:
        target_dir: Directory that will contain the extracted cache
                    (created if it does not already exist).
        version: VEP cache version (e.g. "114").
        assembly: Genome assembly name (e.g. "GRCh38").
        download_url: Full URL of the ``.tar.gz`` cache archive to download.

    Raises:
        RuntimeError: If the download or extraction step fails. Partial
                      downloads are cleaned up before the exception is raised.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / f"homo_sapiens_vep_{version}_{assembly}.tar.gz"

    # Best-effort disk space check. This does not block the download, it only
    # warns the user in case there is clearly not enough free space.
    try:
        usage = shutil.disk_usage(target_dir)
        free_gb = usage.free / (1024 ** 3)
        # Require roughly double the archive size to account for the
        # compressed archive plus its extracted contents co-existing briefly.
        recommended_free_gb = ESTIMATED_VEP_CACHE_SIZE_GB * 2
        if free_gb < recommended_free_gb:
            logger.warning(
                f"Low disk space at '{target_dir}': {free_gb:.1f} GB free, "
                f"~{recommended_free_gb:.0f} GB recommended for download + extraction."
            )
    except OSError:
        pass  # Disk usage check is best-effort only; ignore failures.

    logger.info(f"Downloading VEP cache from: {download_url}")
    logger.info(f"Saving archive to: {archive_path}")

    try:
        _download_with_progress(download_url, archive_path)
    except Exception as e:
        if archive_path.exists():
            archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download VEP cache from '{download_url}': {e}"
        ) from e

    logger.info(f"Download complete. Extracting archive to: {target_dir}")

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract VEP cache archive '{archive_path}': {e}"
        ) from e
    finally:
        # Remove the archive regardless of extraction outcome to save disk
        # space; if extraction failed, the caller will report the error.
        try:
            archive_path.unlink(missing_ok=True)
            logger.info(f"Removed downloaded archive: {archive_path}")
        except OSError:
            logger.warning(f"Could not remove downloaded archive: {archive_path}")


def _build_manual_cache_instructions(default_root: Path, target_dir: Path,
                                     version: str, assembly: str,
                                     download_url: str) -> str:
    """
    Build a human-readable message explaining how to obtain and place the
    VEP cache manually, for cases where automatic download is skipped
    or declined.

    Args:
        default_root: Default VEP cache root directory used by pyMut.
        target_dir: Full expected path of the specific cache directory.
        version: VEP cache version (e.g. "114").
        assembly: Genome assembly name (e.g. "GRCh38").
        download_url: Direct download URL for the required cache archive.

    Returns:
        str: Multi-line instructional message.
    """
    expected_inner_dir = target_dir / "homo_sapiens" / f"{version}_{assembly}"
    return (
        "VEP cache not found automatically.\n"
        f"  Default search location : {default_root}\n"
        f"  Expected cache folder   : {target_dir}\n"
        f"  Expected internal layout: {expected_inner_dir}\n"
        "\n"
        "To resolve this, you can either:\n"
        f"  1) Download the cache manually from:\n"
        f"     {download_url}\n"
        f"     and extract it so that the final directory matches:\n"
        f"     {target_dir}\n"
        "\n"
        "  2) If you already have a compatible VEP cache elsewhere on disk,\n"
        "     either pass its path explicitly via the 'cache_dir' argument,\n"
        "     or copy/symlink it into the expected location shown above.\n"
    )


def _handle_missing_cache(default_root: Path,
                           version: str,
                           assembly: str,
                           auto_download: Optional[bool]) -> Path:
    """
    Handle the case where no valid VEP cache was found automatically.

    Depending on ``auto_download``, this either downloads the cache right
    away, interactively asks the user for confirmation, or raises an error
    with manual installation instructions.

    Args:
        default_root: Default VEP cache root directory used by pyMut.
        version: VEP cache version to use if a download is triggered.
        assembly: Genome assembly to use if a download is triggered.
        auto_download: Controls the download behaviour:
            - True  -> download automatically, without asking.
            - False -> never download; raise with manual instructions.
            - None  -> ask the user interactively (if possible); if the
                       session is not interactive, behaves like False.

    Returns:
        Path: Path to the newly downloaded and extracted cache directory.

    Raises:
        FileNotFoundError: If the cache is missing and the user declines
                            (or cannot be asked) to download it automatically.
        RuntimeError: If the automatic download/extraction fails.
    """
    target_dir = default_root / f"homo_sapiens_vep_{version}_{assembly}"
    download_url = VEP_CACHE_DOWNLOAD_URL_TEMPLATE.format(version=version, assembly=assembly)
    manual_instructions = _build_manual_cache_instructions(
        default_root, target_dir, version, assembly, download_url
    )

    logger.warning(f"VEP cache not found at the default package location: {default_root}")

    should_download = auto_download

    if should_download is None:
        if not _is_interactive_session():
            logger.warning(
                "Non-interactive environment detected; skipping the "
                "interactive cache download prompt."
            )
            should_download = False
        else:
            print("\n" + "=" * 78)
            print("VEP reference cache not found.")
            print(f"Expected location: {target_dir}")
            print(
                "This cache is required for VEP annotation and is "
                f"approximately {ESTIMATED_VEP_CACHE_SIZE_GB} GB in size."
            )
            print("=" * 78)
            should_download = _prompt_yes_no(
                "Do you want to download it automatically now? "
                "This may take a long time depending on your connection"
            )

    if not should_download:
        logger.error(manual_instructions)
        raise FileNotFoundError(
            "VEP cache not found and automatic download was not performed.\n\n"
            f"{manual_instructions}"
        )

    logger.info(f"Starting automatic VEP cache download to: {target_dir}")
    _download_and_extract_vep_cache(
        target_dir=target_dir, version=version, assembly=assembly, download_url=download_url
    )

    # Verify that the extracted archive actually produced the expected layout.
    expected_inner_dir = target_dir / "homo_sapiens" / f"{version}_{assembly}"
    if not expected_inner_dir.exists():
        raise FileNotFoundError(
            "VEP cache download/extraction finished, but the expected "
            f"internal structure was not found: {expected_inner_dir}\n\n"
            f"{manual_instructions}"
        )

    logger.info(f"VEP cache successfully installed at: {target_dir}")
    return target_dir


def _resolve_vep_cache(cache_dir: Optional[Union[str, Path]],
                       assembly: Optional[str],
                       version: Optional[str],
                       auto_download: Optional[bool]) -> Path:
    """
    Resolve the VEP cache directory to use, following this order:

    1. If ``cache_dir`` is explicitly provided, use it as-is (it must exist).
    2. Otherwise, look for a valid, already-extracted cache in the default
       package location (``<package_root>/data/resources/vep``). If
       ``assembly`` and ``version`` were both provided, look specifically
       for that cache; otherwise, auto-detect any valid cache present.
    3. If no extracted cache is found, check whether a downloaded but
       not-yet-extracted ``.tar.gz`` archive is already present. If so,
       raise immediately with clear instructions on how to extract it and
       the exact destination path (no download is attempted in this case).
    4. If neither an extracted cache nor a compressed archive is found,
       offer to download it automatically (see ``_handle_missing_cache``
       for the exact behaviour, which depends on ``auto_download``).

    Args:
        cache_dir: Explicit VEP cache path provided by the caller, or None
                   to trigger auto-detection/auto-download.
        assembly: Genome assembly requested by the caller, if any.
        version: VEP cache version requested by the caller, if any.
        auto_download: Controls automatic download behaviour when the cache
                        is missing (True/False/None); see
                        ``_handle_missing_cache`` for details.

    Returns:
        Path: A validated, existing VEP cache directory.

    Raises:
        FileNotFoundError: If an explicit ``cache_dir`` was given but does
                            not exist, or if no cache could be found/downloaded.
        RuntimeError: If an automatic download was attempted but failed.
    """
    # Case 1: explicit path provided -> preserve original strict behaviour.
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache directory not found: {cache_path}")
        return cache_path

    # Case 2/3: no explicit path -> resolve from the default package location.
    default_root = _get_default_cache_root()

    if assembly is not None and version is not None:
        # The caller requested a specific assembly/version: look for exactly
        # that cache before falling back to the download workflow.
        specific_dir = default_root / f"homo_sapiens_vep_{version}_{assembly}"
        inner_dir = specific_dir / "homo_sapiens" / f"{version}_{assembly}"
        if specific_dir.exists() and inner_dir.exists():
            logger.info(f"Using requested VEP cache found at: {specific_dir}")
            return specific_dir

        # The extracted cache is missing; check whether the compressed
        # archive is already present but simply not extracted yet.
        archive_match = _find_existing_archive(default_root, version=version, assembly=assembly)
        if archive_match is not None:
            archive_path, found_version, found_assembly = archive_match
            instructions = _build_archive_found_instructions(
                archive_path, specific_dir, found_version, found_assembly
            )
            logger.warning(instructions)
            raise FileNotFoundError(
                "VEP cache archive found but not extracted yet.\n\n" + instructions
            )

        target_version, target_assembly = version, assembly
    else:
        # No specific assembly/version requested: auto-detect any valid cache.
        found = _find_existing_cache(default_root)
        if found is not None:
            logger.info(f"Using auto-detected VEP cache at: {found}")
            return found

        # No extracted cache found; check whether a downloaded archive is
        # already present under the default root but not yet extracted.
        archive_match = _find_existing_archive(default_root)
        if archive_match is not None:
            archive_path, found_version, found_assembly = archive_match
            target_dir_for_archive = default_root / f"homo_sapiens_vep_{found_version}_{found_assembly}"
            instructions = _build_archive_found_instructions(
                archive_path, target_dir_for_archive, found_version, found_assembly
            )
            logger.warning(instructions)
            raise FileNotFoundError(
                "VEP cache archive found but not extracted yet.\n\n" + instructions
            )

        target_version = version or DEFAULT_VEP_CACHE_VERSION
        target_assembly = assembly or DEFAULT_VEP_CACHE_ASSEMBLY

    return _handle_missing_cache(default_root, target_version, target_assembly, auto_download)


def _extract_assembly_and_version_from_cache(cache_dir: Union[str, Path]) -> tuple[str, str]:
    """Extract assembly and version information from VEP cache directory name."""
    cache_path = Path(cache_dir)
    cache_name = cache_path.name

    pattern = r'homo_sapiens_vep_(\d+)_([A-Za-z0-9\.]+)'
    match = re.search(pattern, cache_name)

    if not match:
        raise ValueError(
            f"Cache directory name '{cache_name}' doesn't match expected format 'homo_sapiens_vep_{{version}}_{{assembly}}'")

    version = match.group(1)
    assembly = match.group(2)

    return assembly, version


def _get_case_insensitive_column(columns: list, target_column: str) -> str:
    """Find column name case-insensitively."""
    column_map = {col.lower(): col for col in columns}
    target_lower = target_column.lower()
    if target_lower in column_map:
        return column_map[target_lower]
    else:
        raise KeyError(f"Column '{target_column}' not found in MAF file. Available columns: {columns}")


def _detect_text_encoding(raw: bytes) -> str:
    """
    Best-effort detection of the text encoding of a raw byte string.

    Tries, in order, UTF-8 with BOM, UTF-16 (with/without BOM), CP1252 and
    Latin-1, falling back to UTF-8 if nothing else decodes cleanly. This is
    what lets ``_maf_to_region`` cope with MAF files exported from Excel or
    other tools that write UTF-16 (which raises
    ``UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff...`` when
    naively opened as UTF-8).

    Args:
        raw: Raw file content as bytes.

    Returns:
        str: The name of an encoding that successfully decodes ``raw``.
    """
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def _coerce_maf_to_utf8(maf_file: Path) -> Path:
    """
    Ensure a MAF file (plain or ``.gz``) is readable as plain UTF-8 text.

    If the file is already plain UTF-8 with no BOM, the original path is
    returned unchanged (no extra I/O). Otherwise the file is decompressed
    (if needed), decoded using the detected encoding, and re-written as a
    sibling ``*.utf8.maf`` file which is returned instead.

    Args:
        maf_file: Path to the original MAF file (.maf or .maf.gz).

    Returns:
        Path: Path to a file guaranteed to be readable as UTF-8 text.
    """
    if maf_file.suffix == '.gz':
        with gzip.open(maf_file, 'rb') as fh:
            raw = fh.read()
    else:
        raw = maf_file.read_bytes()

    encoding = _detect_text_encoding(raw)

    # Already plain UTF-8, no BOM, uncompressed: nothing to do.
    if maf_file.suffix != '.gz' and encoding == "utf-8" and not raw.startswith(b'\xef\xbb\xbf'):
        return maf_file

    decode_encoding = "utf-8-sig" if encoding == "utf-8" and raw.startswith(b'\xef\xbb\xbf') else encoding
    try:
        text = raw.decode(decode_encoding)
    except UnicodeDecodeError:
        logger.warning(
            f"Could not cleanly decode {maf_file} as '{encoding}'; "
            "falling back to UTF-8 with replacement characters."
        )
        text = raw.decode('utf-8', errors='replace')

    if encoding != "utf-8":
        logger.info(f"Detected '{encoding}' encoding for {maf_file}; converting to UTF-8.")

    converted_path = maf_file.with_name(f"{maf_file.stem}.utf8.maf")
    converted_path.write_text(text, encoding="utf-8", newline="")
    return converted_path


def _maf_to_region(maf_path: Union[str, Path],
                   out_path: Optional[Union[str, Path]] = None) -> Tuple[bool, str]:
    """
    Convert a MAF file to region format.

    Args:
        maf_path: Path to the MAF file (.maf or .maf.gz)
        out_path: Path to the output .region file (optional)

    Returns:
        Tuple[bool, str]: (success_status, output_path)
    """
    maf_file = Path(maf_path)

    if not maf_file.exists():
        logger.error(f"MAF file not found: {maf_file}")
        return False, ""

    if out_path is None:
        if maf_file.suffix == '.gz' and maf_file.stem.endswith('.maf'):
            base_name = maf_file.stem[:-4]
            output_file = maf_file.parent / f"{base_name}.region"
        elif maf_file.suffix == '.maf':
            output_file = maf_file.with_suffix('.region')
        else:
            output_file = maf_file.with_suffix('.region')
    else:
        output_file = Path(out_path)

    logger.info(f"Converting MAF to region format: {maf_file} -> {output_file}")

    try:
        readable_maf_file = _coerce_maf_to_utf8(maf_file)

        with open(readable_maf_file, 'r', encoding='utf-8', newline='') as maf, \
                open(output_file, "w", encoding="utf-8") as out:
            reader = csv.DictReader(maf, delimiter="\t")

            columns = reader.fieldnames
            if not columns:
                raise ValueError("No columns found in MAF file")

            try:
                chrom_col = _get_case_insensitive_column(columns, "Chromosome")
                start_col = _get_case_insensitive_column(columns, "Start_Position")
                end_col = _get_case_insensitive_column(columns, "End_position")
                alt_col = _get_case_insensitive_column(columns, "Tumor_Seq_Allele2")

                try:
                    strand_col = _get_case_insensitive_column(columns, "Strand")
                    has_strand = True
                except KeyError:
                    has_strand = False
                    logger.warning("Strand column not found in MAF file, using default value '+'")

            except KeyError as e:
                logger.error(f"Required column not found: {e}")
                raise

            for row in reader:
                chrom_raw = row[chrom_col]
                chrom = format_chr(chrom_raw)
                start = row[start_col]
                endpos = row[end_col]
                alt = row[alt_col]
                strand = row[strand_col] if has_strand else "+"
                strand_num = "1" if strand == "+" else "-1"

                out.write(f"{chrom}:{start}-{endpos}:{strand_num}/{alt}\n")

        logger.info(f"Successfully converted MAF to region format: {output_file}")
        return True, str(output_file)

    except Exception as e:
        logger.error(f"Error converting MAF to region format: {e}")
        return False, str(output_file) if 'output_file' in locals() else ""


def wrap_maf_vep_annotate_protein(maf_file: Union[str, Path],
                                  fasta: Union[str, Path],
                                  *,
                                  cache_dir: Optional[Union[str, Path]] = None,
                                  output_file: Optional[Union[str, Path]] = None,
                                  synonyms_file: Optional[Union[str, Path]] = None,
                                  assembly: Optional[str] = None,
                                  version: Optional[str] = None,
                                  compress: bool = True,
                                  no_stats: bool = True,
                                  auto_download: Optional[bool] = None) -> Tuple[bool, str]:
    """
    Wrapper method for VEP annotation that accepts MAF files and merges annotations back to MAF.

    This method converts a MAF file to region format internally, runs VEP annotation, and then
    merges the VEP annotations back with the original MAF file. VEP annotation uses the following
    fixed parameters:
    - --offline --cache
    - --protein --uniprot --domains --symbol
    - --synonyms (automatically constructed from cache directory or provided explicitly)
    - --no_stats (only when no_stats=False)

    After successful VEP annotation, the method automatically merges the VEP results with the
    original MAF file, creating an annotated MAF file with VEP_ prefixed columns. The original
    VEP annotation files are preserved and not deleted.

    Assembly and cache version can be provided explicitly or automatically extracted from the cache directory name.
    The chr_synonyms file path can be provided explicitly or automatically constructed as: cache_dir/homo_sapiens/{version}_{assembly}/chr_synonyms.txt

    VEP cache resolution:
        ``cache_dir`` is now OPTIONAL. If it is not provided, the following
        resolution strategy is applied automatically:
          1. Look for an existing, structurally valid cache under the default
             package location: ``<pyMut_install_dir>/data/resources/vep``.
          2. If no cache is found there, the user is asked (interactively,
             on the console) whether to download the Ensembl cache
             automatically. This download is large (~25 GB), so the user is
             clearly warned beforehand.
          3. If the user declines (or the process is running non-interactively
             and ``auto_download`` was not explicitly set), a
             ``FileNotFoundError`` is raised with the direct download URL and
             instructions on where to place an existing cache manually.
        This automatic behaviour can be controlled via ``auto_download``:
        pass True to always download without asking, or False to never
        download and always fail fast with instructions instead.

    NOTE (API change): ``cache_dir`` and all parameters after it are now
    keyword-only. Existing code calling this function positionally as
    ``wrap_maf_vep_annotate_protein(maf_file, cache_dir, fasta)`` must be
    updated to ``wrap_maf_vep_annotate_protein(maf_file, fasta, cache_dir=cache_dir)``.

    Args:
        maf_file: Path to the MAF file to annotate (.maf or .maf.gz)
        fasta: Path to the reference FASTA file
        cache_dir: Path to the VEP cache directory (optional). If None, the
                  cache is auto-detected or its download is offered as
                  described above.
        output_file: Path to the output file (optional). If None, creates a directory
                    in the same location as maf_file with format 'vep_annotation_HHMMDDMMYYYY'
        synonyms_file: Path to the chromosome synonyms file (optional). If None, automatically
                      constructed from cache directory structure
        assembly: Genome assembly name (optional). If None, automatically extracted from cache directory name
                  (or defaults to "GRCh38" if a cache download is triggered and no cache directory
                  name is available yet)
        version: VEP cache version (optional). If None, automatically extracted from cache directory name
                 (or defaults to "114" if a cache download is triggered and no cache directory
                 name is available yet)
        compress: Whether to compress the merged output file with gzip (default: True)
        no_stats: Whether to disable VEP statistics generation (default: True). When True, --no_stats flag is omitted
        auto_download: Controls automatic cache download when no cache is found and
                       ``cache_dir`` was not provided:
                           - True: download automatically without asking.
                           - False: never download; raise immediately with manual instructions.
                           - None (default): ask interactively on the console; in a
                             non-interactive session this behaves like False.

    Returns:
        Tuple[bool, str]: (success_status, output_info) where success_status is True
                         if annotation was successful, and output_info contains information
                         about both the VEP file and the merged MAF file paths

    Raises:
        ValueError: If cache directory name format is invalid and assembly/version not provided
        FileNotFoundError: If required files (MAF, FASTA) don't exist, if an explicitly
                           provided cache_dir doesn't exist, or if no cache could be found
                           and automatic download was declined/skipped
        RuntimeError: If an automatic cache download or extraction fails
    """
    maf_path = Path(maf_file)
    fasta_path = Path(fasta)
    if not maf_path.exists():
        raise FileNotFoundError(f"MAF file not found: {maf_path}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    # Resolve the VEP cache directory: use the explicit path if given,
    # otherwise auto-detect it (or offer to download it) from the default
    # package location. See _resolve_vep_cache for the full strategy.
    cache_path = _resolve_vep_cache(cache_dir, assembly, version, auto_download)

    logger.info(f"Converting MAF file to region format: {maf_path}")
    region_success, region_path = _maf_to_region(maf_path)

    if not region_success:
        logger.error("Failed to convert MAF to region format")
        return False, ""

    region_file = Path(region_path)
    logger.info(f"Successfully converted MAF to region format: {region_file}")

    if output_file is None:
        timestamp = datetime.now().strftime("%H%M%d%m")
        output_dir_name = f"vep_annotation_{timestamp}"
        output_dir = maf_path.parent / output_dir_name
        output_dir.mkdir(exist_ok=True)

        output_filename = f"{maf_path.stem}_vep_protein.txt"
        output_path = output_dir / output_filename
        final_output_path = None
    else:
        output_path = Path(output_file)
        final_output_path = output_path
        if output_path.suffix == '.gz':
            # VEP always writes plain text regardless of the filename we give it, so
            # using a ".gz" name here would make pandas try to gzip-decode plain text
            # later on. Write VEP's raw output without the .gz suffix and let the
            # merge step produce the compressed final file at the requested path.
            output_path = output_path.with_suffix('')
            logger.info(
                f"Requested output '{final_output_path}' ends in .gz; VEP raw output "
                f"will be written to '{output_path}' and the compressed merged file "
                f"will be written to '{final_output_path}'."
            )

    if assembly is None or version is None:
        try:
            extracted_assembly, extracted_version = _extract_assembly_and_version_from_cache(cache_path)
            if assembly is None:
                assembly = extracted_assembly
            if version is None:
                version = extracted_version
            logger.info(f"Extracted from cache: assembly={assembly}, version={version}")
        except ValueError as e:
            logger.error(f"Failed to extract assembly/version from cache: {e}")
            raise
    else:
        logger.info(f"Using provided: assembly={assembly}, version={version}")

    if synonyms_file is None:
        chr_synonyms_path = cache_path / "homo_sapiens" / f"{version}_{assembly}" / "chr_synonyms.txt"
        logger.info(f"Auto-constructed chr synonyms path: {chr_synonyms_path}")
    else:
        chr_synonyms_path = Path(synonyms_file)
        logger.info(f"Using provided chr synonyms path: {chr_synonyms_path}")

    vep_cmd = [
        "vep",
        "--input_file", str(region_file),
        "--format", "region",
        "--offline", "--cache", "--cache_version", version,
        "--dir_cache", str(cache_path),
        "--assembly", assembly,
        "--synonyms", str(chr_synonyms_path),
        "--fasta", str(fasta_path),
        "--protein", "--uniprot", "--domains", "--symbol",
        "--pick",
        "--keep_csq",
        "--force_overwrite",
        "--output_file", str(output_path)
    ]

    # Add --no_stats when no_stats is True
    if no_stats:
        vep_cmd.insert(-2, "--no_stats")

    try:
        logger.info(f"Running VEP annotation: {' '.join(vep_cmd)}")
        result = subprocess.run(vep_cmd, check=True, capture_output=True, text=True)
        logger.info("VEP annotation completed successfully")

        if result.stderr:
            logger.warning(f"VEP warnings/messages: {result.stderr}")

        logger.info("Merging VEP annotations with original MAF file...")
        try:
            merged_df, merged_output_path = merge_maf_with_vep_annotations(
                maf_file=maf_path,
                vep_file=output_path,
                output_file=final_output_path,
                compress=compress
            )
            logger.info(f"Successfully merged VEP annotations. Merged file: {merged_output_path}")

            return True, f"VEP folder: {output_path}, Merged file: {merged_output_path}"

        except Exception as merge_error:
            logger.error(f"Failed to merge VEP annotations: {merge_error}")
            # Success for VEP annotation, merge failure
            return True, f"VEP folder: {output_path}, Merge failed: {merge_error}"

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"VEP annotation failed: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            logger.error(f"VEP error output: {e.stderr}")
        return False, str(output_path)
    except Exception as e:
        logger.error(f"Unexpected error during VEP annotation: {e}")
        return False, str(output_path)
    finally:
        # Clean up temporary region file
        try:
            if region_file.exists():
                region_file.unlink()
                logger.info(f"Cleaned up temporary region file: {region_file}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary region file {region_file}: {e}")


def wrap_vcf_vep_annotate_unified(vcf_file: Union[str, Path],
                                  fasta: Union[str, Path],
                                  *,
                                  cache_dir: Optional[Union[str, Path]] = None,
                                  output_file: Optional[Union[str, Path]] = None,
                                  synonyms_file: Optional[Union[str, Path]] = None,
                                  assembly: Optional[str] = None,
                                  version: Optional[str] = None,
                                  no_stats: bool = True,
                                  # New parameters to control annotations
                                  annotate_protein: bool = False,
                                  annotate_gene: bool = False,
                                  annotate_variant_class: bool = False,
                                  distance: Optional[int] = None,
                                  auto_download: Optional[bool] = None) -> Tuple[bool, str]:
    """
    Unified method for VEP annotation that allows combining different types of annotation.

    VEP cache resolution:
        ``cache_dir`` is OPTIONAL. If not provided, pyMut will automatically
        look for a valid cache under ``<pyMut_install_dir>/data/resources/vep``
        and, if none is found, will offer to download it (~25 GB) after a
        clear warning. See ``auto_download`` to control this behaviour, and
        ``_resolve_vep_cache`` / ``_handle_missing_cache`` for the full
        resolution strategy shared with ``wrap_maf_vep_annotate_protein``.

    NOTE (API change): ``cache_dir`` and all parameters after it are now
    keyword-only. Existing code calling this function positionally as
    ``wrap_vcf_vep_annotate_unified(vcf_file, cache_dir, fasta)`` must be
    updated to ``wrap_vcf_vep_annotate_unified(vcf_file, fasta, cache_dir=cache_dir)``.

    Args:
        vcf_file: Path to the VCF file to annotate
        fasta: Path to the reference FASTA file
        cache_dir: Path to the VEP cache directory (optional). If None, the
                  cache is auto-detected or its download is offered as
                  described above.
        output_file: Path to the output file (optional)
        synonyms_file: Path to the chromosome synonyms file (optional)
        assembly: Genome assembly name (optional; defaults to "GRCh38" if a
                  cache download is triggered without an explicit cache_dir)
        version: VEP cache version (optional; defaults to "114" if a cache
                 download is triggered without an explicit cache_dir)
        no_stats: Whether to disable VEP statistics generation
        annotate_protein: Whether to include protein annotation (--protein --uniprot --domains --symbol)
        annotate_gene: Whether to include gene annotation (--symbol)
        annotate_variant_class: Whether to include variant classification (--variant_class)
        distance: Distance for nearest gene search (only for annotate_gene)
        auto_download: Controls automatic cache download when no cache is found and
                       ``cache_dir`` was not provided:
                           - True: download automatically without asking.
                           - False: never download; raise immediately with manual instructions.
                           - None (default): ask interactively on the console; in a
                             non-interactive session this behaves like False.

    Returns:
        Tuple[bool, str]: (success_status, output_info)

    Raises:
        ValueError: If no annotation option is enabled
        FileNotFoundError: If required files (VCF, FASTA) don't exist, if an explicitly
                           provided cache_dir doesn't exist, or if no cache could be found
                           and automatic download was declined/skipped
        RuntimeError: If an automatic cache download or extraction fails
    """
    # Validation: at least one annotation must be enabled
    if not any([annotate_protein, annotate_gene, annotate_variant_class]):
        raise ValueError("At least one annotation option must be enabled")

    vcf_path = Path(vcf_file)
    fasta_path = Path(fasta)

    # Validate input files
    if not vcf_path.exists():
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    # Resolve the VEP cache directory: use the explicit path if given,
    # otherwise auto-detect it (or offer to download it) from the default
    # package location. See _resolve_vep_cache for the full strategy.
    cache_path = _resolve_vep_cache(cache_dir, assembly, version, auto_download)

    logger.info(f"Starting unified VEP annotation for VCF file: {vcf_path}")

    # Handle output file creation
    if output_file is None:
        timestamp = datetime.now().strftime("%H%M%d%m")
        output_dir_name = f"vep_annotation_{timestamp}"
        output_dir = vcf_path.parent / output_dir_name
        output_dir.mkdir(exist_ok=True)

        # Create descriptive name based on selected annotations
        annotations = []
        if annotate_protein:
            annotations.append("protein")
        if annotate_gene:
            annotations.append("gene")
        if annotate_variant_class:
            annotations.append("variant_class")

        output_filename = f"{vcf_path.stem}_vep_{'_'.join(annotations)}.vcf"
        output_path = output_dir / output_filename
    else:
        output_path = Path(output_file)

    # Extract assembly and version from cache if not provided
    if assembly is None or version is None:
        try:
            extracted_assembly, extracted_version = _extract_assembly_and_version_from_cache(cache_path)
            if assembly is None:
                assembly = extracted_assembly
            if version is None:
                version = extracted_version
            logger.info(f"Extracted from cache: assembly={assembly}, version={version}")
        except ValueError as e:
            logger.error(f"Failed to extract assembly/version from cache: {e}")
            raise
    else:
        logger.info(f"Using provided: assembly={assembly}, version={version}")

    # Handle chromosome synonyms file
    if synonyms_file is None:
        chr_synonyms_path = cache_path / "homo_sapiens" / f"{version}_{assembly}" / "chr_synonyms.txt"
        logger.info(f"Auto-constructed chr synonyms path: {chr_synonyms_path}")
    else:
        chr_synonyms_path = Path(synonyms_file)
        logger.info(f"Using provided chr synonyms path: {chr_synonyms_path}")

    # Build base VEP command
    vep_cmd = [
        "vep",
        "--input_file", str(vcf_path),
        "--vcf",
        "--offline",
        "--cache",
        "--cache_version", version,
        "--dir_cache", str(cache_path),
        "--assembly", assembly,
        "--synonyms", str(chr_synonyms_path),
        "--fasta", str(fasta_path),
        "--pick",
        "--force_overwrite",
        "--output_file", str(output_path)
    ]

    # Add specific parameters based on selected options
    if annotate_protein:
        vep_cmd.extend(["--protein", "--uniprot", "--domains", "--symbol"])

    if annotate_gene:
        if not annotate_protein:  # Avoid duplicating --symbol
            vep_cmd.append("--symbol")

        # Add distance parameters if specified
        if distance is not None:
            vep_cmd.extend(["--nearest", "symbol", "--distance", str(distance)])

    if annotate_variant_class:
        # Add --variant_class WITHOUT --fields to get all variant class fields
        vep_cmd.append("--variant_class")

    # Add --no_stats when no_stats is True
    if no_stats:
        vep_cmd.append("--no_stats")

    try:
        logger.info(f"Running unified VEP annotation: {' '.join(vep_cmd)}")
        result = subprocess.run(vep_cmd, check=True, capture_output=True, text=True)
        logger.info("Unified VEP annotation completed successfully")

        if result.stderr:
            logger.warning(f"VEP warnings/messages: {result.stderr}")

        return True, f"VEP output file: {output_path}"

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Unified VEP annotation failed: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            logger.error(f"VEP error output: {e.stderr}")
        return False, str(output_path)
    except Exception as e:
        logger.error(f"Unexpected error during unified VEP annotation: {e}")
        return False, str(output_path)
