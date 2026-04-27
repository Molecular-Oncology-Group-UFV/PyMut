"""
Module for creating lollipop plots.

This module contains functions for creating lollipop plots that visualize
mutation patterns along protein sequences, including domain annotations
and mutation hotspots.

Lollipop plots are essential in cancer genomics for:
- Visualizing mutation distribution along protein sequences
- Identifying mutation hotspots (recurrent positions)
- Showing protein domain architecture
- Comparing mutation types at specific amino acid positions

Main functions:
- extract_aa_position(): Extracts amino acid position from HGVS notation
- resolve_protein_length(): Determines protein length from data or annotations
- resolve_domains(): Gets PFAM domain information
- aggregate_mutation_events(): Aggregates mutations by position and type
- compute_lollipop_layout(): Calculates coordinates for visualization
- apply_label_repel(): Implements label repulsion algorithm
- _create_lollipop_plot(): Main function to create the lollipop plot
"""

import logging
import re
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

if TYPE_CHECKING:
    from ..core import PyMutation

logger = logging.getLogger(__name__)

# Fixed color palette for variant classifications (reproducibility)
# Supports both standard MAF naming (Mixed_Case) and uppercase (UPPERCASE)
VARIANT_COLOR_PALETTE = {
    # Standard MAF naming (Mixed_Case)
    'Missense_Mutation': '#008000',          # Green
    'Nonsense_Mutation': '#FF0000',          # Red
    'Frame_Shift_Del': '#8B008B',            # Dark magenta
    'Frame_Shift_Ins': '#9400D3',            # Dark violet
    'In_Frame_Del': '#00CED1',               # Dark turquoise
    'In_Frame_Ins': '#4682B4',               # Steel blue
    'Splice_Site': '#FF8C00',                # Dark orange
    'Nonstop_Mutation': '#FF1493',           # Deep pink
    'Translation_Start_Site': '#FFD700',     # Gold
    'Silent': '#A9A9A9',                     # Dark gray
    'Multi_Hit': '#000000',                  # Black
    'Other': '#808080',                      # Gray
    # Uppercase variants (some MAF files use this format)
    'MISSENSE_MUTATION': '#008000',          # Green
    'NONSENSE_MUTATION': '#FF0000',          # Red
    'FRAME_SHIFT_DEL': '#8B008B',            # Dark magenta
    'FRAME_SHIFT_INS': '#9400D3',            # Dark violet
    'IN_FRAME_DEL': '#00CED1',               # Dark turquoise
    'IN_FRAME_INS': '#4682B4',               # Steel blue
    'SPLICE_SITE': '#FF8C00',                # Dark orange
    'NONSTOP_MUTATION': '#FF1493',           # Deep pink
    'TRANSLATION_START_SITE': '#FFD700',     # Gold
    'SILENT': '#A9A9A9',                     # Dark gray
    'MULTI_HIT': '#000000',                  # Black
}

# Default parameters
DEFAULT_LOLLIPOP_FIGSIZE = (16, 6)
DEFAULT_LABEL_TOP_N = 20
DEFAULT_COUNT_BY = "mutations"  # or "samples"

# Non-synonymous variant classifications
# These are the variant types counted for somatic mutation rate calculation
NON_SYNONYMOUS_VARIANTS = {
    'Frame_Shift_Del',
    'Frame_Shift_Ins',
    'In_Frame_Del',
    'In_Frame_Ins',
    'Missense_Mutation',
    'Nonsense_Mutation',
    'Nonstop_Mutation',
    'Splice_Site',
    'Translation_Start_Site',
    # Also include uppercase variants (some MAF files use this format)
    'FRAME_SHIFT_DEL',
    'FRAME_SHIFT_INS',
    'IN_FRAME_DEL',
    'IN_FRAME_INS',
    'MISSENSE_MUTATION',
    'NONSENSE_MUTATION',
    'NONSTOP_MUTATION',
    'SPLICE_SITE',
    'TRANSLATION_START_SITE',
}


def extract_aa_position(protein_change: str) -> Optional[int]:
    """
    Extract amino acid position from HGVS protein notation.
    
    Supports multiple formats:
    - Standard: p.D835Y, p.R132H
    - Stop codon: p.W288*, p.Q61*
    - Frameshift: p.D835fs, p.D835Yfs*3
    - Insertion: p.D835_I836insGLY, p.835_836ins3
    - Deletion: p.D835del, p.D835_Y842del
    - Delins: p.D835_Y842delinsH
    
    Args:
        protein_change: HGVS protein notation string
        
    Returns:
        int: Amino acid position (first affected residue), or None if not parseable
        
    Examples:
        >>> extract_aa_position("p.D835Y")
        835
        >>> extract_aa_position("p.W288*")
        288
        >>> extract_aa_position("p.D835_Y842del")
        835
        >>> extract_aa_position("p.D835fs")
        835
    """
    if pd.isna(protein_change) or not protein_change:
        return None
    
    protein_str = str(protein_change).strip()
    
    # Remove "p." prefix if present
    if protein_str.startswith("p."):
        protein_str = protein_str[2:]
    
    # Pattern 1: Standard single AA change (e.g., D835Y, W288*)
    # Captures: letter(s) + digits + letter(s)/*/fs/etc
    match = re.match(r'^([A-Z*]+)(\d+)', protein_str, re.IGNORECASE)
    if match:
        return int(match.group(2))
    
    # Pattern 2: Range notation (e.g., D835_Y842del, 835_836ins3)
    # Captures the first position
    match = re.match(r'^[A-Z]*(\d+)_', protein_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Pattern 3: Just digits (sometimes in simplified formats)
    match = re.search(r'(\d+)', protein_str)
    if match:
        return int(match.group(1))
    
    logger.debug(f"Could not extract position from: {protein_change}")
    return None


def resolve_protein_length(data: pd.DataFrame,
                          aa_positions: pd.Series,
                          gene: str,
                          protein_id: Optional[str] = None,
                          transcript_id: Optional[str] = None,
                          py_mut: Optional['PyMutation'] = None) -> int:
    """
    Determine protein length from various sources.
    
    Priority order:
    1. Query UniProt for canonical protein (most reliable)
    2. If protein_id/transcript_id provided: query internal database
    3. Use maximum observed position + 10% margin
    4. Use 95th percentile of positions + 20% margin (robust to outliers)
    
    Args:
        data: DataFrame with mutation data
        aa_positions: Series with extracted amino acid positions
        gene: Gene symbol (for UniProt query)
        protein_id: UniProt protein ID (optional)
        transcript_id: RefSeq/Ensembl transcript ID (optional)
        py_mut: PyMutation object for accessing PFAM data (optional)
        
    Returns:
        int: Protein length (from UniProt or estimated)
    """
    # Method 1: Query UniProt for canonical protein length (MOST RELIABLE)
    try:
        _, uniprot_length = query_uniprot_domains(gene)
        if uniprot_length > 0:
            return uniprot_length
    except Exception as e:
        logger.debug(f"Could not get protein length from UniProt: {e}")
    
    # Method 2: Query from database if identifiers provided
    if (protein_id or transcript_id) and py_mut:
        # TODO: Implement database lookup when available
        pass
    
    # Method 3: Estimate from observed positions
    valid_positions = aa_positions.dropna()
    
    if len(valid_positions) == 0:
        logger.warning("No valid amino acid positions found, using default length of 500")
        return 500
    
    # Use 95th percentile for robustness to outliers
    percentile_95 = np.percentile(valid_positions, 95)
    max_position = valid_positions.max()
    
    # Choose the more conservative estimate
    estimated_length = max(percentile_95 * 1.2, max_position * 1.1)
    estimated_length = int(np.ceil(estimated_length))
    
    logger.debug(f"Estimated protein length: {estimated_length} aa (max observed: {max_position})")
    
    return estimated_length


def query_uniprot_domains(gene_symbol: str, organism: str = "9606") -> Tuple[List[Dict], int]:
    """
    Query UniProt directly for protein domains AND protein length.
    
    Args:
        gene_symbol: Gene symbol (e.g., "DNMT3A")
        organism: Organism taxonomy ID (default: 9606 for human)
        
    Returns:
        Tuple of (domains list, protein_length)
        domains: List of domain dictionaries with start, end, name
        protein_length: Length of the canonical protein sequence
    """
    try:
        import requests
        
        # Query UniProt for canonical/reviewed entry
        uniprot_url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": f"gene_exact:{gene_symbol} AND organism_id:{organism} AND reviewed:true",
            "format": "json",
            "size": 1
        }
        
        response = requests.get(uniprot_url, params=params, timeout=10)
        if not response.ok:
            logger.debug(f"UniProt query failed with status {response.status_code}")
            return [], 0
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            logger.debug(f"No UniProt entry found for {gene_symbol}")
            return [], 0
        
        entry = results[0]
        uniprot_id = entry["primaryAccession"]
        sequence_length = entry.get("sequence", {}).get("length", 0)
        
        # Extract domains from features
        domains = []
        features = entry.get("features", [])
        
        for feature in features:
            if feature.get("type") == "Domain":
                location = feature.get("location", {})
                start = location.get("start", {}).get("value")
                end = location.get("end", {}).get("value")
                description = feature.get("description", "Domain")
                
                if start and end:
                    domains.append({
                        "start": int(start),
                        "end": int(end),
                        "name": description
                    })
        
        logger.debug(f"UniProt {uniprot_id}: {sequence_length} aa, {len(domains)} domains")
        
        return domains, sequence_length
        
    except Exception as e:
        logger.debug(f"Error querying UniProt for {gene_symbol}: {e}")
        return [], 0


def load_pfam_domains_from_database(transcript_id: str, gene: str) -> Optional[List[Dict]]:
    """
    Load PFAM domains from internal dataset.
    
    Loads PFAM domain data from a curated internal database
    using transcript ID as the primary key.
    
    Args:
        transcript_id: RefSeq transcript ID (e.g., "NM_022552")
        gene: Gene symbol (for filtering)
        
    Returns:
        List of domain dictionaries, or None if not found
    """
    try:
        import os

        import pandas as pd
        
        # Path to PFAM domains CSV
        module_dir = os.path.dirname(os.path.abspath(__file__))
        domains_file = os.path.join(module_dir, '..', 'data', 'pfam_domains.csv')
        
        if not os.path.exists(domains_file):
            logger.debug(f"PFAM domains file not found: {domains_file}")
            return None
        
        # Load domains with dtype specification to avoid warnings
        domains_df = pd.read_csv(domains_file, low_memory=False)
        
        # Column mapping from internal database:
        # HGNC -> gene name
        # refseq.ID -> transcript ID  
        # Start -> domain start position
        # End -> domain end position
        # Label -> domain name
        
        # Filter by transcript ID and gene
        matched = domains_df[
            (domains_df['refseq.ID'] == transcript_id) &
            (domains_df['HGNC'] == gene)
        ]
        
        if matched.empty:
            logger.debug(f"No PFAM domains found for {transcript_id} ({gene})")
            return None
        
        # Convert to domain list
        domains = []
        for _, row in matched.iterrows():
            domains.append({
                "start": int(row['Start']),
                "end": int(row['End']),
                "name": str(row['Label'])
            })
        
        logger.debug(f"Loaded {len(domains)} PFAM domains for {transcript_id}")
        return domains
        
    except Exception as e:
        logger.debug(f"Error loading PFAM domains: {e}")
        return None


def resolve_domains(gene: str,
                   protein_length: int,
                   py_mut: Optional['PyMutation'] = None,
                   domains_source: Optional[str] = "pfam",
                   custom_domains: Optional[List[Dict]] = None,
                   transcript_id: Optional[str] = None) -> List[Dict]:
    """
    Resolve PFAM domains for the given gene.
    
    Strategy:
    1. Use custom domains if provided
    2. If transcript_id provided, try internal database first
    3. Query UniProt directly (most reliable, no DB required)
    4. Try PFAM database if available
    5. Check annotated data
    6. Fall back to full-length representation
    
    Args:
        gene: Gene symbol
        protein_length: Length of the protein sequence
        py_mut: PyMutation object for accessing PFAM data
        domains_source: Source for domains ("pfam" or None)
        custom_domains: Optional list of custom domain definitions
                       Format: [{"start": 100, "end": 200, "name": "Kinase"}, ...]
        transcript_id: RefSeq transcript ID for domain lookup
        
    Returns:
        List of domain dictionaries with keys: start, end, name
    """
    domains = []
    
    # Use custom domains if provided
    if custom_domains:
        logger.debug(f"Using {len(custom_domains)} custom domains")
        for domain in custom_domains:
            # Validate and clip to protein length
            start = max(1, int(domain.get("start", 1)))
            end = min(protein_length, int(domain.get("end", protein_length)))
            name = domain.get("name", "Domain")
            
            if start <= end:
                domains.append({"start": start, "end": end, "name": name})
        
        return domains
    
    # Try internal database domains if transcript_id provided
    if transcript_id and domains_source == "pfam":
        db_domains = load_pfam_domains_from_database(transcript_id, gene)
        if db_domains:
            return db_domains
    
    # Try to get PFAM domains if requested
    if domains_source == "pfam":
        # FIRST: Try direct UniProt query (most reliable, no database required)
        try:
            domains, _ = query_uniprot_domains(gene)  # Unpack tuple, ignore length
            if domains:
                return domains
        except Exception as e:
            logger.debug(f"UniProt domain query failed: {e}")
        
        # SECOND: Try PFAM database if available and py_mut is provided
        if py_mut:
            # Try to query PFAM database directly by gene symbol
            try:
                from ..utils.database import connect_db
                
                db_conn = connect_db()
                
                # Query PFAM domains for this gene
                query = f"""
                    SELECT DISTINCT
                        pfam_id,
                        pfam_name,
                        seq_start,
                        seq_end
                    FROM pfam_uniprot
                    WHERE gene_name = '{gene}'
                    AND seq_start IS NOT NULL
                    AND seq_end IS NOT NULL
                    ORDER BY seq_start
                """
                
                result = db_conn.execute(query).fetchdf()
                
                if not result.empty:
                    logger.debug(f"Found {len(result)} PFAM domains from database")
                    for _, row in result.iterrows():
                        start = int(row['seq_start'])
                        end = int(row['seq_end'])
                        name = row['pfam_name'] if pd.notna(row['pfam_name']) else row['pfam_id']
                        
                        # Validate domain boundaries
                        if start > 0 and end > start and start <= protein_length:
                            domains.append({
                                "start": start,
                                "end": min(end, protein_length),
                                "name": name
                            })
                    
                    db_conn.close()
                    return domains
                
                db_conn.close()
                
            except Exception as e:
                logger.debug(f"Could not query PFAM database directly: {e}")
            
            # Fallback: Check if PFAM annotation is already present in the data
            has_pfam = 'pfam_id' in py_mut.data.columns or 'PFAM_ID' in py_mut.data.columns
            
            # Now try to extract domains for this specific gene from annotated data
            if has_pfam:
                # Filter data for this gene
                gene_data = py_mut.data[py_mut.data['Hugo_Symbol'] == gene]
                
                # Get unique PFAM domains with their positions for this gene
                pfam_cols = ['pfam_id', 'pfam_name', 'seq_start', 'seq_end']
                gene_pfam = gene_data[pfam_cols].dropna(subset=['pfam_id']).drop_duplicates()
                
                if not gene_pfam.empty:
                    for _, row in gene_pfam.iterrows():
                        start = int(row['seq_start']) if pd.notna(row['seq_start']) else 1
                        end = int(row['seq_end']) if pd.notna(row['seq_end']) else protein_length
                        pfam_id = str(row['pfam_id'])
                        pfam_name = str(row['pfam_name']) if pd.notna(row['pfam_name']) else pfam_id
                        
                        # Create domain name combining ID and name
                        name = f"{pfam_name} ({pfam_id})" if pfam_name != pfam_id else pfam_id
                        
                        # Validate and clip to protein length
                        start = max(1, min(start, protein_length))
                        end = max(start, min(end, protein_length))
                        
                        domains.append({"start": start, "end": end, "name": name})
                    
                    logger.debug(f"Retrieved {len(domains)} PFAM domains from data")
                    return domains
    
    # If no domains found, create a generic "full length" domain
    if not domains:
        logger.debug("No domains found, using full-length representation")
        domains.append({
            "start": 1,
            "end": protein_length,
            "name": gene
        })
    
    return domains


def aggregate_mutation_events(data: pd.DataFrame,
                             aa_positions: pd.Series,
                             variant_column: str = "Variant_Classification",
                             sample_column: str = "Tumor_Sample_Barcode",
                             aa_col: str = "HGVSp_Short",
                             count_by: str = "mutations",
                             filter_non_syn: bool = True) -> Dict[Tuple[int, str], Dict]:
    """
    Aggregate mutation events by (position, amino_acid_change) and variant type.
    
    Critical behavior: Aggregates by (position, specific_change) tuple.
    For example, R882H and R882C are counted as SEPARATE lollipops.
    
    Args:
        data: DataFrame with mutation data
        aa_positions: Series with extracted amino acid positions
        variant_column: Name of variant classification column
        sample_column: Name of sample identifier column
        aa_col: Name of protein change column (for labels)
        count_by: "mutations" (count all events) or "samples" (count unique samples)
        filter_non_syn: If True, only count non-synonymous variants
        
    Returns:
        Dictionary mapping (position, conv) -> {
            "total": int,
            "by_class": {variant_type: count},
            "labels": set of HGVS notations,
            "samples": set of sample IDs,
            "position": int,
            "conv": str
        }
    """
    aggregated: Dict[Tuple[int, str], Dict] = defaultdict(lambda: {
        "total": 0,
        "by_class": defaultdict(int),
        "labels": set(),
        "samples": set(),
        "position": 0,
        "conv": ""
    })
    
    # Prepare working dataframe
    work_df = data.copy()
    work_df['aa_position'] = aa_positions
    work_df = work_df.dropna(subset=['aa_position'])
    work_df['aa_position'] = work_df['aa_position'].astype(int)
    
    # Filter to non-synonymous variants only
    if filter_non_syn:
        work_df = work_df[work_df[variant_column].isin(NON_SYNONYMOUS_VARIANTS)]
    
    if count_by == "samples":
        # Count unique samples per (position, conv)
        for _, row in work_df.iterrows():
            position = int(row['aa_position'])
            variant_type = row[variant_column]
            sample = row[sample_column]
            conv = row[aa_col] if (aa_col in row and pd.notna(row[aa_col])) else f"p.{position}"
            
            key = (position, conv)
            
            if sample not in aggregated[key]["samples"]:
                aggregated[key]["samples"].add(sample)
                aggregated[key]["total"] += 1
            
            # Track variant type for this (position, conv)
            aggregated[key]["by_class"][variant_type] += 1
            aggregated[key]["position"] = position
            aggregated[key]["conv"] = conv
            
            # Collect label (protein change)
            aggregated[key]["labels"].add(conv)
    else:
        # Count ALL mutation events (rows), grouped by (Variant_Classification, conv, pos)
        # This matches: prot.snp.sumamry = prot.dat[,.N, .(Variant_Classification, conv, pos)]
        for _, row in work_df.iterrows():
            position = int(row['aa_position'])
            variant_type = row[variant_column]
            sample = row[sample_column]
            conv = row[aa_col] if (aa_col in row and pd.notna(row[aa_col])) else f"p.{position}"
            
            # CRITICAL: Key is (position, conv) - e.g., (882, "p.R882H")
            key = (position, conv)
            
            # Count EVERY mutation event (row in MAF) for this SPECIFIC AA change
            aggregated[key]["total"] += 1
            
            # Track variant type for this (position, conv)
            aggregated[key]["by_class"][variant_type] += 1
            
            # Track position and conv for later use
            aggregated[key]["position"] = position
            aggregated[key]["conv"] = conv
            
            # Track samples for reference (but don't use for count)
            aggregated[key]["samples"].add(sample)
            
            # Collect label (protein change)
            aggregated[key]["labels"].add(conv)
    
    # Convert defaultdicts to regular dicts
    result: Dict[Tuple[int, str], Dict] = {}
    for key, pos_data in aggregated.items():
        result[key] = {
            "total": pos_data["total"],
            "by_class": dict(pos_data["by_class"]),
            "labels": pos_data["labels"],
            "samples": pos_data["samples"],
            "position": pos_data["position"],
            "conv": pos_data["conv"]
        }
    
    logger.debug(f"Aggregated into {len(result)} unique AA changes")
    
    return result


def compute_lollipop_layout(aggregated_data: Dict[Tuple[int, str], Dict],
                           protein_length: int,
                           strategy: str = "top") -> Dict[Tuple[int, str], Dict]:
    """
    Compute coordinates for lollipop visualization.
    
    Args:
        aggregated_data: Aggregated mutation data by (position, conv) key
        protein_length: Length of the protein
        strategy: Layout strategy (default "top" - all circles at stem top)
        
    Returns:
        Dictionary with layout information for each (position, conv) key
    """
    layout = {}
    
    # Get max count for scaling
    max_count = max(data["total"] for data in aggregated_data.values())
    
    for key, data in aggregated_data.items():
        position = data["position"]  # Extract position from data
        total_count = data["total"]
        by_class = data["by_class"]
        
        # Transform counts to visual scale
        # If maxCount <= 5: use count2 = 1 + count (direct mapping)
        # If maxCount > 5: normalize to 1-6 scale: count2 = 1 + (count * (5/maxCount))
        if max_count <= 5:
            height = 1 + total_count
        else:
            height = 1 + (total_count * (5.0 / max_count))
        
        # Calculate circle sizes for each variant class
        circles = []
        
        # Fixed size circles for consistency
        # All circles have the same size regardless of count
        
        if len(by_class) == 1:
            # Single variant type: one circle at the top
            variant_type = list(by_class.keys())[0]
            count = by_class[variant_type]
            
            circles.append({
                "variant_type": variant_type,
                "count": count,
                "x": position,
                "y": height,
                "radius": 1.5  # Fixed size for consistency
            })
        else:
            # Multiple variant types: show the dominant one
            # Sort by count and take the most frequent
            sorted_variants = sorted(by_class.items(), key=lambda x: x[1], reverse=True)
            dominant_variant, dominant_count = sorted_variants[0]
            
            circles.append({
                "variant_type": dominant_variant,
                "count": total_count,  # Total count for legend
                "x": position,
                "y": height,
                "radius": 1.5  # Fixed size for consistency
            })
        
        layout[key] = {
            "total": total_count,
            "height": height,
            "circles": circles,
            "labels": data["labels"],
            "by_class": by_class,  # Keep for legend
            "max_count": max_count,  # Store for reference
            "position": position,  # Store position for easy access
            "conv": data["conv"]  # Store conv for labels
        }
    
    return layout


def apply_label_repel(layout: Dict[Tuple[int, str], Dict],
                     label_top_n: int,
                     protein_length: int) -> List[Dict]:
    """
    Apply label repulsion algorithm to avoid overlaps.
    
    This implements a simplified force-directed label placement:
    1. Select top N AA changes by mutation count
    2. Initialize label positions above the circles
    3. Apply repulsion forces between overlapping labels
    4. Return final label coordinates with leader lines
    
    Args:
        layout: Layout dictionary with (position, conv) keys
        label_top_n: Number of top AA changes to label
        protein_length: Length of protein (for bounds checking)
        
    Returns:
        List of label dictionaries with keys: position, text, x, y, x_line, y_line
    """
    # Sort AA changes by total count and select top N
    sorted_positions = sorted(
        layout.items(),
        key=lambda x: x[1]["total"],
        reverse=True
    )[:label_top_n]
    
    if not sorted_positions:
        return []
    
    # Initialize label positions
    labels = []
    for key, data in sorted_positions:
        position = data["position"]  # Extract position from data
        # Create label text from HGVS notations (cleaned format)
        label_set = data["labels"]
        
        if len(label_set) == 0:
            label_text = str(position)
        elif len(label_set) == 1:
            # Single mutation: extract clean format (R882H style, without "p.")
            raw_label = str(list(label_set)[0]).strip()
            
            # Parse HGVS notation: p.R882H -> R882H or p.R882* -> R882
            # Support various formats: p.R882H, p.R882*, p.R882fs, etc.
            match = re.match(r'^p\.([A-Z]+)(\d+)(.*)$', raw_label, re.IGNORECASE)
            if match:
                # Format: R882H (ref AA + position + variant)
                ref_aa = match.group(1)
                pos = match.group(2)
                variant = match.group(3)
                
                # p.R882* -> R882 (not R882*)
                # p.R882fs -> R882
                # p.R882H -> R882H
                variant_clean = variant.replace('*', '').replace('fs', '').replace('_', '')
                
                if variant_clean and not variant_clean.startswith('del') and not variant_clean.startswith('ins'):
                    label_text = f"{ref_aa}{pos}{variant_clean}"
                else:
                    # For stop codons, frameshifts, etc.: just show position
                    label_text = f"{ref_aa}{pos}"
            else:
                # Fallback: just remove "p." prefix and asterisks
                label_text = raw_label.replace("p.", "").replace("*", "")
        else:
            # Multiple mutations at same position: show the most frequent one
            # Count occurrence of each label and pick the most common
            label_counts = {}
            for label in label_set:
                label_str = str(label).strip()
                # Extract just the variant part for comparison
                match = re.match(r'^p\.([A-Z]+)(\d+)(.+)$', label_str, re.IGNORECASE)
                if match:
                    variant_part = match.group(3)
                    # Count this variant
                    if variant_part not in label_counts:
                        label_counts[variant_part] = 0
                    label_counts[variant_part] += 1
            
            # Get most frequent variant
            if label_counts:
                most_frequent = max(label_counts.items(), key=lambda x: x[1])[0]
                # Get reference AA from first label
                first_label = str(list(label_set)[0]).strip()
                match = re.match(r'^p\.([A-Z]+)(\d+)', first_label, re.IGNORECASE)
                if match:
                    ref_aa = match.group(1)
                    # Show most frequent variant or just position if multiple different ones
                    if len(label_counts) == 1:
                        label_text = f"{ref_aa}{position}{most_frequent}"
                    else:
                        # Multiple different variants: just show position with asterisk
                        label_text = f"{ref_aa}{position}*"
                else:
                    label_text = f"{position}*"
            else:
                label_text = f"{position}*"
        
        # Initial label position: above the highest circle
        max_y = data["height"]
        
        labels.append({
            "position": position,
            "text": label_text,
            "x": position,
            "y": max_y + 0.45,  # Offset for label positioning
            "x_line": position,
            "y_line": max_y,
            "total": data["total"]
        })
    
    # Apply simple repulsion algorithm (10 iterations) - DISABLED by default
    # Labels are positioned directly above without complex repulsion
    # unless explicitly requested
    # for iteration in range(10):
    #     ...
    
    # Keep labels simple (no repulsion by default)
    
    return labels


def _create_lollipop_plot(py_mut: 'PyMutation',
                         gene: str,
                         aa_col: str = "HGVSp_Short",
                         transcript_id: Optional[str] = None,
                         protein_id: Optional[str] = None,
                         domains_source: Optional[str] = "pfam",
                         custom_domains: Optional[List[Dict]] = None,
                         count_by: str = DEFAULT_COUNT_BY,
                         label_top_n: int = DEFAULT_LABEL_TOP_N,
                         show_lollipops: bool = True,
                         figsize: Tuple[int, int] = DEFAULT_LOLLIPOP_FIGSIZE,
                         title: Optional[str] = None) -> Figure:
    """
    Create a lollipop plot for a specific gene.
    
    This visualization shows:
    - Protein domain architecture (PFAM domains)
    - Mutation distribution along the protein sequence (if show_lollipops=True)
    - Mutation hotspots with size proportional to frequency
    - Top mutated positions with labels
    
    Args:
        py_mut: PyMutation object with mutation data
        gene: Gene symbol to visualize
        aa_col: Column name with protein change annotation (HGVSp_Short, Protein_Change)
        transcript_id: Specific transcript ID to use (optional)
        protein_id: Specific protein/UniProt ID to use (optional)
        domains_source: "pfam" to use PFAM domains, or None
        custom_domains: Custom domain definitions (overrides domains_source)
        count_by: "mutations" (count all) or "samples" (unique samples only)
        label_top_n: Number of top positions to label
        show_lollipops: If True (default), show mutation lollipops; if False, show only protein domains
        figsize: Figure size (width, height) in inches
        title: Plot title (auto-generated if None)
        
    Returns:
        matplotlib Figure object
        
    Raises:
        ValueError: If gene not found or no valid mutation data
        
    Examples:
        >>> fig = py_mut.lollipop_plot(gene="FLT3", label_top_n=15)
        >>> fig.savefig("FLT3_lollipop.png", dpi=300, bbox_inches='tight')
        
        >>> # Show only protein domains without mutations
        >>> fig = py_mut.lollipop_plot(gene="TP53", show_lollipops=False)
    """
    start_time = time.time()
    data = py_mut.data
    
    # Step 1: Filter data for the specified gene
    gene_column = "Hugo_Symbol"
    if gene_column not in data.columns:
        raise ValueError(f"Column '{gene_column}' not found in data")
    
    gene_data = data[data[gene_column] == gene].copy()
    
    if len(gene_data) == 0:
        raise ValueError(f"No mutations found for gene '{gene}'")
    
    # Step 2: Extract amino acid positions
    if aa_col not in gene_data.columns:
        # Try alternative column names
        alternative_cols = ["Protein_Change", "AAChange", "HGVSp", "amino_acid_change"]
        found_col = None
        for alt_col in alternative_cols:
            if alt_col in gene_data.columns:
                found_col = alt_col
                logger.info(f"Using alternative protein change column: {alt_col}")
                break
        
        if found_col:
            aa_col = found_col
        else:
            raise ValueError(
                f"Protein change column '{aa_col}' not found. "
                f"Available columns: {list(gene_data.columns)}"
            )
    
    gene_data['aa_position'] = gene_data[aa_col].apply(extract_aa_position)
    
    # Remove entries without valid position
    valid_positions = gene_data['aa_position'].notna().sum()
    
    if valid_positions == 0:
        raise ValueError(f"Could not extract any valid amino acid positions from column '{aa_col}'")
    
    # Step 3: Resolve protein length
    protein_length = resolve_protein_length(
        gene_data,
        gene_data['aa_position'],
        gene=gene,
        protein_id=protein_id,
        transcript_id=transcript_id,
        py_mut=py_mut
    )
    
    # Step 4: Resolve domains
    domains = resolve_domains(
        gene=gene,
        protein_length=protein_length,
        py_mut=py_mut,
        domains_source=domains_source,
        custom_domains=custom_domains,
        transcript_id=transcript_id
    )
    
    # Step 5: Aggregate mutation events (non-synonymous only)
    variant_column = "Variant_Classification"
    sample_column = "Tumor_Sample_Barcode"
    
    aggregated = aggregate_mutation_events(
        gene_data,
        gene_data['aa_position'],
        variant_column=variant_column,
        sample_column=sample_column,
        aa_col=aa_col,
        count_by=count_by,
        filter_non_syn=True
    )
    
    if not aggregated:
        raise ValueError("No aggregated mutations after filtering")
    
    # Step 6: Compute layout
    layout = compute_lollipop_layout(
        aggregated,
        protein_length,
        strategy="top"
    )
    
    # Step 7: Prepare labels with repel algorithm
    labels = apply_label_repel(layout, label_top_n, protein_length)
    
    # Step 8: Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Step 9: Draw protein backbone
    protein_bar_height = 0.4
    
    ax.add_patch(mpatches.Rectangle(
        (0, 0.3),
        protein_length,
        protein_bar_height,
        facecolor='#95a5a6',
        edgecolor='#000000',
        linewidth=1.0,
        zorder=1
    ))
    
    # Step 10: Draw domains with distinct colors
    from matplotlib import cm
    
    # Sort domains by length (longest first) for proper overlapping
    domains_sorted = sorted(domains, key=lambda d: d['end'] - d['start'], reverse=True)
    
    # Use distinct colors for multiple domains
    if len(domains_sorted) > 1:
        domain_colors = cm.get_cmap('Pastel1')(np.linspace(0.1, 0.9, len(domains_sorted)))
    else:
        domain_colors = ['#D5E5F5']
    
    for i, (domain, color) in enumerate(zip(domains_sorted, domain_colors)):
        start = domain["start"]
        end = domain["end"]
        width = end - start
        name = domain["name"]
        
        # Skip full-length representation
        is_full_length = (name == gene and len(domains) == 1)
        
        if not is_full_length:
            # Draw domain rectangle
            ax.add_patch(mpatches.Rectangle(
                (start, 0.25),
                width,
                0.5,
                facecolor=color,
                edgecolor='#000000',
                linewidth=1.0,
                alpha=1.0,
                zorder=2
            ))
            
            # Add domain label if wide enough
            label_x = start + width / 2
            if width > protein_length * 0.05:
                ax.text(label_x, 0.5, name,
                        ha='center', va='center',
                        fontsize=7, fontweight='normal', style='italic',
                        color='#000000',
                        zorder=3)
    
    # Step 11: Draw lollipops (only if show_lollipops=True)
    if show_lollipops:
        for key, data in layout.items():
            position = data["position"]
            
            # Draw stem (from top of backbone bar to lollipop circle)
            stem_height = data["height"]
            ax.plot([position, position], [0.7, stem_height],
                    color='gray', linewidth=1.2, zorder=4, alpha=0.7)
            
            # Draw circles for each variant type
            for circle in data["circles"]:
                color = VARIANT_COLOR_PALETTE.get(circle["variant_type"], VARIANT_COLOR_PALETTE["Other"])
                marker_size = 120  # Fixed size for all lollipops
                
                ax.scatter(
                    circle["x"], circle["y"],
                    s=marker_size,
                    c=[color],
                    edgecolors='#333333',
                    linewidths=0.8,
                    zorder=5,
                    alpha=0.9,
                    marker='o'
                )
    
    # Step 12: Draw labels with leader lines (only if show_lollipops=True)
    if show_lollipops:
        for label in labels:
            # Leader line from lollipop to label
            ax.plot([label["x_line"], label["x"]],
                    [label["y_line"], label["y"]],
                    color='#999999', linewidth=0.5, linestyle=':',
                    alpha=0.5, zorder=3)
            
            # Label text positioned above the lollipop
            label_y = label["y_line"] + 0.45
            
            ax.text(label["x"], label_y, label["text"],
                    ha='center', va='bottom',
                    fontsize=8, fontweight='normal',
                    color='#000000',
                    zorder=6)
    
    # Step 13: Create legend (only if show_lollipops=True)
    if show_lollipops:
        legend_elements = []
        
        # Get unique variant types from layout data
        all_variant_types = set()
        variant_counts: Dict[str, int] = {}
        
        for data in layout.values():
            for vt, count in data["by_class"].items():
                all_variant_types.add(vt)
                variant_counts[vt] = variant_counts.get(vt, 0) + count
        
        # Sort by frequency
        sorted_variants = sorted(all_variant_types, key=lambda x: variant_counts.get(x, 0), reverse=True)
        
        for variant_type in sorted_variants:
            color = VARIANT_COLOR_PALETTE.get(variant_type, VARIANT_COLOR_PALETTE["Other"])
            count = variant_counts.get(variant_type, 0)
            legend_elements.append(
                mpatches.Patch(facecolor=color, edgecolor='#333333', linewidth=0.8,
                              label=f'{variant_type} ({count})')
            )
        
        # Place legend outside plot area
        ax.legend(handles=legend_elements, 
                 loc='center left',
                 bbox_to_anchor=(1.02, 0.5),
                 frameon=True, 
                 fancybox=False,
                 shadow=False,
                 edgecolor='#CCCCCC',
                 title='Variant Type',
                 title_fontsize=9,
                 fontsize=8)
    
    # Step 14: Configure axes
    ax.set_xlim(-protein_length * 0.02, protein_length * 1.02)
    
    # Y-LIMITS: Reduced height to prevent tall lollipops from being cut off
    # If showing lollipops, use dynamic limit based on max height; otherwise use minimal limit
    if show_lollipops and layout:
        max_layout_height = max(data["height"] for data in layout.values())
        # Add 20% padding above tallest lollipop to prevent cutoff
        y_limit = max_layout_height * 1.2
    else:
        # Minimal limit for domains-only view
        y_limit = 1.5
    
    ax.set_ylim(0, y_limit)
    
    # Set aspect to 'auto' to allow proper scaling, but ensure circles remain circular
    # by adjusting data aspect ratio
    ax.set_aspect('auto')
    
    # Axis labels with cleaner formatting
    # X-axis: show protein length in the label
    if transcript_id:
        xlabel = f'{gene} Protein Position (AA)\n{transcript_id} | {protein_length} AA'
    else:
        xlabel = f'{gene} Protein Position (AA) | {protein_length} AA'
    
    ax.set_xlabel(xlabel, fontsize=9, fontweight='normal', color='#444444')
    
    # Y-axis: No label (meaning is implicit from plot type)
    ax.set_ylabel('', fontsize=10, fontweight='normal', color='#444444')
    
    # Y-axis ticks: Show real counts (only if showing lollipops)
    if show_lollipops and aggregated:
        max_count = max(data["total"] for data in aggregated.values())
        
        # Calculate max_height for tick positions
        if max_count <= 5:
            max_height = 1 + max_count
        else:
            max_height = 1 + (max_count * (5.0 / max_count))
        
        ax.set_yticks([0, max_height])
        ax.set_yticklabels(['0', str(int(max_count))])
    else:
        # Minimal ticks for domains-only view
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0', '1'])
    
    # Title
    if title is None:
        if show_lollipops:
            # Calculate somatic mutation rate
            if hasattr(py_mut, 'samples') and py_mut.samples:
                total_samples_in_data = len(py_mut.samples)
            elif 'Tumor_Sample_Barcode' in py_mut.data.columns:
                total_samples_in_data = py_mut.data['Tumor_Sample_Barcode'].nunique()
            else:
                total_samples_in_data = gene_data['Tumor_Sample_Barcode'].nunique() if 'Tumor_Sample_Barcode' in gene_data.columns else 1
            
            # Get unique samples with non-synonymous mutations
            gene_data_ns = gene_data[gene_data[variant_column].isin(NON_SYNONYMOUS_VARIANTS)]
            n_samples = gene_data_ns['Tumor_Sample_Barcode'].nunique() if 'Tumor_Sample_Barcode' in gene_data_ns.columns else 0
            
            mutation_rate = (n_samples / total_samples_in_data * 100) if total_samples_in_data > 0 else 0
            title = f'{gene} : [Somatic Mutation Rate: {mutation_rate:.2f}%]'
        else:
            # Domains-only view
            title = f'{gene} : Protein Domain Architecture'
        
        # Subtitle with transcript ID
        if transcript_id:
            subtitle = transcript_id
        else:
            subtitle = None
    else:
        subtitle = None
    
    # Main title with bold font
    ax.set_title(title, fontsize=12, fontweight='bold', fontstyle='normal', 
                loc='left', pad=12)
    
    # Add subtitle if provided (transcript ID)
    if subtitle:
        ax.text(0.0, 1.0, subtitle, 
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=10, fontweight='normal', color='#000000')
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # No grid
    ax.grid(False)
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=8, colors='#666666', width=0.8)
    ax.tick_params(axis='both', which='minor', labelsize=7, colors='#666666', width=0.5)
    
    # X-axis tick spacing
    if protein_length > 1000:
        tick_spacing = 200
    elif protein_length > 500:
        tick_spacing = 100
    elif protein_length > 200:
        tick_spacing = 50
    else:
        tick_spacing = 25
    
    ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
    ax.set_xticks([0] + list(range(tick_spacing, protein_length + 1, tick_spacing)))
    
    plt.tight_layout()
    
    # Log creation time and statistics
    elapsed_time = time.time() - start_time
    if show_lollipops:
        logger.info(f"Lollipop plot for {gene} created in {elapsed_time:.2f}s ({len(gene_data)} mutations, {len(aggregated)} AA changes)")
    else:
        logger.info(f"Protein domain plot for {gene} created in {elapsed_time:.2f}s")
    
    return fig
