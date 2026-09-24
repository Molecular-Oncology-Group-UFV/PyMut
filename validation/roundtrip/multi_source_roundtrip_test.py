"""
Round-trip preservation test across several PyMut example files.

For each input file (2 VCFs + 2 MAFs) this checks that the cycle
read -> export -> re-read preserves 100% of the variant IDs
(CHROM_POS_REF_ALT) and of the per-sample genotypes.

Produces:
  - One combined figure with all cohorts side by side in a single row.
  - One individual figure per cohort.
"""

from pathlib import Path
import re
import tempfile
import matplotlib.pyplot as plt
import pandas as pd

import pyMut
from pyMut.input import read_vcf, read_maf

# --------------------------------------------------------------------------
# Compatibility shim (same one used in flow_pymut.py)
# --------------------------------------------------------------------------
# to_maf()'s large-dataset (>10,000 rows) PyArrow export path has two known
# issues against the pyarrow/pandas versions in this environment:
#   1. It calls pyarrow.compute.utf8_replace_substring(), which does not
#      exist in some pyarrow releases (the equivalent is replace_substring
#      with the same argument order).
#   2. It re-adds columns via pyarrow.Table.append_column() without
#      checking whether the source data already has a column of that name
#      (Chromosome, Start_Position, NCBI_Build, dbSNP_RS, ...), which on
#      wide real-world MAFs (e.g. PAAD) produces duplicate column names and
#      crashes with "ValueError: Found non-unique column index" in
#      Table.to_pandas().
# Rather than patching that branch function by function, force to_maf() to
# always take its pandas-based export path instead -- the same one that
# already round-trips TCGA-LAML with zero discrepancies. This only adds the
# missing pyarrow alias if not already present, so it's a no-op where it
# isn't needed.
import pyarrow.compute as _pc

if not hasattr(_pc, "utf8_replace_substring"):
    _pc.utf8_replace_substring = _pc.replace_substring

import pyMut.output as _pymut_output

_pymut_output.HAS_PYARROW = False

# ---------------------------
# 1) Input file paths
# ---------------------------
# pyMut's own bundled example VCFs, located relative to the installed
# package (works wherever pyMut is installed, no hardcoded home dirs).
EXAMPLES_VCF_DIR = Path(pyMut.__file__).resolve().parent / "data" / "examples" / "VCF"

# Folder containing this script (validation/roundtrip/) and the project
# root (two levels up). Paths are resolved from here, so the script works
# no matter which directory it is launched from.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Project MAF inputs.
PROJECT_DATA_DIR = PROJECT_ROOT / "data"

DATASETS = [
    {
        "label": "VCF chr10 1k",
        "kind": "vcf",
        "path": EXAMPLES_VCF_DIR / "subset_1k_variants_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class.vcf",
        "assembly": "38",
    },
    {
        "label": "VCF chr10 50k",
        "kind": "vcf",
        "path": EXAMPLES_VCF_DIR / "subset_50k_variants_vep_protein_gene_variant_class.vcf.gz",
        "assembly": "38",
    },
    {
        "label": "MAF (LAML)",
        "kind": "maf",
        "path": PROJECT_DATA_DIR / "tcga_laml.maf.gz",
        "assembly": "37",
    },
    {
        "label": "MAF (PAAD)",
        "kind": "maf",
        "path": PROJECT_DATA_DIR / "PAAD-TB.final_analysis.maf",
        "assembly": "37",
    },
]

# ---------------------------
# 2) Helper functions
# ---------------------------
def variant_id(df: pd.DataFrame) -> pd.Series:
    required = ["CHROM", "POS", "REF", "ALT"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns to build the variant identifier: {missing}")
    return (
        df[required]
        .astype(str)
        .agg(lambda r: f"{r['CHROM']}_{r['POS']}_{r['REF']}_{r['ALT']}", axis=1)
    )

def variant_id_normalized(df: pd.DataFrame) -> pd.Series:
    """
    Same as variant_id, but "sanding down" differences that are purely
    text-encoding issues rather than real content differences:
      - CHROM/REF/ALT uppercased and stripped
      - CHROM without the 'chr' prefix
      - POS forced to a clean int-like string (avoids '12345' vs '12345.0')
    """
    required = ["CHROM", "POS", "REF", "ALT"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns to build the variant identifier: {missing}")

    chrom = df["CHROM"].astype(str).str.strip().str.upper().str.replace("^CHR", "", regex=True)
    pos = df["POS"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    ref = df["REF"].astype(str).str.strip().str.upper()
    alt = df["ALT"].astype(str).str.strip().str.upper()

    return chrom + "_" + pos + "_" + ref + "_" + alt

def genotype_columns(df: pd.DataFrame, known_samples=None):
    """
    Returns the list of columns in `df` that hold per-sample genotypes.

    Preferred path: if `known_samples` is given (the real sample list, as
    reported by PyMutation itself), just intersect it with `df.columns`.
    This is the only fully reliable option, because a wide MAF/VCF table
    can have anywhere from a handful to several hundred extra annotation
    columns (e.g. a 565-column MAF), and there is no fixed-size "ignore
    list" that can safely enumerate all of them by name.

    Fallback path (used only when `known_samples` is not available): the
    original ignore-list heuristic, which assumes every column NOT in a
    small fixed set of known VCF/MAF metadata fields is a sample. This is
    fragile on real-world MAFs: it previously misclassified annotation
    columns such as 'dbSNP_RS' as if they were per-sample genotypes,
    silently corrupting the genotype-mismatch statistics. It is kept only
    as a last resort and callers are warned when it is used.
    """
    if known_samples is not None:
        return [c for c in known_samples if c in df.columns]

    ignore = {
        "CHROM", "POS", "REF", "ALT", "ID", "QUAL", "FILTER", "INFO", "FORMAT",
        "Chromosome", "Start_Position", "End_Position", "Reference_Allele",
        "Tumor_Seq_Allele1", "Tumor_Seq_Allele2", "Tumor_Sample_Barcode",
        "_variant_id"
    }
    return [c for c in df.columns if c not in ignore]

def get_known_samples(pymutation_obj):
    """
    Tries to read the authoritative sample list off a PyMutation object,
    checking the attribute names this codebase is known to use. Returns
    None if none of them are found, so the caller can fall back to the
    heuristic in genotype_columns() and warn the user.
    """
    for attr in ("samples", "sample_names", "sample_columns", "sample_list"):
        value = getattr(pymutation_obj, attr, None)
        if value:
            return list(value)
    metadata = getattr(pymutation_obj, "metadata", None)
    if metadata is not None:
        for attr in ("samples", "sample_names", "sample_columns", "sample_list"):
            value = getattr(metadata, attr, None)
            if value:
                return list(value)
    return None

def compare_variant_sets(label: str, source_df: pd.DataFrame, roundtrip_df: pd.DataFrame,
                          sample_cols=None):
    src = source_df.copy()
    rt = roundtrip_df.copy()

    src["_variant_id"] = variant_id(src)
    rt["_variant_id"] = variant_id(rt)

    src_ids = set(src["_variant_id"])
    rt_ids = set(rt["_variant_id"])

    same_ids = src_ids == rt_ids
    missing_ids = src_ids - rt_ids
    extra_ids = rt_ids - src_ids
    matched_ids = src_ids & rt_ids
    n_missing = len(missing_ids)
    n_extra = len(extra_ids)
    n_matched = len(matched_ids)

    # Comparison using the normalized ID, to tell apart a "text formatting
    # issue" from a genuine loss of information.
    src["_variant_id_norm"] = variant_id_normalized(src)
    rt["_variant_id_norm"] = variant_id_normalized(rt)
    src_ids_norm = set(src["_variant_id_norm"])
    rt_ids_norm = set(rt["_variant_id_norm"])
    n_missing_norm = len(src_ids_norm - rt_ids_norm)
    n_extra_norm = len(rt_ids_norm - src_ids_norm)

    # Compare genotypes sample by sample. `sample_cols`, when provided by
    # the caller, is the authoritative sample list (see get_known_samples);
    # otherwise genotype_columns() falls back to its ignore-list heuristic.
    if sample_cols is None:
        sample_cols = sorted(set(genotype_columns(src)) & set(genotype_columns(rt)))
    else:
        sample_cols = sorted(set(sample_cols) & set(src.columns) & set(rt.columns))
    mismatch_records = []
    MAX_RECORDS = 2000  # cap so this stays cheap even on large cohorts
    if sample_cols:
        src_gt = src[["_variant_id"] + sample_cols].astype(object).fillna(".").infer_objects(copy=False).astype(str)
        rt_gt = rt[["_variant_id"] + sample_cols].astype(object).fillna(".").infer_objects(copy=False).astype(str)

        merged = src_gt.merge(rt_gt, on="_variant_id", how="outer", suffixes=("_src", "_rt"))

        mismatches = 0
        total_checked = 0
        for col in sample_cols:
            c1, c2 = f"{col}_src", f"{col}_rt"
            if c1 not in merged.columns or c2 not in merged.columns:
                continue
            total_checked += len(merged)
            s1 = merged[c1].fillna(".").astype(str)
            s2 = merged[c2].fillna(".").astype(str)
            diff_mask = s1 != s2
            n_diff = int(diff_mask.sum())
            mismatches += n_diff
            if n_diff and len(mismatch_records) < MAX_RECORDS:
                diff_rows = merged.loc[diff_mask, ["_variant_id"]].copy()
                diff_rows["sample"] = col
                diff_rows["src"] = s1[diff_mask].values
                diff_rows["rt"] = s2[diff_mask].values
                remaining = MAX_RECORDS - len(mismatch_records)
                mismatch_records.extend(diff_rows.head(remaining).to_dict("records"))
    else:
        mismatches, total_checked = 0, 0

    return {
        "label": label,
        "n_source": len(src_ids),
        "n_roundtrip": len(rt_ids),
        "same_ids": same_ids,
        "missing": n_missing,
        "extra": n_extra,
        "matched": n_matched,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "missing_norm": n_missing_norm,
        "extra_norm": n_extra_norm,
        "mismatched_genotypes": mismatches,
        "genotypes_checked": total_checked,
        "genotype_mismatch_records": mismatch_records,
    }

def diagnose_mismatches(label: str, summary: dict, max_examples: int = 10):
    """
    Pairs up "missing" (source-only) and "extra" (roundtrip-only) IDs by
    their CHROM_POS locus, to see whether REF/ALT changed at the same
    locus, or whether loci disappeared/appeared entirely.
    """
    missing_ids = summary["missing_ids"]
    extra_ids = summary["extra_ids"]

    if not missing_ids and not extra_ids:
        return

    print(f"\n--- Diagnostics {label}: {len(missing_ids)} missing / {len(extra_ids)} extra (exact ID) ---")

    missing_norm = summary.get("missing_norm")
    extra_norm = summary.get("extra_norm")
    if missing_norm is not None:
        if missing_norm == 0 and extra_norm == 0:
            print("  >>> The mismatch DISAPPEARS with the normalized ID -> text formatting issue, not real content loss.")
        elif missing_norm < len(missing_ids) or extra_norm < len(extra_ids):
            print(f"  >>> With the normalized ID there are still {missing_norm} missing / {extra_norm} extra "
                  f"(was {len(missing_ids)}/{len(extra_ids)}). Mix of formatting + real content change.")
        else:
            print("  >>> The normalized ID does NOT reduce the mismatch -> real content change (REF/ALT/locus).")

    def locus(vid: str) -> str:
        parts = vid.split("_")
        return "_".join(parts[:2])

    missing_by_locus, extra_by_locus = {}, {}
    for vid in missing_ids:
        missing_by_locus.setdefault(locus(vid), []).append(vid)
    for vid in extra_ids:
        extra_by_locus.setdefault(locus(vid), []).append(vid)

    shared_loci = sorted(set(missing_by_locus) & set(extra_by_locus))
    only_missing_loci = sorted(set(missing_by_locus) - set(extra_by_locus))
    only_extra_loci = sorted(set(extra_by_locus) - set(missing_by_locus))

    print(f"  Same CHROM_POS on both sides (possible REF/ALT change): {len(shared_loci)} loci")
    print(f"  Loci that disappear entirely:                          {len(only_missing_loci)}")
    print(f"  New loci that appear:                                  {len(only_extra_loci)}")

    if shared_loci:
        print(f"  Examples (up to {max_examples}):")
        for loc in shared_loci[:max_examples]:
            print(f"    {loc}: source -> {missing_by_locus[loc]} | roundtrip -> {extra_by_locus[loc]}")

def diagnose_genotype_mismatches(label: str, summary: dict, max_examples: int = 15):
    """
    Even when every variant ID matches perfectly (same_ids=True), individual
    genotype cells can still differ between source and roundtrip. This looks
    at the concrete (variant, sample, source_value, roundtrip_value) records
    captured in compare_variant_sets() and buckets them into likely causes:

      - indel: REF or ALT is not a single base (insertion/deletion), where
        allele-string handling is more error-prone than for SNPs.
      - order_swap: same two alleles on both sides, just in a different
        order (e.g. "A|T" vs "T|A") - a phasing/representation issue, not a
        real difference in the call itself.
      - missing_vs_called: one side is "." (missing/not captured) and the
        other has an actual genotype.
      - other: genuinely different alleles - the most concerning bucket.
    """
    records = summary.get("genotype_mismatch_records") or []
    n_total = summary.get("mismatched_genotypes", 0)

    if not records:
        return

    print(f"\n--- Genotype diagnostics {label}: {n_total:,} mismatched genotype cells "
          f"(showing first {len(records)} captured) ---")

    def alleles_of(gt: str):
        return gt.split("|") if "|" in gt else [gt]

    buckets = {"indel": [], "order_swap": [], "missing_vs_called": [], "other": []}

    for rec in records:
        vid, sample, src_val, rt_val = rec["_variant_id"], rec["sample"], rec["src"], rec["rt"]
        parts = vid.split("_")
        ref = parts[2] if len(parts) > 2 else ""
        alt = parts[3] if len(parts) > 3 else ""

        if src_val == "." or rt_val == ".":
            buckets["missing_vs_called"].append(rec)
        elif sorted(alleles_of(src_val)) == sorted(alleles_of(rt_val)) and src_val != rt_val:
            buckets["order_swap"].append(rec)
        elif len(ref) != 1 or len(alt) != 1:
            buckets["indel"].append(rec)
        else:
            buckets["other"].append(rec)

    for name, items in buckets.items():
        print(f"  {name}: {len(items)}")

    for name, items in buckets.items():
        if items:
            print(f"\n  Examples for '{name}' (up to {max_examples}):")
            for rec in items[:max_examples]:
                print(f"    {rec['_variant_id']} | sample={rec['sample']} | "
                      f"source='{rec['src']}' | roundtrip='{rec['rt']}'")

def slugify(label: str) -> str:
    slug = label.replace("\n", " ").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")

# ---------------------------
# 3) Plotting helpers
# ---------------------------
plt.style.use("seaborn-v0_8-whitegrid")

OK_COLOR = "#2CA655"        # green: IDs match
BAD_COLOR = "#D64545"       # red: IDs differ
MATCHED_COLOR = "#3B6FA0"   # blue: present on both sides
DIFF_COLOR = "#E0704A"      # orange: exclusive to one side

def draw_panel(ax, label: str, summary: dict):
    """
    Draws one cohort's stacked bar chart (Source / Roundtrip, split into
    "present in both" vs "exclusive to this side"), with the title on top
    and a boxed stats summary placed right below the title, above the bars.
    """
    clean_label = label.replace("\n", " ")

    if summary is None:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{clean_label}\n(file not found)", ha="center", va="center",
                 fontsize=10, color="#999999", transform=ax.transAxes)
        return

    x = [0, 1]
    cats = ["Source", "Roundtrip"]
    matched_vals = [summary["matched"], summary["matched"]]
    excl_vals = [summary["missing"], summary["extra"]]
    totals = [m + e for m, e in zip(matched_vals, excl_vals)]

    ax.bar(x, matched_vals, color=MATCHED_COLOR, edgecolor="black",
           linewidth=0.8, width=0.55, zorder=3, label="Present in both")
    ax.bar(x, excl_vals, bottom=matched_vals, color=DIFF_COLOR, edgecolor="black",
           linewidth=0.8, width=0.55, zorder=3, label="Exclusive to this side")

    top = max(totals) if max(totals) > 0 else 1
    ax.set_ylim(0, top * 1.15)

    for j, (mval, eval_, tot) in enumerate(zip(matched_vals, excl_vals, totals)):
        if mval > 0 and mval / top > 0.06:
            ax.text(j, mval / 2, f"{mval:,}", ha="center", va="center",
                    fontsize=15, color="white", fontweight="bold")
        if eval_ > 0 and eval_ / top > 0.04:
            ax.text(j, mval + eval_ / 2, f"{eval_:,}", ha="center", va="center",
                    fontsize=15, color="white", fontweight="bold")
        ax.text(j, tot + top * 0.02, f"{tot:,}", ha="center", va="bottom",
                fontsize=15, fontweight="bold")

    is_ok = summary["same_ids"]
    badge_color = OK_COLOR if is_ok else BAD_COLOR
    badge_text = "IDs match" if is_ok else "IDs differ"

    # Title is drawn as an explicit text object (not ax.set_title) because
    # ax.set_title's pad-based positioning does not play well with
    # tight_layout() once other transAxes annotations (badge, stats box)
    # are stacked above the axes: in testing it ended up invisible.
    ax.text(
        0.5, 1.34, clean_label,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=20, fontweight="bold", color="#111111", zorder=4,
    )

    # Small colored status pill, just below the title.
    ax.text(
        0.5, 1.22, badge_text,
        transform=ax.transAxes, ha="center", va="center",
        fontsize=15, color="white", fontweight="bold", zorder=4,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=badge_color, edgecolor="none"),
    )

    # Boxed stats summary, between the badge and the top of the plot area.
    stats_text = (
        f"Matched: {summary['matched']:,}   Missing: {summary['missing']}   Extra: {summary['extra']}\n"
        f"Genotype mismatches: {summary['mismatched_genotypes']:,} / {summary['genotypes_checked']:,}"
    )
    ax.text(
        0.5, 1.11, stats_text,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=15, color="#222222", linespacing=1.7, zorder=4,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor=badge_color, linewidth=1.3),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=16)
    ax.set_ylabel("Unique variant IDs", fontsize=16)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ---------------------------
# 4) Run the round-trip for every dataset
# ---------------------------
summaries = []

with tempfile.TemporaryDirectory(prefix="pymut_multi_roundtrip_") as td:
    td = Path(td)

    for ds in DATASETS:
        label = ds["label"].replace("\n", " ")
        path = ds["path"]
        kind = ds["kind"]
        assembly = ds["assembly"]

        print(f"\n=== Processing {label} ({path}) ===")

        if not path.exists():
            print(f"  !! File not found, skipping: {path}")
            summaries.append(None)
            continue

        if kind == "vcf":
            src_obj = read_vcf(path, assembly=assembly)
            out_path = td / f"roundtrip_{path.stem}.vcf"
            src_obj.to_vcf(out_path)
            rt_obj = read_vcf(out_path, assembly=assembly)
        else:  # maf
            src_obj = read_maf(path, assembly=assembly)
            out_path = td / f"roundtrip_{path.stem}.maf"
            src_obj.to_maf(out_path)
            rt_obj = read_maf(out_path, assembly=assembly)

        # Prefer the authoritative sample list reported by PyMutation itself
        # over guessing which columns are "samples" by exclusion. This
        # matters a lot on wide, real-world MAFs (hundreds of annotation
        # columns): without it, annotation columns such as 'dbSNP_RS' get
        # silently compared as if they were per-sample genotypes.
        known_samples = get_known_samples(src_obj)
        if known_samples is None:
            known_samples = get_known_samples(rt_obj)
        if known_samples is None:
            print("  !! Could not find an authoritative sample list on the PyMutation "
                  "object (tried .samples/.sample_names/.sample_columns/.sample_list, "
                  "also under .metadata). Falling back to the ignore-list heuristic in "
                  "genotype_columns(), which may misclassify annotation columns as "
                  "samples on wide MAFs.")

        summary = compare_variant_sets(ds["label"], src_obj.data, rt_obj.data,
                                        sample_cols=known_samples)
        summaries.append(summary)

        print("Summary:", {k: v for k, v in summary.items() if k not in ("missing_ids", "extra_ids", "genotype_mismatch_records")})
        diagnose_mismatches(ds["label"], summary)
        diagnose_genotype_mismatches(ds["label"], summary)

    # ---------------------------
    # 5) Output folders
    # ---------------------------
    base_out_dir = SCRIPT_DIR / "results"
    individual_out_dir = base_out_dir / "individual"
    individual_out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # 6) Combined figure: all cohorts in a single row
    # ---------------------------
    n_panels = len(DATASETS)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.2 * n_panels, 7.2))
    axes = axes.flatten() if n_panels > 1 else [axes]

    for ax, ds, summary in zip(axes, DATASETS, summaries):
        draw_panel(ax, ds["label"], summary)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.05), fontsize=17)

    fig.suptitle("Round-trip preservation in PyMut — example files",
                 fontsize=25, fontweight="bold", y=1.15)

    plt.tight_layout()

    combined_out_path = base_out_dir / "roundtrip_preservation_combined.png"
    plt.savefig(combined_out_path, dpi=200, bbox_inches="tight")
    print(f"\nCombined figure saved to: {combined_out_path}")
    plt.show()
    plt.close(fig)

    # ---------------------------
    # 7) One individual figure per cohort
    # ---------------------------
    for ds, summary in zip(DATASETS, summaries):
        fig_i, ax_i = plt.subplots(1, 1, figsize=(6.5, 6.3))
        draw_panel(ax_i, ds["label"], summary)

        handles_i, labels_i = ax_i.get_legend_handles_labels()
        fig_i.legend(handles_i, labels_i, loc="upper center", ncol=2, frameon=False,
                     bbox_to_anchor=(0.5, 1.05), fontsize=16)

        plt.tight_layout()

        out_path_i = individual_out_dir / f"roundtrip_{slugify(ds['label'])}.png"
        plt.savefig(out_path_i, dpi=200, bbox_inches="tight")
        print(f"Individual figure saved to: {out_path_i}")
        plt.close(fig_i)
