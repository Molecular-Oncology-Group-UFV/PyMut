import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from unittest.mock import patch, Mock
from pathlib import Path
from pandas.errors import ParserError

from yaml.parser import ParserError

from src.pyMut.input import read_maf, HAS_PYARROW
from src.pyMut.core import PyMutation


class TestReadMAFAssemblyValidation:
    """Test assembly parameter validation."""
    
    def test_valid_assemblies(self):
        """Test that valid assembly values ('37', '38') are accepted."""
        maf_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf"
        
        # Should not raise for valid assemblies
        result_37 = read_maf(maf_path, assembly="37")
        assert isinstance(result_37, PyMutation)
        assert result_37.metadata.assembly == "37"
        
        result_38 = read_maf(maf_path, assembly="38")
        assert isinstance(result_38, PyMutation)
        assert result_38.metadata.assembly == "38"
    
    def test_invalid_assembly_raises_error(self):
        """Test that invalid assembly values raise ValueError."""
        maf_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf"
        
        with pytest.raises(ValueError, match="Assembly parameter must be either '37' or '38'"):
            read_maf(maf_path, assembly="36")
        
        with pytest.raises(ValueError, match="Assembly parameter must be either '37' or '38'"):
            read_maf(maf_path, assembly="hg19")


class TestReadMAFFileOpening:
    """Test file opening for both .maf and .maf.gz files."""
    
    def test_read_maf_plain_text(self):
        """Test reading plain text .maf file."""
        maf_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf"
        
        result = read_maf(maf_path, assembly="38")
        assert isinstance(result, PyMutation)
        assert len(result.data) > 0
        assert result.metadata.source_format == "MAF"
    
    def test_read_maf_gzipped(self):
        """Test reading gzipped .maf.gz file."""
        maf_gz_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf.gz"
        
        result = read_maf(maf_gz_path, assembly="38")
        assert isinstance(result, PyMutation)
        assert len(result.data) > 0
        assert result.metadata.source_format == "MAF"
    
    def test_file_not_found_raises_error(self):
        """Test that non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_maf("non_existent_file.maf", assembly="38")


class TestReadMAFCommentSeparation:
    """Test comment separation and body reading."""
    
    def test_comment_separation(self):
        """Test that comments are properly separated from data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            # Write MAF with comments
            f.write("# This is a comment\n")
            f.write("# Another comment\n")
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\n")
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2802\n")
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38")
            
            # Should have processed the data correctly
            assert isinstance(result, PyMutation)
            assert len(result.data) == 1
            
            # Comments should be stored in metadata
            assert result.metadata.notes is not None
            assert "This is a comment" in result.metadata.notes
        finally:
            os.unlink(temp_path)
    
    def test_no_comments_handling(self):
        """Test handling of MAF files without comments."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            # Write MAF without comments
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\n")
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2802\n")
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38")
            
            # Should still work without comments
            assert isinstance(result, PyMutation)
            assert len(result.data) == 1
        finally:
            os.unlink(temp_path)


class TestReadMAFColumnValidation:
    """Test required columns validation and standardization."""
    
    def test_required_columns_present(self):
        """Test that required columns are present after reading."""
        maf_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf"
        
        result = read_maf(maf_path, assembly="38")
        
        # Check for VCF-like columns that should be generated
        vcf_columns = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER"]
        for col in vcf_columns:
            assert col in result.data.columns, f"Missing VCF-like column: {col}"
    
    def test_missing_required_columns_raises_error(self):
        """Test that missing required columns raise ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            # Write MAF missing required Reference_Allele column
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\n")
            f.write("TP53\t17\t7674220\tC\tT\tTCGA-AB-2802\n")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Missing required columns in MAF"):
                read_maf(temp_path, assembly="38")
        finally:
            os.unlink(temp_path)
    
    def test_case_insensitive_column_matching(self):
        """Test that column matching is case-insensitive."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            # Write MAF with mixed case column names
            f.write("hugo_symbol\tchromosome\tstart_position\treference_allele\ttumor_seq_allele1\ttumor_seq_allele2\ttumor_sample_barcode\n")
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2802\n")
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38")
            
            # Should work with case-insensitive matching
            assert isinstance(result, PyMutation)
            assert len(result.data) == 1
        finally:
            os.unlink(temp_path)


class TestReadMAFEngineHandling:
    """Test PyArrow and pandas engine handling."""
    
    def test_pyarrow_engine_path(self):
        """Test MAF reading using PyArrow engine when available."""
        if not HAS_PYARROW:
            pytest.skip("PyArrow not available")
        
        maf_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf"
        
        result = read_maf(maf_path, assembly="38")
        
        # Should successfully read with PyArrow
        assert isinstance(result, PyMutation)
        assert len(result.data) > 0
    
    @patch('src.pyMut.input.HAS_PYARROW', False)
    def test_pandas_fallback_engine(self):
        """Test fallback to pandas 'c' engine when PyArrow fails."""
        maf_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf"
        
        result = read_maf(maf_path, assembly="38")
        
        # Should still work with pandas fallback
        assert isinstance(result, PyMutation)
        assert len(result.data) > 0
    
    @patch('pandas.read_csv')
    def test_pyarrow_failure_fallback(self, mock_read_csv):
        """Test fallback when PyArrow engine fails."""
        # Mock PyArrow failure and pandas success
        mock_read_csv.side_effect = [
            ValueError("PyArrow error"),  # First call fails
            pd.DataFrame({  # Second call succeeds
                'Hugo_Symbol': ['TP53'],
                'Chromosome': ['17'],
                'Start_Position': [7674220],
                'Reference_Allele': ['C'],
                'Tumor_Seq_Allele1': ['C'],
                'Tumor_Seq_Allele2': ['T'],
                'Tumor_Sample_Barcode': ['TCGA-AB-2802']
            })
        ]
        
        # Use a temp directory to avoid cache interference
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            maf_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf"
            
            result = read_maf(maf_path, assembly="38", cache_dir=cache_dir)
            
            # Should have fallen back to pandas engine
            assert isinstance(result, PyMutation)
            assert mock_read_csv.call_count == 2  # Called twice due to fallback


class TestReadMAFVCFFieldGeneration:
    """Test generation of VCF-like fields from MAF data."""
    
    def test_chromosome_formatting(self):
        """Test that Chromosome is formatted correctly as CHROM."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\n")
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2802\n")
            f.write("KRAS\tX\t25380275\tG\tG\tA\tTCGA-AB-2803\n")
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38")
            
            # Should have formatted chromosome names
            assert "CHROM" in result.data.columns
            chroms = result.data["CHROM"].tolist()
            assert "chr17" in chroms or "17" in chroms
            assert "chrX" in chroms or "X" in chroms
        finally:
            os.unlink(temp_path)
    
    def test_position_conversion(self):
        """Test that Start_Position is converted to integer POS."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\n")
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2802\n")
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38")
            
            # Should have integer POS column
            assert "POS" in result.data.columns
            assert result.data["POS"].dtype == "int64"
            assert result.data["POS"].iloc[0] == 7674220
        finally:
            os.unlink(temp_path)
    
    def test_id_generation_from_dbsnp(self):
        """Test ID generation from dbSNP_RS column."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\tdbSNP_RS\n")
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2802\trs1042522\n")
            f.write("KRAS\t12\t25380275\tG\tG\tA\tTCGA-AB-2803\t.\n")
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38")
            
            # Should have ID column with formatted RS values
            assert "ID" in result.data.columns
            ids = result.data["ID"].tolist()
            assert "1042522" in ids
            assert "." in ids
        finally:
            os.unlink(temp_path)
    
    def test_id_default_when_no_dbsnp(self):
        """Test ID defaults to '.' when no dbSNP_RS column."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\n")
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2802\n")
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38")
            
            # Should have ID column with default values
            assert "ID" in result.data.columns
            assert all(id_val == "." for id_val in result.data["ID"])
        finally:
            os.unlink(temp_path)
    
    def test_ref_alt_generation(self):
        """Test REF and ALT generation from allele columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\n")
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2802\n")  # ALT from Allele2
            f.write("KRAS\t12\t25380275\tG\tA\t\tTCGA-AB-2803\n")   # ALT from Allele1 when Allele2 is empty
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38")
            
            # Should have REF and ALT columns
            assert "REF" in result.data.columns
            assert "ALT" in result.data.columns
            
            # Check specific values
            df = result.data.set_index("Hugo_Symbol")
            assert df.loc["TP53", "REF"] == "C"
            assert df.loc["TP53", "ALT"] == "T" # FP: ALT from Allele2
            assert df.loc["KRAS", "REF"] == "G"
            assert df.loc["KRAS", "ALT"] == "A" # FP: ALT from Allele1 when Allele2 is empty
        finally:
            os.unlink(temp_path)
    
    def test_default_qual_filter_fields(self):
        """Test that QUAL and FILTER fields default to '.' when not present."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\n")
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2802\n")
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38")
            
            # Should have default QUAL and FILTER values
            assert "QUAL" in result.data.columns
            assert "FILTER" in result.data.columns
            assert all(qual == "." for qual in result.data["QUAL"])
            assert all(filter_val == "." for filter_val in result.data["FILTER"])
        finally:
            os.unlink(temp_path)


class TestReadMAFSampleExpansion:
    """Test expansion of mutations into sample columns."""
    
    def test_sample_column_creation(self):
        """Test that sample columns are created for each unique sample."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\n")
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2802\n")
            f.write("KRAS\t12\t25380275\tG\tG\tA\tTCGA-AB-2803\n")
            f.write("PIK3CA\t3\t178936091\tG\tG\tA\tTCGA-AB-2802\n")  # Same sample as first
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38")
            
            # Should have created columns for unique samples
            expected_samples = ["TCGA-AB-2802", "TCGA-AB-2803"]
            for sample in expected_samples:
                assert sample in result.data.columns, f"Missing sample column: {sample}"
            
            # Should track samples correctly
            assert set(result.samples) == set(expected_samples)
        finally:
            os.unlink(temp_path)
    
    def test_sample_genotype_assignment(self):
        """Test that genotypes are correctly assigned to sample columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\n")
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2802\n")
            f.write("KRAS\t12\t25380275\tG\tG\tA\tTCGA-AB-2803\n")
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38")

            df = result.data.set_index(["CHROM", "POS", "Hugo_Symbol"])

            # TP53
            assert df.loc[("chr17", 7674220, "TP53"), "TCGA-AB-2802"] == "C|T"  # Mutated
            assert df.loc[("chr17", 7674220, "TP53"), "TCGA-AB-2803"] == "C|C"  # Reference

            # KRAS
            assert df.loc[("chr12", 25380275, "KRAS"), "TCGA-AB-2803"] == "G|A"  # Mutated
            assert df.loc[("chr12", 25380275, "KRAS"), "TCGA-AB-2802"] == "G|G"  # Reference

        finally:
            os.unlink(temp_path)


class TestReadMAFVariantConsolidation:
    """Test variant consolidation functionality."""
    
    def test_consolidation_enabled(self):
        """Test variant consolidation when enabled."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\n")
            # Same variant in two different samples
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2802\n")
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2803\n")
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38", consolidate_variants=True)
            
            # Should have consolidated identical variants
            assert len(result.data) == 1, "Variants should be consolidated into one row"
            
            # Both samples should have the mutation
            assert result.data.loc[0, "TCGA-AB-2802"] == "C|T"
            assert result.data.loc[0, "TCGA-AB-2803"] == "C|T"
        finally:
            os.unlink(temp_path)
    
    def test_consolidation_disabled(self):
        """Test behavior when consolidation is disabled."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\n")
            # Same variant in two different samples
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2802\n")
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2803\n")
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38", consolidate_variants=False)
            
            # Should NOT consolidate variants
            assert len(result.data) == 2, "Variants should remain as separate rows"
        finally:
            os.unlink(temp_path)
    
    def test_chunked_consolidation_large_dataset(self):
        """Test chunked consolidation for large datasets."""
        # Create a temporary MAF with many rows to trigger chunked processing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\n")
            
            # Create many identical variants to test chunking (simulate large dataset)
            for i in range(200):  # Create enough rows to potentially trigger chunking
                f.write(f"TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-{i:04d}\n")
            
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38", consolidate_variants=True)
            
            # Should consolidate all identical variants into one row
            assert len(result.data) == 1, "All identical variants should be consolidated"
            
            # Should have 200 sample columns
            assert len(result.samples) == 200
        finally:
            os.unlink(temp_path)


class TestReadMAFVariantClassificationNormalization:
    """Test variant classification normalization."""
    
    def test_variant_classification_normalization(self):
        """Test that Variant_Classification is normalized."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\tVariant_Classification\n")
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2802\tmissense_mutation\n")  # lowercase
            f.write("KRAS\t12\t25380275\tG\tG\tA\tTCGA-AB-2803\tSilent\n")  # mixed case
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38")
            
            # Should have normalized classification values
            if "Variant_Classification" in result.data.columns:
                classifications = result.data["Variant_Classification"].tolist()
                # Should all be uppercase
                assert all(str(cls).isupper() for cls in classifications if cls and str(cls) != "nan")
        finally:
            os.unlink(temp_path)


class TestReadMAFCaching:
    """Test MAF caching functionality."""
    
    def test_cache_hit_loads_from_cache(self):
        """Test that cache hit loads data from cache."""
        maf_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            
            # First read - should create cache
            result1 = read_maf(maf_path, assembly="38", cache_dir=cache_dir)
            
            # Second read - should load from cache
            result2 = read_maf(maf_path, assembly="38", cache_dir=cache_dir)
            
            # Results should be equivalent
            assert len(result1.data) == len(result2.data)
            assert result1.data.columns.tolist() == result2.data.columns.tolist()
            assert result1.metadata.source_format == result2.metadata.source_format
    
    def test_cache_write_saves_parquet_and_json(self):
        """Test that cache write saves both parquet and JSON metadata."""
        maf_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            
            # Read and cache
            result = read_maf(maf_path, assembly="38", cache_dir=cache_dir)
            
            # Check that cache files exist
            cache_files = list(cache_dir.glob("*.parquet"))
            assert len(cache_files) > 0, "No parquet cache files created"
            
            json_files = list(cache_dir.glob("*.json"))
            assert len(json_files) > 0, "No JSON metadata files created"
            
            # Check JSON metadata content
            json_file = json_files[0]
            with open(json_file) as f:
                cache_info = json.load(f)
            
            assert "comments" in cache_info
            assert "samples" in cache_info
            assert "processing_time" in cache_info
    
    def test_cache_fsync_behavior(self):
        """Test that cache files are properly synced to disk."""
        maf_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            
            # Mock fsync to ensure it's called
            with patch('os.fsync') as mock_fsync:
                result = read_maf(maf_path, assembly="38", cache_dir=cache_dir)
                
                # fsync should have been called for cache files
                assert mock_fsync.called, "fsync should be called to ensure data persistence"


class TestReadMAFColumnOrdering:
    """Test column ordering in final output."""
    
    def test_column_ordering(self):
        """Test that columns are ordered correctly."""
        maf_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf"
        
        result = read_maf(maf_path, assembly="38")
        
        # VCF-like columns should come first
        expected_order = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER"]
        actual_columns = result.data.columns.tolist()
        
        for i, expected_col in enumerate(expected_order):
            assert actual_columns[i] == expected_col, f"Column {i} should be {expected_col}, got {actual_columns[i]}"
        
        # Sample columns should come after VCF-like columns
        sample_start_idx = len(expected_order)
        sample_columns_in_df = actual_columns[sample_start_idx:sample_start_idx + len(result.samples)]
        assert set(sample_columns_in_df) == set(result.samples), "Sample columns not in expected position"
    
    def test_chromosome_column_removal(self):
        """Test that original Chromosome column is removed."""
        maf_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf"
        
        result = read_maf(maf_path, assembly="38")
        
        # Original Chromosome column should be removed
        assert "Chromosome" not in result.data.columns, "Original Chromosome column should be removed"


class TestReadMAFPyMutationObject:
    """Test PyMutation object creation and metadata."""
    
    def test_pymutation_object_creation(self):
        """Test that PyMutation object is created correctly."""
        maf_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf"
        
        result = read_maf(maf_path, assembly="38")
        
        assert isinstance(result, PyMutation)
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) > 0
        assert result.metadata.source_format == "MAF"
        assert result.metadata.assembly == "38"
        assert result.metadata.file_path == str(maf_path)
    
    def test_sample_tracking(self):
        """Test that samples are correctly tracked in PyMutation object."""
        maf_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf"
        
        result = read_maf(maf_path, assembly="38")
        
        # Should have samples list
        assert isinstance(result.samples, list)
        assert len(result.samples) > 0
        
        # All sample columns should exist in DataFrame
        for sample in result.samples:
            assert sample in result.data.columns, f"Sample {sample} not found in DataFrame columns"
    
    def test_metadata_filters_default(self):
        """Test that metadata filters default correctly."""
        maf_path = Path(__file__).parent / "fixtures" / "data" / "test_tcga_laml_100variants.maf"
        
        result = read_maf(maf_path, assembly="38")
        
        # Should have default filter value
        assert result.metadata.filters == ["."]
    
    def test_metadata_notes_from_comments(self):
        """Test that metadata notes are populated from comments."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            # Write MAF with comments
            f.write("# Test comment 1\n")
            f.write("# Test comment 2\n")
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\n")
            f.write("TP53\t17\t7674220\tC\tC\tT\tTCGA-AB-2802\n")
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38")
            
            # Should have notes from comments
            assert result.metadata.notes is not None
            assert "Test comment 1" in result.metadata.notes
            assert "Test comment 2" in result.metadata.notes
        finally:
            os.unlink(temp_path)


class TestReadMAFErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_maf_file(self):
        """Test handling of empty MAF files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            # Write only header
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\n")
            temp_path = f.name
        
        try:
            result = read_maf(temp_path, assembly="38")
            
            # Should handle empty data gracefully
            assert isinstance(result, PyMutation)
            assert len(result.data) == 0
            assert len(result.samples) == 0
        finally:
            os.unlink(temp_path)
    
    def test_malformed_maf_handling(self):
        """Test handling of malformed MAF data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False) as f:
            # Write MAF with inconsistent columns
            f.write("Hugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele1\tTumor_Seq_Allele2\tTumor_Sample_Barcode\n")
            f.write("TP53\t17\t7674220\tC\tC\n")  # Missing columns
            temp_path = f.name
        
        try:
            # Should handle gracefully without crashing
            result = read_maf(temp_path, assembly="38")
            assert isinstance(result, PyMutation)
        except Exception as e:
            # If it fails, it should fail gracefully
            assert isinstance(e, (ValueError, IndexError, KeyError, NotImplementedError))
        finally:
            os.unlink(temp_path)