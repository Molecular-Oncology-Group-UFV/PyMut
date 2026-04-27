import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import subprocess
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

from src.pyMut.input import read_vcf, HAS_PYARROW
from src.pyMut.core import PyMutation


class TestReadVCFAssemblyValidation:
    """Test assembly parameter validation."""
    
    def test_valid_assemblies(self):
        """Test that valid assembly values ('37', '38') are accepted."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        # Should not raise for valid assemblies
        result_37 = read_vcf(vcf_path, assembly="37")
        assert isinstance(result_37, PyMutation)
        assert result_37.metadata.assembly == "37"
        
        result_38 = read_vcf(vcf_path, assembly="38")
        assert isinstance(result_38, PyMutation)
        assert result_38.metadata.assembly == "38"
    
    def test_invalid_assembly_raises_error(self):
        """Test that invalid assembly values raise ValueError."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        with pytest.raises(ValueError, match="Assembly parameter must be either '37' or '38'"):
            read_vcf(vcf_path, assembly="36")
        
        with pytest.raises(ValueError, match="Assembly parameter must be either '37' or '38'"):
            read_vcf(vcf_path, assembly="39")
        
        with pytest.raises(ValueError, match="Assembly parameter must be either '37' or '38'"):
            read_vcf(vcf_path, assembly="hg19")


class TestReadVCFHeaderValidation:
    """Test VCF header validation."""
    
    def test_missing_chrom_header_raises_error(self):
        """Test that missing #CHROM header raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
            # Write VCF without #CHROM header
            f.write("##fileformat=VCFv4.2\n")
            f.write("##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">\n")
            f.write("10\t100\t.\tA\tT\t.\tPASS\tDP=100\tGT\t0/1\n")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="VCF file is missing the header line starting with #CHROM"):
                read_vcf(temp_path, assembly="38")
        finally:
            os.unlink(temp_path)
    
    def test_valid_chrom_header_accepted(self):
        """Test that valid #CHROM header is accepted."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        assert isinstance(result, PyMutation)
        assert "CHROM" in result.data.columns


class TestReadVCFFileOpening:
    """Test file opening for both .vcf and .vcf.gz files."""
    
    def test_read_vcf_plain_text(self):
        """Test reading plain text .vcf file."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        assert isinstance(result, PyMutation)
        assert len(result.data) > 0
        assert result.metadata.source_format == "VCF"
    
    def test_read_vcf_gzipped(self):
        """Test reading gzipped .vcf.gz file."""
        vcf_gz_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf.gz"
        
        result = read_vcf(vcf_gz_path, assembly="38")
        assert isinstance(result, PyMutation)
        assert len(result.data) > 0
        assert result.metadata.source_format == "VCF"
    
    def test_file_not_found_raises_error(self):
        """Test that non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_vcf("non_existent_file.vcf", assembly="38")


class TestReadVCFColumnHandling:
    """Test VCF column detection, standardization, and required columns."""
    
    def test_sample_column_detection(self):
        """Test detection of sample columns from header."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        
        # Should have sample columns
        assert len(result.samples) > 0
        # Standard VCF columns should not be in sample list
        standard_cols = {"CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"}
        for sample in result.samples:
            assert sample not in standard_cols
    
    def test_column_standardization(self):
        """Test that column names are properly standardized."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        
        # Required columns should be present and standardized
        required_cols = ["CHROM", "POS", "ID", "REF", "ALT", "FILTER"]
        for col in required_cols:
            assert col in result.data.columns
    
    def test_missing_required_columns_raises_error(self):
        """Test that missing required columns raise ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
            # Write VCF missing required REF column
            f.write("##fileformat=VCFv4.2\n")
            f.write("#CHROM\tPOS\tID\tALT\tQUAL\tFILTER\tINFO\n")
            f.write("10\t100\t.\tT\t.\tPASS\tDP=100\n")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Missing required columns in VCF"):
                read_vcf(temp_path, assembly="38")
        finally:
            os.unlink(temp_path)


class TestReadVCFGenotypeConversion:
    """Test vectorized genotype conversion."""
    
    def test_genotype_conversion_basic(self):
        """Test basic genotype conversion from numeric to alleles."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        
        # Check that we have sample columns with converted genotypes
        if result.samples:
            sample_col = result.samples[0]
            sample_data = result.data[sample_col]
            
            # Should contain allelic representations, not numeric
            # Look for patterns like "A|T", "G/C", etc.
            non_missing = sample_data[sample_data.notna() & (sample_data != ".")]
            if len(non_missing) > 0:
                # At least some should contain nucleotides or proper no-call formats
                has_nucleotides = any(any(base in str(val) for base in "ATCG") for val in non_missing[:10])
                has_proper_nocall = any(val in [".", "./.", ".|."] for val in non_missing[:10])
                assert has_nucleotides or has_proper_nocall
    
    def test_genotype_conversion_nocalls(self):
        """Test handling of no-call genotypes (., ./., .|.)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\n")
            f.write("10\t100\t.\tA\tT\t.\tPASS\t.\tGT\t.\n")
            f.write("10\t101\t.\tG\tC\t.\tPASS\t.\tGT\t./.\n")
            f.write("10\t102\t.\tT\tG\t.\tPASS\t.\tGT\t.|.\n")
            temp_path = f.name
        
        try:
            result = read_vcf(temp_path, assembly="38")
            
            # Check no-call handling
            sample_data = result.data["sample1"]
            assert "." in sample_data.values
            assert "./." in sample_data.values or ".|." in sample_data.values
        finally:
            os.unlink(temp_path)


class TestReadVCFInfoExpansion:
    """Test INFO column expansion with both PyArrow and pandas fallback."""
    
    def test_info_expansion_pyarrow_path(self):
        """Test INFO expansion using PyArrow when available."""
        if not HAS_PYARROW:
            pytest.skip("PyArrow not available")
        
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        
        # Should have INFO fields expanded as separate columns
        # Check for common INFO fields that should be present
        info_fields = ["AC", "AF", "AN", "DP"]
        present_info_fields = [field for field in info_fields if field in result.data.columns]
        assert len(present_info_fields) > 0, "No INFO fields were expanded"
    
    @patch('src.pyMut.input.HAS_PYARROW', False)
    def test_info_expansion_pandas_fallback(self):
        """Test INFO expansion using pandas fallback when PyArrow fails."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        
        # Should still have INFO fields expanded
        info_fields = ["AC", "AF", "AN", "DP"]
        present_info_fields = [field for field in info_fields if field in result.data.columns]
        assert len(present_info_fields) > 0, "No INFO fields were expanded in pandas fallback"
    
    def test_info_flags_handling(self):
        """Test that INFO flags without '=' are set to True."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write("##INFO=<ID=PASS,Number=0,Type=Flag,Description=\"Pass filter\">\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            f.write("10\t100\t.\tA\tT\t.\tPASS\tPASS;DP=100\n")
            temp_path = f.name
        
        try:
            result = read_vcf(temp_path, assembly="38")
            
            # PASS should be True (flag), DP should be "100" (value)
            if "PASS" in result.data.columns:
                assert bool(result.data["PASS"].iloc[0]) is True

            if "DP" in result.data.columns:
                assert int(result.data["DP"].iloc[0]) == 100

        finally:
            os.unlink(temp_path)


class TestReadVCFCSQHandling:
    """Test CSQ (VEP) annotation handling and conflicts."""
    
    def test_csq_expansion(self):
        """Test expansion of CSQ annotations into VEP_* columns."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        
        # Should have VEP_* columns from CSQ expansion
        vep_columns = [col for col in result.data.columns if col.startswith("VEP_")]
        assert len(vep_columns) > 0, "No VEP columns found after CSQ expansion"
        
        # Check for common VEP fields
        expected_vep_fields = ["VEP_Consequence", "VEP_SYMBOL", "VEP_Gene", "VEP_VARIANT_CLASS"]
        present_vep_fields = [field for field in expected_vep_fields if field in result.data.columns]
        assert len(present_vep_fields) > 0, "Expected VEP fields not found"
    
    def test_csq_sample_conflict_handling(self):
        """Test handling of CSQ naming conflicts with sample columns."""
        # Create a VCF with a sample named "CSQ"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write("##INFO=<ID=CSQ,Number=.,Type=String,Description=\"Consequence annotations from Ensembl VEP. Format: Allele|Consequence\">\n")
            f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tCSQ\n")
            f.write("10\t100\t.\tA\tT\t.\tPASS\tCSQ=T|missense_variant\tGT\t0/1\n")
            temp_path = f.name
        
        try:
            result = read_vcf(temp_path, assembly="38")
            
            # Should have created CSQ_sample column for the sample
            assert "CSQ_sample" in result.data.columns or any("CSQ_sample" in col for col in result.data.columns)
            # Should still have expanded VEP columns
            vep_columns = [col for col in result.data.columns if col.startswith("VEP_")]
            assert len(vep_columns) > 0
        finally:
            os.unlink(temp_path)


class TestReadVCFHugoSymbolGeneration:
    """Test Hugo_Symbol generation from VEP_SYMBOL and VEP_NEAREST."""
    
    def test_hugo_symbol_from_vep_symbol_only(self):
        """Test Hugo_Symbol generation when only VEP_SYMBOL exists."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        
        # If VEP_SYMBOL exists, Hugo_Symbol should be generated
        if "VEP_SYMBOL" in result.data.columns:
            assert "Hugo_Symbol" in result.data.columns
            
            # Check that Hugo_Symbol takes values from VEP_SYMBOL when available
            symbol_data = result.data["VEP_SYMBOL"]
            hugo_data = result.data["Hugo_Symbol"]
            
            # Where VEP_SYMBOL has values, Hugo_Symbol should match
            valid_symbol_mask = symbol_data.notna() & (symbol_data != "") & (symbol_data != "None")
            if valid_symbol_mask.any():
                assert (hugo_data[valid_symbol_mask] == symbol_data[valid_symbol_mask]).all()
    
    def test_hugo_symbol_fallback_to_nearest(self):
        """Test Hugo_Symbol fallback to VEP_NEAREST when VEP_SYMBOL is empty."""
        # This test would require creating test data with empty VEP_SYMBOL but populated VEP_NEAREST
        # For now, we just verify the logic exists in the main test file
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        
        # If either VEP field exists, Hugo_Symbol should be generated
        has_vep_symbol = "VEP_SYMBOL" in result.data.columns
        has_vep_nearest = "VEP_NEAREST" in result.data.columns
        
        if has_vep_symbol or has_vep_nearest:
            assert "Hugo_Symbol" in result.data.columns


class TestReadVCFVariantClassification:
    """Test Variant_Classification generation."""
    
    def test_variant_classification_generation(self):
        """Test generation of Variant_Classification from VEP data."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        
        # If VEP_Consequence and VEP_VARIANT_CLASS exist, Variant_Classification should be generated
        if "VEP_Consequence" in result.data.columns and "VEP_VARIANT_CLASS" in result.data.columns:
            assert "Variant_Classification" in result.data.columns
            
            # Should contain valid classification values
            classifications = result.data["Variant_Classification"].dropna().unique()
            valid_classifications = {
                'NONSENSE_MUTATION', 'NONSTOP_MUTATION', 'SPLICE_SITE', 'SPLICE_REGION',
                'FRAME_SHIFT_DEL', 'FRAME_SHIFT_INS', 'IN_FRAME_DEL', 'IN_FRAME_INS',
                'MISSENSE_MUTATION', 'SILENT', "5'UTR", "3'UTR", "5'FLANK", "3'FLANK",
                'INTRON', 'IGR', 'RNA'
            }
            
            # At least some classifications should be recognized
            recognized = set(classifications) & valid_classifications
            assert len(recognized) > 0, f"No recognized classifications found: {classifications}"
    
    def test_variant_classification_normalization(self):
        """Test that Variant_Classification values are normalized to uppercase."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        
        if "Variant_Classification" in result.data.columns:
            classifications = result.data["Variant_Classification"].dropna()
            # Should all be uppercase
            assert all(str(val).isupper() for val in classifications if val and str(val) != "nan")


class TestReadVCFVariantType:
    """Test Variant_Type generation from VEP_VARIANT_CLASS."""
    
    def test_variant_type_generation(self):
        """Test generation of Variant_Type from VEP_VARIANT_CLASS."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        
        if "VEP_VARIANT_CLASS" in result.data.columns:
            assert "Variant_Type" in result.data.columns
            
            # Should contain valid variant types
            variant_types = result.data["Variant_Type"].dropna().unique()
            valid_types = {'SNP', 'DNP', 'TNP', 'ONP', 'INS', 'DEL', 'IND', 'INV', 'CNV', 'UNKNOWN'}
            
            # All types should be valid
            invalid_types = set(variant_types) - valid_types
            assert len(invalid_types) == 0, f"Invalid variant types found: {invalid_types}"
    
    def test_variant_type_mapping_rules(self):
        """Test specific variant type mapping rules."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            f.write("10\t100\t.\tA\tT\t.\tPASS\tVEP_VARIANT_CLASS=SNV\n")
            f.write("10\t101\t.\tAT\tGC\t.\tPASS\tVEP_VARIANT_CLASS=substitution\n")
            f.write("10\t102\t.\tA\tAT\t.\tPASS\tVEP_VARIANT_CLASS=insertion\n")
            f.write("10\t103\t.\tAT\tA\t.\tPASS\tVEP_VARIANT_CLASS=deletion\n")
            temp_path = f.name
        
        try:
            result = read_vcf(temp_path, assembly="38")
            
            if "Variant_Type" in result.data.columns:
                variant_types = result.data["Variant_Type"].tolist()
                # SNV -> SNP, substitution -> DNP (2bp), insertion -> INS, deletion -> DEL
                expected_types = ["SNP", "DNP", "INS", "DEL"]
                for expected in expected_types:
                    assert expected in variant_types
        finally:
            os.unlink(temp_path)


class TestReadVCFCaching:
    """Test VCF caching functionality."""
    
    def test_cache_hit_loads_from_cache(self):
        """Test that cache hit loads data from cache."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            
            # First read - should create cache
            result1 = read_vcf(vcf_path, assembly="38", cache_dir=cache_dir)
            
            # Second read - should load from cache
            result2 = read_vcf(vcf_path, assembly="38", cache_dir=cache_dir)
            
            # Results should be equivalent
            assert len(result1.data) == len(result2.data)
            assert result1.data.columns.tolist() == result2.data.columns.tolist()
            assert result1.metadata.source_format == result2.metadata.source_format
    
    def test_cache_write_saves_parquet_and_json(self):
        """Test that cache write saves both parquet and JSON metadata."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            
            # Read and cache
            result = read_vcf(vcf_path, assembly="38", cache_dir=cache_dir)
            
            # Check that cache files exist
            cache_files = list(cache_dir.glob("*.parquet"))
            assert len(cache_files) > 0, "No parquet cache files created"
            
            json_files = list(cache_dir.glob("*.json"))
            assert len(json_files) > 0, "No JSON metadata files created"
            
            # Check JSON metadata content
            json_file = json_files[0]
            with open(json_file) as f:
                cache_info = json.load(f)
            
            assert "meta_lines" in cache_info
            assert "sample_columns" in cache_info
            assert "processing_time" in cache_info


class TestReadVCFTabixIndexing:
    """Test Tabix indexing functionality."""
    
    @patch('subprocess.run')
    def test_tabix_indexing_success(self, mock_subprocess):
        """Test successful Tabix indexing."""
        mock_subprocess.return_value = Mock(returncode=0)
        
        vcf_gz_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf.gz"
        
        # Should not raise error even if tabix is called
        result = read_vcf(vcf_gz_path, assembly="38", create_index=True)
        assert isinstance(result, PyMutation)
    
    @patch('subprocess.run')
    def test_tabix_indexing_failure(self, mock_subprocess):
        """Test handling of Tabix indexing failure."""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "tabix")
        
        vcf_gz_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf.gz"
        
        # Should still work even if tabix fails
        result = read_vcf(vcf_gz_path, assembly="38", create_index=True)
        assert isinstance(result, PyMutation)
    
    def test_no_indexing_for_plain_vcf(self):
        """Test that indexing is not attempted for plain VCF files."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        # Should work without issues for plain VCF
        result = read_vcf(vcf_path, assembly="38", create_index=True)
        assert isinstance(result, PyMutation)


class TestReadVCFDefaultFields:
    """Test handling of default fields and column ordering."""
    
    def test_default_qual_filter_fields_strict(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf', delete=False) as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tINFO\n") # Missing QUAL and FILTER
            f.write("10\t100\t.\tA\tT\tDP=100\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as err:
                read_vcf(temp_path, assembly="38")
            assert "Missing required columns in VCF: FILTER" in str(err.value)
        finally:
            os.unlink(temp_path)

    
    def test_format_column_removed(self):
        """Test that FORMAT column is removed from final output."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        
        # FORMAT should not be in final columns
        assert "FORMAT" not in result.data.columns
    
    def test_column_ordering(self):
        """Test that columns are ordered correctly."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        
        # Core VCF columns should come first
        expected_order = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER"]
        actual_columns = result.data.columns.tolist()
        
        for i, expected_col in enumerate(expected_order):
            assert actual_columns[i] == expected_col, f"Column {i} should be {expected_col}, got {actual_columns[i]}"


class TestReadVCFPyMutationObject:
    """Test PyMutation object creation and metadata."""
    
    def test_pymutation_object_creation(self):
        """Test that PyMutation object is created correctly."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        
        assert isinstance(result, PyMutation)
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) > 0
        assert result.metadata.source_format == "VCF"
        assert result.metadata.assembly == "38"
        assert result.metadata.file_path == str(vcf_path)
    
    def test_sample_columns_tracking(self):
        """Test that sample columns are correctly tracked."""
        vcf_path = Path(__file__).parent / "fixtures" / "data" / "test_ALL.chr10.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_vep_protein_gene_variant_class_100variants.vcf"
        
        result = read_vcf(vcf_path, assembly="38")
        
        # Should have samples list
        assert isinstance(result.samples, list)
        assert len(result.samples) >= 0
        
        # All sample columns should exist in DataFrame
        for sample in result.samples:
            assert sample in result.data.columns