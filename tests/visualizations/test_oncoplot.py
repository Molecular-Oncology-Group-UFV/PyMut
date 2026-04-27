#!/usr/bin/env python3
"""
Test suite for oncoplot functionality.

This module contains comprehensive tests for all functions related
to oncoplot generation in pyMut, including performance benchmarking
on real datasets.
"""

import unittest
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import time
import warnings

# Set matplotlib backend to Agg to avoid GUI requirements
matplotlib.use('Agg')

# Add src to path to import pyMut
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pyMut.core import PyMutation
from pyMut.input import read_maf
from pyMut.visualizations.oncoplot import (
    is_mutated,
    detect_sample_columns,
    process_mutation_matrix
)


class TestOncoplot(unittest.TestCase):
    """Tests for oncoplot functionality and performance."""
    
    execution_times = {}

    @classmethod
    def setUpClass(cls):
        """Load real dataset once for performance tests."""
        cls.maf_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '../../src/pyMut/data/examples/MAF/tcga_laml.maf.gz'
        ))
        
        if os.path.exists(cls.maf_path):
            print(f"\n[Setup] Loading real dataset from {cls.maf_path}...")
            start_time = time.time()
            cls.real_py_mut = read_maf(cls.maf_path, assembly="37")
            load_time = time.time() - start_time
            print(f"[Setup] Dataset loaded in {load_time:.4f} seconds.")
        else:
            warnings.warn(f"Real dataset not found at {cls.maf_path}. Performance tests may be skipped.")
            cls.real_py_mut = None

    @classmethod
    def tearDownClass(cls):
        """Print performance summary."""
        print("\n" + "="*60)
        print("       ONCOPLOT PERFORMANCE SUMMARY")
        print("="*60)
        print(f"{'Visualization':<45} | {'Time (s)':<10}")
        print("-" * 60)
        for name, duration in sorted(cls.execution_times.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:<45} | {duration:.4f}")
        print("="*60 + "\n")

    def setUp(self):
        """Set up minimal test data (Long Format - Standard MAF)."""
        # Create a minimal MAF-like DataFrame (Long format)
        # This mimics the standard input from read_maf better than a matrix
        self.minimal_maf_data = pd.DataFrame({
            'Hugo_Symbol': ['TP53', 'TP53', 'KRAS', 'PIK3CA', 'PIK3CA', 'BRCA1'],
            'Variant_Classification': ['Missense_Mutation', 'Nonsense_Mutation', 'Missense_Mutation', 
                                     'Missense_Mutation', 'Frame_Shift_Del', 'In_Frame_Ins'],
            'Variant_Type': ['SNP', 'SNP', 'SNP', 'SNP', 'DEL', 'INS'],
            'REF': ['A', 'C', 'G', 'T', 'A', 'C'],
            'ALT': ['G', 'T', 'A', 'C', 'G', 'T'],
            'Tumor_Sample_Barcode': ['Sample_1', 'Sample_1', 'Sample_2', 'Sample_2', 'Sample_3', 'Sample_3']
        })

        # Manually expand to Wide format as read_maf does, because oncoplot requires sample columns
        # (read_maf automatically adds these columns, but here we are creating the DataFrame manually)
        samples = self.minimal_maf_data['Tumor_Sample_Barcode'].unique()
        for sample in samples:
            # Initialize with wildtype (REF|REF) - simplified here as A|A for all
            self.minimal_maf_data[sample] = 'A|A' 
            
        # Set mutations for the specific samples in each row
        for idx, row in self.minimal_maf_data.iterrows():
            sample = row['Tumor_Sample_Barcode']
            ref = row['REF']
            alt = row['ALT']
            # Set genotype to REF|ALT to indicate mutation
            self.minimal_maf_data.at[idx, sample] = f"{ref}|{alt}"

        self.py_mut = PyMutation(self.minimal_maf_data)

    def tearDown(self):
        """Clean up matplotlib figures."""
        plt.close('all')

    def test_oncoplot_basic_execution(self):
        """Test basic oncoplot execution with standard MAF data."""
        print("\n[Test] Basic oncoplot execution (Long format)...")
        start_time = time.time()
        fig = self.py_mut.oncoplot()
        duration = time.time() - start_time
        
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['oncoplot (functional test)'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_performance_real_dataset(self):
        """Benchmark oncoplot on the real TCGA LAML dataset."""
        if self.real_py_mut is None:
            self.skipTest("Real dataset not available")
            
        print("\n[Benchmark] Running oncoplot on real TCGA LAML dataset...")
        
        start_time = time.time()
        fig = self.real_py_mut.oncoplot(top_genes_count=20, max_samples=100)
        total_time = time.time() - start_time
        
        print(f"   -> Total execution time: {total_time:.4f} seconds")
        self.__class__.execution_times['oncoplot (real dataset)'] = total_time
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertLess(total_time, 15.0, "Oncoplot took too long (>15s)")

    def test_oncoplot_custom_parameters(self):
        """Test oncoplot with custom parameters."""
        print("\n[Test] Oncoplot with custom parameters...")
        fig = self.py_mut.oncoplot(
            title="Test Custom Oncoplot",
            top_genes_count=2,
            max_samples=2,
            figsize=(12, 8)
        )
        self.assertIsInstance(fig, plt.Figure)

    def test_wide_format_support(self):
        """Test oncoplot with wide format (matrix-like) data."""
        print("\n[Test] Wide format data support...")
        
        wide_data = pd.DataFrame({
            'Hugo_Symbol': ['TP53', 'KRAS', 'PIK3CA'],
            'Variant_Classification': ['Missense_Mutation', 'Missense_Mutation', 'In_Frame_Del'],
            'REF': ['A', 'C', 'G'],
            'ALT': ['G', 'T', 'A'],
            'TCGA-AB-1234': ['A|G', 'C|C', 'G|A'],
            'TCGA-CD-5678': ['A|A', 'C|T', 'G|G']
        })
        py_mut_wide = PyMutation(wide_data)
        
        start_time = time.time()
        fig = py_mut_wide.oncoplot()
        duration = time.time() - start_time
        
        print(f"   -> Wide format execution time: {duration:.4f}s")
        self.assertIsInstance(fig, plt.Figure)

    def test_oncoplot_invalid_parameters(self):
        """Test invalid parameters for oncoplot."""
        with self.assertRaises(ValueError):
            self.py_mut.oncoplot(top_genes_count=0)
        
        with self.assertRaises(ValueError):
            self.py_mut.oncoplot(max_samples=-1)


class TestOncoplotUtilities(unittest.TestCase):
    """Tests for oncoplot utility functions."""
    
    def test_is_mutated_basic_cases(self):
        """Test basic mutation detection cases."""
        self.assertTrue(is_mutated("A|G", "A", "G"))
        self.assertFalse(is_mutated("A|A", "A", "G"))
    
    def test_detect_sample_columns(self):
        """Test automatic detection of sample columns."""
        df_tcga = pd.DataFrame({
            'Hugo_Symbol': ['TP53'],
            'TCGA-AB-1234': ['A|G'],
            'Other_Column': ['value']
        })
        sample_cols = detect_sample_columns(df_tcga)
        self.assertEqual(sample_cols, ['TCGA-AB-1234'])

    def test_process_mutation_matrix_multi_hit(self):
        """Test Multi_Hit detection."""
        multi_hit_data = pd.DataFrame({
            'Hugo_Symbol': ['TP53', 'TP53'],
            'Variant_Classification': ['Missense_Mutation', 'Nonsense_Mutation'],
            'REF': ['A', 'C'],
            'ALT': ['G', 'T'],
            'TCGA-AB-1234': ['A|G', 'C|T'],
        })
        matrix, counts = process_mutation_matrix(multi_hit_data)
        self.assertEqual(matrix.loc['TP53', 'TCGA-AB-1234'], 'Multi_Hit')


if __name__ == '__main__':
    unittest.main(verbosity=2)
