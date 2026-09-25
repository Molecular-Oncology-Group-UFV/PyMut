#!/usr/bin/env python3
"""
Test suite for somatic interactions visualization.

This module contains comprehensive tests for the somatic_interactions function in pyMut,
including functional tests with dummy data and performance benchmarking on real datasets.
"""

import os
import sys
import time
import unittest
import warnings

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Set matplotlib backend to Agg to avoid GUI requirements
matplotlib.use('Agg')

# Add src to path to import pyMut
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pyMut.core import PyMutation
from pyMut.input import read_maf


class TestSomaticInteractions(unittest.TestCase):
    """Tests for somatic interactions visualization functionality and performance."""
    
    execution_times = {}

    @classmethod
    def setUpClass(cls):
        """Load real dataset once for performance tests."""
        # Define paths to real data
        cls.base_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/pyMut/data'))
        cls.maf_path = os.path.join(cls.base_data_path, 'examples/MAF/tcga_laml.maf.gz')
        
        # Check if real data exists
        if os.path.exists(cls.maf_path):
            print(f"\n[Setup] Loading real dataset from {cls.maf_path}...")
            start_time = time.time()
            # Load with assembly="37" and consolidate_variants=False as per notebook
            cls.real_py_mut = read_maf(cls.maf_path, assembly="37", consolidate_variants=False)
            load_time = time.time() - start_time
            print(f"[Setup] Dataset loaded in {load_time:.4f} seconds.")
        else:
            warnings.warn("Real dataset file not found. Performance tests will be skipped.")
            cls.real_py_mut = None

    @classmethod
    def tearDownClass(cls):
        """Print performance summary."""
        print("\n" + "="*60)
        print("   SOMATIC INTERACTIONS PERFORMANCE SUMMARY")
        print("="*60)
        print(f"{'Test Case':<45} | {'Time (s)':<10}")
        print("-" * 60)
        for name, duration in sorted(cls.execution_times.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:<45} | {duration:.4f}")
        print("="*60 + "\n")

    def setUp(self):
        """Set up minimal test data."""
        # Create a minimal MAF DataFrame for functional tests
        # We simulate 5 genes across 10 samples
        # Gene A and B co-occur in Sample 1, 2, 3
        # Gene C and D are mutually exclusive (C in 4,5,6; D in 7,8,9)
        data = []
        
        # Gene A: Samples 1, 2, 3
        for i in range(1, 4):
            data.append({'Hugo_Symbol': 'GeneA', 'Tumor_Sample_Barcode': f'Sample_{i}'})
            
        # Gene B: Samples 1, 2, 3 (Perfect co-occurrence with A)
        for i in range(1, 4):
            data.append({'Hugo_Symbol': 'GeneB', 'Tumor_Sample_Barcode': f'Sample_{i}'})
            
        # Gene C: Samples 4, 5, 6
        for i in range(4, 7):
            data.append({'Hugo_Symbol': 'GeneC', 'Tumor_Sample_Barcode': f'Sample_{i}'})
            
        # Gene D: Samples 7, 8, 9 (Mutually exclusive with C)
        for i in range(7, 10):
            data.append({'Hugo_Symbol': 'GeneD', 'Tumor_Sample_Barcode': f'Sample_{i}'})
            
        # Gene E: Random samples
        data.append({'Hugo_Symbol': 'GeneE', 'Tumor_Sample_Barcode': 'Sample_1'})
        data.append({'Hugo_Symbol': 'GeneE', 'Tumor_Sample_Barcode': 'Sample_10'})

        self.minimal_maf_data = pd.DataFrame(data)
        # Add required columns that might be checked even if not used for calculation
        self.minimal_maf_data['Variant_Classification'] = 'Missense_Mutation'
        self.minimal_maf_data['Chromosome'] = '1'
        self.minimal_maf_data['Start_Position'] = 100
        self.minimal_maf_data['End_Position'] = 100
        self.minimal_maf_data['Reference_Allele'] = 'A'
        self.minimal_maf_data['Tumor_Seq_Allele2'] = 'T'
        
        self.py_mut = PyMutation(self.minimal_maf_data)

    def tearDown(self):
        """Clean up figures."""
        plt.close('all')

    def test_basic_execution(self):
        """Test basic somatic_interactions execution with dummy data."""
        print("\n[Test] basic_execution...")
        start_time = time.time()
        
        fig = self.py_mut.somatic_interactions(top_genes=5)
        
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['basic_execution'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_show_counts_false(self):
        """Test somatic_interactions with show_counts=False."""
        print("\n[Test] show_counts_false...")
        start_time = time.time()
        
        fig = self.py_mut.somatic_interactions(top_genes=5, show_counts=False)
        
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['show_counts_false'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_custom_pvalues(self):
        """Test somatic_interactions with custom p-value thresholds."""
        print("\n[Test] custom_pvalues...")
        start_time = time.time()
        
        fig = self.py_mut.somatic_interactions(
            top_genes=5,
            pvalue=(0.01, 0.001)
        )
        
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['custom_pvalues'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_performance_real_dataset_default(self):
        """Benchmark somatic_interactions on real dataset (default settings)."""
        if self.real_py_mut is None:
            self.skipTest("Real dataset not available")
            
        print("\n[Benchmark] Running somatic_interactions (default) on real dataset...")
        
        start_time = time.time()
        
        fig = self.real_py_mut.somatic_interactions()
        
        total_time = time.time() - start_time
        
        print(f"   -> Total execution time: {total_time:.4f} seconds")
        self.__class__.execution_times['real_dataset_default'] = total_time
        
        self.assertIsInstance(fig, plt.Figure)

    def test_performance_real_dataset_top15(self):
        """Benchmark somatic_interactions on real dataset (top_genes=15)."""
        if self.real_py_mut is None:
            self.skipTest("Real dataset not available")
            
        print("\n[Benchmark] Running somatic_interactions (top_genes=15) on real dataset...")
        
        start_time = time.time()
        
        fig = self.real_py_mut.somatic_interactions(top_genes=15)
        
        total_time = time.time() - start_time
        
        print(f"   -> Total execution time: {total_time:.4f} seconds")
        self.__class__.execution_times['real_dataset_top15'] = total_time
        
        self.assertIsInstance(fig, plt.Figure)

    def test_performance_real_dataset_custom(self):
        """Benchmark somatic_interactions on real dataset (custom thresholds)."""
        if self.real_py_mut is None:
            self.skipTest("Real dataset not available")
            
        print("\n[Benchmark] Running somatic_interactions (custom) on real dataset...")
        
        start_time = time.time()
        
        fig = self.real_py_mut.somatic_interactions(
            top_genes=20,
            pvalue=(0.01, 0.001)
        )
        
        total_time = time.time() - start_time
        
        print(f"   -> Total execution time: {total_time:.4f} seconds")
        self.__class__.execution_times['real_dataset_custom'] = total_time
        
        self.assertIsInstance(fig, plt.Figure)

if __name__ == '__main__':
    unittest.main(verbosity=2)
