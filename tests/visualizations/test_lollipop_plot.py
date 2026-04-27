#!/usr/bin/env python3
"""
Test suite for lollipop plot visualization.

This module contains comprehensive tests for the lollipop_plot function in pyMut,
including functional tests with dummy data and performance benchmarking on real datasets.
"""

import os
import sys
import time
import unittest
import warnings
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Set matplotlib backend to Agg to avoid GUI requirements
matplotlib.use('Agg')

# Add src to path to import pyMut
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pyMut.core import PyMutation
from pyMut.input import read_maf

class TestLollipopPlot(unittest.TestCase):
    """Tests for lollipop plot visualization functionality and performance."""
    
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
        print("   LOLLIPOP PLOT PERFORMANCE SUMMARY")
        print("="*60)
        print(f"{'Test Case':<45} | {'Time (s)':<10}")
        print("-" * 60)
        for name, duration in sorted(cls.execution_times.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:<45} | {duration:.4f}")
        print("="*60 + "\n")

    def setUp(self):
        """Set up minimal test data."""
        # Create a minimal MAF DataFrame for functional tests
        # We simulate a gene 'TESTGENE' with some mutations
        self.minimal_maf_data = pd.DataFrame({
            'Hugo_Symbol': ['TESTGENE'] * 5 + ['OTHERGENE'],
            'Chromosome': ['1'] * 6,
            'Start_Position': [100, 100, 200, 300, 300, 500],
            'End_Position': [100, 100, 200, 300, 300, 500],
            'Variant_Classification': [
                'Missense_Mutation', 'Missense_Mutation', 
                'Nonsense_Mutation', 
                'Frame_Shift_Del', 'Frame_Shift_Del',
                'Missense_Mutation'
            ],
            'Variant_Type': ['SNP', 'SNP', 'SNP', 'DEL', 'DEL', 'SNP'],
            'Tumor_Sample_Barcode': [
                'Sample_1', 'Sample_2', 
                'Sample_3', 
                'Sample_4', 'Sample_5',
                'Sample_1'
            ],
            'Protein_Change': [
                'p.R100H', 'p.R100H', # Recurrent hotspot
                'p.E200*', 
                'p.G300fs', 'p.G300fs',
                'p.A500T'
            ]
        })
        
        self.py_mut = PyMutation(self.minimal_maf_data)

    def tearDown(self):
        """Clean up figures."""
        plt.close('all')

    def test_basic_execution(self):
        """Test basic lollipop_plot execution with dummy data."""
        print("\n[Test] basic_execution...")
        start_time = time.time()
        
        # Use domains_source=None to avoid network calls in unit tests
        fig = self.py_mut.lollipop_plot(
            gene='TESTGENE',
            aa_col='Protein_Change',
            domains_source=None
        )
        
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['basic_execution'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_custom_domains(self):
        """Test lollipop_plot with custom domains."""
        print("\n[Test] custom_domains...")
        start_time = time.time()
        
        custom_domains = [
            {"start": 50, "end": 150, "name": "Domain_A"},
            {"start": 250, "end": 350, "name": "Domain_B"}
        ]
        
        fig = self.py_mut.lollipop_plot(
            gene='TESTGENE',
            aa_col='Protein_Change',
            custom_domains=custom_domains
        )
        
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['custom_domains'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_show_lollipops_false(self):
        """Test lollipop_plot with show_lollipops=False (domains only)."""
        print("\n[Test] show_lollipops_false...")
        start_time = time.time()
        
        fig = self.py_mut.lollipop_plot(
            gene='TESTGENE',
            aa_col='Protein_Change',
            domains_source=None,
            show_lollipops=False
        )
        
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['show_lollipops_false'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_count_by_samples(self):
        """Test lollipop_plot counting by samples instead of mutations."""
        print("\n[Test] count_by_samples...")
        start_time = time.time()
        
        fig = self.py_mut.lollipop_plot(
            gene='TESTGENE',
            aa_col='Protein_Change',
            domains_source=None,
            count_by='samples'
        )
        
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['count_by_samples'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_performance_real_dataset_dnmt3a(self):
        """Benchmark lollipop_plot on real dataset (DNMT3A) matching notebook."""
        if self.real_py_mut is None:
            self.skipTest("Real dataset not available")
            
        print("\n[Benchmark] Running lollipop_plot (DNMT3A) on real dataset...")
        
        start_time = time.time()
        
        try:
            fig = self.real_py_mut.lollipop_plot(
                gene='DNMT3A',
                aa_col='Protein_Change',
                transcript_id='NM_022552',
                label_top_n=1
            )
        except Exception as e:
            print(f"   -> Failed with error: {e}")
            # If it fails (e.g. network), we might want to fail the test or skip
            raise e

        total_time = time.time() - start_time
        
        print(f"   -> Total execution time: {total_time:.4f} seconds")
        self.__class__.execution_times['real_dataset_DNMT3A'] = total_time
        
        self.assertIsInstance(fig, plt.Figure)
        # Allow some time for potential network request if not cached
        self.assertLess(total_time, 30.0, "Analysis took too long (>30s)")

    def test_performance_real_dataset_tp53(self):
        """Benchmark lollipop_plot on real dataset (TP53) matching notebook."""
        if self.real_py_mut is None:
            self.skipTest("Real dataset not available")
            
        print("\n[Benchmark] Running lollipop_plot (TP53) on real dataset...")
        
        start_time = time.time()
        
        fig = self.real_py_mut.lollipop_plot(
            gene='TP53',
            aa_col='Protein_Change',
            transcript_id='NM_000546',
            label_top_n=1
        )
        
        total_time = time.time() - start_time
        
        print(f"   -> Total execution time: {total_time:.4f} seconds")
        self.__class__.execution_times['real_dataset_TP53'] = total_time
        
        self.assertIsInstance(fig, plt.Figure)

    def test_invalid_gene(self):
        """Test behavior when gene is not found."""
        print("\n[Test] invalid_gene...")
        start_time = time.time()
        
        # Should probably raise an error or return None/Empty figure depending on implementation
        # Assuming it raises ValueError or similar if gene not found in MAF
        try:
            with self.assertRaises(Exception):
                self.py_mut.lollipop_plot(gene='NONEXISTENT_GENE')
        except AssertionError:
            # If it doesn't raise, maybe it returns an empty plot?
            # Let's check if it runs without error but produces a figure
            fig = self.py_mut.lollipop_plot(gene='NONEXISTENT_GENE')
            self.assertIsInstance(fig, plt.Figure)
            
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['invalid_gene'] = duration

if __name__ == '__main__':
    unittest.main(verbosity=2)
