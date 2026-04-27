#!/usr/bin/env python3
"""
Test suite for summary_plot functionality.

This module contains comprehensive tests for the summary_plot visualization
and its individual sub-panels. It includes tests for:
- Basic execution and return types
- Custom parameters
- Data format support (Long vs Wide)
- Performance benchmarking on real datasets
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


class TestSummaryPlot(unittest.TestCase):
    """Tests for the summary_plot visualization and its components."""

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
        print("       VISUALIZATION PERFORMANCE SUMMARY")
        print("="*60)
        print(f"{'Visualization':<45} | {'Time (s)':<10}")
        print("-" * 60)
        for name, duration in sorted(cls.execution_times.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:<45} | {duration:.4f}")
        print("="*60 + "\n")

    def setUp(self):
        """Set up minimal test data (Long Format - Standard MAF)."""
        # Create a minimal MAF-like DataFrame (Long format)
        self.minimal_maf_data = pd.DataFrame({
            'Hugo_Symbol': ['TP53', 'TP53', 'KRAS', 'BRCA1', 'EGFR'] * 10,
            'Variant_Classification': [
                'Missense_Mutation', 'Nonsense_Mutation', 'Missense_Mutation', 
                'Frame_Shift_Del', 'Splice_Site'
            ] * 10,
            'Variant_Type': ['SNP', 'SNP', 'SNP', 'DEL', 'SNP'] * 10,
            'REF': ['A', 'C', 'G', 'T', 'A'] * 10,
            'ALT': ['G', 'T', 'A', 'C', 'G'] * 10,
            'Tumor_Sample_Barcode': [f'Sample_{i%5}' for i in range(50)]
        })
        self.py_mut = PyMutation(self.minimal_maf_data)

    def tearDown(self):
        """Clean up matplotlib figures."""
        plt.close('all')

    def test_summary_plot_basic_execution(self):
        """Test that summary_plot runs without error and returns a Figure."""
        print("\n[Test] Basic summary_plot execution...")
        start_time = time.time()
        
        fig = self.py_mut.summary_plot()
        
        exec_time = time.time() - start_time
        print(f"   -> Execution time: {exec_time:.4f}s")
        
        self.assertIsInstance(fig, plt.Figure)
        # Check if we have roughly the expected number of axes (6 panels)
        self.assertGreaterEqual(len(fig.axes), 6)

    def test_summary_plot_custom_parameters(self):
        """Test summary_plot with custom parameters."""
        print("\n[Test] summary_plot with custom parameters...")
        
        fig = self.py_mut.summary_plot(
            figsize=(20, 15),
            title="Custom Test Summary",
            max_samples=3,
            top_genes_count=5
        )
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig.get_figwidth(), 20)
        self.assertEqual(fig.get_figheight(), 15)
        # We can't easily check the title text on the figure object in a generic way 
        # without inspecting specific text objects, but execution without error is key.

    def test_individual_panels(self):
        """Test execution of individual panel functions."""
        print("\n[Test] Individual panel functions...")
        
        panels = [
            ('variant_classification_plot', {}),
            ('variant_type_plot', {}),
            ('snv_class_plot', {}),
            ('variants_per_sample_plot', {'max_samples': 10}),
            ('variant_classification_summary_plot', {}),
            ('top_mutated_genes_plot', {'mode': 'variants', 'count': 5}),
            ('top_mutated_genes_plot', {'mode': 'samples', 'count': 5})
        ]
        
        for method_name, kwargs in panels:
            with self.subTest(method=method_name):
                if hasattr(self.py_mut, method_name):
                    method = getattr(self.py_mut, method_name)
                    start_t = time.time()
                    fig = method(**kwargs)
                    duration = time.time() - start_t
                    
                    print(f"   -> {method_name}: {duration:.4f}s")
                    
                    # Create a readable name for the report
                    if method_name == 'top_mutated_genes_plot':
                        mode = kwargs.get('mode', 'variants')
                        report_name = f"{method_name} ({mode})"
                    else:
                        report_name = method_name
                    
                    self.__class__.execution_times[report_name] = duration
                    
                    self.assertIsInstance(fig, plt.Figure)
                else:
                    self.fail(f"Method {method_name} not found in PyMutation")

    def test_performance_real_dataset(self):
        """Benchmark summary_plot on the real TCGA LAML dataset."""
        if self.real_py_mut is None:
            self.skipTest("Real dataset not available")
            
        print("\n[Benchmark] Running summary_plot on real TCGA LAML dataset...")
        
        # Measure full summary plot
        start_time = time.time()
        fig = self.real_py_mut.summary_plot()
        total_time = time.time() - start_time
        
        print(f"   -> Total execution time: {total_time:.4f} seconds")
        self.__class__.execution_times['summary_plot (full)'] = total_time
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertLess(total_time, 10.0, "Summary plot took too long (>10s)")

    def test_wide_format_support(self):
        """Test summary_plot with wide format (matrix-like) data."""
        print("\n[Test] Wide format data support...")
        
        # Create wide format data
        wide_data = pd.DataFrame({
            'Hugo_Symbol': ['TP53', 'KRAS', 'BRCA1'],
            'Variant_Classification': ['Missense_Mutation', 'Missense_Mutation', 'Nonsense_Mutation'],
            'Variant_Type': ['SNP', 'SNP', 'SNP'],
            'REF': ['A', 'G', 'C'],
            'ALT': ['T', 'A', 'G'],
            'Sample_1': ['A|T', 'G|G', 'C|C'],
            'Sample_2': ['A|A', 'G|A', 'C|G']
        })
        
        py_mut_wide = PyMutation(wide_data)
        
        start_time = time.time()
        fig = py_mut_wide.summary_plot()
        exec_time = time.time() - start_time
        
        print(f"   -> Wide format execution time: {exec_time:.4f}s")
        self.assertIsInstance(fig, plt.Figure)

if __name__ == '__main__':
    unittest.main(verbosity=2)
