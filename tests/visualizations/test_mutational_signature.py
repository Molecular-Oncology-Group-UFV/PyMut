#!/usr/bin/env python3
"""
Test suite for mutational signature analysis functionality.

This module contains comprehensive tests for all functions related
to mutational signature analysis in pyMut, including performance benchmarking
on real datasets.
"""

import os
import shutil
import sys
import tempfile
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


class TestMutationalSignature(unittest.TestCase):
    """Tests for mutational signature analysis functionality and performance."""
    
    execution_times = {}

    @classmethod
    def setUpClass(cls):
        """Load real dataset once for performance tests."""
        # Define paths to real data (using TCGA LAML as it is lighter than ms.maf.gz)
        cls.base_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/pyMut/data'))
        # cls.maf_path = os.path.join(cls.base_data_path, 'examples/MAF/ms.maf.gz')
        cls.maf_path = os.path.join(cls.base_data_path, 'examples/MAF/tcga_laml.maf.gz')
        cls.ref_genome_path = os.path.join(cls.base_data_path, 'resources/genome/GRCh37/Homo_sapiens.GRCh37.dna.toplevel.fa')
        cls.cosmic_path = os.path.join(cls.base_data_path, 'examples/COSMIC_catalogue-signatures_SBS96_v3.4/COSMIC_72.tsv')
        
        # Check if real data exists
        if os.path.exists(cls.maf_path) and os.path.exists(cls.ref_genome_path) and os.path.exists(cls.cosmic_path):
            print(f"\n[Setup] Loading real dataset from {cls.maf_path}...")
            start_time = time.time()
            # We need to load with assembly="37" as per the notebook example
            cls.real_py_mut = read_maf(cls.maf_path, assembly="37")
            load_time = time.time() - start_time
            print(f"[Setup] Dataset loaded in {load_time:.4f} seconds.")
        else:
            warnings.warn("Real dataset files not found. Performance tests will be skipped.")
            cls.real_py_mut = None

    @classmethod
    def tearDownClass(cls):
        """Print performance summary."""
        print("\n" + "="*60)
        print("   MUTATIONAL SIGNATURE PERFORMANCE SUMMARY")
        print("="*60)
        print(f"{'Visualization':<45} | {'Time (s)':<10}")
        print("-" * 60)
        for name, duration in sorted(cls.execution_times.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:<45} | {duration:.4f}")
        print("="*60 + "\n")

    def setUp(self):
        """Set up minimal test data and temporary files."""
        # Create a temporary directory for dummy genome and cosmic files
        self.test_dir = tempfile.mkdtemp()
        
        # 1. Create a dummy Reference Genome (FASTA)
        # We create a simple genome with one chromosome "1"
        # Sequence: AAAAA C AAAAA G AAAAA T AAAAA (Length 20ish)
        # Pos 6: C (Context A[C]A)
        # Pos 12: G (Context A[G]A) -> C on reverse strand
        # Pos 18: T (Context A[T]A)
        self.temp_genome_path = os.path.join(self.test_dir, "temp_genome.fa")
        with open(self.temp_genome_path, "w") as f:
            f.write(">1\n")
            f.write("AAAAACAAAAAGAAAAATAAAA\n")
            # Index file is usually created automatically by pyfaidx, 
            # but we ensure the file exists.

        # 2. Create a dummy COSMIC signatures file
        self.temp_cosmic_path = os.path.join(self.test_dir, "temp_cosmic.tsv")
        self._create_dummy_cosmic_file(self.temp_cosmic_path)

        # 3. Create a minimal MAF DataFrame
        # We add mutations that align with our dummy genome
        self.minimal_maf_data = pd.DataFrame({
            'Hugo_Symbol': ['GENE1', 'GENE1', 'GENE2'],
            'Chromosome': ['1', '1', '1'],
            'Start_Position': [6, 12, 18],
            'End_Position': [6, 12, 18],
            'Reference_Allele': ['C', 'G', 'T'],
            'Tumor_Seq_Allele2': ['T', 'A', 'G'], # C>T, G>A (C>T), T>G (T>G)
            'Variant_Classification': ['Missense_Mutation', 'Missense_Mutation', 'Missense_Mutation'],
            'Variant_Type': ['SNP', 'SNP', 'SNP'],
            'Tumor_Sample_Barcode': ['Sample_1', 'Sample_2', 'Sample_1']
        })
        
        self.py_mut = PyMutation(self.minimal_maf_data)

    def tearDown(self):
        """Clean up temporary files and figures."""
        shutil.rmtree(self.test_dir)
        plt.close('all')

    def _create_dummy_cosmic_file(self, filepath):
        """Helper to create a valid COSMIC signature file structure."""
        bases = ['A', 'C', 'G', 'T']
        contexts = []
        # Generate 96 contexts: 5'[REF>ALT]3'
        # REF is C or T
        for ref in ['C', 'T']:
            for alt in bases:
                if alt == ref:
                    continue
                for b5 in bases:
                    for b3 in bases:
                        contexts.append(f"{b5}[{ref}>{alt}]{b3}")
        
        # Create DataFrame
        df = pd.DataFrame({'MutationType': contexts})
        # Add dummy signatures SBS1, SBS2
        df['SBS1'] = 1.0 / 96  # Flat distribution
        df['SBS2'] = [1.0 if i == 0 else 0.0 for i in range(96)] # Spike at first context
        
        df.to_csv(filepath, sep='\t', index=False)

    def test_signature_bar_chart(self):
        """Test signature_bar_chart execution."""
        print("\n[Test] signature_bar_chart execution...")
        start_time = time.time()
        fig = self.py_mut.signature_bar_chart(
            ref_genome=self.temp_genome_path,
            n_signatures=2
        )
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['signature_bar_chart (functional)'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_cosine_similarity_heatmap(self):
        """Test cosine_similarity_heatmap execution."""
        print("\n[Test] cosine_similarity_heatmap execution...")
        start_time = time.time()
        fig = self.py_mut.cosine_similarity_heatmap(
            ref_genome=self.temp_genome_path,
            cosmic_path=self.temp_cosmic_path,
            n_signatures=2
        )
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['cosine_similarity_heatmap (functional)'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_signature_contribution_heatmap(self):
        """Test signature_contribution_heatmap execution."""
        print("\n[Test] signature_contribution_heatmap execution...")
        start_time = time.time()
        fig = self.py_mut.signature_contribution_heatmap(
            ref_genome=self.temp_genome_path,
            n_signatures=2
        )
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['signature_contribution_heatmap (functional)'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_signature_stacked_bar_chart(self):
        """Test signature_stacked_bar_chart execution."""
        print("\n[Test] signature_stacked_bar_chart execution...")
        start_time = time.time()
        fig = self.py_mut.signature_stacked_bar_chart(
            ref_genome=self.temp_genome_path,
            n_signatures=2
        )
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['signature_stacked_bar_chart (functional)'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_signature_donut_plot(self):
        """Test signature_donut_plot execution."""
        print("\n[Test] signature_donut_plot execution...")
        start_time = time.time()
        fig = self.py_mut.signature_donut_plot(
            ref_genome=self.temp_genome_path,
            n_signatures=2
        )
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['signature_donut_plot (functional)'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_mutational_signature_analysis_full(self):
        """Test the complete mutational_signature_analysis visualization."""
        print("\n[Test] mutational_signature_analysis (full) execution...")
        start_time = time.time()
        fig = self.py_mut.mutational_signature_analysis(
            ref_genome=self.temp_genome_path,
            cosmic_path=self.temp_cosmic_path,
            n_signatures=2
        )
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['mutational_signature_analysis (functional)'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_performance_real_dataset(self):
        """Benchmark mutational_signature_analysis on the real dataset."""
        if self.real_py_mut is None:
            self.skipTest("Real dataset not available")
            
        print("\n[Benchmark] Running mutational_signature_analysis on real dataset...")
        
        start_time = time.time()
        # We use n_signatures=3 as in the notebook example
        fig = self.real_py_mut.mutational_signature_analysis(
            ref_genome=self.ref_genome_path,
            cosmic_path=self.cosmic_path,
            n_signatures=3
        )
        total_time = time.time() - start_time
        
        print(f"   -> Total execution time: {total_time:.4f} seconds")
        self.__class__.execution_times['mutational_signature_analysis (real dataset)'] = total_time
        
        self.assertIsInstance(fig, plt.Figure)
        # This is a heavy operation, so we allow more time (e.g., 60s)
        self.assertLess(total_time, 60.0, "Analysis took too long (>60s)")

    def test_invalid_parameters(self):
        """Test invalid parameters."""
        # Test with invalid n_signatures
        with self.assertRaises(ValueError):
            self.py_mut.signature_bar_chart(
                ref_genome=self.temp_genome_path,
                n_signatures=0
            )
        
        # Test with missing reference genome file
        with self.assertRaises(Exception): # Could be FileNotFoundError or other IO error
            self.py_mut.signature_bar_chart(
                ref_genome="non_existent_file.fa"
            )

if __name__ == '__main__':
    unittest.main(verbosity=2)
