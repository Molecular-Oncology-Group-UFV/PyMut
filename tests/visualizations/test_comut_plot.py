#!/usr/bin/env python3
"""
Test suite for CoMut plot visualization.

This module contains comprehensive tests for the CoMut plot functionality in pyMut,
including individual panels (A-G) and the complete multi-panel visualization.
Includes functional tests with dummy data and performance benchmarking on real datasets.
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


class TestCoMutPlot(unittest.TestCase):
    """Tests for CoMut plot visualization functionality and performance."""
    
    execution_times = {}

    @classmethod
    def setUpClass(cls):
        """Load real dataset once for performance tests."""
        # Define paths to real data
        cls.base_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/pyMut/data'))
        cls.maf_path = os.path.join(cls.base_data_path, 'examples/MAF/TCGA_test.maf')
        cls.tsv_path = os.path.join(cls.base_data_path, 'examples/TSV')
        
        # Check if real data exists
        if os.path.exists(cls.maf_path) and os.path.exists(cls.tsv_path):
            print(f"\n[Setup] Loading real dataset from {cls.maf_path}...")
            start_time = time.time()
            # Load with assembly="37" and consolidate_variants=False as per notebook
            cls.real_py_mut = read_maf(cls.maf_path, assembly="37", consolidate_variants=False)
            load_time = time.time() - start_time
            print(f"[Setup] Dataset loaded in {load_time:.4f} seconds.")
            
            # Load reference sample order from mutation_burden.tsv as in notebook
            ref_mb_path = os.path.join(cls.tsv_path, "mutation_burden.tsv")
            if os.path.exists(ref_mb_path):
                ref_mb = pd.read_csv(ref_mb_path, sep="\t")
                # Find sample column robustly
                sample_col = next(c for c in ref_mb.columns if c.lower() in {"sample","sample_id","tumor_sample_barcode"})
                cls.sample_order = ref_mb[sample_col].astype(str).tolist()
            else:
                cls.sample_order = None
                warnings.warn("mutation_burden.tsv not found. Sample order for benchmarks might be incorrect.")

            # Define paths for other TSVs
            cls.sig_tsv = os.path.join(cls.tsv_path, "sig_contribution.tsv")
            cls.purity_tsv = os.path.join(cls.tsv_path, "purity.tsv")
            cls.mut_data_tsv = os.path.join(cls.tsv_path, "mutation_data.tsv")
            cls.cna_tsv = os.path.join(cls.tsv_path, "cna.tsv")
            cls.wgd_tsv = os.path.join(cls.tsv_path, "wgd.tsv")
            cls.sp_tsv = os.path.join(cls.tsv_path, "sp.tsv")
            
        else:
            warnings.warn("Real dataset files not found. Performance tests will be skipped.")
            cls.real_py_mut = None

    @classmethod
    def tearDownClass(cls):
        """Print performance summary."""
        print("\n" + "="*60)
        print("   COMUT PLOT PERFORMANCE SUMMARY")
        print("="*60)
        print(f"{'Test Case':<45} | {'Time (s)':<10}")
        print("-" * 60)
        for name, duration in sorted(cls.execution_times.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:<45} | {duration:.4f}")
        print("="*60 + "\n")

    def setUp(self):
        """Set up minimal test data for functional tests."""
        # 1. Dummy MAF for Panel A
        self.dummy_samples = ['S1', 'S2', 'S3']
        self.minimal_maf_data = pd.DataFrame({
            'Hugo_Symbol': ['GeneA', 'GeneB', 'GeneC'],
            'Tumor_Sample_Barcode': ['S1', 'S2', 'S3'],
            'Variant_Classification': ['Missense_Mutation', 'Nonsense_Mutation', 'Silent'],
            'Chromosome': ['1', '2', '3'],
            'Start_Position': [100, 200, 300],
            'End_Position': [100, 200, 300],
            'Reference_Allele': ['A', 'C', 'G'],
            'Tumor_Seq_Allele2': ['T', 'G', 'A']
        })
        self.py_mut = PyMutation(self.minimal_maf_data)
        
        # 2. Dummy DataFrames for other panels
        # Panel B: Signatures
        self.dummy_sig_df = pd.DataFrame({
            'Sig1': [0.2, 0.8, 0.5],
            'Sig2': [0.8, 0.2, 0.5]
        }, index=self.dummy_samples)
        
        # Panel C: Purity
        self.dummy_purity_df = pd.DataFrame({
            'sample': self.dummy_samples,
            'value': [0.9, 0.6, 0.8]
        })
        
        # Panel D: Mutation Type
        self.dummy_mut_type_df = pd.DataFrame({
            'sample': ['S1', 'S2', 'S3'],
            'category': ['GeneX', 'GeneX', 'GeneY'],
            'value': ['Missense', 'Nonsense', 'Frameshift indel']
        })
        
        # Panel E: CNA
        self.dummy_cna_df = pd.DataFrame({
            'sample': ['S1', 'S2', 'S3'],
            'category': ['GeneZ', 'GeneZ', 'GeneZ'],
            'value': ['Allelic amplification', 'Baseline', 'Allelic deletion']
        })
        
        # Panel F: WGD
        self.dummy_wgd_df = pd.DataFrame({
            'sample': self.dummy_samples,
            'value': ['Yes', 'No', 'Yes']
        })
        
        # Panel G: Same Patient
        self.dummy_sp_df = pd.DataFrame({
            'sample': self.dummy_samples,
            'group': ['P1', 'P1', 'P2']
        })

    def tearDown(self):
        """Clean up figures."""
        plt.close('all')

    def test_panel_a_mutation_burden(self):
        """Test Panel A: Mutation Burden."""
        print("\n[Test] panel_a_mutation_burden...")
        start_time = time.time()
        fig = self.py_mut.comut_mutation_burden(
            sample_ids=self.dummy_samples,
            territory_bp=1000000
        )
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['panel_a_functional'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_panel_b_signatures(self):
        """Test Panel B: Mutational Signatures."""
        print("\n[Test] panel_b_signatures...")
        start_time = time.time()
        fig = self.py_mut.comut_mutation_signatures_plot(
            signatures_df=self.dummy_sig_df,
            sample_order=self.dummy_samples
        )
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['panel_b_functional'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_panel_c_purity(self):
        """Test Panel C: Purity."""
        print("\n[Test] panel_c_purity...")
        start_time = time.time()
        fig = self.py_mut.comut_purity_plot(
            purity_df=self.dummy_purity_df,
            sample_order=self.dummy_samples
        )
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['panel_c_functional'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_panel_d_mutation_type(self):
        """Test Panel D: Mutation Type."""
        print("\n[Test] panel_d_mutation_type...")
        start_time = time.time()
        fig = self.py_mut.comut_mutation_type_plot(
            mutation_data_df=self.dummy_mut_type_df,
            sample_order=self.dummy_samples,
            gene_order=['GeneX', 'GeneY']
        )
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['panel_d_functional'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_panel_e_cna(self):
        """Test Panel E: CNA."""
        print("\n[Test] panel_e_cna...")
        start_time = time.time()
        fig = self.py_mut.comut_cna_plot(
            cna_df=self.dummy_cna_df,
            sample_order=self.dummy_samples,
            gene_order=['GeneZ']
        )
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['panel_e_functional'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_panel_f_wgd(self):
        """Test Panel F: WGD."""
        print("\n[Test] panel_f_wgd...")
        start_time = time.time()
        fig = self.py_mut.comut_wgd_plot(
            wgd_df=self.dummy_wgd_df,
            sample_order=self.dummy_samples
        )
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['panel_f_functional'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_panel_g_same_patient(self):
        """Test Panel G: Same Patient."""
        print("\n[Test] panel_g_same_patient...")
        start_time = time.time()
        fig = self.py_mut.comut_same_patient_plot(
            sp_df=self.dummy_sp_df,
            sample_order=self.dummy_samples
        )
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['panel_g_functional'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_full_comut_plot_functional(self):
        """Test the complete create_comut_plot with dummy data."""
        print("\n[Test] full_comut_plot_functional...")
        start_time = time.time()
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            sig_path = os.path.join(tmpdir, 'sig.tsv')
            # Panel B expects wide format for TSV in docs: sample, Sig1, Sig2
            # My dummy_sig_df has index as sample.
            self.dummy_sig_df.reset_index().rename(columns={'index': 'sample'}).to_csv(sig_path, sep='\t', index=False)
            
            purity_path = os.path.join(tmpdir, 'purity.tsv')
            self.dummy_purity_df.to_csv(purity_path, sep='\t', index=False)
            
            mut_path = os.path.join(tmpdir, 'mut.tsv')
            self.dummy_mut_type_df.to_csv(mut_path, sep='\t', index=False)
            
            cna_path = os.path.join(tmpdir, 'cna.tsv')
            self.dummy_cna_df.to_csv(cna_path, sep='\t', index=False)
            
            wgd_path = os.path.join(tmpdir, 'wgd.tsv')
            self.dummy_wgd_df.to_csv(wgd_path, sep='\t', index=False)
            
            sp_path = os.path.join(tmpdir, 'sp.tsv')
            self.dummy_sp_df.to_csv(sp_path, sep='\t', index=False)
            
            fig = self.py_mut.create_comut_plot(
                sample_order=self.dummy_samples,
                gene_order=['GeneX', 'GeneY'],
                cna_gene_order=['GeneZ'],
                signatures_tsv=sig_path,
                purity_tsv=purity_path,
                mutation_data_tsv=mut_path,
                cna_tsv=cna_path,
                wgd_tsv=wgd_path,
                sp_tsv=sp_path,
                territory_bp=1000000,
                figsize=(10, 10)
            )
            
        duration = time.time() - start_time
        print(f"   -> Execution time: {duration:.4f}s")
        self.__class__.execution_times['full_comut_functional'] = duration
        self.assertIsInstance(fig, plt.Figure)

    def test_performance_real_dataset_full(self):
        """Benchmark the full CoMut plot on real dataset."""
        if self.real_py_mut is None:
            self.skipTest("Real dataset not available")
            
        print("\n[Benchmark] Running create_comut_plot on real dataset...")
        
        start_time = time.time()
        
        gene_order = ["MAP3K1", "CDH1", "TP53", "PIK3CA"]
        cna_gene_order = ["ERBB2", "CDKN2A", "MYC"]
        
        fig = self.real_py_mut.create_comut_plot(
            sample_order=self.sample_order,
            gene_order=gene_order,
            cna_gene_order=cna_gene_order,
            signatures_tsv=self.sig_tsv,
            purity_tsv=self.purity_tsv,
            mutation_data_tsv=self.mut_data_tsv,
            cna_tsv=self.cna_tsv,
            wgd_tsv=self.wgd_tsv,
            sp_tsv=self.sp_tsv,
            territory_bp=60456963,
            somatic_only=False,
            pass_only=False,
            signature_labels=["Signature 1", "Signature 2", "Signature 3", "Signature 4"],
            figsize=(14, 14.9),
            max_samples=None
        )
        
        total_time = time.time() - start_time
        
        print(f"   -> Total execution time: {total_time:.4f} seconds")
        self.__class__.execution_times['real_dataset_full'] = total_time
        
        self.assertIsInstance(fig, plt.Figure)

if __name__ == '__main__':
    unittest.main(verbosity=2)
