#!/usr/bin/env python3
"""
Example: Weight Perturbation Analysis

This script demonstrates how to:
1. Load a neural network from a .nnet file
2. For a fixed alpha (α), perturb all weights by ±α
3. Recompute bounds for both original and perturbed networks
4. Classify neurons into 6 categories based on bound changes
5. Compare perturbation strategies

Key Concepts:
- α (alpha): Maximum allowed weight perturbation
- Worst-case: Maximize negative weights, minimize positive (expand bounds)
- Best-case: Minimize negative weights, maximize positive (shrink bounds)
- Classification: How bounds change (Identical, Shrinkage, Expansion, etc.)
"""

import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '../..'))

from abstract_reuse.varry_alpha import PerturbedNetworkAnalyzer
import numpy as np

BASE_DIR = os.getcwd()
nnet_path = os.path.join(BASE_DIR, "nnet", "ACASXU_run2a_1_1_batch_2000.nnet")

def example_single_alpha():
    """Example 1: Analyze perturbation for a single alpha value."""
    print("=" * 70)
    print("EXAMPLE 1: Single Alpha Perturbation Analysis")
    print("=" * 70)
    
    # nnet_path = "/home/subhankar-jana/artifact/nnet/ACASXU_run2a_1_1_batch_2000.nnet"
    input_bounds = [(-0.5, 0.5) for _ in range(5)]
    
    # Create analyzer
    analyzer = PerturbedNetworkAnalyzer(nnet_path, input_bounds)
    
    # Analyze for alpha = 0.1
    alpha = 0.1
    print(f"\nAnalyzing weight perturbation for α = {alpha}")
    analyzer.analyze_for_alpha(alpha, strategies=['worst-case', 'best-case'])
    
    # Get output bounds comparison
    analysis_worst = analyzer.perturbation_analysis[alpha]['worst-case']
    analysis_best = analyzer.perturbation_analysis[alpha]['best-case']
    
    print(f"\nOutput Bound Expansion:")
    print("-" * 70)
    for i in range(5):
        orig_upper = analysis_worst['original_output_bounds']['upper'][i]
        pert_worst = analysis_worst['perturbed_output_bounds']['upper'][i]
        pert_best = analysis_best['perturbed_output_bounds']['upper'][i]
        
        expansion_worst = pert_worst / orig_upper if orig_upper > 0 else 0
        expansion_best = pert_best / orig_upper if orig_upper > 0 else 0
        
        print(f"Output {i}:")
        print(f"  Original: {orig_upper:.2f}")
        print(f"  Worst-case: {pert_worst:.2f} ({expansion_worst:.2f}x)")
        print(f"  Best-case: {pert_best:.2f} ({expansion_best:.2f}x)")


def example_multiple_alphas():
    """Example 2: Compare perturbations across multiple alpha values."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Multiple Alpha Values Comparison")
    print("=" * 70)
    
    # nnet_path = "/home/subhankar-jana/artifact/nnet/ACASXU_run2a_1_1_batch_2000.nnet"
    input_bounds = [(-0.5, 0.5) for _ in range(5)]
    
    analyzer = PerturbedNetworkAnalyzer(nnet_path, input_bounds)
    
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    print(f"\nAnalyzing for α = {alphas}")
    print("This will take a moment...")
    
    for alpha in alphas:
        analyzer.analyze_for_alpha(alpha, strategies=['worst-case'])
    
    # Summary table
    print(f"\n\nWORST-CASE PERTURBATION SUMMARY")
    print("-" * 70)
    print(f"{'α':<10} {'Expansion Classification':<20} {'% Neurons':<12}")
    print("-" * 70)
    
    for alpha in alphas:
        analysis = analyzer.perturbation_analysis[alpha]['worst-case']
        total_stats = analysis['total_stats']
        total = sum(total_stats.values())
        
        # Handle both string and enum keys
        expansion_key = 'Expansion' if 'Expansion' in total_stats else 'EXPANSION'
        expansion_pct = (total_stats.get(expansion_key, 0) / total * 100) if total > 0 else 0
        print(f"{alpha:<10.2f} {'Expansion':<20} {expansion_pct:>10.1f}%")
    
    # Output bound expansion
    print(f"\n\nOUTPUT BOUND EXPANSION (Output 0)")
    print("-" * 70)
    print(f"{'α':<10} {'Original Upper':<15} {'Perturbed Upper':<18} {'Expansion':<12}")
    print("-" * 70)
    
    for alpha in alphas:
        analysis = analyzer.perturbation_analysis[alpha]['worst-case']
        orig_upper = analysis['original_output_bounds']['upper'][0]
        pert_upper = analysis['perturbed_output_bounds']['upper'][0]
        expansion = pert_upper / orig_upper if orig_upper > 0 else 0
        
        print(f"{alpha:<10.2f} {orig_upper:<15.2f} {pert_upper:<18.2f} {expansion:<12.2f}x")


def example_analyze_specific_layer():
    """Example 3: Detailed analysis of specific layer."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Detailed Layer Analysis")
    print("=" * 70)
    
    # nnet_path = "/home/subhankar-jana/artifact/nnet/ACASXU_run2a_1_1_batch_2000.nnet"
    input_bounds = [(-0.5, 0.5) for _ in range(5)]
    
    analyzer = PerturbedNetworkAnalyzer(nnet_path, input_bounds)
    
    alpha = 0.5
    print(f"\nAnalyzing Layer 1 with α = {alpha}")
    analyzer.analyze_for_alpha(alpha, strategies=['worst-case', 'best-case'])
    
    # Detailed layer comparison for worst-case
    print(f"\nWorst-Case Perturbation - Top Affected Neurons:")
    analyzer.print_detailed_layer_comparison(
        alpha=alpha, strategy='worst-case', layer_idx=1, top_n=10
    )
    
    # Also show best-case
    print(f"\n\nBest-Case Perturbation - Top Affected Neurons:")
    analyzer.print_detailed_layer_comparison(
        alpha=alpha, strategy='best-case', layer_idx=1, top_n=10
    )


def example_compare_strategies():
    """Example 4: Compare worst-case vs best-case vs random strategies."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 4: Strategy Comparison (Worst-Case vs Best-Case vs Random)")
    print("=" * 70)
    
    # nnet_path = "/home/subhankar-jana/artifact/nnet/ACASXU_run2a_1_1_batch_2000.nnet"
    input_bounds = [(-0.5, 0.5) for _ in range(5)]
    
    analyzer = PerturbedNetworkAnalyzer(nnet_path, input_bounds)
    
    alpha = 0.2
    print(f"\nComparing strategies for α = {alpha}")
    print("(Running worst-case, best-case, and random perturbations...)")
    analyzer.analyze_for_alpha(alpha, strategies=['worst-case', 'best-case', 'random'])
    
    analysis_worst = analyzer.perturbation_analysis[alpha]['worst-case']
    analysis_best = analyzer.perturbation_analysis[alpha]['best-case']
    analysis_random = analyzer.perturbation_analysis[alpha]['random']
    
    # Helper function to robustly retrieve statistics
    def _get_stat(total_stats: dict, key_name: str) -> int:
        """Retrieve a statistic count robustly whether keys are Enum members or strings."""
        # Try Enum keys (e.g., BoundChangeType.EXPANSION)
        for k in total_stats.keys():
            try:
                if hasattr(k, 'name') and k.name == key_name:
                    return total_stats[k]
            except Exception:
                pass

        # Try common string variants
        return int(total_stats.get(key_name, total_stats.get(key_name.upper(), total_stats.get(key_name.capitalize(), 0))))

    print(f"\nStrategy Comparison:")
    print("-" * 80)
    print(f"{'Metric':<30} {'Worst-Case':<17} {'Best-Case':<17} {'Random':<17}")
    print("-" * 80)
    
    # Compare expansion percentages
    worst_expansion = _get_stat(analysis_worst['total_stats'], 'EXPANSION')
    best_expansion = _get_stat(analysis_best['total_stats'], 'EXPANSION')
    random_expansion = _get_stat(analysis_random['total_stats'], 'EXPANSION')
    total = sum(analysis_worst['total_stats'].values()) if analysis_worst['total_stats'] else 0

    worst_pct = (worst_expansion / total * 100) if total > 0 else 0
    best_pct = (best_expansion / total * 100) if total > 0 else 0
    random_pct = (random_expansion / total * 100) if total > 0 else 0

    print(f"{'Neurons with Expansion':<30} {worst_pct:>16.1f}% {best_pct:>16.1f}% {random_pct:>16.1f}%")
    
    # Compare output bounds
    orig_upper = analysis_worst['original_output_bounds']['upper'][0]
    worst_upper = analysis_worst['perturbed_output_bounds']['upper'][0]
    best_upper = analysis_best['perturbed_output_bounds']['upper'][0]
    random_upper = analysis_random['perturbed_output_bounds']['upper'][0]
    
    print(f"{'Output 0 - Upper Bound':<30} {worst_upper:>16.2f} {best_upper:>16.2f} {random_upper:>16.2f}")
    print(f"{'Output 0 - Expansion Factor':<30} {worst_upper/orig_upper:>16.2f}x {best_upper/orig_upper:>16.2f}x {random_upper/orig_upper:>16.2f}x")
    
    # Compare shrinkage percentages (opposite of expansion)
    worst_shrinkage = _get_stat(analysis_worst['total_stats'], 'SHRINKAGE')
    best_shrinkage = _get_stat(analysis_best['total_stats'], 'SHRINKAGE')
    random_shrinkage = _get_stat(analysis_random['total_stats'], 'SHRINKAGE')
    
    worst_shrink_pct = (worst_shrinkage / total * 100) if total > 0 else 0
    best_shrink_pct = (best_shrinkage / total * 100) if total > 0 else 0
    random_shrink_pct = (random_shrinkage / total * 100) if total > 0 else 0
    
    print(f"{'Neurons with Shrinkage':<30} {worst_shrink_pct:>16.1f}% {best_shrink_pct:>16.1f}% {random_shrink_pct:>16.1f}%")
    
    # Print detailed insights
    print(f"\n\nDetailed Insights:")
    print("-" * 80)
    print(f"Worst-Case Strategy:")
    print(f"  - Maximizes negative weights, minimizes positive weights")
    print(f"  - Results in maximum bound expansion")
    print(f"  - Output 0 expansion factor: {worst_upper/orig_upper:.2f}x")
    print(f"\nBest-Case Strategy:")
    print(f"  - Minimizes negative weights, maximizes positive weights")
    print(f"  - Results in maximum bound shrinkage/contraction")
    print(f"  - Output 0 expansion factor: {best_upper/orig_upper:.2f}x")
    print(f"\nRandom Strategy:")
    print(f"  - Randomly perturbs weights by ±α")
    print(f"  - Results in average-case behavior (typically between best and worst)")
    print(f"  - Output 0 expansion factor: {random_upper/orig_upper:.2f}x")


def example_sensitivity_analysis():
    """Example 5: Sensitivity analysis - how bounds change with alpha."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 5: Sensitivity Analysis")
    print("=" * 70)
    
    # nnet_path = "/home/subhankar-jana/artifact/nnet/ACASXU_run2a_1_1_batch_2000.nnet"
    input_bounds = [(-0.5, 0.5) for _ in range(5)]
    
    analyzer = PerturbedNetworkAnalyzer(nnet_path, input_bounds)
    
    alphas = np.linspace(0.01, 1.0, 10)
    
    print(f"\nAnalyzing sensitivity over α range: {alphas[0]:.2f} to {alphas[-1]:.2f}")
    
    results = {'alpha': [], 'worst_case_expansion': [], 'best_case_expansion': []}
    
    for alpha in alphas:
        analyzer.analyze_for_alpha(alpha, strategies=['worst-case', 'best-case'])
        
        analysis_worst = analyzer.perturbation_analysis[alpha]['worst-case']
        analysis_best = analyzer.perturbation_analysis[alpha]['best-case']
        
        orig_upper = analysis_worst['original_output_bounds']['upper'][0]
        worst_upper = analysis_worst['perturbed_output_bounds']['upper'][0]
        best_upper = analysis_best['perturbed_output_bounds']['upper'][0]
        
        results['alpha'].append(alpha)
        results['worst_case_expansion'].append(worst_upper / orig_upper if orig_upper > 0 else 0)
        results['best_case_expansion'].append(best_upper / orig_upper if orig_upper > 0 else 0)
    
    # Print results
    print(f"\n{'α':<10} {'Worst-Case Expansion':<25} {'Best-Case Expansion':<25}")
    print("-" * 60)
    for i, alpha in enumerate(alphas):
        print(f"{alpha:<10.3f} {results['worst_case_expansion'][i]:<25.2f}x {results['best_case_expansion'][i]:<25.2f}x")
    
    # Find critical alpha
    print(f"\n\nSensitivity Insights:")
    print("-" * 70)
    print(f"As α increases from 0 to 1:")
    print(f"  Worst-case expansion goes from {results['worst_case_expansion'][0]:.2f}x to {results['worst_case_expansion'][-1]:.2f}x")
    print(f"  Best-case expansion goes from {results['best_case_expansion'][0]:.2f}x to {results['best_case_expansion'][-1]:.2f}x")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("WEIGHT PERTURBATION ANALYSIS EXAMPLES")
    print("=" * 70)
    
    # Run examples
    example_single_alpha()
    example_multiple_alphas()
    example_analyze_specific_layer()
    example_compare_strategies()
    example_sensitivity_analysis()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)
