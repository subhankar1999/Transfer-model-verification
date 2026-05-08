#!/usr/bin/env python3
"""
Visualization: Weight Perturbation Analysis Across Alpha Values

This script generates 3 comprehensive plots showing how perturbation strategies
behave as alpha (maximum weight perturbation) varies:

1. WORST-CASE Strategy Plot
2. BEST-CASE Strategy Plot
3. RANDOM Strategy Plot

For each strategy, the plots show:
- Expansion factor of output bounds (primary y-axis)
- Percentage of neurons with Expansion classification (secondary y-axis)
- Percentage of neurons with Shrinkage classification (secondary y-axis)
"""

import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '../..'))

from abstract_reuse.varry_alpha import PerturbedNetworkAnalyzer, BoundChangeType
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings

plt.rcParams.update({
    "font.size": 28,          # base size
    "axes.titlesize": 28,
    "axes.labelsize": 28,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.fontsize": 28,
    "figure.titlesize": 28
})

warnings.filterwarnings('ignore')

def analyze_all_alphas(nnet_path, input_bounds, alphas, strategies):
    """
    Analyze perturbations for multiple alpha values and strategies.
    
    Returns:
        Dictionary with results organized by alpha and strategy
    """
    print("=" * 70)
    print("ANALYZING PERTURBATIONS FOR MULTIPLE ALPHA VALUES")
    print("=" * 70)
    
    analyzer = PerturbedNetworkAnalyzer(nnet_path, input_bounds)
    results = {alpha: {} for alpha in alphas}
    
    for i, alpha in enumerate(alphas):
        print(f"\n[{i+1}/{len(alphas)}] Analyzing α = {alpha:.3f}")
        analyzer.analyze_for_alpha(alpha, strategies=strategies)
        
        # Extract results for each strategy
        for strategy in strategies:
            analysis = analyzer.perturbation_analysis[alpha][strategy]
            
            # Get output bounds
            orig_upper = analysis['original_output_bounds']['upper'][0]
            pert_upper = analysis['perturbed_output_bounds']['upper'][0]
            expansion_factor = pert_upper / orig_upper if orig_upper > 0 else 0
            
            # Get classification statistics
            total_stats = analysis['total_stats']
            total_neurons = sum(total_stats.values())
            
            # Robustly retrieve statistics
            def _get_stat(key_name):
                for k in total_stats.keys():
                    try:
                        if hasattr(k, 'name') and k.name == key_name:
                            return total_stats[k]
                    except:
                        pass
                return int(total_stats.get(key_name, total_stats.get(key_name.upper(), 0)))
            
            category_names = [
                'IDENTICAL',
                'SHRINKAGE',
                'EXPANSION',
                'PARTIAL_OVERLAP',
                'DISJOINT',
                'DEGENERATE'
            ]

            category_counts = {}
            category_pcts = {}

            for name in category_names:
                count = _get_stat(name)
                category_counts[name] = count
                category_pcts[name] = (count / total_neurons * 100) if total_neurons > 0 else 0
            
            results[alpha][strategy] = {
                'expansion_factor': expansion_factor,
                'category_counts': category_counts,
                'category_pcts': category_pcts,
                'total_neurons': total_neurons
            }
                
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return results


def plot_all_categories(alphas, results, strategy, output_path):
    """
    Plot ALL 6 categories vs alpha
    """

    categories = [
        'IDENTICAL',
        'SHRINKAGE',
        'EXPANSION',
        'PARTIAL_OVERLAP',
        'DISJOINT',
        'DEGENERATE'
    ]

    colors = {
        'IDENTICAL': '#8E44AD',
        'SHRINKAGE': '#1E8449',
        'EXPANSION': '#C0392B',
        'PARTIAL_OVERLAP': '#F39C12',
        'DISJOINT': '#2C3E50',
        'DEGENERATE': '#7F8C8D'
    }

    plt.figure(figsize=(12, 7))

    for category in categories:
        values = []
        for alpha in alphas:
            if strategy in results[alpha]:
                values.append(results[alpha][strategy]['category_pcts'][category])
            else:
                values.append(0)

        plt.plot(alphas, values, marker='o', linewidth=2,
                 label=category.replace('_', ' ').title(),
                 color=colors[category])

    plt.xlabel('Alpha (α)', fontsize=12, fontweight='bold')
    plt.ylabel('Neuron Percentage (%)', fontsize=12, fontweight='bold')
    plt.title(f'{strategy.replace("-", " ").title()} Strategy - All Categories vs Alpha',
              fontsize=14, fontweight='bold')

    plt.ylim([0, 100])
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_strategy_analysis(alphas, results, strategy, output_path):
    """
    Create a comprehensive plot for a single perturbation strategy.
    
    Args:
        alphas: List of alpha values
        results: Results dictionary from analyze_all_alphas
        strategy: Strategy name ('worst-case', 'best-case', or 'random')
        output_path: Path to save the plot
    """
    
    # Extract data for this strategy
    expansion_factors = []
    expansion_pcts = []
    shrinkage_pcts = []
    identical_pcts = []
    
    for alpha in alphas:
        if strategy in results[alpha]:
            data = results[alpha][strategy]
            expansion_factors.append(data['expansion_factor'])
            expansion_pcts.append(data['expansion_pct'])
            shrinkage_pcts.append(data['shrinkage_pct'])
            identical_pcts.append(data['identical_pct'])
        else:
            expansion_factors.append(0)
            expansion_pcts.append(0)
            shrinkage_pcts.append(0)
            identical_pcts.append(0)
    
    # Create figure with twin axes
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot expansion factor on primary y-axis
    color1 = '#E74C3C' if strategy == 'worst-case' else '#27AE60' if strategy == 'best-case' else '#3498DB'
    ax1.set_xlabel('Alpha (α) - Maximum Weight Perturbation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Output Bound Expansion Factor (x)', fontsize=12, fontweight='bold', color=color1)
    line1 = ax1.plot(alphas, expansion_factors, marker='o', linewidth=2.5, 
                     markersize=8, color=color1, label='Expansion Factor', zorder=3)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Create secondary y-axis for percentages
    ax2 = ax1.twinx()
    color2_exp = '#C0392B'  # Darker red for expansion
    color2_shrink = '#1E8449'  # Darker green for shrinkage
    
    ax2.set_ylabel('Neuron Classification (%)', fontsize=12, fontweight='bold')
    line2 = ax2.plot(alphas, expansion_pcts, marker='s', linewidth=2, 
                     markersize=7, color=color2_exp, linestyle='--', 
                     label='% Neurons with Expansion', zorder=2)
    line3 = ax2.plot(alphas, shrinkage_pcts, marker='^', linewidth=2, 
                     markersize=7, color=color2_shrink, linestyle='--', 
                     label='% Neurons with Shrinkage', zorder=2)
    ax2.set_ylim([-5, 105])
    ax2.tick_params(axis='y')
    
    # Title and legend
    strategy_title = strategy.replace('-', ' ').title()
    fig.suptitle(f'{strategy_title} Strategy: Bound Expansion vs Neuron Classification\n(as Alpha varies)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Combine legends from both axes
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.95)
    
    # Add background color based on strategy
    if strategy == 'worst-case':
        fig.patch.set_facecolor('#FFF5F5')
    elif strategy == 'best-case':
        fig.patch.set_facecolor('#F5FFF5')
    else:  # random
        fig.patch.set_facecolor('#F5F5FF')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_path}")
    plt.close()


def plot_combined_comparison(alphas, results, output_path):
    """
    Create a combined comparison plot showing all three strategies.
    
    Args:
        alphas: List of alpha values
        results: Results dictionary from analyze_all_alphas
        output_path: Path to save the plot
    """
    
    strategies = ['worst-case', 'best-case', 'random']
    colors = ['#E74C3C', '#27AE60', '#3498DB']
    
    # Extract expansion factors for all strategies
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Expansion Factor Comparison
    for strategy, color in zip(strategies, colors):
        expansion_factors = []
        for alpha in alphas:
            if strategy in results[alpha]:
                expansion_factors.append(results[alpha][strategy]['expansion_factor'])
            else:
                expansion_factors.append(0)
        
        ax1.plot(alphas, expansion_factors, marker='o', linewidth=2.5, 
                markersize=8, color=color, label=strategy.replace('-', ' ').title())
    
    ax1.set_xlabel('Alpha (α)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Output Bound Expansion Factor (x)', fontsize=12, fontweight='bold')
    ax1.set_title('Expansion Factor Comparison Across Strategies', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: Neurons with Expansion %
    for strategy, color in zip(strategies, colors):
        expansion_pcts = []
        for alpha in alphas:
            if strategy in results[alpha]:
                expansion_pcts.append(results[alpha][strategy]['expansion_pct'])
            else:
                expansion_pcts.append(0)
        
        ax2.plot(alphas, expansion_pcts, marker='s', linewidth=2.5, 
                markersize=8, color=color, label=strategy.replace('-', ' ').title())
    
    ax2.set_xlabel('Alpha (α)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Neurons with Expansion (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Neuron Expansion Classification Across Strategies', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([-5, 105])
    
    fig.suptitle('Combined Strategy Comparison', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined plot: {output_path}")
    plt.close()


def plot_bar_graph_by_strategy(alphas, results, strategy, output_path):
    """
    Create bar graphs for a single perturbation strategy showing metrics by alpha.
    
    Args:
        alphas: List of alpha values
        results: Results dictionary from analyze_all_alphas
        strategy: Strategy name ('worst-case', 'best-case', or 'random')
        output_path: Path to save the plot
    """
    
    # Extract data for this strategy
    expansion_factors = []
    expansion_pcts = []
    shrinkage_pcts = []
    identical_pcts = []
    
    for alpha in alphas:
        if strategy in results[alpha]:
            data = results[alpha][strategy]
            expansion_factors.append(data['expansion_factor'])
            expansion_pcts.append(data['expansion_pct'])
            shrinkage_pcts.append(data['shrinkage_pct'])
            identical_pcts.append(data['identical_pct'])
        else:
            expansion_factors.append(0)
            expansion_pcts.append(0)
            shrinkage_pcts.append(0)
            identical_pcts.append(0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{strategy.replace("-", " ").title()} Strategy - Bar Graph Analysis', 
                 fontsize=15, fontweight='bold')
    
    # Color for strategy
    bar_color = '#E74C3C' if strategy == 'worst-case' else '#27AE60' if strategy == 'best-case' else '#3498DB'
    
    # Convert alphas to strings for x-axis labels
    alpha_labels = [f'α={a:.2f}' for a in alphas]
    x_pos = np.arange(len(alphas))
    bar_width = 0.7
    
    # Plot 1: Shrinkage %
    axes[1, 0].bar(x_pos, shrinkage_pcts, bar_width, color='#1E8449', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_xlabel('Alpha (α) Values', fontweight='bold')
    axes[1, 0].set_ylabel('Neurons with Shrinkage (%)', fontweight='bold')
    axes[1, 0].set_title('Neurons with Shrinkage Classification', fontweight='bold', fontsize=12)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(alpha_labels, rotation=45, ha='right')
    axes[1, 0].set_ylim([0, 105])
    axes[1, 0].grid(True, axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
    for i, v in enumerate(shrinkage_pcts):
        axes[1, 0].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)


    
    # Plot 2: Expansion %
    axes[0, 1].bar(x_pos, expansion_pcts, bar_width, color='#C0392B', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_xlabel('Alpha (α) Values', fontweight='bold')
    axes[0, 1].set_ylabel('Neurons with Expansion (%)', fontweight='bold')
    axes[0, 1].set_title('Neurons with Expansion Classification', fontweight='bold', fontsize=12)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(alpha_labels, rotation=45, ha='right')
    axes[0, 1].set_ylim([0, 105])
    axes[0, 1].grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, v in enumerate(expansion_pcts):
        axes[0, 1].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    

    # Plot 3: Expansion Factor
    axes[0, 0].bar(x_pos, expansion_factors, bar_width, color=bar_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_xlabel('Alpha (α) Values', fontweight='bold')
    axes[0, 0].set_ylabel('Expansion Factor (x)', fontweight='bold')
    axes[0, 0].set_title('Output Bound Expansion Factor', fontweight='bold', fontsize=12)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(alpha_labels, rotation=45, ha='right')
    axes[0, 0].grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, v in enumerate(expansion_factors):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    
    # Plot 4: Identical %
    axes[1, 1].bar(x_pos, identical_pcts, bar_width, color='#8E44AD', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_xlabel('Alpha (α) Values', fontweight='bold')
    axes[1, 1].set_ylabel('Neurons Identical (%)', fontweight='bold')
    axes[1, 1].set_title('Neurons with Identical Classification', fontweight='bold', fontsize=12)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(alpha_labels, rotation=45, ha='right')
    axes[1, 1].set_ylim([0, 105])
    axes[1, 1].grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, v in enumerate(identical_pcts):
        axes[1, 1].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved bar graph: {output_path}")
    plt.close()


def plot_bar_comparison_strategies(alphas, results, output_path):
    """
    Create grouped bar charts comparing all strategies at each alpha value.
    
    Args:
        alphas: List of alpha values
        results: Results dictionary from analyze_all_alphas
        output_path: Path to save the plot
    """
    
    strategies = ['worst-case', 'best-case', 'random']
    colors = ['#E74C3C', '#27AE60', '#3498DB']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Strategy Comparison - Bar Graphs by Alpha Value', 
                 fontsize=15, fontweight='bold')
    
    alpha_labels = [f'α={a:.2f}' for a in alphas]
    x_pos = np.arange(len(alphas))
    bar_width = 0.25
    
    # Extract data for all strategies
    data_by_metric = {
        'expansion_factor': {s: [] for s in strategies},
        'expansion_pct': {s: [] for s in strategies},
        'shrinkage_pct': {s: [] for s in strategies},
        'identical_pct': {s: [] for s in strategies}
    }
    
    for alpha in alphas:
        for strategy in strategies:
            if strategy in results[alpha]:
                data = results[alpha][strategy]
                data_by_metric['expansion_factor'][strategy].append(data['expansion_factor'])
                data_by_metric['expansion_pct'][strategy].append(data['expansion_pct'])
                data_by_metric['shrinkage_pct'][strategy].append(data['shrinkage_pct'])
                data_by_metric['identical_pct'][strategy].append(data['identical_pct'])
            else:
                data_by_metric['expansion_factor'][strategy].append(0)
                data_by_metric['expansion_pct'][strategy].append(0)
                data_by_metric['shrinkage_pct'][strategy].append(0)
                data_by_metric['identical_pct'][strategy].append(0)
    
    # Plot 1: Expansion Factor Comparison
    for i, (strategy, color) in enumerate(zip(strategies, colors)):
        offset = (i - 1) * bar_width
        axes[0, 0].bar(x_pos + offset, data_by_metric['expansion_factor'][strategy], 
                      bar_width, label=strategy.replace('-', ' ').title(), 
                      color=color, alpha=0.8, edgecolor='black', linewidth=1)
    
    axes[0, 0].set_xlabel('Alpha (α) Values', fontweight='bold')
    axes[0, 0].set_ylabel('Expansion Factor (x)', fontweight='bold')
    axes[0, 0].set_title('Expansion Factor Comparison', fontweight='bold', fontsize=12)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(alpha_labels, rotation=45, ha='right')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Plot 2: Expansion % Comparison
    for i, (strategy, color) in enumerate(zip(strategies, colors)):
        offset = (i - 1) * bar_width
        axes[0, 1].bar(x_pos + offset, data_by_metric['expansion_pct'][strategy], 
                      bar_width, label=strategy.replace('-', ' ').title(), 
                      color=color, alpha=0.8, edgecolor='black', linewidth=1)
    
    axes[0, 1].set_xlabel('Alpha (α) Values', fontweight='bold')
    axes[0, 1].set_ylabel('Neurons with Expansion (%)', fontweight='bold')
    axes[0, 1].set_title('Expansion Classification Comparison', fontweight='bold', fontsize=12)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(alpha_labels, rotation=45, ha='right')
    axes[0, 1].set_ylim([0, 105])
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Plot 3: Shrinkage % Comparison
    for i, (strategy, color) in enumerate(zip(strategies, colors)):
        offset = (i - 1) * bar_width
        axes[1, 0].bar(x_pos + offset, data_by_metric['shrinkage_pct'][strategy], 
                      bar_width, label=strategy.replace('-', ' ').title(), 
                      color=color, alpha=0.8, edgecolor='black', linewidth=1)
    
    axes[1, 0].set_xlabel('Alpha (α) Values', fontweight='bold')
    axes[1, 0].set_ylabel('Neurons with Shrinkage (%)', fontweight='bold')
    axes[1, 0].set_title('Shrinkage Classification Comparison', fontweight='bold', fontsize=12)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(alpha_labels, rotation=45, ha='right')
    axes[1, 0].set_ylim([0, 105])
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Plot 4: Identical % Comparison
    for i, (strategy, color) in enumerate(zip(strategies, colors)):
        offset = (i - 1) * bar_width
        axes[1, 1].bar(x_pos + offset, data_by_metric['identical_pct'][strategy], 
                      bar_width, label=strategy.replace('-', ' ').title(), 
                      color=color, alpha=0.8, edgecolor='black', linewidth=1)
    
    axes[1, 1].set_xlabel('Alpha (α) Values', fontweight='bold')
    axes[1, 1].set_ylabel('Neurons Identical (%)', fontweight='bold')
    axes[1, 1].set_title('Identical Classification Comparison', fontweight='bold', fontsize=12)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(alpha_labels, rotation=45, ha='right')
    axes[1, 1].set_ylim([0, 105])
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved strategy comparison bar graphs: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("PERTURBATION ANALYSIS VISUALIZATION")
    print("=" * 70)
    
    # Configuration
    BASE_DIR = os.getcwd()
    nnet_path = os.path.join(BASE_DIR, "nnet", "ACASXU_run2a_1_1_batch_2000.nnet")
    # nnet_path = "/home/subhankar-jana/artifact/nnet/ACASXU_run2a_1_1_batch_2000.nnet"
    input_bounds = [(-0.5, 0.5) for _ in range(5)]
    
    # Alpha values to analyze
    alphas = np.linspace(0.01, 1.0, 15)
    strategies = ['worst-case', 'best-case', 'random']
    
    print(f"\nConfiguration:")
    print(f"  Network: ACASXU_run2a_1_1_batch_2000.nnet")
    print(f"  Input bounds: [{-0.5}, {0.5}] for 5 inputs")
    print(f"  Alpha values: {len(alphas)} values from {alphas[0]:.3f} to {alphas[-1]:.3f}")
    print(f"  Strategies: {', '.join(strategies)}")
    
    # Run analysis
    results = analyze_all_alphas(nnet_path, input_bounds, alphas, strategies)
    output1_path = 'all_categories_random_case,png'
    plot_all_categories(alphas, results, 'random', output1_path)
    output2_path = 'all_categories_best_case,png'
    plot_all_categories(alphas, results, 'best-case', output2_path)
    output3_path = 'all_categories_worst_case,png'
    plot_all_categories(alphas, results, 'worst-case', output3_path)
    
    # # Create output directory
    # output_dir = "/home/subhankar-jana/artifact/perturbation_plots"
    # os.makedirs(output_dir, exist_ok=True)
    
    # # Generate individual strategy plots
    # print("\n" + "=" * 70)
    # print("GENERATING PLOTS")
    # print("=" * 70)
    
    # for strategy in strategies:
    #     output_path = os.path.join(output_dir, f"strategy_{strategy.replace('-', '_')}.png")
    #     plot_strategy_analysis(alphas, results, strategy, output_path)
    
    # # Generate combined comparison plot
    # combined_path = os.path.join(output_dir, "strategies_comparison.png")
    # plot_combined_comparison(alphas, results, combined_path)
    
    # # Generate bar graphs for each strategy
    # print("\n" + "=" * 70)
    # print("GENERATING BAR GRAPHS")
    # print("=" * 70)
    
    # for strategy in strategies:
    #     output_path = os.path.join(output_dir, f"bar_graph_{strategy.replace('-', '_')}.png")
    #     plot_bar_graph_by_strategy(alphas, results, strategy, output_path)
    
    # # Generate combined bar graph comparison
    # bar_comparison_path = os.path.join(output_dir, "bar_comparison_strategies.png")
    # plot_bar_comparison_strategies(alphas, results, bar_comparison_path)
    
    # # Print summary statistics
    # print("\n" + "=" * 70)
    # print("SUMMARY STATISTICS")
    # print("=" * 70)
    
    # for strategy in strategies:
    #     print(f"\n{strategy.upper()} STRATEGY:")
    #     print("-" * 70)
    #     print(f"{'Alpha':<10} {'Expansion Factor':<20} {'% Expansion':<15} {'% Shrinkage':<15}")
    #     print("-" * 70)
        
    #     for alpha in alphas[::2]:  # Print every other alpha for brevity
    #         data = results[alpha].get(strategy, {})
    #         exp_factor = data.get('expansion_factor', 0)
    #         exp_pct = data.get('expansion_pct', 0)
    #         shrink_pct = data.get('shrinkage_pct', 0)
            
    #         print(f"{alpha:<10.3f} {exp_factor:<20.2f}x {exp_pct:<15.2f}% {shrink_pct:<15.2f}%")
    
    # print("\n" + "=" * 70)
    # print("VISUALIZATION COMPLETE!")
    # print("=" * 70)
    # print(f"\nPlots saved to: {output_dir}/")
    # print(f"\nLine Plots:")
    # print(f"  - strategy_worst_case.png")
    # print(f"  - strategy_best_case.png")
    # print(f"  - strategy_random.png")
    # print(f"  - strategies_comparison.png")
    # print(f"\nBar Graphs:")
    # print(f"  - bar_graph_worst_case.png")
    # print(f"  - bar_graph_best_case.png")
    # print(f"  - bar_graph_random.png")
    # print(f"  - bar_comparison_strategies.png")


if __name__ == "__main__":
    main()
