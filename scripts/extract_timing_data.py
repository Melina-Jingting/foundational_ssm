#!/usr/bin/env python3
"""
Script to extract timing data from wandb logs and analyze training performance.
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import argparse

def extract_timing_data_from_wandb(run_id: str, project_name: str) -> Dict[str, List[float]]:
    """Extract timing data from a wandb run."""
    api = wandb.Api()
    
    try:
        run = api.run(f"{project_name}/{run_id}")
        history = run.history()
        
        # Extract timing columns
        timing_data = {}
        for col in history.columns:
            if col.startswith('timing_'):
                component = col.replace('timing_', '')
                timing_data[component] = history[col].dropna().tolist()
        
        return timing_data
    
    except Exception as e:
        print(f"Error extracting data from wandb: {e}")
        return {}

def analyze_training_performance(timing_data: Dict[str, List[float]]):
    """Analyze training performance from timing data."""
    if not timing_data:
        print("No timing data available")
        return
    
    print("=" * 60)
    print("TRAINING PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Calculate statistics for each component
    analysis = {}
    for component, times in timing_data.items():
        if times:
            analysis[component] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': np.sum(times),
                'count': len(times),
                'p50': np.percentile(times, 50),
                'p95': np.percentile(times, 95),
                'p99': np.percentile(times, 99)
            }
    
    # Sort by total time
    sorted_components = sorted(analysis.items(), key=lambda x: x[1]['total'], reverse=True)
    
    print(f"{'Component':<25} {'Mean (s)':<10} {'Total (s)':<10} {'Count':<8} {'P95 (s)':<10}")
    print("-" * 65)
    
    total_time = sum(stats['total'] for stats in analysis.values())
    
    for component, stats in sorted_components:
        percentage = (stats['total'] / total_time) * 100
        print(f"{component:<25} {stats['mean']:<10.4f} {stats['total']:<10.4f} "
              f"{stats['count']:<8} {stats['p95']:<10.4f} ({percentage:.1f}%)")
    
    print(f"\nTotal training time: {total_time:.2f} seconds")
    
    # Identify bottlenecks
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    for component, stats in sorted_components[:3]:
        percentage = (stats['total'] / total_time) * 100
        print(f"🔍 {component}: {percentage:.1f}% of total time")
        
        if stats['p95'] > stats['mean'] * 2:
            print(f"   ⚠️  High variance: P95 is {stats['p95']/stats['mean']:.1f}x mean")
        
        if stats['count'] > 1000:
            print(f"   📊 High frequency: {stats['count']} operations")
    
    return analysis

def plot_timing_analysis(analysis: Dict[str, Dict[str, float]], save_path: str = "training_timing_analysis.png"):
    """Plot timing analysis."""
    if not analysis:
        print("No analysis data to plot")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    components = list(analysis.keys())
    means = [analysis[comp]['mean'] for comp in components]
    totals = [analysis[comp]['total'] for comp in components]
    p95s = [analysis[comp]['p95'] for comp in components]
    counts = [analysis[comp]['count'] for comp in components]
    
    # Mean time per operation
    bars1 = ax1.bar(range(len(components)), means, color='skyblue')
    ax1.set_title('Mean Time per Operation', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xticks(range(len(components)))
    ax1.set_xticklabels(components, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, mean in zip(bars1, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(means)*0.01,
                f'{mean:.3f}s', ha='center', va='bottom', fontsize=8)
    
    # Total time spent
    bars2 = ax2.bar(range(len(components)), totals, color='lightcoral')
    ax2.set_title('Total Time Spent', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_xticks(range(len(components)))
    ax2.set_xticklabels(components, rotation=45, ha='right')
    
    # Add percentage labels
    total_time = sum(totals)
    for bar, total in zip(bars2, totals):
        height = bar.get_height()
        percentage = (total / total_time) * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(totals)*0.01,
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 95th percentile
    bars3 = ax3.bar(range(len(components)), p95s, color='lightgreen')
    ax3.set_title('95th Percentile Time', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_xticks(range(len(components)))
    ax3.set_xticklabels(components, rotation=45, ha='right')
    
    # Operation counts
    bars4 = ax4.bar(range(len(components)), counts, color='gold')
    ax4.set_title('Number of Operations', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Count')
    ax4.set_xticks(range(len(components)))
    ax4.set_xticklabels(components, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Timing analysis plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract and analyze timing data from wandb')
    parser.add_argument('--run_id', type=str, required=True, help='Wandb run ID')
    parser.add_argument('--project', type=str, default='foundational_ssm_pretrain_decoding', 
                       help='Wandb project name')
    parser.add_argument('--save_plot', type=str, default='training_timing_analysis.png',
                       help='Path to save the timing analysis plot')
    
    args = parser.parse_args()
    
    print(f"Extracting timing data from wandb run: {args.run_id}")
    
    # Extract timing data
    timing_data = extract_timing_data_from_wandb(args.run_id, args.project)
    
    if not timing_data:
        print("No timing data found. Make sure:")
        print("1. The run ID is correct")
        print("2. The project name is correct")
        print("3. The run has timing data logged")
        return
    
    print(f"Found timing data for {len(timing_data)} components")
    
    # Analyze performance
    analysis = analyze_training_performance(timing_data)
    
    # Plot results
    plot_timing_analysis(analysis, args.save_plot)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 