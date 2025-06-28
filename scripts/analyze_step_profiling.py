#!/usr/bin/env python3
"""
Script to analyze step-by-step profiling data from training runs.
This shows the detailed breakdown of time spent in each component of the training step.
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import argparse

def extract_step_timing_data(run_id: str, project_name: str) -> Dict[str, List[float]]:
    """Extract step-by-step timing data from a wandb run."""
    api = wandb.Api()
    
    try:
        run = api.run(f"{project_name}/{run_id}")
        history = run.history()
        
        # Extract step-specific timing columns
        step_timing_data = {}
        for col in history.columns:
            if col.startswith('timing_step_'):
                component = col.replace('timing_step_', '')
                step_timing_data[component] = history[col].dropna().tolist()
        
        return step_timing_data
    
    except Exception as e:
        print(f"Error extracting data from wandb: {e}")
        return {}

def analyze_step_breakdown(timing_data: Dict[str, List[float]]):
    """Analyze the breakdown of time within each training step."""
    if not timing_data:
        print("No step timing data available")
        return
    
    print("=" * 80)
    print("STEP-BY-STEP TRAINING BREAKDOWN ANALYSIS")
    print("=" * 80)
    
    # Calculate statistics for each step component
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
    
    # Sort by mean time
    sorted_components = sorted(analysis.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print(f"{'Component':<25} {'Mean (s)':<10} {'Total (s)':<10} {'Count':<8} {'P95 (s)':<10} {'% of Step':<10}")
    print("-" * 80)
    
    total_step_time = sum(stats['total'] for stats in analysis.values())
    
    for component, stats in sorted_components:
        percentage = (stats['total'] / total_step_time) * 100
        print(f"{component:<25} {stats['mean']:<10.4f} {stats['total']:<10.4f} "
              f"{stats['count']:<8} {stats['p95']:<10.4f} {percentage:<10.1f}%")
    
    print(f"\nTotal step time: {total_step_time:.4f} seconds")
    
    # Identify bottlenecks
    print("\n" + "=" * 80)
    print("STEP BOTTLENECK ANALYSIS")
    print("=" * 80)
    
    for component, stats in sorted_components[:3]:
        percentage = (stats['total'] / total_step_time) * 100
        print(f"🔍 {component}: {percentage:.1f}% of step time ({stats['mean']:.4f}s avg)")
        
        if stats['p95'] > stats['mean'] * 2:
            print(f"   ⚠️  High variance: P95 is {stats['p95']/stats['mean']:.1f}x mean")
        
        if stats['count'] > 1000:
            print(f"   📊 High frequency: {stats['count']} operations")
    
    return analysis

def plot_step_breakdown(analysis: Dict[str, Dict[str, float]], save_path: str = "step_breakdown_analysis.png"):
    """Plot step breakdown analysis."""
    if not analysis:
        print("No analysis data to plot")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    components = list(analysis.keys())
    means = [analysis[comp]['mean'] for comp in components]
    totals = [analysis[comp]['total'] for comp in components]
    p95s = [analysis[comp]['p95'] for comp in components]
    counts = [analysis[comp]['count'] for comp in components]
    
    # Calculate percentages
    total_time = sum(totals)
    percentages = [(total / total_time) * 100 for total in totals]
    
    # Mean time per operation
    bars1 = ax1.bar(range(len(components)), means, color='skyblue')
    ax1.set_title('Mean Time per Step Component', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xticks(range(len(components)))
    ax1.set_xticklabels(components, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, mean in zip(bars1, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(means)*0.01,
                f'{mean:.3f}s', ha='center', va='bottom', fontsize=8)
    
    # Percentage of total step time
    bars2 = ax2.bar(range(len(components)), percentages, color='lightcoral')
    ax2.set_title('Percentage of Total Step Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_xticks(range(len(components)))
    ax2.set_xticklabels(components, rotation=45, ha='right')
    
    # Add percentage labels
    for bar, pct in zip(bars2, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(percentages)*0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 95th percentile
    bars3 = ax3.bar(range(len(components)), p95s, color='lightgreen')
    ax3.set_title('95th Percentile Time', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_xticks(range(len(components)))
    ax3.set_xticklabels(components, rotation=45, ha='right')
    
    # Pie chart of total time distribution
    ax4.pie(percentages, labels=components, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Time Distribution in Training Step', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Step breakdown analysis plot saved to {save_path}")

def generate_optimization_recommendations(analysis: Dict[str, Dict[str, float]]):
    """Generate specific optimization recommendations based on step analysis."""
    if not analysis:
        return
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    
    sorted_components = sorted(analysis.items(), key=lambda x: x[1]['mean'], reverse=True)
    total_time = sum(stats['total'] for stats in analysis.values())
    
    for component, stats in sorted_components[:3]:
        percentage = (stats['total'] / total_time) * 100
        
        if component == "forward_backward" and percentage > 50:
            print(f"🔧 {component} ({percentage:.1f}%): Consider model optimization")
            print("   - Reduce model complexity or number of layers")
            print("   - Use mixed precision training")
            print("   - Optimize JAX compilation")
        
        elif component == "model_partition" and stats['mean'] > 0.01:
            print(f"🔧 {component} ({percentage:.1f}%): Consider caching partition")
            print("   - Cache model partition if filter_spec doesn't change")
            print("   - Optimize partition operation")
        
        elif component == "optimizer_update" and stats['mean'] > 0.01:
            print(f"🔧 {component} ({percentage:.1f}%): Consider optimizer optimization")
            print("   - Use simpler optimizer (SGD instead of Adam)")
            print("   - Reduce weight decay")
            print("   - Optimize update computation")
        
        elif component == "model_update" and stats['mean'] > 0.01:
            print(f"🔧 {component} ({percentage:.1f}%): Consider update optimization")
            print("   - Batch model updates")
            print("   - Optimize apply_updates operation")
        
        elif "forward_pass" in component and percentage > 30:
            print(f"🔧 {component} ({percentage:.1f}%): Forward pass optimization")
            print("   - Reduce batch size")
            print("   - Use model parallelism")
            print("   - Optimize model architecture")
        
        elif "backward_pass" in component and percentage > 30:
            print(f"🔧 {component} ({percentage:.1f}%): Backward pass optimization")
            print("   - Use gradient checkpointing")
            print("   - Optimize gradient computation")
            print("   - Consider mixed precision")

def main():
    parser = argparse.ArgumentParser(description='Analyze step-by-step profiling data from wandb')
    parser.add_argument('--run_id', type=str, required=True, help='Wandb run ID')
    parser.add_argument('--project', type=str, default='foundational_ssm_pretrain_decoding', 
                       help='Wandb project name')
    parser.add_argument('--save_plot', type=str, default='step_breakdown_analysis.png',
                       help='Path to save the step breakdown plot')
    
    args = parser.parse_args()
    
    print(f"Extracting step timing data from wandb run: {args.run_id}")
    
    # Extract step timing data
    timing_data = extract_step_timing_data(args.run_id, args.project)
    
    if not timing_data:
        print("No step timing data found. Make sure:")
        print("1. The run ID is correct")
        print("2. The project name is correct")
        print("3. The run used 'detailed' or 'separate_fb' profiling level")
        return
    
    print(f"Found step timing data for {len(timing_data)} components")
    
    # Analyze step breakdown
    analysis = analyze_step_breakdown(timing_data)
    
    # Plot results
    plot_step_breakdown(analysis, args.save_plot)
    
    # Generate recommendations
    generate_optimization_recommendations(analysis)
    
    print("\nStep analysis complete!")

if __name__ == "__main__":
    main() 