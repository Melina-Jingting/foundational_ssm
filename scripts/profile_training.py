#!/usr/bin/env python3
"""
Comprehensive profiling script for training performance analysis.
This script provides detailed timing breakdowns and performance metrics.
"""

import os
import sys
import time
import cProfile
import pstats
import io
from collections import defaultdict
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# JAX imports
import jax
import jax.numpy as jnp
import jax.profiler as profiler
from jax.profiler import trace

def profile_function(func, *args, **kwargs):
    """Profile a function using cProfile."""
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 functions by cumulative time
    
    return result, s.getvalue()

def analyze_timing_data(timing_logs: Dict[str, List[float]]):
    """Analyze timing data and generate insights."""
    analysis = {}
    
    for component, times in timing_logs.items():
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
    
    return analysis

def plot_timing_breakdown(analysis: Dict[str, Dict[str, float]], save_path: str = "timing_breakdown.png"):
    """Plot timing breakdown analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    components = list(analysis.keys())
    means = [analysis[comp]['mean'] for comp in components]
    totals = [analysis[comp]['total'] for comp in components]
    p95s = [analysis[comp]['p95'] for comp in components]
    counts = [analysis[comp]['count'] for comp in components]
    
    # Mean time per operation
    ax1.bar(components, means)
    ax1.set_title('Mean Time per Operation')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Total time spent
    ax2.bar(components, totals)
    ax2.set_title('Total Time Spent')
    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 95th percentile
    ax3.bar(components, p95s)
    ax3.set_title('95th Percentile Time')
    ax3.set_ylabel('Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Operation counts
    ax4.bar(components, counts)
    ax4.set_title('Number of Operations')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Timing breakdown saved to {save_path}")

def generate_performance_report(analysis: Dict[str, Dict[str, float]]):
    """Generate a comprehensive performance report."""
    print("=" * 60)
    print("PERFORMANCE ANALYSIS REPORT")
    print("=" * 60)
    
    # Sort by total time
    sorted_components = sorted(analysis.items(), key=lambda x: x[1]['total'], reverse=True)
    
    print(f"{'Component':<20} {'Mean (s)':<10} {'Total (s)':<10} {'Count':<8} {'P95 (s)':<10}")
    print("-" * 60)
    
    for component, stats in sorted_components:
        print(f"{component:<20} {stats['mean']:<10.4f} {stats['total']:<10.4f} "
              f"{stats['count']:<8} {stats['p95']:<10.4f}")
    
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    # Identify bottlenecks
    total_time = sum(stats['total'] for stats in analysis.values())
    
    for component, stats in sorted_components[:3]:  # Top 3 bottlenecks
        percentage = (stats['total'] / total_time) * 100
        print(f"{component}: {percentage:.1f}% of total time ({stats['total']:.2f}s)")
        
        if stats['p95'] > stats['mean'] * 2:
            print(f"  ⚠️  High variance: P95 is {stats['p95']/stats['mean']:.1f}x mean")
        
        if stats['count'] > 1000:
            print(f"  📊 High frequency: {stats['count']} operations")
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)
    
    # Generate recommendations
    for component, stats in sorted_components[:3]:
        if component == "training_step" and stats['mean'] > 0.1:
            print(f"🔧 {component}: Consider reducing batch size or model complexity")
        elif component == "data_loading" and stats['mean'] > 0.05:
            print(f"🔧 {component}: Consider data prefetching or caching")
        elif component == "validation" and stats['total'] > total_time * 0.2:
            print(f"🔧 {component}: Consider reducing validation frequency")
        elif "forward_pass" in component and stats['mean'] > 0.05:
            print(f"🔧 {component}: Consider model optimization or JIT compilation")
    
    print(f"\nTotal training time: {total_time:.2f} seconds")

def profile_jax_operations():
    """Profile JAX-specific operations."""
    print("JAX Profiling Information:")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default device: {jax.default_backend()}")
    
    # Check if GPU is available
    try:
        gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
        if gpu_devices:
            print(f"GPU devices: {len(gpu_devices)}")
            print(f"GPU memory: {jax.device_get(jax.device_count())} devices")
        else:
            print("No GPU devices found")
    except Exception as e:
        print(f"Could not check GPU status: {e}")

def main():
    """Main profiling function."""
    print("Starting comprehensive training profiling...")
    
    # Profile JAX setup
    profile_jax_operations()
    
    # Example timing data (replace with actual data from your training)
    example_timing_data = {
        'data_loading': [0.001, 0.002, 0.001, 0.003, 0.001],
        'training_step': [0.05, 0.06, 0.04, 0.07, 0.05],
        'forward_pass': [0.03, 0.04, 0.02, 0.05, 0.03],
        'backward_pass': [0.02, 0.02, 0.02, 0.02, 0.02],
        'optimization': [0.01, 0.01, 0.01, 0.01, 0.01],
        'validation': [0.1, 0.12, 0.08, 0.15, 0.11],
        'logging': [0.001, 0.001, 0.002, 0.001, 0.001]
    }
    
    # Analyze timing data
    analysis = analyze_timing_data(example_timing_data)
    
    # Generate report
    generate_performance_report(analysis)
    
    # Plot results
    plot_timing_breakdown(analysis)
    
    print("\nProfiling complete! Check the generated plots and reports.")

if __name__ == "__main__":
    main() 