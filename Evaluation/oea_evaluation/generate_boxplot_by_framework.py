#!/usr/bin/env python3
"""
Generate BottleScore box plots grouped by framework (vLLM, SGLang, RTP-LLM)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Read data
data_file = Path(__file__).parent / 'all_operator_instances_fixed.csv'
df = pd.read_csv(data_file)

print(f"Total instances: {len(df)}")
print(f"Frameworks: {df['framework'].unique()}")
print(f"Operators: {df['operator'].unique()}")

# Filter to key operators (n >= 5 instances)
operator_counts = df['operator'].value_counts()
key_operators = operator_counts[operator_counts >= 5].index.tolist()
print(f"\nKey operators (n >= 5): {len(key_operators)}")
print(key_operators)

df_filtered = df[df['operator'].isin(key_operators)].copy()
print(f"\nFiltered instances: {len(df_filtered)}")

# Get median bottleneck score per operator for sorting
operator_medians = df_filtered.groupby('operator')['bottleneck_score'].median().sort_values(ascending=False)
print("\nOperator median BottleScore (sorted):")
print(operator_medians)

# Create figure with subplots for each framework
frameworks = ['vLLM', 'SGLang', 'RTP-LLM']
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

for idx, framework in enumerate(frameworks):
    ax = axes[idx]
    
    # Filter data for this framework
    df_fw = df_filtered[df_filtered['framework'] == framework].copy()
    
    if len(df_fw) == 0:
        ax.text(0.5, 0.5, f'No data for {framework}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'{framework}\n(n=0)', fontsize=14, fontweight='bold')
        continue
    
    # Get operators present in this framework
    fw_operators = df_fw['operator'].unique()
    # Sort by global median (to keep consistent order across frameworks)
    fw_operators_sorted = [op for op in operator_medians.index if op in fw_operators]
    
    # Prepare data for box plot
    data_to_plot = []
    labels = []
    counts = []
    
    for operator in fw_operators_sorted:
        op_data = df_fw[df_fw['operator'] == operator]['bottleneck_score'].values
        if len(op_data) > 0:
            data_to_plot.append(op_data)
            labels.append(operator)
            counts.append(len(op_data))
    
    if len(data_to_plot) == 0:
        ax.text(0.5, 0.5, f'No valid data for {framework}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'{framework}\n(n=0)', fontsize=14, fontweight='bold')
        continue
    
    # Create box plot
    bp = ax.boxplot(data_to_plot, 
                    labels=labels,
                    patch_artist=True,
                    widths=0.6,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6, markeredgecolor='darkred'))
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(data_to_plot)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Styling
    ax.set_ylabel('BottleScore', fontsize=12, fontweight='bold')
    ax.set_xlabel('Operator', fontsize=12, fontweight='bold')
    ax.set_title(f'{framework}\n(n={len(df_fw)} instances)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add instance counts as text
    for i, (label, count) in enumerate(zip(labels, counts)):
        ax.text(i+1, ax.get_ylim()[1] * 0.95, f'n={count}', 
                ha='center', va='top', fontsize=8, color='gray')

plt.tight_layout()

# Save figure
output_file = Path(__file__).parent / 'oea_operator_bottlescore_by_framework.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_file.name}")

# Print statistics by framework
print("\n" + "="*80)
print("BottleScore Statistics by Framework")
print("="*80)

for framework in frameworks:
    df_fw = df_filtered[df_filtered['framework'] == framework]
    if len(df_fw) == 0:
        print(f"\n{framework}: No data")
        continue
    
    print(f"\n{framework} (n={len(df_fw)} instances):")
    print("-" * 60)
    
    for operator in operator_medians.index:
        op_data = df_fw[df_fw['operator'] == operator]['bottleneck_score']
        if len(op_data) == 0:
            continue
        
        print(f"{operator:15s}: n={len(op_data):3d}, "
              f"Min={op_data.min():.4f}, "
              f"P25={op_data.quantile(0.25):.4f}, "
              f"P50={op_data.median():.4f}, "
              f"P75={op_data.quantile(0.75):.4f}, "
              f"Max={op_data.max():.4f}")

print("\n" + "="*80)
print("Framework comparison completed!")
print("="*80)
