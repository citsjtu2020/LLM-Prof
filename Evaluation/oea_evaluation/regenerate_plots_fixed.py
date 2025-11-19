#!/usr/bin/env python3
"""
使用修复后的数据重新生成OEA图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 16

# Read FIXED data
df = pd.read_csv('Evaluation/oea_evaluation/all_operator_instances_fixed.csv')

print(f"Total instances (fixed): {len(df)}")
print(f"Unique operators: {df['operator'].nunique()}")
print(f"Operators: {sorted(df['operator'].unique())}")

# ============================================================================
# 1. Generate BottleScore Box Plot
# ============================================================================

print("\n" + "="*80)
print("Generating BottleScore Box Plot...")
print("="*80)

# Calculate median bottleneck score for each operator
operator_medians = df.groupby('operator')['bottleneck_score'].median().sort_values(ascending=False)
print("\nOperator median BottleScore (sorted):")
print(operator_medians)

# Select operators with sufficient instances (n >= 5) for meaningful box plots
operator_counts = df['operator'].value_counts()
key_operators = [op for op in operator_medians.index
                 if operator_counts[op] >= 5]

print(f"\nKey operators (n >= 5): {len(key_operators)}")
print(key_operators)

# Filter data
df_filtered = df[df['operator'].isin(key_operators)].copy()
print(f"\nFiltered instances: {len(df_filtered)}")

# Define operator category colors
category_colors = {
    'Linear (MLP projections)': '#2ECC71',  # Green
    'Linear (qkv_proj)': '#3498DB',         # Blue
    'Linear (o_proj)': '#9B59B6',           # Purple
    'Attention': '#E74C3C',                 # Red
    'Activation': '#E67E22',                # Orange
    'Other': '#95A5A6'                      # Gray
}

# Create figure
fig, ax = plt.subplots(figsize=(14, 7))

# Prepare data for box plot
box_data = []
box_positions = []
box_colors = []
x_labels = []

for i, operator in enumerate(key_operators):
    data = df_filtered[df_filtered['operator'] == operator]['bottleneck_score']
    n = len(data)
    category = df_filtered[df_filtered['operator'] == operator]['operator_category'].iloc[0]

    box_data.append(data)
    box_positions.append(i)
    box_colors.append(category_colors.get(category, '#95A5A6'))
    x_labels.append(operator)

# Create box plot
bp = ax.boxplot(box_data, positions=box_positions, widths=0.6, patch_artist=True,
                showfliers=True,
                boxprops=dict(linewidth=1.5),
                medianprops=dict(color='black', linewidth=2.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))

# Color boxes by category
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(box_colors[i])
    patch.set_alpha(0.7)

# Add horizontal reference lines
ax.axhline(0.3, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='BottleScore = 0.3')
ax.axhline(0.1, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, label='BottleScore = 0.1')

# Set x-axis labels
ax.set_xticks(box_positions)
ax.set_xticklabels(x_labels, rotation=45, ha='right')

# Labels and title
ax.set_xlabel('Operator Type', fontweight='bold', fontsize=20)
ax.set_ylabel('BottleScore', fontweight='bold', fontsize=20)
ax.set_ylim(-0.02, 0.5)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# Create legend
from matplotlib.patches import Patch
legend_elements = []
for category, color in category_colors.items():
    if category in df_filtered['operator_category'].values:
        legend_elements.append(Patch(facecolor=color, alpha=0.7, label=category))

legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--', alpha=0.5, label='0.3 threshold'))
legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle=':', alpha=0.5, label='0.1 threshold'))

ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, ncol=1, fontsize=18)

plt.tight_layout()
plt.savefig('Evaluation/oea_evaluation/oea_operator_bottlescore_boxplot.png',
            dpi=300, bbox_inches='tight')
print("\n✓ Saved: oea_operator_bottlescore_boxplot.png")

# Print statistics
print("\n" + "="*80)
print("BottleScore Statistics (Fixed Data)")
print("="*80)
for operator in key_operators:
    data = df_filtered[df_filtered['operator'] == operator]['bottleneck_score']
    print(f"\n{operator:15s}: n={len(data):3d}, "
          f"Min={data.min():6.4f}, P25={np.percentile(data, 25):6.4f}, "
          f"P50={np.percentile(data, 50):6.4f}, P75={np.percentile(data, 75):6.4f}, "
          f"Max={data.max():6.4f}, IQR={np.percentile(data, 75) - np.percentile(data, 25):6.4f}")

plt.close()

# ============================================================================
# 2. Generate Efficiency CDF Plot
# ============================================================================

print("\n" + "="*80)
print("Generating Efficiency CDF Plot...")
print("="*80)

# Select key operators for CDF (those with sufficient data)
cdf_operators = ['up_proj', 'gate_proj', 'down_proj', 'qkv_proj', 'o_proj', 'attention']
cdf_operators = [op for op in cdf_operators if op in df['operator'].values]

print(f"CDF operators: {cdf_operators}")

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

# Define colors for operators
operator_colors = {
    'up_proj': '#2ECC71',
    'gate_proj': '#27AE60',
    'down_proj': '#16A085',
    'qkv_proj': '#3498DB',
    'o_proj': '#9B59B6',
    'attention': '#E74C3C'
}

# Plot CDF for each operator
for operator in cdf_operators:
    data = df[df['operator'] == operator]['efficiency'].values
    data_sorted = np.sort(data)
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    
    color = operator_colors.get(operator, '#95A5A6')
    ax.plot(data_sorted * 100, cdf, label=f'{operator} (n={len(data)})',
            linewidth=2.5, color=color)
    
    # Print statistics
    p25, p50, p75 = np.percentile(data, [25, 50, 75])
    print(f"{operator:15s}: P25={p25*100:5.1f}%, P50={p50*100:5.1f}%, P75={p75*100:5.1f}%")

# Labels and formatting
ax.set_xlabel('Operator Efficiency (%)', fontweight='bold', fontsize=20)
ax.set_ylabel('Cumulative Probability', fontweight='bold', fontsize=20)
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='lower right', framealpha=0.9, fontsize=16)

# Add reference lines
ax.axvline(50, color='gray', linestyle='--', alpha=0.3, linewidth=1.5)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1.5)

plt.tight_layout()
plt.savefig('Evaluation/oea_evaluation/oea_operator_efficiency_cdf.png',
            dpi=300, bbox_inches='tight')
print("\n✓ Saved: oea_operator_efficiency_cdf.png")

plt.close()

# ============================================================================
# 3. Generate Summary Statistics
# ============================================================================

print("\n" + "="*80)
print("Summary Statistics (Fixed Data)")
print("="*80)

summary = df.groupby('operator').agg({
    'efficiency': ['count', 'mean', 'median', 'std', 'min', 'max'],
    'time_proportion': ['mean', 'sum'],
    'bottleneck_score': ['mean', 'median', 'max']
}).round(4)

print(summary)

# Save summary to CSV
summary.to_csv('Evaluation/oea_evaluation/operator_summary_fixed.csv')
print("\n✓ Saved: operator_summary_fixed.csv")

print("\n" + "="*80)
print("All plots regenerated successfully with fixed data!")
print("="*80)
