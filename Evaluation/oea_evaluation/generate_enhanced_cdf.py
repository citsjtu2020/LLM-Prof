#!/usr/bin/env python3
"""
Generate enhaned CDF plot with P50 markers and efficiency thresholds
Uses fixed data (gate_up_proj split into gate_proj and up_proj)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Read fixed data
data_file = Path(__file__).parent / 'all_operator_instances_fixed.csv'
df = pd.read_csv(data_file)

print(f"Total instances: {len(df)}")

# Focus on key operators for CDF
cdf_operators = ['up_proj', 'gate_proj', 'down_proj', 'qkv_proj', 'o_proj', 'attention']
df_cdf = df[df['operator'].isin(cdf_operators)].copy()

print(f"\nCDF operators: {cdf_operators}")
print(f"Filtered instances: {len(df_cdf)}")

# Calculate statistics for each operator
stats = {}
for op in cdf_operators:
    op_data = df_cdf[df_cdf['operator'] == op]['efficiency'].values
    if len(op_data) > 0:
        stats[op] = {
            'data': op_data,
            'n': len(op_data),
            'p25': np.percentile(op_data, 25),
            'p50': np.percentile(op_data, 50),
            'p75': np.percentile(op_data, 75),
            'min': op_data.min(),
            'max': op_data.max()
        }
        print(f"{op:15s}: P25={stats[op]['p25']:6.1%}, P50={stats[op]['p50']:6.1%}, P75={stats[op]['p75']:6.1%}")

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

# Define colors for each operator
colors = {
    'up_proj': '#1f77b4',
    'gate_proj': '#ff7f0e', 
    'down_proj': '#2ca02c',
    'qkv_proj': '#d62728',
    'o_proj': '#9467bd',
    'attention': '#8c564b'
}

# Plot CDF for each operator
for op in cdf_operators:
    if op not in stats:
        continue
    
    data = stats[op]['data']
    data_sorted = np.sort(data)
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    
    # Plot CDF line
    label = f"{op} (P50={stats[op]['p50']:.1%}, n={stats[op]['n']})"
    ax.plot(data_sorted, cdf, label=label, linewidth=2.5, color=colors[op], alpha=0.8)
    
    # Mark P50 point
    p50_val = stats[op]['p50']
    ax.plot(p50_val, 0.5, 'o', markersize=10, color=colors[op], 
            markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    
    # Add P50 text annotation (offset to avoid overlap)
    offset_y = 0.02 if op in ['up_proj', 'gate_proj'] else -0.05
    # ax.annotate(f'{p50_val:.1%}', 
    #             xy=(p50_val, 0.5), 
    #             xytext=(p50_val, 0.5 + offset_y),
    #             fontsize=9, 
    #             ha='center',
    #             bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[op], alpha=0.3, edgecolor='none'))

# Add vertical reference lines for efficiency thresholds
thresholds = [0.10, 0.30, 0.50, 0.80]
threshold_labels = ['10%', '30%', '50%', '80%']

for threshold, label in zip(thresholds, threshold_labels):
    ax.axvline(x=threshold, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
    # ax.text(threshold, 0.98, label, 
    #         ha='center', va='top', 
    #         fontsize=16, 
    #         color='gray',
    #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

# Styling
ax.set_xlabel('Operator Efficiency', fontsize=20, fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=20, fontweight='bold')
# ax.set_title('Cumulative Distribution of Operator Efficiency (165 instances)', 
#              fontsize=14, fontweight='bold', pad=15)

# Set axis limits and format
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.0%}'))

# Set tick label font size - 放大刻度标签字体
ax.tick_params(axis='both', which='major', labelsize=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Legend - 放大图例字体，添加浅灰色背景以提高可见性
ax.legend(loc='lower right', fontsize=20, framealpha=0.98, 
          edgecolor='darkgray', fancybox=True, shadow=True,
          facecolor='whitesmoke', frameon=True)

# Tight layout
plt.tight_layout()

# Save figure
output_file = Path(__file__).parent / 'oea_operator_efficiency_cdf.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_file.name}")

# Print detailed statistics
print("\n" + "="*80)
print("Efficiency Statistics (Fixed Data)")
print("="*80)

for op in cdf_operators:
    if op not in stats:
        continue
    s = stats[op]
    print(f"\n{op}:")
    print(f"  n={s['n']:3d}, Min={s['min']:6.1%}, P25={s['p25']:6.1%}, "
          f"P50={s['p50']:6.1%}, P75={s['p75']:6.1%}, Max={s['max']:6.1%}")

print("\n" + "="*80)
print("Enhanced CDF plot generated successfully!")
print("="*80)