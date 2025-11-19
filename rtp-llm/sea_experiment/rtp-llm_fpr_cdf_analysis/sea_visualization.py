#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEA层可视化分析脚本
生成三个核心图表：
1. GPU类型效率对比小提琴图  
2. 模型参数vs FPR和Token Size双纵轴箱线图
3. Token Size、FPR和GPU利用率的CDF分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

class SEAVisualizer:
    def __init__(self, data_file='extract_all_cases_fpr_data_with_k_support.txt'):
        """初始化可视化器"""
        self.data_file = data_file
        self.df = None
        self.load_data()
        self.preprocess_data()
    
    def load_data(self):
        """加载数据"""
        print("Loading data from:", self.data_file)
        
        # 读取数据文件，跳过头部说明
        with open(self.data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 找到数据开始行
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('序号'):
                data_start = i
                break
        
        # 解析数据
        data_lines = []
        for line in lines[data_start+2:]:  # 跳过表头和分隔线
            if line.strip() and not line.startswith('='):
                if '统计信息' in line:
                    break
                data_lines.append(line.strip())
        
        # 解析每行数据
        records = []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 10:
                try:
                    record = {
                        'id': int(parts[0]),
                        'pod_name': ' '.join(parts[1:-8]),
                        'model': parts[-8],
                        'gpu_type': parts[-7],
                        'n_gpu': int(parts[-6]),
                        'qps': float(parts[-5]),
                        'u_gpu': float(parts[-4]),
                        'f_peak': float(parts[-3]),
                        'fpr': float(parts[-2]),
                        'token_size': int(parts[-1])
                    }
                    records.append(record)
                except (ValueError, IndexError) as e:
                    print(f"Skipping line due to parsing error: {line[:50]}...")
                    continue
        
        self.df = pd.DataFrame(records)
        print(f"Loaded {len(self.df)} records")
    
    def preprocess_data(self):
        """数据预处理"""
        # 计算资源权重
        self.df['resource_weight'] = self.df['n_gpu'] * self.df['f_peak'] * self.df['u_gpu']
        
        # 模型规模分类
        def classify_model_size(model):
            model_lower = model.lower()
            if any(x in model_lower for x in ['0.5b', '0.6b']):
                return '<1B'
            elif any(x in model_lower for x in ['1.5b', '1.8b', '1.7b']):
                return '1-2B'
            elif any(x in model_lower for x in ['3b', '4b', '7b', '8b']):
                return '3-8B'
            elif any(x in model_lower for x in ['13b', '14b']):
                return '13-14B'
            elif any(x in model_lower for x in ['30b', '32b']):
                return '30-32B'
            else:
                return 'Other'
        
        self.df['model_size'] = self.df['model'].apply(classify_model_size)
        
        # Token长度分类
        def classify_token_length(token_size):
            if token_size < 100:
                return '<100'
            elif token_size < 1000:
                return '100-1K'
            elif token_size < 5000:
                return '1K-5K'
            else:
                return '>5K'
        
        self.df['token_category'] = self.df['token_size'].apply(classify_token_length)
        
        # GPU类型排序（按F_peak）
        gpu_order = ['custom accelerator', 'L20', 'A10', 'H20', 'A800', 'A100', 'H800']
        self.df['gpu_type'] = pd.Categorical(self.df['gpu_type'], categories=gpu_order, ordered=True)
        
        print("Data preprocessing completed")
    
    def plot_gpu_efficiency_violin(self, save_path='sea_figure1_gpu_violin.png'):
        """图1: GPU类型效率对比小提琴图"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # 准备数据
        gpu_data = []
        gpu_fpeak = {}
        
        for gpu_type in self.df['gpu_type'].cat.categories:
            gpu_subset = self.df[self.df['gpu_type'] == gpu_type]
            if len(gpu_subset) > 0:
                gpu_data.extend([(gpu_type, fpr) for fpr in gpu_subset['fpr']])
                gpu_fpeak[gpu_type] = gpu_subset['f_peak'].iloc[0]
        
        gpu_df = pd.DataFrame(gpu_data, columns=['GPU_Type', 'FPR'])
        
        # 绘制小提琴图
        parts = ax.violinplot([gpu_df[gpu_df['GPU_Type']==gpu]['FPR'].values 
                              for gpu in gpu_df['GPU_Type'].unique()], 
                             positions=range(len(gpu_df['GPU_Type'].unique())), 
                             showmeans=True, showmedians=True)
        
        # 美化小提琴图
        colors = plt.cm.Set3(np.linspace(0, 1, len(gpu_df['GPU_Type'].unique())))
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # 叠加散点图
        for i, gpu_type in enumerate(gpu_df['GPU_Type'].unique()):
            gpu_subset = self.df[self.df['gpu_type'] == gpu_type]
            y_values = gpu_subset['fpr'].values
            x_values = np.random.normal(i, 0.04, size=len(y_values))  # 添加随机抖动
            ax.scatter(x_values, y_values, alpha=0.6, s=30, color='black')
        
        # 设置x轴标签和F_peak标注
        gpu_labels = []
        for gpu_type in gpu_df['GPU_Type'].unique():
            fpeak = gpu_fpeak.get(gpu_type, 0)
            count = len(self.df[self.df['gpu_type'] == gpu_type])
            gpu_labels.append(f'{gpu_type}\n(F_peak={fpeak:.0f})\n(n={count})')
        
        ax.set_xticks(range(len(gpu_df['GPU_Type'].unique())))
        ax.set_xticklabels(gpu_labels, rotation=45, ha='right')
        ax.set_ylabel('FPR (FLOPs Per Request)', fontsize=12)
        ax.set_yscale('log')
        ax.set_title('SEA Figure 1: GPU Type Efficiency Comparison\n(Higher F_peak ≠ Better Efficiency)', 
                    fontsize=14, fontweight='bold')
        
        # 添加统计信息到右下角
        stats_text = []
        for gpu_type in gpu_df['GPU_Type'].unique():
            gpu_subset = self.df[self.df['gpu_type'] == gpu_type]
            if len(gpu_subset) > 0:
                median_fpr = gpu_subset['fpr'].median()
                stats_text.append(f'{gpu_type}: {median_fpr:.2f}')
        
        ax.text(0.98, 0.02, 'Median FPR:\n' + '\n'.join(stats_text), 
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8),
                horizontalalignment='right', verticalalignment='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure 1 saved to: {save_path}")
        plt.show()
    
    def plot_multidimensional_heatmap(self, save_path='sea_figure2_model_params_dual_axis.png'):
        """图2: 模型参数vs FPR和Token Size双纵轴箱线图"""
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
        
        # 准备数据 - 按模型参数分组
        size_order = ['<1B', '1-2B', '3-8B', '13-14B', '30-32B', 'Other']
        
        # 过滤掉没有数据的分组
        valid_sizes = []
        fpr_data = []
        token_data = []
        
        for size in size_order:
            size_subset = self.df[self.df['model_size'] == size]
            if len(size_subset) > 0:
                valid_sizes.append(size)
                fpr_data.append(size_subset['fpr'].values)
                token_data.append(size_subset['token_size'].values)
        
        # 绘制FPR箱线图（左纵轴）
        positions = np.arange(len(valid_sizes))
        bp1 = ax1.boxplot(fpr_data, positions=positions, widths=0.35, 
                         patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='darkblue', linewidth=2))
        
        # 设置左纵轴
        ax1.set_xlabel('Model Parameters', fontsize=12, fontweight='bold')
        ax1.set_ylabel('FPR (FLOPs Per Request)', fontsize=12, fontweight='bold', color='darkblue')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor='darkblue')
        ax1.set_xticks(positions)
        ax1.set_xticklabels(valid_sizes)
        
        # 创建右纵轴
        ax2 = ax1.twinx()
        
        # 绘制Token Size箱线图（右纵轴）
        bp2 = ax2.boxplot(token_data, positions=positions + 0.4, widths=0.35,
                         patch_artist=True, boxprops=dict(facecolor='lightcoral', alpha=0.7),
                         medianprops=dict(color='darkred', linewidth=2))
        
        # 设置右纵轴
        ax2.set_ylabel('Token Size', fontsize=12, fontweight='bold', color='darkred')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor='darkred')
        
        # 添加图例（白色框，右上角）
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', alpha=0.7, label='FPR (Left Axis)'),
            Patch(facecolor='lightcoral', alpha=0.7, label='Token Size (Right Axis)')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=10,
                  bbox_to_anchor=(0.98, 0.98), frameon=True, facecolor='white', edgecolor='black')
        
        # 设置标题
        plt.title('SEA Figure 2: Model Parameters vs FPR and Token Size\n(Dual Y-axis Boxplot Analysis)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # 添加统计信息（黄色框，右下角）
        stats_text = []
        for i, size in enumerate(valid_sizes):
            size_subset = self.df[self.df['model_size'] == size]
            median_fpr = size_subset['fpr'].median()
            median_token = size_subset['token_size'].median()
            count = len(size_subset)
            stats_text.append(f'{size}: FPR={median_fpr:.2f}, Token={median_token:.0f} (n={count})')
        
        ax1.text(0.98, 0.02, 'Median Statistics:\n' + '\n'.join(stats_text), 
                transform=ax1.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8),
                horizontalalignment='right', verticalalignment='bottom')
        
        # 添加网格
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure 2 saved to: {save_path}")
        plt.show()
    
    def plot_correlation_analysis(self, save_path='sea_figure3_cdf_analysis.png'):
        """图3: Token Size、FPR和GPU利用率的CDF分析"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 准备数据
        token_size = self.df['token_size'].values
        fpr = self.df['fpr'].values
        gpu_util = self.df['u_gpu'].values
        
        # 计算CDF
        def compute_cdf(data):
            sorted_data = np.sort(data)
            n = len(sorted_data)
            cdf = np.arange(1, n + 1) / n
            return sorted_data, cdf
        
        # 计算三个变量的CDF
        token_sorted, token_cdf = compute_cdf(token_size)
        fpr_sorted, fpr_cdf = compute_cdf(fpr)
        gpu_sorted, gpu_cdf = compute_cdf(gpu_util)
        
        # 绘制CDF曲线
        ax.plot(token_sorted, token_cdf, 'b-', linewidth=3, label='Token Size', alpha=0.8)
        ax.plot(fpr_sorted, fpr_cdf, 'r-', linewidth=3, label='FPR', alpha=0.8)
        ax.plot(gpu_sorted, gpu_cdf, 'g-', linewidth=3, label='GPU Utilization', alpha=0.8)
        
        # 设置对数刻度（Token Size和FPR范围较大）
        ax.set_xscale('log')
        
        # 设置坐标轴标签
        ax.set_xlabel('Value (Log Scale)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        ax.set_title('SEA Figure 3: Cumulative Distribution Functions\n(Token Size, FPR, and GPU Utilization)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置y轴范围
        ax.set_ylim(0, 1)
        
        # 添加百分位数标记线
        percentiles = [0.25, 0.5, 0.75, 0.9]
        colors = ['orange', 'purple', 'brown', 'pink']
        
        for p, color in zip(percentiles, colors):
            ax.axhline(y=p, color=color, linestyle=':', alpha=0.6, linewidth=1.5)
            ax.text(0.02, p + 0.02, f'P{int(p*100)}', transform=ax.get_yaxis_transform(), 
                   fontsize=10, color=color, fontweight='bold')
        
        # 添加图例
        ax.legend(loc='center right', fontsize=11, frameon=True, 
                 fancybox=True, shadow=True, framealpha=0.9)
        
        # 添加统计信息框（移动到右下角）
        stats_text = [
            f'Dataset: {len(self.df)} services',
            '',
            'Token Size:',
            f'  Min: {token_size.min()}',
            f'  Median: {np.median(token_size):.0f}',
            f'  Max: {token_size.max()}',
            f'  P90: {np.percentile(token_size, 90):.0f}',
            '',
            'FPR:',
            f'  Min: {fpr.min():.3f}',
            f'  Median: {np.median(fpr):.2f}',
            f'  Max: {fpr.max():.1f}',
            f'  P90: {np.percentile(fpr, 90):.1f}',
            '',
            'GPU Utilization:',
            f'  Min: {gpu_util.min():.3f}',
            f'  Median: {np.median(gpu_util):.3f}',
            f'  Max: {gpu_util.max():.3f}',
            f'  P90: {np.percentile(gpu_util, 90):.3f}'
        ]
        
        ax.text(0.98, 0.02, '\n'.join(stats_text), 
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8),
                horizontalalignment='right', verticalalignment='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure 3 saved to: {save_path}")
        plt.show()
    
    def generate_all_figures(self):
        """生成所有图表"""
        print("Generating SEA visualization figures...")
        print("="*50)
        
        self.plot_gpu_efficiency_violin()
        print()
        
        self.plot_multidimensional_heatmap()
        print()
        
        self.plot_correlation_analysis()
        print()
        
        print("All SEA figures generated successfully!")
        print("Files saved:")
        print("- sea_figure1_gpu_violin.png") 
        print("- sea_figure2_model_params_dual_axis.png")
        print("- sea_figure3_cdf_analysis.png")

def main():
    """主函数"""
    # 创建可视化器
    visualizer = SEAVisualizer()
    
    # 生成所有图表
    visualizer.generate_all_figures()
    
    # 打印数据统计摘要
    print("\n" + "="*50)
    print("DATA SUMMARY:")
    print("="*50)
    print(f"Total cases: {len(visualizer.df)}")
    print(f"FPR range: {visualizer.df['fpr'].min():.3f} - {visualizer.df['fpr'].max():.1f}")
    print(f"Efficiency gap: {visualizer.df['fpr'].max() / visualizer.df['fpr'].min():.1f}×")
    print(f"GPU types: {len(visualizer.df['gpu_type'].unique())}")
    print(f"Model types: {len(visualizer.df['model'].unique())}")
    
    # GPU类型效率排名
    print("\nGPU Efficiency Ranking (by median FPR):")
    gpu_efficiency = visualizer.df.groupby('gpu_type')['fpr'].agg(['median', 'count']).sort_values('median')
    for gpu, stats in gpu_efficiency.iterrows():
        print(f"  {gpu}: {stats['median']:.2f} (n={stats['count']})")

if __name__ == "__main__":
    main()