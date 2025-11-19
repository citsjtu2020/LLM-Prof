#!/usr/bin/env python3
"""
FPR CDF分析脚本
分析merged_top_token_size_gpu_results.csv中的FPR数据并生成CDF可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

class FPRCDFAnalyzer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = None
        
    def load_data(self):
        """加载CSV数据"""
        try:
            # 读取CSV文件，使用第一行作为表头
            self.data = pd.read_csv(self.csv_file)
            
            # 获取列名
            columns = self.data.columns.tolist()
            print(f"CSV文件列名: {columns}")
            
            # 验证是否包含FPR列
            if 'FPR' not in columns:
                raise ValueError(f"CSV文件中未找到FPR列。实际列名: {columns}")
            
            # 验证数据完整性
            expected_columns = ['pod_name', 'model_name', 'qps', 'token_size', 
                              'GPU_type', 'iteration', 'GPU_util', 'qps', 'seq_len', 'FPR']
            
            missing_columns = [col for col in expected_columns if col not in columns]
            if missing_columns:
                print(f"警告: 缺少预期的列: {missing_columns}")
            
            print(f"成功加载数据，共{len(self.data)}行")
            
            # 检查FPR列的数据类型和范围
            fpr_data = self.data['FPR'].dropna()
            print(f"FPR数据统计:")
            print(f"  有效数据点: {len(fpr_data)}")
            print(f"  数据类型: {fpr_data.dtype}")
            print(f"  数据范围: {fpr_data.min():.6f} - {fpr_data.max():.6f}")
            print(f"  平均值: {fpr_data.mean():.6f}")
            
            # 检查是否有异常值
            if len(fpr_data) != len(self.data):
                print(f"警告: 发现 {len(self.data) - len(fpr_data)} 个FPR缺失值")
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            raise
    
    def calculate_cdf(self, data_series):
        """计算累积分布函数"""
        sorted_data = np.sort(data_series.dropna())
        n = len(sorted_data)
        cdf_y = np.arange(1, n + 1) / n
        return sorted_data, cdf_y
    
    def plot_overall_cdf(self, output_dir="./"):
        """绘制整体FPR的CDF图"""
        plt.figure(figsize=(10, 6))
        
        # 计算整体CDF
        fpr_values, cdf_values = self.calculate_cdf(self.data['FPR'])
        
        plt.plot(fpr_values, cdf_values, linewidth=2, label='Overall FPR CDF')
        plt.xlabel('FPR Value')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Function of FPR')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加统计信息
        mean_fpr = self.data['FPR'].mean()
        median_fpr = self.data['FPR'].median()
        plt.axvline(mean_fpr, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_fpr:.4f}')
        plt.axvline(median_fpr, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_fpr:.4f}')
        plt.legend()
        
        plt.tight_layout()
        output_file = Path(output_dir) / 'fpr_overall_cdf.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"整体CDF图已保存到: {output_file}")
        plt.show()
    
    def plot_cdf_by_gpu_type(self, output_dir="./"):
        """按GPU类型分组绘制FPR的CDF图"""
        plt.figure(figsize=(12, 8))
        
        gpu_types = self.data['GPU_type'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(gpu_types)))
        
        for i, gpu_type in enumerate(gpu_types):
            gpu_data = self.data[self.data['GPU_type'] == gpu_type]['FPR']
            if len(gpu_data) > 0:
                fpr_values, cdf_values = self.calculate_cdf(gpu_data)
                plt.plot(fpr_values, cdf_values, linewidth=2, 
                        color=colors[i], label=f'{gpu_type} (n={len(gpu_data)})')
        
        plt.xlabel('FPR Value')
        plt.ylabel('Cumulative Probability')
        plt.title('FPR CDF by GPU Type')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        output_file = Path(output_dir) / 'fpr_cdf_by_gpu_type.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"按GPU类型分组的CDF图已保存到: {output_file}")
        plt.show()
    
    def plot_cdf_by_model(self, output_dir="./", top_n=10):
        """按模型分组绘制FPR的CDF图（显示前N个模型）"""
        plt.figure(figsize=(14, 10))
        
        # 获取数据量最多的前N个模型
        model_counts = self.data['model_name'].value_counts()
        top_models = model_counts.head(top_n).index
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_models)))
        
        for i, model in enumerate(top_models):
            model_data = self.data[self.data['model_name'] == model]['FPR']
            if len(model_data) > 0:
                fpr_values, cdf_values = self.calculate_cdf(model_data)
                plt.plot(fpr_values, cdf_values, linewidth=2, 
                        color=colors[i], label=f'{model} (n={len(model_data)})')
        
        plt.xlabel('FPR Value')
        plt.ylabel('Cumulative Probability')
        plt.title(f'FPR CDF by Model (Top {top_n} Models)')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        output_file = Path(output_dir) / f'fpr_cdf_by_model_top{top_n}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"按模型分组的CDF图已保存到: {output_file}")
        plt.show()
    
    def plot_cdf_by_qps(self, output_dir="./"):
        """按qps分组绘制FPR的CDF图"""
        plt.figure(figsize=(10, 6))
        
        qpss = sorted(self.data['qps'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(qpss)))
        
        for i, qps in enumerate(qpss):
            batch_data = self.data[self.data['qps'] == qps]['FPR']
            if len(batch_data) > 0:
                fpr_values, cdf_values = self.calculate_cdf(batch_data)
                plt.plot(fpr_values, cdf_values, linewidth=2, 
                        color=colors[i], label=f'QPS {qps} (n={len(batch_data)})')
        
        plt.xlabel('FPR Value')
        plt.ylabel('Cumulative Probability')
        plt.title('FPR CDF by QPS')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        output_file = Path(output_dir) / 'fpr_cdf_by_qps.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"按qps分组的CDF图已保存到: {output_file}")
        plt.show()
    
    def generate_statistics_report(self):
        """生成FPR统计报告"""
        print("\n" + "="*60)
        print("FPR统计报告")
        print("="*60)
        
        # 整体统计
        print("整体统计:")
        print(f"  样本数: {len(self.data)}")
        print(f"  平均值: {self.data['FPR'].mean():.6f}")
        print(f"  中位数: {self.data['FPR'].median():.6f}")
        print(f"  标准差: {self.data['FPR'].std():.6f}")
        print(f"  最小值: {self.data['FPR'].min():.6f}")
        print(f"  最大值: {self.data['FPR'].max():.6f}")
        
        # 分位数
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"\n分位数:")
        for p in percentiles:
            value = np.percentile(self.data['FPR'], p)
            print(f"  {p}th percentile: {value:.6f}")
        
        # 按GPU类型统计
        print(f"\n按GPU类型统计:")
        for gpu_type in self.data['GPU_type'].unique():
            gpu_data = self.data[self.data['GPU_type'] == gpu_type]['FPR']
            print(f"  {gpu_type}:")
            print(f"    样本数: {len(gpu_data)}")
            print(f"    平均值: {gpu_data.mean():.6f}")
            print(f"    标准差: {gpu_data.std():.6f}")
        
        # 按qps统计
        print(f"\n按qps统计:")
        for qps in sorted(self.data['qps'].unique()):
            batch_data = self.data[self.data['qps'] == qps]['FPR']
            print(f"  Batch {qps}:")
            print(f"    样本数: {len(batch_data)}")
            print(f"    平均值: {batch_data.mean():.6f}")
            print(f"    标准差: {batch_data.std():.6f}")
    
    def run_analysis(self, output_dir="./"):
        """运行完整的FPR CDF分析"""
        print("开始FPR CDF分析...")
        
        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)
        
        # 加载数据
        self.load_data()
        
        # 生成统计报告
        self.generate_statistics_report()
        
        # 生成各种CDF图表
        print("\n生成CDF图表...")
        self.plot_overall_cdf(output_dir)
        self.plot_cdf_by_gpu_type(output_dir)
        self.plot_cdf_by_model(output_dir)
        self.plot_cdf_by_qps(output_dir)
        
        print(f"\n分析完成！所有图表已保存到: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='FPR CDF分析工具')
    parser.add_argument('--input', '-i', default='merged_two_cases_gpu_results.csv',
                       help='输入CSV文件路径 (默认: merged_two_cases_gpu_results.csv)')
    parser.add_argument('--output', '-o', default='./fpr_cdf_analysis',
                       help='输出目录 (默认: ./fpr_cdf_analysis)')
    parser.add_argument('--top-models', '-t', type=int, default=10,
                       help='显示前N个模型的CDF (默认: 10)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.input).exists():
        print(f"错误: 输入文件 {args.input} 不存在")
        return
    
    # 创建分析器并运行分析
    analyzer = FPRCDFAnalyzer(args.input)
    analyzer.run_analysis(args.output)

if __name__ == "__main__":
    main()