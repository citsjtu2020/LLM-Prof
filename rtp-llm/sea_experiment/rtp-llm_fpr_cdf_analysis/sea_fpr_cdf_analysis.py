#!/usr/bin/env python3
"""
SEA层FPR CDF分析脚本
基于extract_all_cases_fpr_data_with_k_support.txt数据计算FPR并生成CDF可视化图表
适配SEA实验的数据结构和分析需求
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import re
from datetime import datetime

class SEAFPRCDFAnalyzer:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = None
        self.fpr_data = None
        
    def load_data(self):
        """加载FPR数据文件并预处理"""
        try:
            print(f"从文件加载FPR数据: {self.data_file}")
            
            # 读取文件内容
            with open(self.data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 解析数据行
            data_records = []
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # 跳过空行和分隔线
                if not line or line.startswith('=') or 'pod_name' in line:
                    continue
                
                # 跳过统计信息部分
                if '统计信息:' in line or 'Token Size统计:' in line or 'GPU类型和F_peak分布:' in line or '模型分布:' in line:
                    break
                
                # 解析数据行
                try:
                    # 使用正则表达式解析固定格式的数据行
                    # 格式: 序号 pod_name model_name gpu_type n_gpu qps u_gpu f_peak fpr token_size
                    parts = line.split()
                    
                    if len(parts) >= 10:
                        idx = int(parts[0])
                        pod_name = parts[1]
                        model_name = parts[2]
                        gpu_type = parts[3]
                        n_gpu = int(parts[4])
                        qps = float(parts[5])
                        u_gpu = float(parts[6])
                        f_peak = float(parts[7])
                        fpr = float(parts[8])
                        token_size = int(parts[9]) if len(parts) > 9 else 0
                        
                        # 构建服务签名
                        service_signature = f"{model_name}@{gpu_type}×{n_gpu}"
                        
                        record = {
                            'idx': idx,
                            'pod_name': pod_name,
                            'service_signature': service_signature,
                            'model_name': model_name,
                            'gpu_type': gpu_type,
                            'n_gpu': n_gpu,
                            'qps': qps,
                            'u_gpu': u_gpu,
                            'f_peak': f_peak,
                            'fpr': fpr,
                            'token_size': token_size,
                            'inference_engine': 'RTP-LLM',  # 根据文档，当前都是RTP-LLM
                            'task_function': 'LLM_Inference'
                        }
                        
                        data_records.append(record)
                        
                except (ValueError, IndexError) as e:
                    print(f"解析第{line_num}行时出错: {e}")
                    print(f"  行内容: {line}")
                    continue
            
            # 创建DataFrame
            self.fpr_data = pd.DataFrame(data_records)
            
            if len(self.fpr_data) == 0:
                raise ValueError("没有成功解析任何数据行")
            
            print(f"成功加载数据，共{len(self.fpr_data)}个服务")
            print(f"FPR数据范围: {self.fpr_data['fpr'].min():.6f} - {self.fpr_data['fpr'].max():.6f}")
            
            # 数据预处理
            self._preprocess_data()
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            raise
    
    def _preprocess_data(self):
        """数据预处理：验证和清理数据"""
        print("开始数据预处理...")
        
        # 过滤掉无效的FPR值
        valid_fpr = self.fpr_data[
            (self.fpr_data['fpr'] > 0) & 
            (self.fpr_data['qps'] > 0) &
            (self.fpr_data['fpr'] != np.inf)
        ].copy()
        
        print(f"预处理完成，有效FPR数据点: {len(valid_fpr)}")
        if len(valid_fpr) > 0:
            print(f"FPR数据范围: {valid_fpr['fpr'].min():.6f} - {valid_fpr['fpr'].max():.6f}")
            print(f"效率差异倍数: {valid_fpr['fpr'].max()/valid_fpr['fpr'].min():.1f}×")
        
        self.fpr_data = valid_fpr
    
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
        fpr_values, cdf_values = self.calculate_cdf(self.fpr_data['fpr'])
        
        plt.plot(fpr_values, cdf_values, linewidth=2, label='Overall FPR CDF')
        plt.xlabel('FPR Value')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Function of FPR')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加统计信息
        mean_fpr = self.fpr_data['fpr'].mean()
        median_fpr = self.fpr_data['fpr'].median()
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
        
        gpu_types = self.fpr_data['gpu_type'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(gpu_types)))
        
        for i, gpu_type in enumerate(gpu_types):
            gpu_data = self.fpr_data[self.fpr_data['gpu_type'] == gpu_type]['fpr']
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
    
    def plot_cdf_by_model_family(self, output_dir="./", top_n=10):
        """按模型分组绘制FPR的CDF图（显示前N个模型）"""
        plt.figure(figsize=(14, 10))
        
        # 提取模型系列（如qwen3, QwQ等）
        self.fpr_data['model_family'] = self.fpr_data['model_name'].apply(self._extract_model_family)
        
        # 获取数据量最多的前N个模型系列
        model_counts = self.fpr_data['model_family'].value_counts()
        top_families = model_counts.head(top_n).index
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_families)))
        
        for i, family in enumerate(top_families):
            family_data = self.fpr_data[self.fpr_data['model_family'] == family]['fpr']
            if len(family_data) > 0:
                fpr_values, cdf_values = self.calculate_cdf(family_data)
                plt.plot(fpr_values, cdf_values, linewidth=2, 
                        color=colors[i], label=f'{family} (n={len(family_data)})')
        
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
    
    def plot_cdf_by_gpu_count(self, output_dir="./"):
        """按batch size分组绘制FPR的CDF图"""
        plt.figure(figsize=(10, 6))
        
        gpu_counts = sorted(self.fpr_data['n_gpu'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(gpu_counts)))
        
        for i, gpu_count in enumerate(gpu_counts):
            gpu_count_data = self.fpr_data[self.fpr_data['n_gpu'] == gpu_count]['fpr']
            if len(gpu_count_data) > 0:
                fpr_values, cdf_values = self.calculate_cdf(gpu_count_data)
                plt.plot(fpr_values, cdf_values, linewidth=2, 
                        color=colors[i], label=f'GPU {gpu_count} (n={len(gpu_count_data)})')
        
        plt.xlabel('FPR Value')
        plt.ylabel('Cumulative Probability')
        plt.title('FPR CDF by Batch Size')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        output_file = Path(output_dir) / 'fpr_cdf_by_batch_size.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"按batch size分组的CDF图已保存到: {output_file}")
        plt.show()
    
    def generate_statistics_report(self):
        """生成FPR统计报告"""
        print("\n" + "="*60)
        print("FPR统计报告")
        print("="*60)
        
        # 整体统计
        print("整体统计:")
        print(f"  样本数: {len(self.fpr_data)}")
        print(f"  平均值: {self.fpr_data['fpr'].mean():.6f}")
        print(f"  中位数: {self.fpr_data['fpr'].median():.6f}")
        print(f"  标准差: {self.fpr_data['fpr'].std():.6f}")
        print(f"  最小值: {self.fpr_data['fpr'].min():.6f}")
        print(f"  最大值: {self.fpr_data['fpr'].max():.6f}")
        
        # 分位数
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"\n分位数:")
        for p in percentiles:
            value = np.percentile(self.fpr_data['fpr'], p)
            print(f"  {p}th percentile: {value:.6f}")
        
        # 按GPU类型统计
        print(f"\n按GPU类型统计:")
        for gpu_type in self.fpr_data['gpu_type'].unique():
            gpu_data = self.fpr_data[self.fpr_data['gpu_type'] == gpu_type]['fpr']
            print(f"  {gpu_type}:")
            print(f"    样本数: {len(gpu_data)}")
            print(f"    平均值: {gpu_data.mean():.6f}")
            print(f"    标准差: {gpu_data.std():.6f}")
        
        # 按GPU数量统计
        print(f"\n按GPU数量统计:")
        for gpu_count in sorted(self.fpr_data['n_gpu'].unique()):
            gpu_count_data = self.fpr_data[self.fpr_data['n_gpu'] == gpu_count]['fpr']
            print(f"  GPU {gpu_count}:")
            print(f"    样本数: {len(gpu_count_data)}")
            print(f"    平均值: {gpu_count_data.mean():.6f}")
            print(f"    标准差: {gpu_count_data.std():.6f}")
    
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
        self.plot_cdf_by_model_family(output_dir)
        self.plot_cdf_by_gpu_count(output_dir)
        
        print(f"\n分析完成！所有图表已保存到: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='FPR CDF分析工具')
    parser.add_argument('--input', '-i', default='extract_all_cases_fpr_data_with_k_support.txt',
                       help='输入FPR数据文件路径 (默认: extract_all_cases_fpr_data_with_k_support.txt)')
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
    analyzer = SEAFPRCDFAnalyzer(args.input)
    analyzer.run_analysis(args.output)

if __name__ == "__main__":
    main()