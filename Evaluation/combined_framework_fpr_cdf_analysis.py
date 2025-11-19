#!/usr/bin/env python3
"""
综合框架FPR CDF分析脚本
整合RTP-LLM、SGLang、vLLM三个框架的FPR数据并生成对比CDF图表
包含P85位置标识和大字体显示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

class CombinedFrameworkFPRAnalyzer:
    def __init__(self):
        self.rtp_llm_data = None
        self.sglang_data = None
        self.vllm_data = None
        self.combined_data = None
        
    def load_rtp_llm_data(self, txt_file):
        """加载RTP-LLM的TXT格式FPR数据"""
        print(f"加载RTP-LLM数据: {txt_file}")
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
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
                        
                        record = {
                            'framework': 'RTP-LLM',
                            'pod_name': pod_name,
                            'model_name': model_name,
                            'gpu_type': gpu_type,
                            'n_gpu': n_gpu,
                            'qps': qps,
                            'u_gpu': u_gpu,
                            'f_peak': f_peak,
                            'fpr': fpr,
                            'token_size': token_size
                        }
                        
                        data_records.append(record)
                        
                except (ValueError, IndexError) as e:
                    continue
            
            self.rtp_llm_data = pd.DataFrame(data_records)
            
            # 过滤有效数据
            self.rtp_llm_data = self.rtp_llm_data[
                (self.rtp_llm_data['fpr'] > 0) & 
                (self.rtp_llm_data['qps'] > 0) &
                (self.rtp_llm_data['fpr'] != np.inf)
            ].copy()
            
            print(f"RTP-LLM数据加载完成，有效数据点: {len(self.rtp_llm_data)}")
            if len(self.rtp_llm_data) > 0:
                print(f"FPR范围: {self.rtp_llm_data['fpr'].min():.6f} - {self.rtp_llm_data['fpr'].max():.6f}")
            
        except Exception as e:
            print(f"加载RTP-LLM数据时出错: {e}")
            raise
    
    def load_csv_data(self, csv_file, framework_name):
        """加载CSV格式的FPR数据（SGLang和vLLM）"""
        print(f"加载{framework_name}数据: {csv_file}")
        
        try:
            data = pd.read_csv(csv_file)
            
            # 验证FPR列存在
            if 'FPR' not in data.columns:
                raise ValueError(f"CSV文件中未找到FPR列")
            
            # 添加框架标识
            data['framework'] = framework_name
            
            # 过滤有效数据
            valid_data = data[
                (data['FPR'] > 0) & 
                (data['FPR'] != np.inf) &
                (data['FPR'].notna())
            ].copy()
            
            print(f"{framework_name}数据加载完成，有效数据点: {len(valid_data)}")
            if len(valid_data) > 0:
                print(f"FPR范围: {valid_data['FPR'].min():.6f} - {valid_data['FPR'].max():.6f}")
            
            return valid_data
            
        except Exception as e:
            print(f"加载{framework_name}数据时出错: {e}")
            raise
    
    def load_all_data(self):
        """加载所有框架的数据"""
        # 加载RTP-LLM数据
        rtp_llm_file = "sea_experiment/rtp-llm_fpr_cdf_analysis/extract_all_cases_fpr_data_with_k_support.txt"
        self.load_rtp_llm_data(rtp_llm_file)
        
        # 加载SGLang数据
        sglang_file = "sea_experiment/sglang_fpr_cdf_analysis/merged_two_cases_gpu_results.csv"
        self.sglang_data = self.load_csv_data(sglang_file, 'SGLang')
        
        # 加载vLLM数据
        vllm_file = "sea_experiment/vllm_fpr_cdf_analysis/merged_two_cases_gpu_results.csv"
        self.vllm_data = self.load_csv_data(vllm_file, 'vLLM')
        
        # 统一数据格式
        self.standardize_data()
    
    def standardize_data(self):
        """统一三个框架的数据格式"""
        # RTP-LLM数据已经有fpr列
        rtp_llm_std = self.rtp_llm_data[['framework', 'fpr']].copy()
        
        # SGLang和vLLM数据需要重命名FPR列为fpr
        sglang_std = self.sglang_data[['framework', 'FPR']].copy()
        sglang_std = sglang_std.rename(columns={'FPR': 'fpr'})
        
        vllm_std = self.vllm_data[['framework', 'FPR']].copy()
        vllm_std = vllm_std.rename(columns={'FPR': 'fpr'})
        
        # 合并所有数据
        self.combined_data = pd.concat([rtp_llm_std, sglang_std, vllm_std], ignore_index=True)
        
        print(f"\n数据统一完成:")
        print(f"总数据点: {len(self.combined_data)}")
        for framework in self.combined_data['framework'].unique():
            framework_data = self.combined_data[self.combined_data['framework'] == framework]
            print(f"{framework}: {len(framework_data)} 个数据点")
    
    def calculate_cdf(self, data_series):
        """计算累积分布函数"""
        sorted_data = np.sort(data_series.dropna())
        n = len(sorted_data)
        cdf_y = np.arange(1, n + 1) / n
        return sorted_data, cdf_y
    
    def calculate_percentile(self, data_series, percentile):
        """计算指定分位数"""
        return np.percentile(data_series.dropna(), percentile)
    
    def plot_combined_cdf_with_p85(self, output_dir="./"):
        """绘制包含所有框架的FPR CDF图，并标识P85位置"""
        # 设置大字体
        plt.rcParams.update({
            'font.size': 25,
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 20,  # 横轴刻度数字字体增大
            'ytick.labelsize': 20,  # 纵轴刻度数字字体增大
            'legend.fontsize': 16
        })
        
        plt.figure(figsize=(10, 6))
        
        # 定义框架颜色
        framework_colors = {
            'RTP-LLM': '#1f77b4',  # 蓝色
            'SGLang': '#ff7f0e',   # 橙色
            'vLLM': '#2ca02c'      # 绿色
        }
        
        frameworks = self.combined_data['framework'].unique()
        p85_values = {}
        
        # 为每个框架绘制CDF曲线
        for framework in frameworks:
            framework_data = self.combined_data[self.combined_data['framework'] == framework]['fpr']
            
            if len(framework_data) > 0:
                fpr_values, cdf_values = self.calculate_cdf(framework_data)
                
                # 计算P85
                p85 = self.calculate_percentile(framework_data, 85)
                p85_values[framework] = p85
                
                # 绘制CDF曲线
                plt.plot(fpr_values, cdf_values, 
                        linewidth=3, 
                        color=framework_colors[framework],
                        label=f'{framework} (n={len(framework_data)})')
        
        # 设置对数刻度横轴
        plt.xscale('log')
        
        # 添加P85水平虚线（使用更醒目的红色虚线）
        plt.axhline(0.85, color='red', linestyle='--', alpha=0.8, linewidth=3)
        plt.text(plt.xlim()[0] * 1.5, 0.87, 'P85', fontsize=16, color='red', weight='bold')
        
        # 设置图表属性
        plt.xlabel('FPR Value (log scale)', fontsize=24, weight='bold')
        plt.ylabel('Cumulative Probability', fontsize=24, weight='bold')
        
        # 设置网格
        plt.grid(True, alpha=0.3, linewidth=1, which='both')
        
        # 设置图例（紧凑的图例）
        legend = plt.legend(loc='lower right', fontsize=20, framealpha=0.9, 
                           fancybox=True, shadow=True, frameon=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.2)
        
        # 设置坐标轴范围
        plt.xlim(left=0.1)  # 设置最小值避免log(0)
        plt.ylim(0, 1)
        
        # 调整布局，减少边距
        plt.tight_layout(pad=1.0)
        
        # 保存图片
        output_file = Path(output_dir) / 'combined_framework_fpr_cdf_with_p85_log.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
        print(f"综合CDF图（对数刻度）已保存到: {output_file}")
        
        # 显示图片
        plt.show()
        
        # 打印P85统计信息
        print(f"\nP85统计信息:")
        for framework, p85 in p85_values.items():
            print(f"{framework}: {p85:.6f}")
    
    def generate_comprehensive_report(self):
        """生成综合统计报告"""
        print("\n" + "="*80)
        print("综合框架FPR统计报告")
        print("="*80)
        
        frameworks = self.combined_data['framework'].unique()
        
        for framework in frameworks:
            framework_data = self.combined_data[self.combined_data['framework'] == framework]['fpr']
            
            print(f"\n{framework}框架统计:")
            print(f"  样本数: {len(framework_data)}")
            print(f"  平均值: {framework_data.mean():.6f}")
            print(f"  中位数: {framework_data.median():.6f}")
            print(f"  标准差: {framework_data.std():.6f}")
            print(f"  最小值: {framework_data.min():.6f}")
            print(f"  最大值: {framework_data.max():.6f}")
            
            # 关键分位数
            percentiles = [25, 50, 75, 85, 90, 95, 99]
            print(f"  分位数:")
            for p in percentiles:
                value = np.percentile(framework_data, p)
                print(f"    P{p}: {value:.6f}")
        
        # 框架间比较
        print(f"\n框架间P85比较:")
        p85_comparison = {}
        for framework in frameworks:
            framework_data = self.combined_data[self.combined_data['framework'] == framework]['fpr']
            p85_comparison[framework] = np.percentile(framework_data, 85)
        
        # 按P85排序
        sorted_frameworks = sorted(p85_comparison.items(), key=lambda x: x[1])
        for i, (framework, p85) in enumerate(sorted_frameworks, 1):
            print(f"  {i}. {framework}: {p85:.6f}")
    
    def run_analysis(self, output_dir="./"):
        """运行完整的综合分析"""
        print("开始综合框架FPR CDF分析...")
        
        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)
        
        # 加载所有数据
        self.load_all_data()
        
        # 生成统计报告
        self.generate_comprehensive_report()
        
        # 生成综合CDF图表
        print("\n生成综合CDF图表...")
        self.plot_combined_cdf_with_p85(output_dir)
        
        print(f"\n分析完成！图表已保存到: {output_dir}")

def main():
    analyzer = CombinedFrameworkFPRAnalyzer()
    analyzer.run_analysis("./sea_experiment/")

if __name__ == "__main__":
    main()