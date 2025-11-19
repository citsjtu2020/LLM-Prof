#!/usr/bin/env python3
"""
SEA (Service Efficiency Analyzer) FPR Filter
筛选 FPR 最大的 15% 案例，用于后续 MEA 分析
"""

import pandas as pd
import numpy as np
import sys
import os

def filter_top_fpr_cases(input_csv, output_csv, top_percentage=0.15):
    """
    从输入 CSV 文件中筛选 FPR 最大的指定百分比案例
    
    Args:
        input_csv: 输入 CSV 文件路径
        output_csv: 输出 CSV 文件路径  
        top_percentage: 筛选的百分比 (默认 0.15 即 15%)
    """
    
    print(f"Loading data from {input_csv}...")
    
    # 读取 CSV 文件
    df = pd.read_csv(input_csv)
    
    print(f"Total cases loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # 检查 FPR 列是否存在
    fpr_column = None
    for col in df.columns:
        if 'fpr' in col.lower() or 'FPR' in col:
            fpr_column = col
            break
    
    if fpr_column is None:
        # 如果没有找到 FPR 列，假设最后一列是 FPR
        fpr_column = df.columns[-1]
        print(f"FPR column not found by name, using last column: {fpr_column}")
    else:
        print(f"Found FPR column: {fpr_column}")
    
    # 确保 FPR 列是数值类型
    df[fpr_column] = pd.to_numeric(df[fpr_column], errors='coerce')
    
    # 移除 FPR 为 NaN 的行
    df_clean = df.dropna(subset=[fpr_column])
    print(f"Cases after removing NaN FPR values: {len(df_clean)}")
    
    # 计算 FPR 统计信息
    fpr_stats = df_clean[fpr_column].describe()
    print(f"\nFPR Statistics:")
    print(fpr_stats)
    
    # 计算 P85 阈值 (top 15%)
    p85_threshold = df_clean[fpr_column].quantile(0.85)
    print(f"\nP85 threshold (top 15%): {p85_threshold:.6f}")
    
    # 筛选 FPR 最大的 15% 案例
    top_cases = df_clean[df_clean[fpr_column] >= p85_threshold].copy()
    
    # 按 FPR 降序排列
    top_cases = top_cases.sort_values(by=fpr_column, ascending=False)
    
    print(f"\nFiltered cases (top 15%): {len(top_cases)}")
    print(f"FPR range in filtered cases: {top_cases[fpr_column].min():.6f} - {top_cases[fpr_column].max():.6f}")
    
    # 显示前几个案例
    print(f"\nTop 5 cases by FPR:")
    for i, (idx, row) in enumerate(top_cases.head().iterrows()):
        case_name = row.iloc[0] if len(row) > 0 else f"Case_{idx}"
        fpr_value = row[fpr_column]
        print(f"  {i+1}. {case_name}: FPR = {fpr_value:.6f}")
    
    # 保存筛选结果
    top_cases.to_csv(output_csv, index=False)
    print(f"\nFiltered cases saved to: {output_csv}")
    
    # 生成统计报告
    print(f"\n=== SEA Layer Filtering Summary ===")
    print(f"Input file: {input_csv}")
    print(f"Output file: {output_csv}")
    print(f"Total input cases: {len(df)}")
    print(f"Valid cases (non-NaN FPR): {len(df_clean)}")
    print(f"Filtered cases (top 15%): {len(top_cases)}")
    print(f"P85 FPR threshold: {p85_threshold:.6f}")
    print(f"Filtering rate: {len(top_cases)/len(df_clean)*100:.1f}%")
    
    return top_cases

def main():
    # 设置输入输出文件路径
    input_file = "merged_two_cases_gpu_results.csv"
    output_file = "cases_after_sea.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        sys.exit(1)
    
    try:
        # 执行筛选
        filtered_cases = filter_top_fpr_cases(input_file, output_file, top_percentage=0.15)
        
        print(f"\n✅ SEA layer filtering completed successfully!")
        print(f"📊 {len(filtered_cases)} high-FPR cases selected for MEA analysis")
        print(f"📁 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during filtering: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
