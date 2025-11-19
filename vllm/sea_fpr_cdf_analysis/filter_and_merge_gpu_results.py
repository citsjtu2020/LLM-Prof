#!/usr/bin/env python3
"""
筛选和合并GPU利用率结果脚本
对于每个CSV文件中的每个distinct model_name，随机保留2个案例
然后将所有筛选后的案例汇总到一个CSV文件中
"""

import pandas as pd
import os
import random
from pathlib import Path

def filter_random_cases(df, num_cases=2):
    """
    对于每个distinct model_name，随机保留num_cases个案例
    
    Args:
        df: DataFrame containing GPU utilization results
        num_cases: Number of random cases to keep for each model (default: 2)
    
    Returns:
        Filtered DataFrame
    """
    filtered_cases = []
    
    # 设置随机种子以确保结果可重现
    random.seed(42)
    
    # 按model_name分组
    for model_name, group in df.groupby('model_name'):
        # 如果案例数量少于等于num_cases，保留所有案例
        if len(group) <= num_cases:
            random_cases = group
        else:
            # 随机选择num_cases个案例
            random_cases = group.sample(n=num_cases, random_state=42)
        
        filtered_cases.append(random_cases)
    
    # 合并所有筛选后的案例
    result_df = pd.concat(filtered_cases, ignore_index=True)
    return result_df

def process_gpu_csv_files(csv_files, output_file='merged_two_cases_gpu_results.csv'):
    """
    处理多个GPU CSV文件，筛选并合并结果
    
    Args:
        csv_files: List of CSV file paths
        output_file: Output merged CSV file path
    """
    all_filtered_cases = []
    
    print("开始处理GPU利用率CSV文件...")
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"警告: 文件 {csv_file} 不存在，跳过")
            continue
            
        print(f"\n处理文件: {csv_file}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 检查必要的列是否存在
            required_columns = ['model_name']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"错误: 文件 {csv_file} 缺少必要的列: {missing_columns}")
                continue
            
            # 显示原始统计信息
            unique_models = df['model_name'].nunique()
            total_cases = len(df)
            print(f"  原始数据: {unique_models} 个不同模型, {total_cases} 个案例")
            
            # 随机筛选每个模型的2个案例
            filtered_df = filter_random_cases(df, num_cases=2)
            
            # 显示筛选后统计信息
            filtered_cases_count = len(filtered_df)
            print(f"  筛选后: {filtered_cases_count} 个案例")
            
            # 显示每个模型的筛选结果
            print("  各模型筛选结果:")
            for model_name, group in filtered_df.groupby('model_name'):
                cases_info = []
                for _, row in group.iterrows():
                    if 'token_size' in row:
                        cases_info.append(f"token_size={row['token_size']}")
                    elif 'batch_size' in row:
                        cases_info.append(f"batch_size={row['batch_size']}")
                    else:
                        cases_info.append("case")
                print(f"    {model_name}: {len(group)} 个案例 ({', '.join(cases_info)})")
            
            # 添加到总结果中
            all_filtered_cases.append(filtered_df)
            
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")
            continue
    
    if not all_filtered_cases:
        print("错误: 没有成功处理任何文件")
        return
    
    # 合并所有筛选后的结果
    print(f"\n合并所有筛选后的结果...")
    merged_df = pd.concat(all_filtered_cases, ignore_index=True)
    
    # 按GPU_type和model_name排序
    if 'GPU_type' in merged_df.columns:
        merged_df = merged_df.sort_values(['GPU_type', 'model_name'], 
                                        ascending=[True, True])
    else:
        merged_df = merged_df.sort_values(['model_name'], 
                                        ascending=[True])
    
    # 保存合并后的结果
    merged_df.to_csv(output_file, index=False)
    
    # 显示最终统计信息
    total_cases = len(merged_df)
    unique_models = merged_df['model_name'].nunique()
    
    if 'GPU_type' in merged_df.columns:
        unique_gpu_types = merged_df['GPU_type'].nunique()
        print(f"\n最终结果:")
        print(f"  总案例数: {total_cases}")
        print(f"  不同模型数: {unique_models}")
        print(f"  不同GPU类型数: {unique_gpu_types}")
        
        print(f"\n按GPU类型统计:")
        for gpu_type, group in merged_df.groupby('GPU_type'):
            models_count = group['model_name'].nunique()
            cases_count = len(group)
            print(f"  {gpu_type}: {models_count} 个模型, {cases_count} 个案例")
    else:
        print(f"\n最终结果: {total_cases} 个案例, {unique_models} 个不同模型")
    
    print(f"\n结果已保存到: {output_file}")
    
    # 显示前几行作为预览
    print(f"\n预览前10行:")
    print(merged_df.head(10).to_string(index=False))

def main():
    # 定义要处理的CSV文件列表
    csv_files = [
        'A100_gpu_utilization_results.csv',
        'A800_gpu_utilization_results.csv', 
        'H20_gpu_utilization_results.csv',
        'H800_gpu_utilization_results.csv',
        'L20_gpu_utilization_results.csv'
    ]
    
    # 检查文件是否存在
    existing_files = []
    for file in csv_files:
        if os.path.exists(file):
            existing_files.append(file)
        else:
            print(f"文件不存在: {file}")
    
    if not existing_files:
        print("错误: 没有找到任何CSV文件")
        return
    
    print(f"找到 {len(existing_files)} 个CSV文件: {existing_files}")
    
    # 处理文件并生成合并结果
    output_file = 'merged_two_cases_gpu_results.csv'
    process_gpu_csv_files(existing_files, output_file)

if __name__ == "__main__":
    main()