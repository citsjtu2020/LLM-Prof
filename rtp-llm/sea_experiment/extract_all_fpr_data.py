#!/usr/bin/env python3
"""
提取所有案例的FPR数据到txt文件
包括：pod_name、base_model、GPU类型、GPU数量、Pod QPS、GPU利用率、FPR值、execute_token_size
"""

import pandas as pd
import numpy as np
import re

def parse_qps_timeseries(qps_str):
    """解析QPS时间序列，计算平均值"""
    try:
        if pd.isna(qps_str) or qps_str == '':
            return 0.0
        
        # 按行分割，跳过时间戳行，只提取QPS数值
        lines = str(qps_str).strip().split('\n')
        qps_values = []
        
        for line in lines:
            line = line.strip()
            # 跳过时间戳行（包含日期格式）
            if '-' in line and ':' in line:
                continue
            # 尝试解析为浮点数
            try:
                qps_val = float(line)
                if qps_val > 0:  # 只考虑正值
                    qps_values.append(qps_val)
            except ValueError:
                continue
        
        return np.mean(qps_values) if qps_values else 0.0
    except:
        return 0.0

def extract_peak_qps(qps_str):
    """提取峰值QPS"""
    try:
        if pd.isna(qps_str) or qps_str == '':
            return 0.0
        
        # 按行分割，跳过时间戳行，只提取QPS数值
        lines = str(qps_str).strip().split('\n')
        qps_values = []
        
        for line in lines:
            line = line.strip()
            # 跳过时间戳行（包含日期格式）
            if '-' in line and ':' in line:
                continue
            # 尝试解析为浮点数
            try:
                qps_val = float(line)
                if qps_val > 0:  # 只考虑正值
                    qps_values.append(qps_val)
            except ValueError:
                continue
        
        return max(qps_values) if qps_values else 0.0
    except:
        return 0.0

def extract_max_token_size(token_str):
    """提取最大token size，支持K后缀（表示千）"""
    try:
        if pd.isna(token_str) or token_str == '':
            return 0.0
        
        # 按行分割，跳过时间戳行，只提取token size数值
        lines = str(token_str).strip().split('\n')
        token_values = []
        
        for line in lines:
            line = line.strip()
            # 跳过时间戳行（包含日期格式）
            if '-' in line and ':' in line:
                continue
            
            # 处理带K后缀的数值（支持空格分隔，如 "1.03 K" 或 "1.03K"）
            if line.upper().endswith('K') or ' K' in line.upper():
                try:
                    # 去掉K后缀和可能的空格，然后转换为数值，乘以1000
                    # 处理 "1.03 K" 或 "1.03K" 格式
                    clean_line = line.upper().replace(' K', '').replace('K', '')
                    token_val = float(clean_line) * 1000
                    if token_val >= 0:
                        token_values.append(token_val)
                except ValueError:
                    continue
            else:
                # 尝试解析为普通浮点数
                try:
                    token_val = float(line)
                    if token_val >= 0:  # 包含0值
                        token_values.append(token_val)
                except ValueError:
                    continue
        
        return max(token_values) if token_values else 0.0
    except:
        return 0.0

def calculate_fpr(f_peak, u_gpu, n_gpu, qps):
    """计算FPR = (F_peak × U_GPU × N_GPU) / QPS"""
    if qps > 0 and not pd.isna(f_peak) and not pd.isna(u_gpu):
        return (f_peak * u_gpu * n_gpu) / qps
    return None

def main():
    # 加载CSV数据
    csv_file = 'sea_experiment/essay_traces_exact_structure.csv'
    df = pd.read_csv(csv_file)
    
    print(f"加载了 {len(df)} 条记录")
    
    # 解析QPS数据
    df['avg_qps'] = df['qps_timeseries'].apply(parse_qps_timeseries)
    df['peak_qps'] = df['qps_timeseries'].apply(extract_peak_qps)
    
    # 解析token size数据（支持K后缀）
    df['max_token_size'] = df['execute_token_size_timeseries'].apply(extract_max_token_size)
    
    # 计算FPR
    results = []
    
    for idx, row in df.iterrows():
        try:
            # 获取基础信息
            pod_name = row['pod_name']
            base_model = row['model_name']
            gpu_type = row['gpu_type']
            gpu_count = row['n_gpu_hardware']
            gpu_utilization = row['u_gpu']
            
            # 获取QPS (优先使用峰值QPS，然后平均QPS，最后app_qps)
            qps = row['peak_qps'] if row['peak_qps'] > 0 else row['avg_qps']
            if qps <= 0:
                qps = row['app_qps']
            
            # 正确获取f_peak - 从CSV的f_peak列读取
            f_peak = row['f_peak']
            
            # 获取最大token size
            max_token_size = row['max_token_size']
            
            # 计算FPR
            fpr = calculate_fpr(f_peak, gpu_utilization, gpu_count, qps)
            
            result = {
                'pod_name': pod_name,
                'base_model': base_model,
                'gpu_type': gpu_type,
                'gpu_count': int(gpu_count),
                'pod_qps': round(qps, 2),
                'gpu_utilization': round(gpu_utilization, 3),
                'f_peak': round(f_peak, 1),
                'fpr': round(fpr, 6) if fpr is not None else 'N/A',
                'max_token_size': int(max_token_size)
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"处理第 {idx} 行时出错: {e}")
            continue
    
    # 输出到txt文件
    output_file = 'extract_all_cases_fpr_data_with_k_support.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("=" * 140 + "\n")
        f.write("所有案例的FPR数据详情 (支持K后缀的Token Size)\n")
        f.write("=" * 140 + "\n")
        f.write(f"总案例数: {len(results)}\n")
        f.write(f"FPR计算公式: FPR = (F_peak × U_GPU × N_GPU) / QPS\n")
        f.write(f"Token Size: 从execute_token_size_timeseries中提取的最大值，支持K后缀（K=1000）\n")
        f.write("=" * 140 + "\n\n")
        
        # 写入列标题
        header = f"{'序号':<4} {'Pod名称':<60} {'模型':<20} {'GPU类型':<15} {'GPU数':<6} {'QPS':<8} {'利用率':<8} {'F_peak':<8} {'FPR':<12} {'Token Size':<10}"
        f.write(header + "\n")
        f.write("-" * 140 + "\n")
        
        # 写入数据
        for i, result in enumerate(results, 1):
            line = f"{i:<4} {result['pod_name']:<60} {result['base_model']:<20} {result['gpu_type']:<15} {result['gpu_count']:<6} {result['pod_qps']:<8} {result['gpu_utilization']:<8} {result['f_peak']:<8} {result['fpr']:<12} {result['max_token_size']:<10}"
            f.write(line + "\n")
        
        # 写入统计信息
        f.write("\n" + "=" * 140 + "\n")
        f.write("统计信息:\n")
        f.write("=" * 140 + "\n")
        
        # FPR统计
        valid_fprs = [r['fpr'] for r in results if r['fpr'] != 'N/A']
        if valid_fprs:
            f.write(f"有效FPR计算数量: {len(valid_fprs)}\n")
            f.write(f"FPR最小值: {min(valid_fprs):.6f}\n")
            f.write(f"FPR最大值: {max(valid_fprs):.6f}\n")
            f.write(f"FPR平均值: {np.mean(valid_fprs):.6f}\n")
            f.write(f"FPR中位数: {np.median(valid_fprs):.6f}\n")
            f.write(f"FPR效率差异倍数: {max(valid_fprs)/min(valid_fprs):.1f}×\n")
        
        # Token Size统计
        token_sizes = [r['max_token_size'] for r in results if r['max_token_size'] > 0]
        if token_sizes:
            f.write(f"\nToken Size统计:\n")
            f.write(f"有效Token Size数量: {len(token_sizes)}\n")
            f.write(f"Token Size最小值: {min(token_sizes)}\n")
            f.write(f"Token Size最大值: {max(token_sizes)}\n")
            f.write(f"Token Size平均值: {np.mean(token_sizes):.1f}\n")
            f.write(f"Token Size中位数: {np.median(token_sizes):.1f}\n")
        
        # GPU类型和F_peak统计
        f.write(f"\nGPU类型和F_peak分布:\n")
        gpu_types = {}
        for result in results:
            gpu_type = result['gpu_type']
            f_peak = result['f_peak']
            if gpu_type not in gpu_types:
                gpu_types[gpu_type] = {'count': 0, 'total_gpus': 0, 'f_peak': f_peak}
            gpu_types[gpu_type]['count'] += 1
            gpu_types[gpu_type]['total_gpus'] += result['gpu_count']
        
        for gpu_type, stats in gpu_types.items():
            f.write(f"  {gpu_type}: {stats['count']}个服务, 共{stats['total_gpus']}张GPU, F_peak={stats['f_peak']} TFLOPs\n")
        
        # 模型统计
        f.write(f"\n模型分布:\n")
        models = {}
        for result in results:
            model = result['base_model']
            if model not in models:
                models[model] = 0
            models[model] += 1
        
        for model, count in sorted(models.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {model}: {count}个服务\n")
    
    print(f"修正后的数据已输出到: {output_file}")
    print(f"处理了 {len(results)} 个案例")
    
    # 输出简要统计到控制台
    if valid_fprs:
        print(f"\nFPR统计:")
        print(f"  有效计算: {len(valid_fprs)}/{len(results)}")
        print(f"  范围: {min(valid_fprs):.6f} - {max(valid_fprs):.6f}")
        print(f"  效率差异: {max(valid_fprs)/min(valid_fprs):.1f}×")
    
    # Token Size统计
    if token_sizes:
        print(f"\nToken Size统计:")
        print(f"  有效数量: {len(token_sizes)}/{len(results)}")
        print(f"  范围: {min(token_sizes)} - {max(token_sizes)}")
        print(f"  平均值: {np.mean(token_sizes):.1f}")
    
    # 显示GPU类型的F_peak值
    print(f"\nGPU类型F_peak验证:")
    unique_gpu_fpeak = {}
    for result in results:
        gpu_type = result['gpu_type']
        f_peak = result['f_peak']
        if gpu_type not in unique_gpu_fpeak:
            unique_gpu_fpeak[gpu_type] = f_peak
    
    for gpu_type, f_peak in unique_gpu_fpeak.items():
        print(f"  {gpu_type}: {f_peak} TFLOPs")

if __name__ == "__main__":
    main()