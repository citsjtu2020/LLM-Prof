#!/usr/bin/env python3
"""
分析trace文件中的distinct kernel统计脚本
"""

import json
import pandas as pd
from collections import Counter
import sys
import os

def analyze_distinct_kernels(trace_file_path: str, case_name: str = None):
    """
    分析单个trace文件中的distinct kernel
    
    Args:
        trace_file_path: trace文件路径
        case_name: case名称
        
    Returns:
        Dict: kernel统计信息
    """
    
    if case_name is None:
        case_name = os.path.basename(os.path.dirname(trace_file_path))
    
    try:
        with open(trace_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件 {trace_file_path} 失败: {e}")
        return None
    
    # 统计所有kernel名称
    kernel_names = []
    total_duration = 0
    
    for event in data.get('traceEvents', []):
        if event.get('cat', '') == 'kernel':
            kernel_name = event.get('name', '')
            duration = event.get('dur', 0)
            if kernel_name:
                kernel_names.append(kernel_name)
                total_duration += duration
    
    # 统计distinct kernel数量和名称
    kernel_counter = Counter(kernel_names)
    
    print(f"\n=== {case_name} Kernel分析 ===")
    print(f"总共有 {len(kernel_counter)} 种不同的kernel")
    print(f"总kernel执行次数: {sum(kernel_counter.values())}")
    print(f"总kernel执行时间: {total_duration/1000:.2f} ms")
    print()
    print("Kernel名称和执行次数:")
    print("=" * 100)
    
    # 按执行次数排序
    sorted_kernels = sorted(kernel_counter.items(), key=lambda x: x[1], reverse=True)
    
    for kernel_name, count in sorted_kernels:
        print(f"{count:6d} 次: {kernel_name}")
    
    return {
        'case_name': case_name,
        'total_distinct_kernels': len(kernel_counter),
        'total_executions': sum(kernel_counter.values()),
        'total_duration_ms': total_duration / 1000,
        'kernel_details': sorted_kernels
    }

def analyze_all_cases_distinct():
    """
    分析所有case的distinct kernel
    """
    
    # 四个case的trace文件路径
    trace_files = [
        ("qwen2-7b-query", "qwen2-7b-query-na61-l20.inference-part0-caa0648d-b-a069/20251015133756-94bccfb15a4fdb67e7f0/ecos-trace-6310-1760506703964612498-0.json"),
        ("qwen2-ast-sft", "qwen2-ast-sft-modelv2-kto-na61-a800.inference-part0-b5b19bcf-b-5289/20251014220050-85fee9ce26117168738f/ecos-trace-62775-1760450468039310885-7.json"),
        ("qwen3-32b-omega", "qwen3-32b-for-omega-na61-h20-2tp-spot.inference-part0-798b1850-a-56a1/20251015133923-e47c901cf95633524d74/ecos-trace-43179-1760506792708329739-0.json"),
        ("recommend-intent", "recommend-intent-qwen3-4b-na61-h20.inference-part0-928ea825-a-9825/20251012145847-46fe46fa1780babddde2/ecos-trace-59086-1760252346235516818-4.json")
    ]
    
    all_results = []
    all_kernel_details = []
    
    for case_name, trace_file in trace_files:
        if os.path.exists(trace_file):
            result = analyze_distinct_kernels(trace_file, case_name)
            if result:
                all_results.append({
                    'Case': result['case_name'],
                    'Total_Distinct_Kernels': result['total_distinct_kernels'],
                    'Total_Executions': result['total_executions'],
                    'Total_Duration_ms': result['total_duration_ms']
                })
                
                # 保存每个kernel的详细信息
                for kernel_name, count in result['kernel_details']:
                    all_kernel_details.append({
                        'Case': result['case_name'],
                        'Kernel_Name': kernel_name,
                        'Execution_Count': count
                    })
        else:
            print(f"文件不存在: {trace_file}")
    
    # 保存汇总结果
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv('distinct_kernels_summary.csv', index=False)
        print(f"\n=== 汇总结果 ===")
        print(summary_df.to_string(index=False))
        print(f"\n汇总结果已保存到: distinct_kernels_summary.csv")
    
    # 保存详细结果
    if all_kernel_details:
        details_df = pd.DataFrame(all_kernel_details)
        details_df.to_csv('distinct_kernels_details.csv', index=False)
        print(f"详细结果已保存到: distinct_kernels_details.csv")

def main():
    """
    主函数
    """
    
    if len(sys.argv) > 1:
        # 分析指定文件
        trace_file = sys.argv[1]
        case_name = sys.argv[2] if len(sys.argv) > 2 else None
        analyze_distinct_kernels(trace_file, case_name)
    else:
        # 分析所有case
        analyze_all_cases_distinct()

if __name__ == "__main__":
    main()