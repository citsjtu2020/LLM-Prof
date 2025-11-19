#!/usr/bin/env python3
"""
基于oea_experiment方法分析四个case的算子时间占比
"""

import json
import os
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def load_case_info_from_diff():
    """
    从diff.txt文件中读取case的模型信息
    
    Returns:
        Dict: case_name -> {'model': str, 'gpu_card': str, 'gpu_num': int}
    """
    case_info = {}
    
    try:
        with open('diff.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按空行分割不同的case
        cases = content.strip().split('\n\n')
        
        for case_block in cases:
            lines = case_block.strip().split('\n')
            if len(lines) < 4:
                continue
                
            # 第一行是case名称
            case_name_line = lines[0]
            
            # 提取简化的case名称
            if 'qwen2-7b-query' in case_name_line:
                case_name = 'qwen2-7b-query'
            elif 'qwen2-ast-sft' in case_name_line:
                case_name = 'qwen2-ast-sft'
            elif 'qwen3-32b-for-omega' in case_name_line:
                case_name = 'qwen3-32b-omega'
            elif 'recommend-intent' in case_name_line:
                case_name = 'recommend-intent'
            else:
                continue
            
            # 解析模型信息
            model = None
            gpu_card = None
            gpu_num = None
            
            for line in lines[1:]:
                if line.startswith('model '):
                    model = line.split('model ')[1].strip()
                elif line.startswith('gpu_card '):
                    gpu_card = line.split('gpu_card ')[1].strip().upper()
                elif line.startswith('gpu_num '):
                    gpu_num = int(line.split('gpu_num ')[1].strip())
            
            if model and gpu_card and gpu_num is not None:
                case_info[case_name] = {
                    'model': model,
                    'gpu_card': gpu_card,
                    'gpu_num': gpu_num
                }
    
    except Exception as e:
        print(f"读取diff.txt文件失败: {e}")
        # 返回默认信息
        case_info = {
            'qwen2-7b-query': {'model': 'qwen2-7b', 'gpu_card': 'L20', 'gpu_num': 1},
            'qwen2-ast-sft': {'model': 'qwen2-7b', 'gpu_card': 'A800', 'gpu_num': 1},
            'qwen3-32b-omega': {'model': 'qwen3-32b', 'gpu_card': 'H20', 'gpu_num': 2},
            'recommend-intent': {'model': 'qwen3-4b', 'gpu_card': 'H20', 'gpu_num': 1}
        }
    
    return case_info

def analyze_kernel_breakdown(trace_file_path: str) -> Dict[str, float]:
    """
    分析单个trace文件的算子时间占比
    
    Args:
        trace_file_path: trace文件路径
        
    Returns:
        Dict: 各类算子的时间占比
    """
    
    try:
        with open(trace_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件 {trace_file_path} 失败: {e}")
        return {}
    
    # 计算各类算子的执行时间
    total_kernel_duration = 0
    gemm_duration = 0           # GEMM/矩阵乘法算子
    attention_duration = 0      # Attention算子
    normalization_duration = 0  # 归一化算子
    activation_duration = 0     # 激活函数算子
    sampling_duration = 0       # 采样算子
    memory_duration = 0         # 内存操作算子
    communication_duration = 0  # 通信算子
    embedding_duration = 0      # 嵌入算子
    utility_duration = 0        # 工具算子
    
    for event in data.get("traceEvents", []):
        if event.get('cat', '') == 'kernel':
            duration = event.get("dur", 0)
            kernel_name = event.get('name', '').lower()
            
            total_kernel_duration += duration
            
            # Communication算子关键词 (最高优先级)
            if any(keyword in kernel_name for keyword in [
                'nccl', 'allreduce', 'allgather', 'reducescatter', 'broadcast',
                'p2p', 'send', 'recv', 'communication'
            ]):
                communication_duration += duration
            
            # GEMM算子关键词 (第二优先级，包含nvjet等重要关键词)
            elif any(keyword in kernel_name for keyword in [
                'gemm', 'sgemm', 'dgemm', 'hgemm', 'cublaslt', 'cutlass', 'ampere_sgemm', 
                'ampere_hgemm', 'turing_sgemm', 'turing_hgemm', 'volta_sgemm', 'volta_hgemm',
                'sm80_gemm', 'sm86_gemm', 'sm75_gemm', 'sm70_gemm', 'nvjet', 'splitkreduce',
                'ampere_fp16_s16816gemm', 'ampere_fp16_s1688gemm', 'ampere_s16816gemm'
            ]):
                gemm_duration += duration
            
            # Attention算子关键词  
            elif any(keyword in kernel_name for keyword in [
                'flash_attention', 'fmha', 'multihead_attention', 'masked_multihead_attention',
                'batchprefillwithpagedkvcache', 'add_fusedqkv_bias_transpose', 
                'decode_add_fusedqkv_bias_transpose'
            ]):
                attention_duration += duration
            
            # Normalization算子关键词
            elif any(keyword in kernel_name for keyword in [
                'rmsnorm', 'layernorm', 'batchnorm', 'groupnorm', 'instancenorm',
                'fusedqkrmsnorm', 'generalrmsnorm', 'rms_norm', 'layer_norm'
            ]):
                normalization_duration += duration
            
            # Activation算子关键词
            elif any(keyword in kernel_name for keyword in [
                'silu', 'relu', 'gelu', 'tanh', 'sigmoid', 'swish', 'mish', 'elu',
                'leaky_relu', 'prelu', 'generic_activation', 'act_and_mul', 'softmax', 'addbiasSoftmax'
            ]):
                activation_duration += duration
            
            # Sampling算子关键词
            elif any(keyword in kernel_name for keyword in [
                'topk', 'topp', 'sampling', 'repetitionpenalty', 'temperature', 'radixsort',
                'curand', 'setup_topk', 'set_topp', 'computetoppdecay', 'toppinitialize', 
                'topp_beam_topk', 'topp_sampling', 'topktoppsampling', 'batchapplyrepetitionpenalty',
                'batchapplyminlengthpenalty', 'batchapplytemperaturepenalty', 'minlengthpenalty'
            ]):
                sampling_duration += duration
            
            # Embedding算子关键词
            elif any(keyword in kernel_name for keyword in [
                'embedding_lookup', 'lookuphiddenstateoflaststoken', 'embedding', 'embed'
            ]):
                embedding_duration += duration
            
            # Memory算子关键词 (移除nvjet相关，这些应该是GEMM)
            elif any(keyword in kernel_name for keyword in [
                'memcpy', 'memset', 'unrolled_elementwise', 'elementwise_kernel', 
                'transpose4d', 'transposeaxis01', 'indexselect', 'vectorized_elementwise',
                'reduce_kernel', 'distribution_elementwise'
            ]):
                memory_duration += duration
            
            # Utility算子关键词
            elif any(keyword in kernel_name for keyword in [
                'convertoffsettoaddr', 'convertoffsettoblockarraydata', 'getpaddingoffset',
                'getcuseqlens', 'addbiasresidual', 'cast', 'convert', 'dtype', 'quantize', 
                'dequantize', 'scale', 'clip', 'clamp', 'mask', 'pad', 'unpad', 'fillfunction'
            ]):
                utility_duration += duration
    
    if total_kernel_duration == 0:
        return {
            "GEMM": 0.0,
            "Attention": 0.0,
            "Normalization": 0.0,
            "Activation": 0.0,
            "Sampling": 0.0,
            "Embedding": 0.0,
            "Memory": 0.0,
            "Utility": 0.0,
            "Communication": 0.0,
            "Others": 0.0,
            "Total_Duration_us": 0
        }
    
    # 计算其他算子时间
    classified_duration = (gemm_duration + attention_duration + normalization_duration + 
                          activation_duration + sampling_duration + embedding_duration +
                          memory_duration + utility_duration + communication_duration)
    other_duration = total_kernel_duration - classified_duration
    
    # 计算占比
    breakdown = {
        "GEMM": gemm_duration / total_kernel_duration,
        "Attention": attention_duration / total_kernel_duration,
        "Normalization": normalization_duration / total_kernel_duration,
        "Activation": activation_duration / total_kernel_duration,
        "Sampling": sampling_duration / total_kernel_duration,
        "Embedding": embedding_duration / total_kernel_duration,
        "Memory": memory_duration / total_kernel_duration,
        "Utility": utility_duration / total_kernel_duration,
        "Communication": communication_duration / total_kernel_duration,
        "Others": other_duration / total_kernel_duration,
        "Total_Duration_us": total_kernel_duration
    }
    
    return breakdown

def find_trace_files() -> List[Tuple[str, str]]:
    """
    查找项目中的四个case的trace文件
    
    Returns:
        List[Tuple]: (case_name, trace_file_path) 的列表
    """
    
    trace_files = []
    
    # 根据实际找到的文件路径更新case目录
    case_patterns = [
        ("qwen2-7b-query", "qwen2-7b-query-na61-l20.inference-part0-caa0648d-b-a069"),
        ("qwen2-ast-sft", "qwen2-ast-sft-modelv2-kto-na61-a800.inference-part0-b5b19bcf-b-5289"), 
        ("qwen3-32b-omega", "qwen3-32b-for-omega-na61-h20-2tp-spot.inference-part0-798b1850-a-56a1"),
        ("recommend-intent", "recommend-intent-qwen3-4b-na61-h20.inference-part0-928ea825-a-9825")
    ]
    
    for case_name, case_dir in case_patterns:
        print(f"检查目录: {case_dir}")
        if os.path.exists(case_dir):
            print(f"  目录存在，查找trace文件...")
            # 查找该目录下的子目录中的ecos-trace JSON文件
            found_trace = False
            for root, dirs, files in os.walk(case_dir):
                for file in files:
                    # 查找ecos-trace开头的JSON文件
                    if file.startswith('ecos-trace') and file.endswith('.json'):
                        trace_file = os.path.join(root, file)
                        trace_files.append((case_name, trace_file))
                        print(f"  找到trace文件: {trace_file}")
                        found_trace = True
                        break
                if found_trace:
                    break
            if not found_trace:
                print(f"  未找到trace文件")
        else:
            print(f"  目录不存在")
    
    return trace_files

def analyze_all_cases():
    """
    分析所有case的算子时间占比
    """
    
    print("=== 四个Case算子时间占比分析 ===")
    print("基于完善的10类分类方法\n")
    
    # 查找trace文件
    trace_files = find_trace_files()
    
    if not trace_files:
        print("未找到任何trace文件，请检查文件路径")
        return
    
    print(f"找到 {len(trace_files)} 个case的trace文件:")
    for case_name, trace_file in trace_files:
        print(f"  - {case_name}: {trace_file}")
    print()
    
    # 分析每个case
    results = {}
    detailed_results = []
    
    for case_name, trace_file in trace_files:
        print(f"正在分析 {case_name}...")
        breakdown = analyze_kernel_breakdown(trace_file)
        
        if breakdown:
            results[case_name] = breakdown
            
            # 打印结果
            total_ms = breakdown["Total_Duration_us"] / 1000
            print(f"  总kernel执行时间: {total_ms:.2f} ms")
            print(f"  GEMM算子:         {breakdown['GEMM']:.1%}")
            print(f"  Attention算子:    {breakdown['Attention']:.1%}")
            print(f"  Normalization算子: {breakdown['Normalization']:.1%}")
            print(f"  Activation算子:   {breakdown['Activation']:.1%}")
            print(f"  Sampling算子:     {breakdown['Sampling']:.1%}")
            print(f"  Embedding算子:    {breakdown['Embedding']:.1%}")
            print(f"  Memory算子:       {breakdown['Memory']:.1%}")
            print(f"  Utility算子:      {breakdown['Utility']:.1%}")
            print(f"  Communication算子: {breakdown['Communication']:.1%}")
            print(f"  其他算子:         {breakdown['Others']:.1%}")
            print()
            
            # 保存详细结果
            detailed_results.append({
                'Case': case_name,
                'GEMM_Percentage': breakdown['GEMM'] * 100,
                'Attention_Percentage': breakdown['Attention'] * 100,
                'Normalization_Percentage': breakdown['Normalization'] * 100,
                'Activation_Percentage': breakdown['Activation'] * 100,
                'Sampling_Percentage': breakdown['Sampling'] * 100,
                'Embedding_Percentage': breakdown['Embedding'] * 100,
                'Memory_Percentage': breakdown['Memory'] * 100,
                'Utility_Percentage': breakdown['Utility'] * 100,
                'Communication_Percentage': breakdown['Communication'] * 100,
                'Others_Percentage': breakdown['Others'] * 100,
                'Total_Duration_ms': total_ms
            })
        else:
            print(f"  分析失败或无有效数据")
            print()
    
    # 保存结果到CSV
    if detailed_results:
        df = pd.DataFrame(detailed_results)
        df.to_csv('operator_breakdown_analysis_enhanced.csv', index=False)
        print(f"详细结果已保存到: operator_breakdown_analysis_enhanced.csv")
        
        # 创建可视化图表
        create_visualization_enhanced(df)
        
        # 打印汇总对比
        print("\n=== 汇总对比 ===")
        print(df.to_string(index=False, float_format='%.1f'))
    
    return results

def create_visualization_enhanced(df: pd.DataFrame):
    """
    创建增强的算子占比可视化图表 - 顶会风格
    """

    # 加载case信息
    case_info = load_case_info_from_diff()

    # 设置专业的绘图参数（顶会论文风格）
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 18,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'axes.linewidth': 1.5,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2,
        'patch.linewidth': 0.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
    })

    # 创建图表 - 使用更大的尺寸和更高的DPI
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 堆叠柱状图 - 所有数据都已经是百分比形式，天然归一化
    categories = ['GEMM_Percentage', 'Attention_Percentage', 'Normalization_Percentage',
                 'Activation_Percentage', 'Sampling_Percentage', 'Embedding_Percentage',
                 'Memory_Percentage', 'Utility_Percentage', 'Communication_Percentage',
                 'Others_Percentage']
    category_labels = ['GEMM', 'Attention', 'Normalization', 'Activation',
                      'Sampling', 'Embedding', 'Memory', 'Utility',
                      'Communication', 'Others']

    # 使用专业的配色方案（ColorBrewer Set3，适合打印和色盲友好）
    colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072',
              '#80b1d3', '#fdb462', '#b3de69', '#fccde5',
              '#d9d9d9', '#bc80bd']

    # 设置柱子宽度
    bar_width = 0.5
    x_pos = range(len(df))

    bottom = [0] * len(df)
    bars = []
    for i, category in enumerate(categories):
        bar = ax.bar(x_pos, df[category], bottom=bottom, width=bar_width,
                     label=category_labels[i], color=colors[i],
                     edgecolor='white', linewidth=1.5)
        bars.append(bar)
        bottom = [b + v for b, v in zip(bottom, df[category])]

    # 设置标题和标签
    ax.set_ylabel('Execution Time (%)', fontweight='bold')
    # ax.set_xlabel('Model Configuration', fontweight='bold')
    ax.set_ylim(0, 105)

    # 设置网格（仅Y轴，虚线）
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_axisbelow(True)

    # 创建两行标签：第一行为模型名，第二行为卡型和数量
    new_labels = []
    for case in df['Case']:
        if case in case_info:
            info = case_info[case]
            model_name = info['model']
            gpu_info = f"{info['gpu_card']}×{info['gpu_num']}"
            new_labels.append(f"{model_name}\n({gpu_info})")
        else:
            new_labels.append(f"{case}\n(Unknown)")

    # 设置x轴标签
    ax.set_xticks(x_pos)
    ax.set_xticklabels(new_labels, fontweight='bold')

    # 在每个柱子上添加主要算子的百分比标签（只显示占比>8%的）
    for i, case in enumerate(df['Case']):
        y_pos = 0
        for j, category in enumerate(categories):
            percentage = df.iloc[i][category]
            if percentage > 8:  # 只显示占比大于8%的标签
                y_pos += percentage / 2
                ax.text(i, y_pos, f'{percentage:.1f}',
                       ha='center', va='center', fontweight='bold',
                       fontsize=18, color='black')
                y_pos += percentage / 2
            else:
                y_pos += percentage

    # 图例放在图表下方，分两行显示
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
             ncol=5, frameon=True, fancybox=False, shadow=False,
             edgecolor='black', facecolor='white', framealpha=1)

    # 调整布局
    plt.tight_layout()

    # 保存为高分辨率图片
    plt.savefig('operator_breakdown_normalized_visualization.png',
                dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')

    # 同时保存为PDF格式（论文常用）
    plt.savefig('operator_breakdown_normalized_visualization.pdf',
                bbox_inches='tight', facecolor='white', edgecolor='none')

    plt.show()

    print("归一化算子占比可视化图表已保存:")
    print("  - operator_breakdown_normalized_visualization.png (600 DPI)")
    print("  - operator_breakdown_normalized_visualization.pdf (矢量格式)")
    print("\n图表特点（顶会风格）：")
    print("- 使用专业的Times New Roman字体")
    print("- 采用ColorBrewer配色方案，适合打印和色盲友好")
    print("- 600 DPI高分辨率，适合论文发表")
    print("- 同时提供PDF矢量格式")
    print("- 简洁的设计，符合ISCA/OSDI等顶会要求")

def main():
    """
    主函数
    """
    
    # 分析所有case
    results = analyze_all_cases()
    
    if results:
        print(f"\n分析完成! 共分析了 {len(results)} 个case")
        print("结果文件:")
        print("  - operator_breakdown_analysis_enhanced.csv (详细数据)")
        print("  - operator_breakdown_normalized_visualization.png (可视化图表)")
    else:
        print("未找到有效的分析结果")

if __name__ == "__main__":
    main()