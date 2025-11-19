#!/usr/bin/env python3
"""
从distinct_kernels_details.csv读取kernel数据，使用修正后的分类规则进行分类
生成包含所有kernel及其operator分类的CSV文件，供人工审核
"""

import csv
from collections import defaultdict

def classify_kernel(kernel_name):
    """
    根据关键词对kernel进行分类 - 修正版本
    """
    kernel_lower = kernel_name.lower()
    
    # Communication算子关键词 (最高优先级)
    communication_keywords = ['nccl', 'allreduce', 'allgather', 'reducescatter', 'broadcast',
                             'p2p', 'send', 'recv', 'communication']
    
    # GEMM算子关键词 (第二优先级，包含nvjet等重要关键词)
    gemm_keywords = ['gemm', 'sgemm', 'dgemm', 'hgemm', 'cublaslt', 'cutlass', 'ampere_sgemm', 
                     'ampere_hgemm', 'turing_sgemm', 'turing_hgemm', 'volta_sgemm', 'volta_hgemm',
                     'sm80_gemm', 'sm86_gemm', 'sm75_gemm', 'sm70_gemm', 'nvjet', 'splitkreduce',
                     'ampere_fp16_s16816gemm', 'ampere_fp16_s1688gemm', 'ampere_s16816gemm']
    
    # Attention算子关键词  
    attention_keywords = ['flash_attention', 'fmha', 'multihead_attention', 'masked_multihead_attention',
                         'batchprefillwithpagedkvcache', 'add_fusedqkv_bias_transpose', 
                         'decode_add_fusedqkv_bias_transpose', 'addbiasSoftmax']
    
    # Normalization算子关键词
    norm_keywords = ['rmsnorm', 'layernorm', 'batchnorm', 'groupnorm', 'instancenorm',
                    'fusedqkrmsnorm', 'generalrmsnorm', 'rms_norm', 'layer_norm']
    
    # Activation算子关键词
    activation_keywords = ['silu', 'relu', 'gelu', 'tanh', 'sigmoid', 'swish', 'mish', 'elu',
                          'leaky_relu', 'prelu', 'generic_activation', 'act_and_mul', 'softmax']
    
    # Sampling算子关键词
    sampling_keywords = ['topk', 'topp', 'sampling', 'repetitionpenalty', 'temperature', 'radixsort',
                        'curand', 'setup_topk', 'set_topp', 'computetoppdecay', 'toppinitialize', 
                        'topp_beam_topk', 'topp_sampling', 'topktoppsampling', 'batchapplyrepetitionpenalty',
                        'batchapplyminlengthpenalty', 'batchapplytemperaturepenalty', 'minlengthpenalty']
    
    # Embedding算子关键词
    embedding_keywords = ['embedding_lookup', 'lookuphiddenstateoflaststoken', 'embedding', 'embed']
    
    # Memory算子关键词 (移除nvjet相关，这些应该是GEMM)
    memory_keywords = ['memcpy', 'memset', 'unrolled_elementwise', 'elementwise_kernel', 
                      'transpose4d', 'transposeaxis01', 'indexselect', 'vectorized_elementwise',
                      'reduce_kernel', 'distribution_elementwise']
    
    # Utility算子关键词
    utility_keywords = ['convertoffsettoaddr', 'convertoffsettoblockarraydata', 'getpaddingoffset',
                       'getcuseqlens', 'addbiasresidual', 'cast', 'convert', 'dtype', 'quantize', 
                       'dequantize', 'scale', 'clip', 'clamp', 'mask', 'pad', 'unpad', 'fillfunction']
    
    # 按优先级进行分类（避免重复匹配）
    for keyword in communication_keywords:
        if keyword in kernel_lower:
            return 'Communication'
    
    for keyword in gemm_keywords:
        if keyword in kernel_lower:
            return 'GEMM'
    
    for keyword in attention_keywords:
        if keyword in kernel_lower:
            return 'Attention'
    
    for keyword in norm_keywords:
        if keyword in kernel_lower:
            return 'Normalization'
    
    for keyword in activation_keywords:
        if keyword in kernel_lower:
            return 'Activation'
    
    for keyword in sampling_keywords:
        if keyword in kernel_lower:
            return 'Sampling'
    
    for keyword in embedding_keywords:
        if keyword in kernel_lower:
            return 'Embedding'
    
    for keyword in memory_keywords:
        if keyword in kernel_lower:
            return 'Memory'
    
    for keyword in utility_keywords:
        if keyword in kernel_lower:
            return 'Utility'
    
    return 'Others'

def analyze_potential_conflicts(kernel_name, operator_type):
    """
    分析潜在的分类冲突和问题
    """
    kernel_lower = kernel_name.lower()
    potential_issues = []
    
    # 检测GEMM和Attention的冲突
    has_gemm = any(kw in kernel_lower for kw in ['gemm', 'sgemm', 'hgemm', 'cutlass', 'nvjet'])
    has_attention = any(kw in kernel_lower for kw in ['qkv', 'attention', 'attn', 'multihead'])
    has_norm = any(kw in kernel_lower for kw in ['norm', 'rmsnorm', 'layernorm'])
    has_activation = any(kw in kernel_lower for kw in ['silu', 'gelu', 'relu', 'activation'])
    
    if has_gemm and has_attention:
        potential_issues.append("GEMM+Attention混合")
    if has_attention and has_norm:
        potential_issues.append("Attention+Norm混合")
    if has_gemm and has_norm:
        potential_issues.append("GEMM+Norm混合")
    if has_activation and (has_gemm or has_attention):
        potential_issues.append("包含Activation")
    
    # 特殊关键词检测
    if 'fused' in kernel_lower:
        potential_issues.append("Fused操作")
    if 'qkv' in kernel_lower and operator_type == 'GEMM':
        potential_issues.append("QKV相关但分类为GEMM")
    if 'rmsnorm' in kernel_lower and operator_type != 'Normalization':
        potential_issues.append("RMSNorm但未分类为Normalization")
    
    # 检查nvjet是否被正确分类为GEMM
    if 'nvjet' in kernel_lower and operator_type != 'GEMM':
        potential_issues.append("nvjet应该是GEMM")
    
    if operator_type == 'Others':
        potential_issues.append("需要人工分类")
    
    return '; '.join(potential_issues) if potential_issues else ""

def main():
    """主函数"""
    input_file = 'distinct_kernels_details.csv'
    output_file = 'kernel_operator_classification_fixed.csv'
    
    # 读取distinct_kernels_details.csv
    kernels_data = []
    kernel_cases = defaultdict(list)
    kernel_counts = defaultdict(int)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = row['Case']
            kernel_name = row['Kernel_Name']
            execution_count = int(row['Execution_Count'])
            
            kernel_cases[kernel_name].append(case)
            kernel_counts[kernel_name] += execution_count
    
    # 对每个unique kernel进行分类
    classified_kernels = []
    operator_stats = defaultdict(int)
    
    for kernel_name in sorted(kernel_cases.keys()):
        operator_type = classify_kernel(kernel_name)
        cases = ', '.join(sorted(set(kernel_cases[kernel_name])))
        total_count = kernel_counts[kernel_name]
        potential_issues = analyze_potential_conflicts(kernel_name, operator_type)
        
        classified_kernels.append({
            'Kernel_Name': kernel_name,
            'Operator_Type': operator_type,
            'Appears_In_Cases': cases,
            'Total_Execution_Count': total_count,
            'Potential_Issues': potential_issues
        })
        
        operator_stats[operator_type] += 1
    
    # 写入输出CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Kernel_Name', 'Operator_Type', 'Appears_In_Cases', 'Total_Execution_Count', 'Potential_Issues']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # 按operator类型分组写入
        for operator_type in ['GEMM', 'Attention', 'Normalization', 'Activation', 'Sampling', 
                             'Embedding', 'Memory', 'Utility', 'Communication', 'Others']:
            type_kernels = [k for k in classified_kernels if k['Operator_Type'] == operator_type]
            for kernel in type_kernels:
                writer.writerow(kernel)
    
    # 生成统计报告
    print(f"=== Kernel分类结果 (修正版) ===")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"总计unique kernel数量: {len(classified_kernels)}")
    print("\n各operator类型kernel数量:")
    
    for operator_type in ['GEMM', 'Attention', 'Normalization', 'Activation', 'Sampling', 
                         'Embedding', 'Memory', 'Utility', 'Communication', 'Others']:
        count = operator_stats.get(operator_type, 0)
        if count > 0:
            print(f"  {operator_type}: {count}")
    
    # 检查nvjet分类情况
    nvjet_kernels = [k for k in classified_kernels if 'nvjet' in k['Kernel_Name'].lower()]
    nvjet_gemm = [k for k in nvjet_kernels if k['Operator_Type'] == 'GEMM']
    nvjet_others = [k for k in nvjet_kernels if k['Operator_Type'] != 'GEMM']
    
    print(f"\nnvjet kernel分类情况:")
    print(f"  总计nvjet kernel: {len(nvjet_kernels)}")
    print(f"  正确分类为GEMM: {len(nvjet_gemm)}")
    print(f"  错误分类为其他: {len(nvjet_others)}")
    
    if nvjet_others:
        print(f"\n错误分类的nvjet kernel:")
        for kernel in nvjet_others[:5]:
            print(f"  - {kernel['Kernel_Name'][:60]}... -> {kernel['Operator_Type']}")
    
    # 重点关注需要人工审核的kernel
    conflict_kernels = [k for k in classified_kernels if k['Potential_Issues'] and 'GEMM+Attention' in k['Potential_Issues']]
    others_kernels = [k for k in classified_kernels if k['Operator_Type'] == 'Others']
    fused_kernels = [k for k in classified_kernels if k['Potential_Issues'] and 'Fused' in k['Potential_Issues']]
    
    if conflict_kernels:
        print(f"\n需要重点审核的GEMM/Attention冲突kernel ({len(conflict_kernels)}个):")
        for kernel in conflict_kernels[:5]:
            print(f"  - {kernel['Kernel_Name'][:60]}...")
        if len(conflict_kernels) > 5:
            print(f"  ... 还有{len(conflict_kernels)-5}个")
    
    if fused_kernels:
        print(f"\nFused操作kernel ({len(fused_kernels)}个):")
        for kernel in fused_kernels[:5]:
            print(f"  - {kernel['Kernel_Name'][:60]}...")
        if len(fused_kernels) > 5:
            print(f"  ... 还有{len(fused_kernels)-5}个")
    
    if others_kernels:
        print(f"\n需要人工分类的Others kernel ({len(others_kernels)}个):")
        for kernel in others_kernels[:5]:
            print(f"  - {kernel['Kernel_Name'][:60]}...")
        if len(others_kernels) > 5:
            print(f"  ... 还有{len(others_kernels)-5}个")
    
    print(f"\n请查看 {output_file} 文件进行人工审核和调整")

if __name__ == "__main__":
    main()