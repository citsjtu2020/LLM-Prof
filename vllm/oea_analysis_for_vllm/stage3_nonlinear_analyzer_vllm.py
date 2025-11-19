#!/usr/bin/env python3
"""
OEA Stage 3: vLLM非Linear算子分析脚本 (高效版本)
基于stage1的iteration信息，对不同阶段的算子采用不同的计算公式

主要改进:
1. 优化正则匹配性能，避免重复匹配
2. 预先对所有kernels进行分类缓存
3. 使用更高效的数据结构
4. 减少不必要的循环

使用方法:
python stage3_nonlinear_analyzer_vllm.py --input oea_stage1_Qwen2.5-32B_batch8_input128_output10_processed.json
"""

import json
import argparse
import os
import sys
import re
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

class VLLMNonLinearAnalyzerV2:
    def __init__(self):
        """初始化vLLM非Linear算子分析器"""
        
        # fwd_splitkv_kernel模式（用于识别Transformer层）
        self.fwd_splitkv_pattern = re.compile(r'.*flash.*fwd.*splitkv.*kernel.*', re.IGNORECASE)
        
        # 各类算子的匹配模式（简化版，只保留最关键的）
        self.operator_patterns = {
            'attention': re.compile(r'.*(attention|attn|flash|fmha|splitkv|reshape.*cache).*', re.IGNORECASE),
            'activation': re.compile(r'.*(silu|gelu|relu|act_and_mul).*', re.IGNORECASE),
            'memory': re.compile(r'.*(memcpy|memset).*', re.IGNORECASE),
            'reduction': re.compile(r'.*reduce.*', re.IGNORECASE),
            'layernorm': re.compile(r'.*(rmsnorm|rms_norm|layernorm|norm).*', re.IGNORECASE),
        }
        
        # kernel分类缓存
        self.kernel_classification_cache = {}
        
        print("初始化完成")
    
    def is_fwd_splitkv_kernel(self, kernel_name: str) -> bool:
        """
        判断是否为attention kernel (非combine类型)
        支持两种类型:
        1. Ampere系列: fwd_splitkv_kernel
        2. Hopper系列: FlashAttnFwdSm90
        """
        if 'combine' in kernel_name.lower():
            return False
        
        # 检查Hopper系列的FlashAttnFwdSm90算子
        if 'flashattnfwdsm90' in kernel_name.lower():
            return True
        
        # 检查Ampere系列的fwd_splitkv算子
        return self.fwd_splitkv_pattern.match(kernel_name) is not None
    
    def is_attention_related_kernel(self, kernel_name: str) -> bool:
        """判断是否为attention相关的kernel"""
        return self.operator_patterns['attention'].search(kernel_name.lower()) is not None
    
    def classify_kernel(self, kernel_name: str) -> str:
        """对kernel进行分类（带缓存）"""
        if kernel_name in self.kernel_classification_cache:
            return self.kernel_classification_cache[kernel_name]
        
        kernel_name_lower = kernel_name.lower()
        
        # 按优先级分类
        for op_type, pattern in self.operator_patterns.items():
            if pattern.search(kernel_name_lower):
                self.kernel_classification_cache[kernel_name] = op_type
                return op_type
        
        self.kernel_classification_cache[kernel_name] = 'unknown'
        return 'unknown'
    
    def calculate_attention_performance_prefill(self, iterations: List[Dict], model_config: Dict) -> Dict:
        """计算Attention算子性能 - Prefill阶段"""
        
        prefill_iterations = [it for it in iterations if it.get('phase') == 'prefill']
        
        if not prefill_iterations:
            return {'message': 'No prefill iterations found', 'kernels': []}

        # 从模型配置中读取参数
        hidden_size = model_config["hidden_size"]
        num_heads = model_config.get("num_attention_heads", model_config.get("num_heads"))
        num_kv_heads = model_config.get("num_key_value_heads", model_config.get("num_kv_heads", num_heads))
        
        tp_size = self.hardware_spec.get('tensor_parallelism', 1)
        head_dim = hidden_size // num_heads
        num_heads = num_heads // tp_size
        num_kv_heads = num_kv_heads // tp_size
        dtype_bytes = 2
        
        # 统计Transformer层数和收集kernels
        total_transformer_layers = 0
        all_attention_kernels = []
        
        for iteration in prefill_iterations:
            for kernel in iteration.get('kernels', []):
                kernel_name = kernel.get('name', '')
                
                if self.is_fwd_splitkv_kernel(kernel_name):
                    total_transformer_layers += 1
                
                if self.is_attention_related_kernel(kernel_name):
                    all_attention_kernels.append(kernel)
        
        print(f"\n=== Attention性能分析 (Prefill阶段) ===")
        print(f"Prefill iterations数: {len(prefill_iterations)}")
        print(f"检测到的Transformer层数: {total_transformer_layers}")
        print(f"所有attention相关kernels: {len(all_attention_kernels)}")
        
        # 获取token_size
        token_size = 2048
        if prefill_iterations and prefill_iterations[0].get('kernels'):
            first_kernel = prefill_iterations[0]['kernels'][0]
            token_size = first_kernel.get('token_size', 2048)
        
        seq_len = token_size
        
        # 计算单层的Attention FLOPS
        qk_flops = 2 * 1 * num_heads * seq_len * seq_len * head_dim
        softmax_flops = 3 * 1 * num_heads * seq_len * seq_len
        av_flops = 2 * 1 * num_heads * seq_len * seq_len * head_dim
        single_layer_flops = qk_flops + softmax_flops + av_flops
        total_flops = single_layer_flops * total_transformer_layers
        
        # 内存访问计算
        q_memory = 1 * seq_len * num_heads * head_dim * dtype_bytes
        k_memory = 1 * seq_len * num_kv_heads * head_dim * dtype_bytes
        v_memory = 1 * seq_len * num_kv_heads * head_dim * dtype_bytes
        scores_memory = 1 * num_heads * seq_len * seq_len * dtype_bytes
        output_memory = 1 * seq_len * num_heads * head_dim * dtype_bytes
        single_layer_memory = q_memory + k_memory + v_memory + scores_memory + output_memory
        total_memory_access = single_layer_memory * total_transformer_layers
        
        # 统计执行时间
        total_duration = sum(k.get('dur', 0) for k in all_attention_kernels)
        
        avg_ai = total_flops / total_memory_access if total_memory_access > 0 else 0
        
        print(f"  总FLOPS: {total_flops/1e12:.4f} TFLOPs ({total_transformer_layers}层)")
        print(f"  总时间: {total_duration/1000:.2f} ms")
        print(f"  平均AI: {avg_ai:.2f}")
        
        return {
            'phase': 'prefill',
            'total_flops': total_flops,
            'total_memory_access': total_memory_access,
            'total_duration_us': total_duration,
            'kernel_count': len(all_attention_kernels),
            'transformer_layers': total_transformer_layers,
            'avg_arithmetic_intensity': avg_ai,
            'complexity_note': 'O(seq_len^2) - quadratic complexity'
        }
    
    def calculate_attention_performance_decode(self, iterations: List[Dict], model_config: Dict) -> Dict:
        """计算Attention算子性能 - Decode阶段"""
        
        decode_iterations = [it for it in iterations if it.get('phase') == 'decode']
        
        if not decode_iterations:
            return {'message': 'No decode iterations found', 'kernels': []}

        # 从模型配置中读取参数
        hidden_size = model_config["hidden_size"]
        num_heads = model_config.get("num_attention_heads", model_config.get("num_heads"))
        num_kv_heads = model_config.get("num_key_value_heads", model_config.get("num_kv_heads", num_heads))
        num_layers = model_config.get("num_hidden_layers", 64)
        
        tp_size = self.hardware_spec.get('tensor_parallelism', 1)
        head_dim = hidden_size // num_heads
        num_heads = num_heads // tp_size
        num_kv_heads = num_kv_heads // tp_size
        dtype_bytes = 2
        
        batch_size = 1
        query_len = 1
        
        # 统计fwd_splitkv_kernel总数
        total_fwd_splitkv_count = 0
        all_attention_kernels = []
        
        for iteration in decode_iterations:
            for kernel in iteration.get('kernels', []):
                kernel_name = kernel.get('name', '')
                
                if self.is_fwd_splitkv_kernel(kernel_name):
                    total_fwd_splitkv_count += 1
                
                if self.is_attention_related_kernel(kernel_name):
                    all_attention_kernels.append(kernel)
        
        decode_iterations_count = len(decode_iterations)
        effective_layers = num_layers
        
        print(f"\n=== Attention性能分析 (Decode阶段) ===")
        print(f"Decode iterations数: {decode_iterations_count}")
        print(f"检测到的fwd_splitkv_kernel总数: {total_fwd_splitkv_count}")
        print(f"推断: {effective_layers}层 × {decode_iterations_count}次decode")
        print(f"所有attention相关kernels: {len(all_attention_kernels)}")
        
        # 获取kv_len
        token_size = 2048
        if decode_iterations and decode_iterations[0].get('kernels'):
            first_kernel = decode_iterations[0]['kernels'][0]
            token_size = first_kernel.get('token_size', 2048)
        
        kv_len = token_size
        
        # 计算单次decode单层的Attention FLOPS
        qk_flops = 2 * batch_size * num_heads * query_len * kv_len * head_dim
        softmax_flops = 3 * batch_size * num_heads * query_len * kv_len
        av_flops = 2 * batch_size * num_heads * query_len * kv_len * head_dim
        single_decode_single_layer_flops = qk_flops + softmax_flops + av_flops
        total_flops = single_decode_single_layer_flops * effective_layers * decode_iterations_count
        
        # 内存访问计算
        q_memory = batch_size * query_len * num_heads * head_dim * dtype_bytes
        k_memory = batch_size * kv_len * num_kv_heads * head_dim * dtype_bytes
        v_memory = batch_size * kv_len * num_kv_heads * head_dim * dtype_bytes
        scores_memory = batch_size * num_heads * query_len * kv_len * dtype_bytes
        output_memory = batch_size * query_len * num_heads * head_dim * dtype_bytes
        single_decode_single_layer_memory = q_memory + k_memory + v_memory + scores_memory + output_memory
        total_memory_access = single_decode_single_layer_memory * effective_layers * decode_iterations_count
        
        # 统计执行时间
        total_duration = sum(k.get('dur', 0) for k in all_attention_kernels)
        
        avg_ai = total_flops / total_memory_access if total_memory_access > 0 else 0
        
        print(f"  总FLOPS: {total_flops/1e12:.4f} TFLOPs ({effective_layers}层 × {decode_iterations_count}次)")
        print(f"  总时间: {total_duration/1000:.2f} ms")
        print(f"  平均AI: {avg_ai:.2f}")
        
        return {
            'phase': 'decode',
            'total_flops': total_flops,
            'total_memory_access': total_memory_access,
            'total_duration_us': total_duration,
            'kernel_count': len(all_attention_kernels),
            'transformer_layers': effective_layers,
            'decode_iterations': decode_iterations_count,
            'avg_arithmetic_intensity': avg_ai,
            'complexity_note': 'O(kv_len) - linear complexity'
        }
    
    def calculate_other_operator_performance(self, iterations: List[Dict], model_config: Dict, 
                                            operator_type: str, phase: str) -> Dict:
        """计算其他非linear算子性能"""
        
        phase_iterations = [it for it in iterations if it.get('phase') == phase]
        
        if not phase_iterations:
            return {'message': f'No {phase} iterations found'}
        
        # 收集该operator_type的所有kernels（使用缓存的分类结果）
        operator_kernels = []
        for iteration in phase_iterations:
            for kernel in iteration.get('kernels', []):
                if self.classify_kernel(kernel.get('name', '')) == operator_type:
                    operator_kernels.append(kernel)
        
        if not operator_kernels:
            return {'message': f'No {operator_type} kernels found'}
        
        hidden_size = model_config["hidden_size"]
        dtype_bytes = 2
        batch_size = 1
        
        total_flops = 0
        total_memory_access = 0
        total_duration = sum(k.get('dur', 0) for k in operator_kernels)
        
        print(f"分析{operator_type}算子 ({phase}阶段): {len(operator_kernels)}个kernels")
        
        # 简化计算：使用统一的估算方法
        for kernel in operator_kernels:
            token_size = kernel.get('token_size', 2048)
            seq_len = token_size if phase == 'prefill' else 1
            
            if operator_type == 'activation':
                intermediate_size = model_config.get("intermediate_size", hidden_size * 4)
                kernel_flops = batch_size * seq_len * intermediate_size * 2
                kernel_memory = batch_size * seq_len * intermediate_size * dtype_bytes * 2
            elif operator_type == 'layernorm':
                kernel_flops = batch_size * seq_len * hidden_size * 3
                kernel_memory = batch_size * seq_len * hidden_size * dtype_bytes * 3
            else:
                kernel_flops = batch_size * seq_len * hidden_size
                kernel_memory = batch_size * seq_len * hidden_size * dtype_bytes * 2
            
            total_flops += kernel_flops
            total_memory_access += kernel_memory
        
        avg_ai = total_flops / total_memory_access if total_memory_access > 0 else 0
        
        print(f"  {operator_type} ({phase}): 总FLOPS={total_flops/1e12:.4f}T, 平均AI={avg_ai:.2f}")
        
        return {
            'operator_type': operator_type,
            'phase': phase,
            'total_flops': total_flops,
            'total_memory_access': total_memory_access,
            'total_duration_us': total_duration,
            'kernel_count': len(operator_kernels),
            'avg_arithmetic_intensity': avg_ai
        }
    
    def analyze_nonlinear_operators(self, stage1_data: Dict) -> Dict:
        """分析非Linear算子"""
        
        print(f"\n=== vLLM非Linear算子分析 (高效版本) ===")
        
        # 提取基本信息
        case_info = stage1_data.get('case_info', {})
        self.hardware_spec = stage1_data.get('hardware_spec', {})
        iteration_analysis = stage1_data.get('iteration_analysis', {})
        
        iterations = iteration_analysis.get('iterations', [])
        
        if not iterations:
            print("错误: 未找到iterations数据")
            return {}
        
        # 获取模型配置
        model_name = case_info.get('model_name', 'Unknown')
        
        model_configs = {
            'Qwen2.5-32B': {
                'hidden_size': 5120,
                'intermediate_size': 27648,
                'num_attention_heads': 40,
                'num_key_value_heads': 8,
                'num_hidden_layers': 64,
            },
            'Qwen2.5-7B': {
                'hidden_size': 3584,
                'intermediate_size': 18944,
                'num_attention_heads': 28,
                'num_key_value_heads': 4,
                'num_hidden_layers': 28,
            },
        }
        
        # 获取模型配置
        model_config = model_configs.get(model_name)
        if not model_config:
            for config_name, config_data in model_configs.items():
                if config_name.lower() in model_name.lower():
                    model_config = config_data
                    break
            if not model_config:
                model_config = model_configs['Qwen2.5-32B']
        
        print(f"模型: {model_name}")
        print(f"硬件: {self.hardware_spec.get('name', 'Unknown')}")
        print(f"总iterations: {len(iterations)}")
        
        # 统计各phase的iterations
        prefill_count = sum(1 for it in iterations if it.get('phase') == 'prefill')
        decode_count = sum(1 for it in iterations if it.get('phase') == 'decode')
        print(f"Prefill iterations: {prefill_count}, Decode iterations: {decode_count}")
        
        # 分析结果
        analysis_results = {
            'prefill_analysis': {},
            'decode_analysis': {},
            'summary': {
                'total_iterations': len(iterations),
                'prefill_iterations': prefill_count,
                'decode_iterations': decode_count,
                'framework': 'vllm'
            }
        }
        
        # 分析Attention算子
        print(f"\n=== PREFILL阶段分析 ===")
        prefill_attention_result = self.calculate_attention_performance_prefill(iterations, model_config)
        if 'message' not in prefill_attention_result:
            analysis_results['prefill_analysis']['attention'] = prefill_attention_result
        
        print(f"\n=== DECODE阶段分析 ===")
        decode_attention_result = self.calculate_attention_performance_decode(iterations, model_config)
        if 'message' not in decode_attention_result:
            analysis_results['decode_analysis']['attention'] = decode_attention_result
        
        # 分析其他算子
        print(f"\n=== 其他算子分析 ===")
        for phase in ['prefill', 'decode']:
            for op_type in ['activation', 'memory', 'reduction', 'layernorm']:
                result = self.calculate_other_operator_performance(iterations, model_config, op_type, phase)
                if 'message' not in result:
                    analysis_results[f'{phase}_analysis'][op_type] = result
        
        # 计算总体统计
        total_prefill_flops = sum(r.get('total_flops', 0) for r in analysis_results['prefill_analysis'].values())
        total_decode_flops = sum(r.get('total_flops', 0) for r in analysis_results['decode_analysis'].values())
        total_prefill_memory = sum(r.get('total_memory_access', 0) for r in analysis_results['prefill_analysis'].values())
        total_decode_memory = sum(r.get('total_memory_access', 0) for r in analysis_results['decode_analysis'].values())
        
        analysis_results['summary'].update({
            'prefill_total_flops': total_prefill_flops,
            'decode_total_flops': total_decode_flops,
            'prefill_total_memory': total_prefill_memory,
            'decode_total_memory': total_decode_memory,
            'prefill_avg_arithmetic_intensity': total_prefill_flops / total_prefill_memory if total_prefill_memory > 0 else 0,
            'decode_avg_arithmetic_intensity': total_decode_flops / total_decode_memory if total_decode_memory > 0 else 0,
            'flops_ratio_decode_to_prefill': total_decode_flops / total_prefill_flops if total_prefill_flops > 0 else 0
        })
        
        print(f"\n=== 总体统计 ===")
        print(f"Prefill总FLOPS: {total_prefill_flops/1e12:.2f} TFLOPs")
        print(f"Decode总FLOPS: {total_decode_flops/1e12:.2f} TFLOPs")
        print(f"Decode/Prefill FLOPS比值: {analysis_results['summary']['flops_ratio_decode_to_prefill']:.3f}")
        
        return analysis_results

def main():
    parser = argparse.ArgumentParser(description='vLLM非Linear算子分析 (高效版本)')
    parser.add_argument('--input', required=True, help='Stage 1输出的JSON文件路径')
    parser.add_argument('--output', help='输出JSON文件路径 (默认: 自动生成)')
    
    args = parser.parse_args()
    
    # 加载Stage 1数据
    try:
        with open(args.input, 'r') as f:
            stage1_data = json.load(f)
        print(f"✓ 成功加载Stage 1数据")
    except Exception as e:
        print(f"✗ 加载Stage 1数据失败: {e}")
        sys.exit(1)
    
    # 设置输出文件路径
    if args.output:
        output_file = args.output
    else:
        pod_name = stage1_data.get('case_info', {}).get('pod_name', 'unknown')
        input_dir = os.path.dirname(args.input)
        output_file = os.path.join(input_dir, f'oea_stage3_{pod_name}_processed.json')
    
    print(f"=== OEA Stage 3: vLLM非Linear算子分析 (高效版本) ===")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {output_file}")
    
    analyzer = VLLMNonLinearAnalyzerV2()
    results = analyzer.analyze_nonlinear_operators(stage1_data)
    
    # 添加元数据
    results['metadata'] = {
        'analysis_type': 'vllm_nonlinear_analysis',
        'input_file': args.input,
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'description': 'vLLM nonlinear operator analysis (optimized version)',
        'framework': 'vllm'
    }
    
    # 保存结果到JSON文件
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✓ 结果已保存到 {output_file}")

if __name__ == "__main__":
    main()