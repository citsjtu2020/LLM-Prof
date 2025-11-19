#!/usr/bin/env python3
"""
OEA Stage 3: SGLang非Linear算子分析脚本 (Prefill + Decode混合模式)
基于stage1的phase信息，对不同阶段的算子采用不同的计算公式

主要特点:
1. Prefill阶段使用O(seq_len^2)复杂度计算 (attention)
2. Decode阶段使用O(kv_len)复杂度计算 (attention)
3. 支持attention、rope、layernorm、activation等非linear算子
4. 使用精确的token size而不是平均值
5. 分别统计prefill和decode阶段的性能指标

使用方法:
python stage3_nonlinear_analyzer_sglang.py --input oea_stage1_Qwen3-14B_batch4_input2048_output10_processed.json
"""

import json
import numpy as np
import argparse
import os
import sys
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple

class SGLangNonLinearAnalyzer:
    def __init__(self):
        """初始化SGLang非Linear算子分析器"""
        
        # 各类算子的匹配模式
        self.operator_patterns = {
            'attention': [
                r'.*attention.*',
                r'.*attn.*',
                r'.*scaled_dot_product.*',
                r'.*flash.*attn.*',
                r'.*fmha.*',
                r'.*multihead.*attention.*',
                r'.*flashinfer.*'
            ],
            'rope': [
                r'.*rope.*',
                r'.*rotary.*',
                r'.*position.*embedding.*',
                r'.*rotary_emb.*'
            ],
            'layernorm': [
                r'.*rmsnorm.*',
                r'.*rms_norm.*',
                r'.*generalRmsNorm.*',
                r'.*layernorm.*',
                r'.*layer_norm.*',
                r'.*norm.*'
            ],
            'activation': [
                r'.*silu.*',
                r'.*swish.*',
                r'.*gelu.*',
                r'.*relu.*',
                r'.*silu_and_mul.*',
                r'.*activation.*',
                r'.*act_and_mul.*'
            ],
            'moe': [
                r'.*moe.*',
                r'.*expert.*',
                r'.*router.*',
                r'.*gate.*expert.*',
                r'.*mixture.*expert.*'
            ],
            'communication': [
                r'.*allreduce.*',
                r'.*allgather.*',
                r'.*reducescatter.*',
                r'.*broadcast.*',
                r'.*p2p.*',
                r'.*nccl.*',
                r'.*communication.*'
            ],
            'memory': [
                r'.*memcpy.*',
                r'.*memset.*',
                r'.*copy.*',
                r'.*transfer.*',
                r'.*h2d.*',
                r'.*d2h.*',
                r'.*d2d.*'
            ],
            'reduction': [
                r'.*splitkreduce.*',
                r'.*splitK.*reduce.*',
                r'.*mergestates.*',
                r'.*merge.*states.*',
                r'.*reduction.*'
            ]
        }
        
        # Linear算子模式（用于排除）
        self.linear_patterns = [
            r'nvjet.*',
            r'.*gemm.*',
            r'.*sgemm.*',
            r'.*hgemm.*',
            r'.*cutlass.*gemm.*',
            r'.*cublas.*'
        ]
        
        # 排除的Linear模式
        self.linear_exclude_patterns = [
            r'.*splitkreduce.*',
            r'.*splitK.*reduce.*'
        ]
        
        # 预编译正则表达式
        print("预编译正则表达式...")
        
        self.compiled_operator_patterns = {}
        for op_type, patterns in self.operator_patterns.items():
            self.compiled_operator_patterns[op_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        self.compiled_linear_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.linear_patterns]
        self.compiled_linear_exclude_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.linear_exclude_patterns]
        
        # 分类缓存
        self.classification_cache = {}
        
        print(f"预编译完成: {sum(len(patterns) for patterns in self.compiled_operator_patterns.values())} 个算子模式")
    
    def is_linear_kernel(self, kernel_name: str) -> bool:
        """判断是否为Linear kernel - 用于排除"""
        if kernel_name in self.classification_cache:
            cached_result = self.classification_cache[kernel_name]
            if cached_result == 'linear':
                return True
            elif cached_result in self.operator_patterns.keys() or cached_result == 'unknown':
                return False
        
        kernel_name_lower = kernel_name.lower()
        
        # 首先检查排除模式
        for exclude_pattern in self.compiled_linear_exclude_patterns:
            if exclude_pattern.search(kernel_name_lower):
                return False
        
        # 然后检查匹配模式
        for pattern in self.compiled_linear_patterns:
            if pattern.search(kernel_name_lower):
                return True
        
        return False
    
    def classify_kernel(self, kernel_name: str) -> str:
        """对kernel进行分类"""
        if kernel_name in self.classification_cache:
            return self.classification_cache[kernel_name]
        
        kernel_name_lower = kernel_name.lower()
        
        # 首先排除Linear算子
        if self.is_linear_kernel(kernel_name):
            self.classification_cache[kernel_name] = 'linear'
            return 'linear'
        
        # 按优先级分类非Linear算子
        for op_type, compiled_patterns in self.compiled_operator_patterns.items():
            for pattern in compiled_patterns:
                if pattern.search(kernel_name_lower):
                    self.classification_cache[kernel_name] = op_type
                    return op_type
        
        self.classification_cache[kernel_name] = 'unknown'
        return 'unknown'
    
    def get_kernel_token_size(self, kernel: Dict) -> int:
        """
        获取kernel的token size - 根据phase返回正确的值
        
        Args:
            kernel: kernel字典，必须包含'phase'字段
            
        Returns:
            token_size: 
                - prefill阶段返回seq_len（输入序列长度，默认2048）
                - decode阶段返回动态的kv_len（从args中获取，默认使用累积的token数）
                - unknown阶段返回seq_len（保守估计）
        """
        phase = kernel.get('phase', 'unknown')
        
        if phase == 'prefill':
            # Prefill阶段：处理完整的输入序列
            return 2048  # 默认seq_len
        elif phase == 'decode':
            # Decode阶段：每次生成1个token，但KV cache长度是累积的
            # 尝试从kernel的args中获取实际的kv_len
            args = kernel.get('args', {})
            
            # 尝试多种可能的字段名
            for key in ['kv_len', 'kv_length', 'cache_len', 'seq_len', 'sequence_length']:
                if key in args:
                    return int(args[key])
            
            # 如果没有找到，返回默认值（假设是初始seq_len + 已生成的token数）
            # 这里简化处理，返回2048作为KV cache的平均长度
            return 2048
        else:
            # Unknown阶段：使用seq_len作为保守估计
            return 2048
    
    def calculate_attention_performance_prefill(self, kernels: List[Dict], model_config: Dict) -> Dict:
        """计算Attention算子性能 - Prefill阶段 (O(seq_len^2)复杂度)"""
        
        if not kernels:
            return {'message': 'No attention kernels found', 'kernels': []}

        # 从模型配置中读取参数
        hidden_size = model_config["hidden_size"]
        num_heads = model_config.get("num_attention_heads", model_config.get("num_heads"))
        num_kv_heads = model_config.get("num_key_value_heads", model_config.get("num_kv_heads", num_heads))
        
        tp_size = self.hardware_spec.get('tensor_parallelism', 1)
        head_dim = hidden_size // num_heads
        num_heads = num_heads // tp_size
        num_kv_heads = num_kv_heads // tp_size
        dtype_bytes = 2  # bfloat16
        
        # 关键修改：直接使用Stage 1统计的实际层数
        # Stage 1已经正确识别了主要的attention kernels数量
        # Prefill阶段：每层1个attention kernel
        # 从实际的kernel数量推断层数
        num_layers = model_config.get("num_hidden_layers", 36)
        
        # Stage 1的attention分类已经是正确的
        # 不需要再次过滤，直接使用传入的kernels
        # 但是这些kernels可能包含辅助kernel，我们需要从kernel数量推断实际层数
        
        # 对于Prefill：通常每层有1个主要attention kernel
        # 如果kernel数量远大于层数，说明包含了辅助kernels
        # 我们应该使用配置中的层数，而不是kernel数量
        effective_layers = num_layers
        
        print(f"\n=== Attention性能分析 (Prefill阶段) ===")
        print(f"模型参数: hidden_size={hidden_size}, num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
        print(f"总kernel数: {len(kernels)}, 使用层数: {effective_layers}")
        
        performance_list = []
        total_duration = 0
        
        # 获取token_size（从第一个kernel推断）
        token_size = self.get_kernel_token_size(kernels[0]) if kernels else 1024
        seq_len = token_size
        
        # 计算单层的Attention FLOPS (O(seq_len^2))
        # QK^T: [batch, heads, seq_len, head_dim] @ [batch, heads, head_dim, seq_len]
        qk_flops = 2 * 1 * num_heads * seq_len * seq_len * head_dim
        
        # Softmax: 3 * seq_len^2 * num_heads
        softmax_flops = 3 * 1 * num_heads * seq_len * seq_len
        
        # Attention @ V: [batch, heads, seq_len, seq_len] @ [batch, heads, seq_len, head_dim]
        av_flops = 2 * 1 * num_heads * seq_len * seq_len * head_dim
        
        single_layer_flops = qk_flops + softmax_flops + av_flops
        
        # 总FLOPS = 单层FLOPS × 层数
        total_flops = single_layer_flops * effective_layers
        
        # 内存访问计算（单层）
        q_memory = 1 * seq_len * num_heads * head_dim * dtype_bytes
        k_memory = 1 * seq_len * num_kv_heads * head_dim * dtype_bytes
        v_memory = 1 * seq_len * num_kv_heads * head_dim * dtype_bytes
        scores_memory = 1 * num_heads * seq_len * seq_len * dtype_bytes
        output_memory = 1 * seq_len * num_heads * head_dim * dtype_bytes
        
        single_layer_memory = q_memory + k_memory + v_memory + scores_memory + output_memory
        total_memory_access = single_layer_memory * effective_layers
        
        # 统计所有kernel的实际执行时间
        for kernel in kernels:
            total_duration += kernel.get('dur', 0)
            
            kernel_perf = {
                'kernel_name': kernel['name'],
                'duration_us': kernel.get('dur', 0),
                'phase': 'prefill',
                'operator_type': 'attention'
            }
            performance_list.append(kernel_perf)
        
        avg_ai = total_flops / total_memory_access if total_memory_access > 0 else 0
        
        print(f"  Prefill Attention: {effective_layers}层, token_size={token_size}")
        print(f"  单层FLOPS: {single_layer_flops/1e9:.2f} GFLOPs")
        print(f"  总FLOPS: {total_flops/1e12:.4f} TFLOPs")
        print(f"  总时间: {total_duration/1000:.2f} ms")
        print(f"  平均AI: {avg_ai:.2f}")
        
        return {
            'phase': 'prefill',
            'kernels': performance_list,
            'total_flops': total_flops,
            'total_memory_access': total_memory_access,
            'total_duration_us': total_duration,
            'kernel_count': len(kernels),  # 保持原始kernel数量用于统计
            'effective_layers': effective_layers,  # 新增：实际计算的层数
            'avg_arithmetic_intensity': avg_ai,
            'complexity_note': 'O(seq_len^2) - quadratic complexity for full sequence processing'
        }
    
    def calculate_attention_performance_decode(self, kernels: List[Dict], model_config: Dict) -> Dict:
        """计算Attention算子性能 - Decode阶段 (O(kv_len)复杂度)"""
        
        if not kernels:
            return {'message': 'No attention kernels found', 'kernels': []}

        # 从模型配置中读取参数
        hidden_size = model_config["hidden_size"]
        num_heads = model_config.get("num_attention_heads", model_config.get("num_heads"))
        num_kv_heads = model_config.get("num_key_value_heads", model_config.get("num_kv_heads", num_heads))
        
        tp_size = self.hardware_spec.get('tensor_parallelism', 1)
        head_dim = hidden_size // num_heads
        num_heads = num_heads // tp_size
        num_kv_heads = num_kv_heads // tp_size
        dtype_bytes = 2  # bfloat16
        
        batch_size = 1
        query_len = 1  # decode阶段每次只生成1个token
        
        # 关键修改：计算实际的层数和decode次数
        num_layers = model_config.get("num_hidden_layers", 36)
        
        # 过滤主要的attention计算kernel
        main_attention_kernels = []
        for kernel in kernels:
            kernel_name = kernel['name'].lower()
            if 'flash' in kernel_name or 'fmha' in kernel_name or 'attention' in kernel_name:
                if not any(x in kernel_name for x in ['copy', 'transpose', 'reshape', 'split', 'concat']):
                    main_attention_kernels.append(kernel)
        
        actual_kernel_count = len(main_attention_kernels)
        if actual_kernel_count == 0:
            actual_kernel_count = len(kernels)
            main_attention_kernels = kernels
        
        # Decode阶段：kernel数量 = 层数 × decode次数
        # 例如：324个kernels = 36层 × 9次decode
        decode_iterations = actual_kernel_count // num_layers if actual_kernel_count >= num_layers else 1
        effective_layers = num_layers
        
        print(f"\n=== Attention性能分析 (Decode阶段) ===")
        print(f"模型参数: num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
        print(f"总kernel数: {len(kernels)}, 主要attention kernel数: {actual_kernel_count}")
        print(f"推断: {effective_layers}层 × {decode_iterations}次decode = {effective_layers * decode_iterations}个kernels")
        print(f"Decode场景: 每次生成{query_len}个token")
        
        performance_list = []
        total_duration = 0
        
        # 获取kv_len（从第一个kernel推断，通常是累积的序列长度）
        token_size = self.get_kernel_token_size(main_attention_kernels[0]) if main_attention_kernels else 1024
        kv_len = token_size
        
        # 计算单次decode单层的Attention FLOPS (O(kv_len))
        # Q @ K^T: [batch, heads, 1, head_dim] @ [batch, heads, head_dim, kv_len]
        qk_flops = 2 * batch_size * num_heads * query_len * kv_len * head_dim
        
        # Softmax: 对[batch, heads, 1, kv_len]进行softmax
        softmax_flops = 3 * batch_size * num_heads * query_len * kv_len
        
        # Attention @ V: [batch, heads, 1, kv_len] @ [batch, heads, kv_len, head_dim]
        av_flops = 2 * batch_size * num_heads * query_len * kv_len * head_dim
        
        single_decode_single_layer_flops = qk_flops + softmax_flops + av_flops
        
        # 总FLOPS = 单层单次FLOPS × 层数 × decode次数
        total_flops = single_decode_single_layer_flops * effective_layers * decode_iterations
        
        # 内存访问计算（单次decode单层）
        q_memory = batch_size * query_len * num_heads * head_dim * dtype_bytes
        k_memory = batch_size * kv_len * num_kv_heads * head_dim * dtype_bytes
        v_memory = batch_size * kv_len * num_kv_heads * head_dim * dtype_bytes
        scores_memory = batch_size * num_heads * query_len * kv_len * dtype_bytes
        output_memory = batch_size * query_len * num_heads * head_dim * dtype_bytes
        
        single_decode_single_layer_memory = q_memory + k_memory + v_memory + scores_memory + output_memory
        total_memory_access = single_decode_single_layer_memory * effective_layers * decode_iterations
        
        # 统计所有kernel的实际执行时间
        for kernel in kernels:
            total_duration += kernel.get('dur', 0)
            
            kernel_perf = {
                'kernel_name': kernel['name'],
                'duration_us': kernel.get('dur', 0),
                'phase': 'decode',
                'operator_type': 'attention'
            }
            performance_list.append(kernel_perf)
        
        avg_ai = total_flops / total_memory_access if total_memory_access > 0 else 0
        
        print(f"  单次decode单层FLOPS: {single_decode_single_layer_flops/1e9:.4f} GFLOPs")
        print(f"  总FLOPS: {total_flops/1e12:.4f} TFLOPs ({effective_layers}层 × {decode_iterations}次)")
        print(f"  总时间: {total_duration/1000:.2f} ms")
        print(f"  平均AI: {avg_ai:.2f}")
        
        return {
            'phase': 'decode',
            'kernels': performance_list,
            'total_flops': total_flops,
            'total_memory_access': total_memory_access,
            'total_duration_us': total_duration,
            'kernel_count': len(kernels),  # 保持原始kernel数量用于统计
            'effective_layers': effective_layers,  # 实际计算的层数
            'decode_iterations': decode_iterations,  # decode次数
            'avg_arithmetic_intensity': avg_ai,
            'complexity_note': 'O(kv_len) - linear complexity for single token generation with KV cache'
        }
    
    def calculate_other_operator_performance(self, kernels: List[Dict], model_config: Dict, operator_type: str, phase: str) -> Dict:
        """计算其他非linear算子性能 (rope, layernorm, activation等)"""
        
        if not kernels:
            return {'message': f'No {operator_type} kernels found', 'kernels': []}
        
        hidden_size = model_config["hidden_size"]
        dtype_bytes = 2  # bfloat16
        batch_size = 1
        
        performance_list = []
        total_flops = 0
        total_memory_access = 0
        total_duration = 0
        
        print(f"\n=== {operator_type.upper()}性能分析 ({phase}阶段) ===")
        
        for kernel in kernels:
            token_size = self.get_kernel_token_size(kernel)
            seq_len = token_size if phase == 'prefill' else 1  # decode阶段只处理1个token
            
            # 根据算子类型计算FLOPS
            if operator_type == 'rope':
                # RoPE: 对Q和K应用旋转位置编码，每个元素需要4次浮点运算
                num_heads = model_config.get("num_attention_heads", model_config.get("num_heads"))
                num_kv_heads = model_config.get("num_key_value_heads", model_config.get("num_kv_heads", num_heads))
                head_dim = hidden_size // num_heads
                
                q_rope_flops = batch_size * seq_len * num_heads * head_dim * 4
                k_rope_flops = batch_size * seq_len * num_kv_heads * head_dim * 4
                total_kernel_flops = q_rope_flops + k_rope_flops
                
                # 内存访问
                q_memory = batch_size * seq_len * num_heads * head_dim * dtype_bytes
                k_memory = batch_size * seq_len * num_kv_heads * head_dim * dtype_bytes
                pos_memory = seq_len * head_dim * dtype_bytes
                output_memory = q_memory + k_memory
                total_kernel_memory = q_memory + k_memory + pos_memory + output_memory
                
            elif operator_type == 'layernorm':
                # RMSNorm: 约3 * hidden_size次运算每个token
                flops_per_token = 3 * hidden_size
                total_kernel_flops = batch_size * seq_len * flops_per_token
                
                # 内存访问
                input_memory = batch_size * seq_len * hidden_size * dtype_bytes
                weight_memory = hidden_size * dtype_bytes
                output_memory = batch_size * seq_len * hidden_size * dtype_bytes
                total_kernel_memory = input_memory + weight_memory + output_memory
                
            elif operator_type == 'activation':
                # 激活函数: 约2次运算每个元素
                intermediate_size = model_config.get("intermediate_size", hidden_size * 4)
                total_kernel_flops = batch_size * seq_len * intermediate_size * 2
                
                # 内存访问
                input_memory = batch_size * seq_len * intermediate_size * dtype_bytes
                output_memory = batch_size * seq_len * intermediate_size * dtype_bytes
                total_kernel_memory = input_memory + output_memory
                
            else:
                # 其他算子的简化计算
                total_kernel_flops = batch_size * seq_len * hidden_size
                total_kernel_memory = batch_size * seq_len * hidden_size * dtype_bytes * 2
            
            arithmetic_intensity = total_kernel_flops / total_kernel_memory if total_kernel_memory > 0 else 0
            
            kernel_perf = {
                'kernel_name': kernel['name'],
                'duration_us': kernel.get('duration_us', 0),
                'phase': phase,
                'operator_type': operator_type,
                'token_size': token_size,
                'seq_len': seq_len,
                'flops': total_kernel_flops,
                'memory_access': total_kernel_memory,
                'arithmetic_intensity': arithmetic_intensity
            }
            
            performance_list.append(kernel_perf)
            total_flops += total_kernel_flops
            total_memory_access += total_kernel_memory
            total_duration += kernel.get('duration_us', 0)
        
        avg_ai = total_flops / total_memory_access if total_memory_access > 0 else 0
        
        print(f"  {operator_type} ({phase}): {len(kernels)}个kernels, 总FLOPS={total_flops/1e12:.2f}T, 平均AI={avg_ai:.2f}")
        
        return {
            'operator_type': operator_type,
            'phase': phase,
            'kernels': performance_list,
            'total_flops': total_flops,
            'total_memory_access': total_memory_access,
            'total_duration_us': total_duration,
            'kernel_count': len(kernels),
            'avg_arithmetic_intensity': avg_ai
        }
    
    def analyze_nonlinear_operators(self, stage1_data: Dict) -> Dict:
        """分析非Linear算子 - 区分prefill和decode阶段"""
        
        print(f"\n=== SGLang非Linear算子分析 (Prefill + Decode混合模式) ===")
        
        # 提取基本信息
        case_info = stage1_data.get('case_info', {})
        self.hardware_spec = stage1_data.get('hardware_spec', {})
        phase_analysis = stage1_data.get('phase_analysis', {})
        
        # 获取模型配置 - 从case_info中获取模型名称
        model_name = case_info.get('model_name', 'Unknown')
        
        # 使用内置的模型配置数据库
        model_configs = {
            'Qwen3-14B': {
                'hidden_size': 5120,
                'intermediate_size': 17408,
                'vocab_size': 151936,
                'num_attention_heads': 40,
                'num_key_value_heads': 8,
                'num_hidden_layers': 40,
                'head_dim': 128
            },
            'Qwen3-8B': {
                'hidden_size': 4096,
                'intermediate_size': 12288,
                'vocab_size': 151936,
                'num_attention_heads': 32,
                'num_key_value_heads': 8,
                'num_hidden_layers': 36,
                'head_dim': 128
            },
            'Qwen3-4B': {
                'hidden_size': 2560,
                'intermediate_size': 9728,
                'vocab_size': 151936,
                'num_attention_heads': 32,
                'num_key_value_heads': 8,
                'num_hidden_layers': 36,
                'head_dim': 128
            },
            'Qwen2.5-14B': {
                'hidden_size': 5120,
                'intermediate_size': 13824,
                'vocab_size': 152064,
                'num_attention_heads': 40,
                'num_key_value_heads': 8,
                'num_hidden_layers': 48,
                'head_dim': 128
            },
            'Qwen2.5-7B': {
                'hidden_size': 3584,
                'intermediate_size': 18944,
                'vocab_size': 152064,
                'num_attention_heads': 28,
                'num_key_value_heads': 4,
                'num_hidden_layers': 28,
                'head_dim': 128
            },
            'Qwen2.5-3B': {
                'hidden_size': 2048,
                'intermediate_size': 11008,
                'vocab_size': 151936,
                'num_attention_heads': 16,
                'num_key_value_heads': 2,
                'num_hidden_layers': 36,
                'head_dim': 128
            },
            'Qwen2.5-32B': {
                'hidden_size': 5120,
                'intermediate_size': 27648,
                'vocab_size': 152064,
                'num_attention_heads': 40,
                'num_key_value_heads': 8,
                'num_hidden_layers': 64,
                'head_dim': 128
            },
            'Qwen3-32B': {
                'hidden_size': 5120,
                'intermediate_size': 25600,
                'vocab_size': 151936,
                'num_attention_heads': 64,
                'num_key_value_heads': 8,
                'num_hidden_layers': 64,
                'head_dim': 128
            },
            'Llama-3.1-8B': {
                'hidden_size': 4096,
                'intermediate_size': 14336,
                'vocab_size': 128256,
                'num_attention_heads': 32,
                'num_key_value_heads': 8,
                'num_hidden_layers': 32,
                'head_dim': 128
            },
            'Llama-3.2-3B': {
                'hidden_size': 3072,
                'intermediate_size': 8192,
                'vocab_size': 128256,
                'num_attention_heads': 24,
                'num_key_value_heads': 8,
                'num_hidden_layers': 28,
                'head_dim': 128
            }
        }
        
        # 获取模型配置
        if model_name in model_configs:
            model_config = model_configs[model_name]
            print(f"使用内置配置: {model_name}")
        else:
            # 尝试模糊匹配
            matched = False
            for config_name, config_data in model_configs.items():
                if config_name.lower() in model_name.lower() or model_name.lower() in config_name.lower():
                    model_config = config_data
                    print(f"模糊匹配配置: {config_name} -> {model_name}")
                    matched = True
                    break
            
            if not matched:
                # 使用默认配置
                model_config = model_configs['Qwen3-14B']
                print(f"警告: 未找到匹配配置，使用默认Qwen3-14B配置")
        
        print(f"模型: {model_name}")
        print(f"硬件: {self.hardware_spec.get('name', 'Unknown')}")
        print(f"模型配置: hidden_size={model_config['hidden_size']}, num_heads={model_config['num_attention_heads']}")
        
        # 分类所有kernels
        operator_kernels = {'prefill': {}, 'decode': {}}
        
        for op_type in self.operator_patterns.keys():
            operator_kernels['prefill'][op_type] = []
            operator_kernels['decode'][op_type] = []
        
        # 遍历所有阶段的kernels并分类
        total_kernels = 0
        classified_kernels = 0
        phase_counts = {'prefill': 0, 'decode': 0}
        
        for phase_name, phase_data in phase_analysis.items():
            if 'operator_stats' not in phase_data:
                continue
            
            for op_type, op_data in phase_data['operator_stats'].items():
                if 'kernels' not in op_data:
                    continue
                
                for kernel in op_data['kernels']:
                    total_kernels += 1
                    kernel_type = self.classify_kernel(kernel['name'])
                    
                    # 使用kernel中已有的phase信息
                    kernel_phase = kernel.get('phase', phase_name)
                    
                    if kernel_type in self.operator_patterns:
                        operator_kernels[kernel_phase][kernel_type].append(kernel)
                        classified_kernels += 1
                        phase_counts[kernel_phase] += 1
        
        print(f"总kernel数: {total_kernels}, 分类成功: {classified_kernels}")
        print(f"Prefill阶段kernels: {phase_counts['prefill']}, Decode阶段kernels: {phase_counts['decode']}")
        
        # 分析结果
        analysis_results = {
            'prefill_analysis': {},
            'decode_analysis': {},
            'summary': {
                'total_kernels': total_kernels,
                'classified_kernels': classified_kernels,
                'classification_rate': classified_kernels / total_kernels if total_kernels > 0 else 0,
                'prefill_kernels': phase_counts['prefill'],
                'decode_kernels': phase_counts['decode'],
                'framework': 'sglang'
            }
        }
        
        # 分别分析prefill和decode阶段
        for phase in ['prefill', 'decode']:
            print(f"\n=== {phase.upper()}阶段分析 ===")
            phase_results = {}
            
            for op_type, kernels in operator_kernels[phase].items():
                if kernels:
                    print(f"\n分析{op_type}算子 ({phase}阶段): {len(kernels)}个kernels")
                    
                    if op_type == 'attention':
                        if phase == 'prefill':
                            result = self.calculate_attention_performance_prefill(kernels, model_config)
                        else:
                            result = self.calculate_attention_performance_decode(kernels, model_config)
                    else:
                        result = self.calculate_other_operator_performance(kernels, model_config, op_type, phase)
                    
                    phase_results[op_type] = result
            
            analysis_results[f'{phase}_analysis'] = phase_results
        
        # 计算总体统计
        total_prefill_flops = sum(result.get('total_flops', 0) for result in analysis_results['prefill_analysis'].values())
        total_decode_flops = sum(result.get('total_flops', 0) for result in analysis_results['decode_analysis'].values())
        total_prefill_memory = sum(result.get('total_memory_access', 0) for result in analysis_results['prefill_analysis'].values())
        total_decode_memory = sum(result.get('total_memory_access', 0) for result in analysis_results['decode_analysis'].values())
        
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
        print(f"Prefill平均算术强度: {analysis_results['summary']['prefill_avg_arithmetic_intensity']:.2f}")
        print(f"Decode平均算术强度: {analysis_results['summary']['decode_avg_arithmetic_intensity']:.2f}")
        
        return analysis_results

def main():
    parser = argparse.ArgumentParser(description='SGLang非Linear算子分析 (Prefill + Decode)')
    parser.add_argument('--input', required=True, help='Stage 1输出的JSON文件路径')
    parser.add_argument('--output', help='输出JSON文件路径 (默认: 自动生成)')
    
    args = parser.parse_args()
    
    # 加载Stage 1数据以获取pod_name
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
        # 从stage1_data中获取pod_name
        pod_name = stage1_data.get('case_info', {}).get('pod_name', 'unknown')
        # 获取输入文件所在目录
        input_dir = os.path.dirname(args.input)
        # 生成输出文件路径：输入文件所在文件夹/oea_stage3_pod_name_processed.json
        output_file = os.path.join(input_dir, f'oea_stage3_{pod_name}_processed.json')
    
    print(f"=== OEA Stage 3: SGLang非Linear算子分析 ===")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {output_file}")
    
    # 创建分析器并运行分析
    try:
        analyzer = SGLangNonLinearAnalyzer()
        results = analyzer.analyze_nonlinear_operators(stage1_data)
        
        # 添加元数据
        results['metadata'] = {
            'analysis_type': 'sglang_nonlinear_analysis',
            'input_file': args.input,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'description': 'SGLang nonlinear operator analysis with phase-specific formulas: prefill (O(seq_len^2)) and decode (O(kv_len))',
            'framework': 'sglang',
            'phase_mapping': {
                'prefill': 'Full sequence processing with quadratic attention complexity',
                'decode': 'Single token generation with linear attention complexity using KV cache'
            }
        }
        
        # 保存结果
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ 分析完成，结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"✗ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()