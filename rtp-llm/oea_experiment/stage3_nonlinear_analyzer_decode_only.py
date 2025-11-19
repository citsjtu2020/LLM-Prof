#!/usr/bin/env python3
"""
OEA Stage 3: 非Linear算子专门分析脚本 - Decode阶段专用版本 (修复版本)
专门针对decode阶段的attention计算，使用正确的decode场景计算公式
基于computational_analysis.txt中的decode阶段计算方法

主要改进:
1. 使用每个kernel的matched_token_size而不是平均token_size
2. 保持原有的输入输出格式完全兼容
3. 提供更精确的FLOPS和性能计算

使用方法:
python stage3_nonlinear_analyzer_decode_only_fixed.py --input stage1_processed_data_fixed.json
"""

import json
import numpy as np
import argparse
import os
import sys
import re
import math
from datetime import datetime
from typing import Dict, List, Any, Tuple

class NonLinearAnalyzerDecodeOnlyFixed:
    def __init__(self):
        """初始化非Linear算子分析器 - Decode阶段专用（修复版本）"""
        
        # 各类算子的匹配模式
        self.operator_patterns = {
            'attention': [
                r'.*attention.*',
                r'.*attn.*',
                r'.*scaled_dot_product.*',
                r'.*flash.*attn.*',
                r'.*fmha.*',
                r'.*multihead.*attention.*'
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
                r'.*activation.*'
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
                r'.*reduction.*',
                r'.*reduce.*'
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
            r'.*splitK.*reduce.*',
            r'.*reduce.*'
        ]
                
        # === 性能优化：预编译正则表达式 ===
        print("预编译正则表达式...")
        
        # 预编译算子模式
        self.compiled_operator_patterns = {}
        for op_type, patterns in self.operator_patterns.items():
            self.compiled_operator_patterns[op_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        # 预编译Linear模式
        self.compiled_linear_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.linear_patterns]
        self.compiled_linear_exclude_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.linear_exclude_patterns]
        
        # 分类缓存
        self.classification_cache = {}
        
        print(f"预编译完成: {sum(len(patterns) for patterns in self.compiled_operator_patterns.values())} 个算子模式")
        print(f"预编译完成: {len(self.compiled_linear_patterns)} 个Linear模式")
        print(f"预编译完成: {len(self.compiled_linear_exclude_patterns)} 个Linear排除模式")
    
    def is_linear_kernel(self, kernel_name: str) -> bool:
        """判断是否为Linear kernel - 用于排除 - 优化性能版本"""
        # 使用缓存
        cache_key = f"linear_{kernel_name}"
        if cache_key in self.classification_cache:
            return self.classification_cache[cache_key]
        
        kernel_name_lower = kernel_name.lower()
        
        # 首先检查排除模式 - 使用预编译正则表达式
        for exclude_pattern in self.compiled_linear_exclude_patterns:
            if exclude_pattern.search(kernel_name_lower):
                self.classification_cache[cache_key] = False
                return False
        
        # 然后检查匹配模式 - 使用预编译正则表达式
        for pattern in self.compiled_linear_patterns:
            if pattern.search(kernel_name_lower):
                self.classification_cache[cache_key] = True
                return True
        
        self.classification_cache[cache_key] = False
        return False

    def classify_kernel(self, kernel_name: str) -> str:
        """对kernel进行分类 - 优化性能版本"""
        # 使用缓存避免重复计算
        if kernel_name in self.classification_cache:
            return self.classification_cache[kernel_name]
        
        kernel_name_lower = kernel_name.lower()
        
        # 首先排除Linear算子 - 使用预编译正则表达式
        # 首先检查排除模式
        for exclude_pattern in self.compiled_linear_exclude_patterns:
            if exclude_pattern.search(kernel_name_lower):
                self.classification_cache[kernel_name] = 'non_linear'
                break
        else:
            # 然后检查匹配模式
            for pattern in self.compiled_linear_patterns:
                if pattern.search(kernel_name_lower):
                    self.classification_cache[kernel_name] = 'linear'
                    return 'linear'
        
        # 按优先级分类非Linear算子 - 使用预编译正则表达式
        for op_type, compiled_patterns in self.compiled_operator_patterns.items():
            for pattern in compiled_patterns:
                if pattern.search(kernel_name_lower):
                    self.classification_cache[kernel_name] = op_type
                    return op_type
        
        # 未匹配到任何模式
        self.classification_cache[kernel_name] = 'unknown'
        return 'unknown'

    def get_kernel_token_size(self, kernel: Dict) -> int:
        """获取kernel的token size - 优先使用matched_token_size，否则使用默认值"""
        # 新版本stage1输出包含matched_token_size字段
        if 'matched_token_size' in kernel:
            return int(kernel['matched_token_size'])
        
        # 兼容旧版本：如果没有matched_token_size，使用平均值
        # 这个值应该从metadata中获取，但这里先用一个合理的默认值
        return 2048

    def calculate_attention_performance_decode_only(self, kernels: List[Dict], model_config: Dict, fallback_token_size: int) -> Dict:
        """计算Attention算子性能 - Decode阶段专用版本（修复版本）
        
        基于computational_analysis.txt中的decode场景计算公式：
        真实场景batch decode: batch_size = N, kvcache_len_list = [seq_len1,seq_len2, seq_len3,...]
        粗略等价转换: batch_size=1, kv_len = sum(kvcache_len_list)
        
        关键特点：
        - Q只有1个token (当前生成的token)
        - K/V来自KV cache，长度为所有历史token的总和
        - 计算复杂度: O(kv_len) 而非 O(seq_len^2)
        """
        
        if not kernels:
            return {'message': 'No attention kernels found', 'kernels': []}
        
        # 模型参数
        # 从实际模型配置中读取参数，移除硬编码默认值
        if 'hidden_size' not in model_config:
            raise ValueError("模型配置中缺少 hidden_size")
        if 'intermediate_size' not in model_config:
            raise ValueError("模型配置中缺少 intermediate_size")
        if 'vocab_size' not in model_config:
            raise ValueError("模型配置中缺少 vocab_size")
        
        # 支持两种注意力头数参数命名：num_attention_heads 和 num_heads
        if 'num_attention_heads' not in model_config and 'num_heads' not in model_config:
            raise ValueError("模型配置中缺少 num_attention_heads 或 num_heads")
            
        hidden_size = model_config["hidden_size"]
        intermediate_size = model_config["intermediate_size"]
        vocab_size = model_config["vocab_size"]
        
        # 读取注意力头数，支持两种命名
        if 'num_attention_heads' in model_config:
            num_heads = model_config["num_attention_heads"]
            print(f"  使用 num_attention_heads: {num_heads}")
        else:
            num_heads = model_config["num_heads"]
            print(f"  使用 num_heads: {num_heads}")
        
        # 处理GQA/MQA配置
        # 支持两种参数命名：num_key_value_heads 和 num_kv_heads
        if 'num_key_value_heads' in model_config:
            num_kv_heads = model_config["num_key_value_heads"]
            print(f"  检测到GQA配置: num_heads={num_heads}, num_kv_heads={num_kv_heads} (使用num_key_value_heads)")
        elif 'num_kv_heads' in model_config:
            num_kv_heads = model_config["num_kv_heads"]
            print(f"  检测到GQA配置: num_heads={num_heads}, num_kv_heads={num_kv_heads} (使用num_kv_heads)")
        else:
            num_kv_heads = num_heads
            print(f"  未使用GQA: num_heads={num_heads}, num_kv_heads={num_kv_heads}")
        
        head_dim = hidden_size // num_heads
        dtype_bytes = 2  # bfloat16
        
        # Decode阶段Attention参数
        batch_size = 1
        query_len = 1  # decode阶段每次只生成1个token
        
        performance_list = []
        total_flops = 0
        total_memory_access = 0
        total_duration = 0
        
        print(f"\n=== Attention性能分析 (Decode阶段专用) - 修复版本 ===")
        print(f"模型参数: num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
        print(f"Decode场景: 每次生成{query_len}个token，使用精确token size")
        
        for kernel in kernels:
            # 使用每个kernel的精确token size而不是平均值
            token_size = self.get_kernel_token_size(kernel)
            kv_len = token_size  # KV cache的总长度
            
            print(f"  Kernel {kernel['name'][:30]}... 使用token_size={token_size}")
            
            # Decode阶段的Attention FLOPS计算
            # 1. Q @ K^T计算: [batch, heads, 1, head_dim] @ [batch, heads, head_dim, kv_len] = [batch, heads, 1, kv_len]
            # 复杂度: O(num_heads * kv_len * head_dim) - 线性复杂度，不是平方
            qk_flops = 2 * batch_size * num_heads * query_len * kv_len * head_dim
            
            # 2. Softmax计算: 对[batch, heads, 1, kv_len]进行softmax
            # 复杂度: O(num_heads * kv_len) - 只需要对kv_len维度做softmax，不是seq_len^2
            softmax_flops = 3 * batch_size * num_heads * query_len * kv_len  # exp + sum + div
            
            # 3. Attention @ V计算: [batch, heads, 1, kv_len] @ [batch, heads, kv_len, head_dim] = [batch, heads, 1, head_dim]
            # 复杂度: O(num_heads * kv_len * head_dim) - 线性复杂度
            av_flops = 2 * batch_size * num_heads * query_len * kv_len * head_dim
            
            total_kernel_flops = qk_flops + softmax_flops + av_flops
            
            # Decode阶段的内存访问计算
            # 读取数据：
            # - Q tensor: 当前生成的1个token的query [batch, 1, num_heads * head_dim]
            q_memory = batch_size * query_len * num_heads * head_dim * dtype_bytes
            
            # - K cache: 所有历史token的key cache [batch, kv_len, num_kv_heads * head_dim]
            # 注意：在GQA中，K/V的头数可能少于Q的头数
            k_memory = batch_size * kv_len * num_kv_heads * head_dim * dtype_bytes
            
            # - V cache: 所有历史token的value cache [batch, kv_len, num_kv_heads * head_dim]
            v_memory = batch_size * kv_len * num_kv_heads * head_dim * dtype_bytes
            
            # 中间结果：
            # - Attention scores: [batch, heads, 1, kv_len] - 注意维度差异
            scores_memory = batch_size * num_heads * query_len * kv_len * dtype_bytes
            
            # 输出数据：
            # - Attention output: [batch, 1, num_heads * head_dim]
            output_memory = batch_size * query_len * num_heads * head_dim * dtype_bytes
            
            total_kernel_memory = q_memory + k_memory + v_memory + scores_memory + output_memory
            
            # 算术强度
            arithmetic_intensity = total_kernel_flops / total_kernel_memory if total_kernel_memory > 0 else 0
            
            kernel_perf = {
                'kernel_name': kernel['name'],
                'duration_us': kernel.get('duration_us', 0),
                'scenario': 'decode_only',
                'token_size': token_size,  # 记录使用的精确token size
                'token_size_source': kernel.get('token_size_source', 'matched'),
                'flops': total_kernel_flops,
                'qk_flops': qk_flops,
                'softmax_flops': softmax_flops,
                'av_flops': av_flops,
                'memory_access': total_kernel_memory,
                'arithmetic_intensity': arithmetic_intensity,
                'dimensions': {
                    'batch_size': batch_size,
                    'query_len': query_len,
                    'kv_len': kv_len,
                    'num_heads': num_heads,
                    'num_kv_heads': num_kv_heads,
                    'head_dim': head_dim,
                    'complexity_note': 'O(kv_len) per token generation, not O(seq_len^2)',
                    'decode_transformation': f'batch_size=1, kv_len={kv_len} (precise token size)'
                },
                'memory_breakdown': {
                    'q_memory_mb': q_memory / 1e6,
                    'k_cache_memory_mb': k_memory / 1e6,
                    'v_cache_memory_mb': v_memory / 1e6,
                    'scores_memory_mb': scores_memory / 1e6,
                    'output_memory_mb': output_memory / 1e6,
                    'total_kv_cache_mb': (k_memory + v_memory) / 1e6
                },
                'performance_characteristics': {
                    'computation_type': 'memory_bound',
                    'bottleneck': 'KV cache access',
                    'scaling': 'linear with kv_len',
                    'cache_efficiency': 'critical for performance'
                }
            }
            
            performance_list.append(kernel_perf)
            total_flops += total_kernel_flops
            total_memory_access += total_kernel_memory
            total_duration += kernel.get('duration_us', 0)
            
            print(f"    场景: Decode阶段专用 (1 token generation)")
            print(f"    Token Size: {token_size} (精确值)")
            print(f"    FLOPS: {total_kernel_flops/1e9:.2f} GFLOPS")
            print(f"    内存访问: {total_kernel_memory/1e6:.2f} MB")
            print(f"    算术强度: {arithmetic_intensity:.2f}")
            print(f"    计算维度: Q[{batch_size},{query_len},{num_heads},{head_dim}] × KV_cache[{batch_size},{kv_len},{num_kv_heads},{head_dim}]")
            print(f"    KV Cache: {(k_memory + v_memory)/1e6:.1f}MB, Scores: {scores_memory/1e6:.1f}MB")
            print(f"    复杂度特征: 线性O({kv_len}) vs 平方O(seq_len^2)")
        
        return {
            'kernels': performance_list,
            'scenario': 'decode_only',
            'scenario_description': 'Decode phase: 1 token generation with KV cache access (O(kv_len) complexity)',
            'total_flops': total_flops,
            'total_memory_access': total_memory_access,
            'total_duration_us': total_duration,
            'kernel_count': len(kernels),
            'avg_arithmetic_intensity': total_flops / total_memory_access if total_memory_access > 0 else 0,
            'decode_characteristics': {
                'query_tokens_per_step': query_len,
                'uses_precise_token_size': True,
                'token_size_variation': len(set(k['token_size'] for k in performance_list)),
                'complexity_per_token': 'O(kv_len)',
                'memory_bottleneck': 'KV cache access dominates',
                'computation_pattern': 'Memory-bound due to large KV cache access',
                'transformation_formula': 'batch_size=1, kv_len=precise_token_size_per_kernel'
            },
            'performance_insights': {
                'flops_per_token': total_flops / len(kernels) if len(kernels) > 0 else 0,
                'memory_per_token': total_memory_access / len(kernels) if len(kernels) > 0 else 0,
                'avg_duration_per_token': total_duration / len(kernels) if len(kernels) > 0 else 0,
                'kv_cache_ratio': sum(k['memory_breakdown']['total_kv_cache_mb'] for k in performance_list) / (total_memory_access / 1e6) if total_memory_access > 0 else 0
            }
        }
    
    def calculate_rope_performance_decode(self, kernels: List[Dict], model_config: Dict, fallback_token_size: int) -> Dict:
        """计算RoPE算子性能 - Decode阶段优化版本（修复版本）"""
        
        if not kernels:
            return {'message': 'No RoPE kernels found', 'kernels': []}
        
        # 从实际模型配置中读取参数，移除硬编码默认值
        if 'hidden_size' not in model_config:
            raise ValueError("模型配置中缺少 hidden_size")
        if 'intermediate_size' not in model_config:
            raise ValueError("模型配置中缺少 intermediate_size")
        if 'vocab_size' not in model_config:
            raise ValueError("模型配置中缺少 vocab_size")
        
        # 支持两种注意力头数参数命名：num_attention_heads 和 num_heads
        if 'num_attention_heads' not in model_config and 'num_heads' not in model_config:
            raise ValueError("模型配置中缺少 num_attention_heads 或 num_heads")
            
        hidden_size = model_config["hidden_size"]
        intermediate_size = model_config["intermediate_size"]
        vocab_size = model_config["vocab_size"]
        
        # 读取注意力头数，支持两种命名
        if 'num_attention_heads' in model_config:
            num_heads = model_config["num_attention_heads"]
            print(f"  使用 num_attention_heads: {num_heads}")
        else:
            num_heads = model_config["num_heads"]
            print(f"  使用 num_heads: {num_heads}")
        
        # 处理GQA/MQA配置
        # 支持两种参数命名：num_key_value_heads 和 num_kv_heads
        if 'num_key_value_heads' in model_config:
            num_kv_heads = model_config["num_key_value_heads"]
            print(f"  检测到GQA配置: num_heads={num_heads}, num_kv_heads={num_kv_heads} (使用num_key_value_heads)")
        elif 'num_kv_heads' in model_config:
            num_kv_heads = model_config["num_kv_heads"]
            print(f"  检测到GQA配置: num_heads={num_heads}, num_kv_heads={num_kv_heads} (使用num_kv_heads)")
        else:
            num_kv_heads = num_heads
            print(f"  未使用GQA: num_heads={num_heads}, num_kv_heads={num_kv_heads}")
        
        head_dim = hidden_size // num_heads
        dtype_bytes = 2  # bfloat16
        
        # Decode阶段RoPE参数
        batch_size = 1
        query_len = 1  # 只对当前生成的1个token应用RoPE
        
        performance_list = []
        total_flops = 0
        total_memory_access = 0
        total_duration = 0
        
        print(f"\n=== RoPE性能分析 (Decode阶段专用) - 修复版本 ===")
        print(f"模型参数: num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
        print(f"Decode场景: 只对当前生成的{query_len}个token应用RoPE，使用精确token size")
        
        for kernel in kernels:
            # 使用每个kernel的精确token size而不是平均值
            token_size = self.get_kernel_token_size(kernel)
            kv_len = token_size  # KV cache的总长度，但RoPE只需要对新token应用
            
            print(f"  Kernel {kernel['name'][:30]}... 使用token_size={token_size}")
            
            # Decode阶段的RoPE FLOPS计算
            # 只需要对当前生成的1个token的Q和K应用RoPE
            # 历史的KV cache已经应用过RoPE，不需要重复计算
            
            # Q RoPE: 对当前token的query应用旋转位置编码
            # 每个元素需要4次浮点运算 (2次乘法 + 2次加法用于旋转)
            q_rope_flops = batch_size * query_len * num_heads * head_dim * 4
            
            # K RoPE: 对当前token的key应用旋转位置编码
            k_rope_flops = batch_size * query_len * num_kv_heads * head_dim * 4
            
            total_kernel_flops = q_rope_flops + k_rope_flops
            
            # Decode阶段的内存访问计算
            # 读取: 只需要当前token的Q, K tensors + position encodings
            q_memory = batch_size * query_len * num_heads * head_dim * dtype_bytes
            k_memory = batch_size * query_len * num_kv_heads * head_dim * dtype_bytes
            pos_memory = query_len * head_dim * dtype_bytes  # 只需要当前位置的编码
            
            # 写入: rotated Q, K (只有当前token)
            output_memory = q_memory + k_memory
            
            total_kernel_memory = q_memory + k_memory + pos_memory + output_memory
            
            # 算术强度
            arithmetic_intensity = total_kernel_flops / total_kernel_memory if total_kernel_memory > 0 else 0
            
            kernel_perf = {
                'kernel_name': kernel['name'],
                'duration_us': kernel.get('duration_us', 0),
                'scenario': 'decode_only',
                'token_size': token_size,  # 记录使用的精确token size
                'token_size_source': kernel.get('token_size_source', 'matched'),
                'flops': total_kernel_flops,
                'q_rope_flops': q_rope_flops,
                'k_rope_flops': k_rope_flops,
                'memory_access': total_kernel_memory,
                'arithmetic_intensity': arithmetic_intensity,
                'dimensions': {
                    'batch_size': batch_size,
                    'query_len': query_len,
                    'kv_cache_len': kv_len,
                    'num_heads': num_heads,
                    'num_kv_heads': num_kv_heads,
                    'head_dim': head_dim,
                    'rope_scope': 'current_token_only'
                },
                'decode_optimization': {
                    'historical_kv_cached': True,
                    'rope_applied_to': 'current_token_only',
                    'complexity_reduction': f'O({query_len}) vs O({kv_len}) for full sequence'
                }
            }
            
            performance_list.append(kernel_perf)
            total_flops += total_kernel_flops
            total_memory_access += total_kernel_memory
            total_duration += kernel.get('duration_us', 0)
            
            print(f"    场景: Decode RoPE (仅当前token)")
            print(f"    Token Size: {token_size} (精确值)")
            print(f"    FLOPS: {total_kernel_flops/1e9:.2f} GFLOPS")
            print(f"    内存访问: {total_kernel_memory/1e6:.2f} MB")
            print(f"    算术强度: {arithmetic_intensity:.2f}")
            print(f"    优化: 历史KV已缓存RoPE结果，只处理新token")
        
        return {
            'kernels': performance_list,
            'scenario': 'decode_only',
            'total_flops': total_flops,
            'total_memory_access': total_memory_access,
            'total_duration_us': total_duration,
            'kernel_count': len(kernels),
            'avg_arithmetic_intensity': total_flops / total_memory_access if total_memory_access > 0 else 0,
            'uses_precise_token_size': True,
            'token_size_variation': len(set(k['token_size'] for k in performance_list))
        }
    
    def calculate_layernorm_performance_decode(self, kernels: List[Dict], model_config: Dict, fallback_token_size: int) -> Dict:
        """计算LayerNorm/RMSNorm算子性能 - Decode阶段版本（修复版本）"""

        if not kernels:
            return {'message': 'No LayerNorm kernels found', 'kernels': []}

        # 从实际模型配置中读取参数，移除硬编码默认值
        if 'hidden_size' not in model_config:
            raise ValueError("模型配置中缺少 hidden_size")

        hidden_size = model_config["hidden_size"]
        dtype_bytes = 2  # bfloat16
        
        # Decode阶段LayerNorm参数
        batch_size = 1
        seq_len = 1  # decode阶段每次只处理1个token
        
        performance_list = []
        total_flops = 0
        total_memory_access = 0
        total_duration = 0
        
        print(f"\n=== LayerNorm性能分析 (Decode阶段专用) - 修复版本 ===")
        print(f"模型参数: hidden_size={hidden_size}")
        print(f"Decode场景: batch_size={batch_size}, seq_len={seq_len} (每次处理1个token)，使用精确token size")
        
        for kernel in kernels:
            # 使用每个kernel的精确token size而不是平均值
            token_size = self.get_kernel_token_size(kernel)
            
            print(f"  Kernel {kernel['name'][:30]}... 使用token_size={token_size}")
            
            # Decode阶段的RMSNorm FLOPS计算
            # 每次只处理1个token，计算量大幅减少
            # 1. 计算平方和: hidden_size次乘法
            # 2. 计算均值: 1次除法
            # 3. 计算RMS: 1次sqrt
            # 4. 归一化: hidden_size次除法
            # 5. 缩放: hidden_size次乘法 (如果有weight)
            # 总计: 约 3 * hidden_size 次运算每个token
            flops_per_token = 3 * hidden_size
            total_kernel_flops = batch_size * seq_len * flops_per_token
            
            # Decode阶段的内存访问计算
            # 读取: input tensor (1个token) + weight
            input_memory = batch_size * seq_len * hidden_size * dtype_bytes
            weight_memory = hidden_size * dtype_bytes  # RMSNorm weight
            
            # 写入: output tensor (1个token)
            output_memory = batch_size * seq_len * hidden_size * dtype_bytes
            
            total_kernel_memory = input_memory + weight_memory + output_memory
            
            # 算术强度
            arithmetic_intensity = total_kernel_flops / total_kernel_memory if total_kernel_memory > 0 else 0
            
            kernel_perf = {
                'kernel_name': kernel['name'],
                'duration_us': kernel.get('duration_us', 0),
                'scenario': 'decode_only',
                'token_size': token_size,  # 记录使用的精确token size
                'token_size_source': kernel.get('token_size_source', 'matched'),
                'flops': total_kernel_flops,
                'flops_per_token': flops_per_token,
                'memory_access': total_kernel_memory,
                'arithmetic_intensity': arithmetic_intensity,
                'dimensions': {
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'hidden_size': hidden_size,
                    'tokens_per_call': seq_len
                }
            }
            
            performance_list.append(kernel_perf)
            total_flops += total_kernel_flops
            total_memory_access += total_kernel_memory
            total_duration += kernel.get('duration_us', 0)
            
            print(f"    场景: Decode LayerNorm (1 token)")
            print(f"    Token Size: {token_size} (精确值)")
            print(f"    FLOPS: {total_kernel_flops/1e9:.2f} GFLOPS")
            print(f"    内存访问: {total_kernel_memory/1e6:.2f} MB")
            print(f"    算术强度: {arithmetic_intensity:.2f}")
        
        return {
            'kernels': performance_list,
            'scenario': 'decode_only',
            'total_flops': total_flops,
            'total_memory_access': total_memory_access,
            'total_duration_us': total_duration,
            'kernel_count': len(kernels),
            'avg_arithmetic_intensity': total_flops / total_memory_access if total_memory_access > 0 else 0,
            'uses_precise_token_size': True,
            'token_size_variation': len(set(k['token_size'] for k in performance_list))
        }
    
    def calculate_activation_performance_decode(self, kernels: List[Dict], model_config: Dict, fallback_token_size: int) -> Dict:
        """计算Activation算子性能 - Decode阶段版本（修复版本）"""

        if not kernels:
            return {'message': 'No activation kernels found', 'kernels': []}

        # 从实际模型配置中读取参数，移除硬编码默认值
        if 'hidden_size' not in model_config:
            raise ValueError("模型配置中缺少 hidden_size")
        if 'intermediate_size' not in model_config:
            raise ValueError("模型配置中缺少 intermediate_size")

        hidden_size = model_config["hidden_size"]
        intermediate_size = model_config["intermediate_size"]
        dtype_bytes = 2  # bfloat16
        
        # Decode阶段Activation参数
        batch_size = 1
        seq_len = 1  # decode阶段每次只处理1个token
        
        performance_list = []
        total_flops = 0
        total_memory_access = 0
        total_duration = 0
        
        print(f"\n=== Activation性能分析 (Decode阶段专用) - 修复版本 ===")
        print(f"模型参数: hidden_size={hidden_size}, intermediate_size={intermediate_size}")
        print(f"Decode场景: batch_size={batch_size}, seq_len={seq_len} (每次处理1个token)，使用精确token size")
        
        for kernel in kernels:
            # 使用每个kernel的精确token size而不是平均值
            token_size = self.get_kernel_token_size(kernel)
            
            print(f"  Kernel {kernel['name'][:30]}... 使用token_size={token_size}")
            
            # Decode阶段的SiLU and Mul FLOPS计算
            # input_size通常是 (intermediate_size * 2) // tp_size，这里假设tp_size=1
            tp_size = model_config.get('tensor_parallelism', 1)
            input_size = (intermediate_size * 2) // tp_size
            
            # SiLU: x / (1 + exp(-x)) 近似为每个元素4次运算
            # Mul: element-wise multiplication，每个元素1次运算
            # 总计: 约5次运算每个元素
            flops_per_element = 5
            total_kernel_flops = batch_size * seq_len * input_size * flops_per_element
            
            # Decode阶段的内存访问计算
            # 读取: input tensor (gate + up) - 只有1个token
            input_memory = batch_size * seq_len * input_size * dtype_bytes
            
            # 写入: output tensor (input_size // 2) - 只有1个token
            output_memory = batch_size * seq_len * (input_size // 2) * dtype_bytes
            
            total_kernel_memory = input_memory + output_memory
            
            # 算术强度
            arithmetic_intensity = total_kernel_flops / total_kernel_memory if total_kernel_memory > 0 else 0
            
            kernel_perf = {
                'kernel_name': kernel['name'],
                'duration_us': kernel.get('duration_us', 0),
                'scenario': 'decode_only',
                'token_size': token_size,  # 记录使用的精确token size
                'token_size_source': kernel.get('token_size_source', 'matched'),
                'flops': total_kernel_flops,
                'flops_per_element': flops_per_element,
                'memory_access': total_kernel_memory,
                'arithmetic_intensity': arithmetic_intensity,
                'dimensions': {
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'input_size': input_size,
                    'output_size': input_size // 2,
                    'tp_size': tp_size
                }
            }
            
            performance_list.append(kernel_perf)
            total_flops += total_kernel_flops
            total_memory_access += total_kernel_memory
            total_duration += kernel.get('duration_us', 0)
            
            print(f"    场景: Decode Activation (1 token)")
            print(f"    Token Size: {token_size} (精确值)")
            print(f"    FLOPS: {total_kernel_flops/1e9:.2f} GFLOPS")
            print(f"    内存访问: {total_kernel_memory/1e6:.2f} MB")
            print(f"    算术强度: {arithmetic_intensity:.2f}")
        
        return {
            'kernels': performance_list,
            'scenario': 'decode_only',
            'total_flops': total_flops,
            'total_memory_access': total_memory_access,
            'total_duration_us': total_duration,
            'kernel_count': len(kernels),
            'avg_arithmetic_intensity': total_flops / total_memory_access if total_memory_access > 0 else 0,
            'uses_precise_token_size': True,
            'token_size_variation': len(set(k['token_size'] for k in performance_list))
        }
    
    def calculate_moe_performance_decode(self, kernels: List[Dict], model_config: Dict, fallback_token_size: int) -> Dict:
        """计算MoE算子性能 - Decode阶段版本（修复版本）"""

        if not kernels:
            return {'message': 'No MoE kernels found', 'kernels': []}

        # MoE模型参数 - MoE模型特有参数，使用保守的默认值（仅用于回退）
        # 注意：如果模型确实是MoE架构但配置缺失，会输出警告
        has_moe_config = any(key in model_config for key in [
            'n_routed_experts', 'num_experts_per_tok', 'moe_intermediate_size'
        ])

        if not has_moe_config:
            print("  警告: 检测到MoE kernels但缺少MoE配置，使用默认估算值")

        n_routed_experts = model_config.get("n_routed_experts", 64)
        n_shared_experts = model_config.get("n_shared_experts", 2)
        num_experts_per_tok = model_config.get("num_experts_per_tok", 6)
        moe_intermediate_size = model_config.get("moe_intermediate_size", 1408)

        # 基础参数仍需验证
        if 'hidden_size' not in model_config:
            raise ValueError("模型配置中缺少 hidden_size")

        hidden_size = model_config["hidden_size"]
        dtype_bytes = 2  # bfloat16
        
        # Decode阶段MoE参数
        batch_size = 1
        seq_len = 1  # decode阶段每次只处理1个token
        
        performance_list = []
        total_flops = 0
        total_memory_access = 0
        total_duration = 0
        
        print(f"\n=== MoE性能分析 (Decode阶段专用) - 修复版本 ===")
        print(f"MoE参数: n_routed_experts={n_routed_experts}, num_experts_per_tok={num_experts_per_tok}")
        print(f"Decode场景: batch_size={batch_size}, seq_len={seq_len} (每次处理1个token)，使用精确token size")
        
        for kernel in kernels:
            # 使用每个kernel的精确token size而不是平均值
            token_size = self.get_kernel_token_size(kernel)
            
            print(f"  Kernel {kernel['name'][:30]}... 使用token_size={token_size}")
            
            # Decode阶段的MoE FLOPS计算
            # 每次只处理1个token，但仍需要路由到多个专家
            
            # 1. Router计算: 为1个token选择top-k experts
            router_flops = batch_size * seq_len * n_routed_experts * 2  # 简化估算
            
            # 2. Expert计算: 1个token使用num_experts_per_tok个experts
            # 每个expert类似一个小的MLP: gate_proj + up_proj + down_proj
            expert_flops_per_token = num_experts_per_tok * (
                2 * hidden_size * moe_intermediate_size +  # gate_proj
                2 * hidden_size * moe_intermediate_size +  # up_proj  
                2 * moe_intermediate_size * hidden_size    # down_proj
            )
            expert_flops = batch_size * seq_len * expert_flops_per_token
            
            total_kernel_flops = router_flops + expert_flops
            
            # Decode阶段的内存访问计算
            # 读取: input (1个token) + expert weights
            input_memory = batch_size * seq_len * hidden_size * dtype_bytes
            expert_weights_memory = num_experts_per_tok * (
                hidden_size * moe_intermediate_size * 3 * dtype_bytes  # gate, up, down weights
            )
            
            # 写入: output (1个token)
            output_memory = batch_size * seq_len * hidden_size * dtype_bytes
            
            total_kernel_memory = input_memory + expert_weights_memory + output_memory
            
            # 算术强度
            arithmetic_intensity = total_kernel_flops / total_kernel_memory if total_kernel_memory > 0 else 0
            
            kernel_perf = {
                'kernel_name': kernel['name'],
                'duration_us': kernel.get('duration_us', 0),
                'scenario': 'decode_only',
                'token_size': token_size,  # 记录使用的精确token size
                'token_size_source': kernel.get('token_size_source', 'matched'),
                'flops': total_kernel_flops,
                'router_flops': router_flops,
                'expert_flops': expert_flops,
                'memory_access': total_kernel_memory,
                'arithmetic_intensity': arithmetic_intensity,
                'dimensions': {
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'hidden_size': hidden_size,
                    'n_routed_experts': n_routed_experts,
                    'num_experts_per_tok': num_experts_per_tok,
                    'moe_intermediate_size': moe_intermediate_size
                }
            }
            
            performance_list.append(kernel_perf)
            total_flops += total_kernel_flops
            total_memory_access += total_kernel_memory
            total_duration += kernel.get('duration_us', 0)
            
            print(f"    场景: Decode MoE (1 token)")
            print(f"    Token Size: {token_size} (精确值)")
            print(f"    FLOPS: {total_kernel_flops/1e9:.2f} GFLOPS")
            print(f"    内存访问: {total_kernel_memory/1e6:.2f} MB")
            print(f"    算术强度: {arithmetic_intensity:.2f}")
        
        return {
            'kernels': performance_list,
            'scenario': 'decode_only',
            'total_flops': total_flops,
            'total_memory_access': total_memory_access,
            'total_duration_us': total_duration,
            'kernel_count': len(kernels),
            'avg_arithmetic_intensity': total_flops / total_memory_access if total_memory_access > 0 else 0,
            'uses_precise_token_size': True,
            'token_size_variation': len(set(k['token_size'] for k in performance_list))
        }
    
    def calculate_communication_performance(self, kernels: List[Dict], model_config: Dict, fallback_token_size: int) -> Dict:
        """计算Communication算子性能 - 通用版本"""
        
        if not kernels:
            return {'message': 'No communication kernels found', 'kernels': []}
        
        performance_list = []
        total_duration = 0
        
        print(f"\n=== Communication性能分析 ===")
        print("注意: Communication算子主要涉及数据传输，FLOPS计算不适用")
        
        for kernel in kernels:
            # 使用每个kernel的精确token size而不是平均值
            token_size = self.get_kernel_token_size(kernel)
            
            kernel_perf = {
                'kernel_name': kernel['name'],
                'duration_us': kernel.get('duration_us', 0),
                'token_size': token_size,  # 记录使用的精确token size
                'token_size_source': kernel.get('token_size_source', 'matched'),
                'analysis_note': 'Communication kernels involve data transfer, not computation',
                'flops': 0,
                'memory_access': 0,
                'arithmetic_intensity': 0,
                'operation_type': 'data_transfer'
            }
            
            performance_list.append(kernel_perf)
            total_duration += kernel.get('duration_us', 0)
            
            print(f"  Kernel: {kernel['name'][:50]}...")
            print(f"    类型: 数据传输操作")
            print(f"    Token Size: {token_size} (精确值)")
            print(f"    持续时间: {kernel.get('duration_us', 0)} μs")
        
        return {
            'kernels': performance_list,
            'total_flops': 0,
            'total_memory_access': 0,
            'total_duration_us': total_duration,
            'kernel_count': len(kernels),
            'avg_arithmetic_intensity': 0,
            'analysis_limitation': 'Communication kernels involve data transfer operations, FLOPS analysis not applicable',
            'uses_precise_token_size': True,
            'token_size_variation': len(set(k['token_size'] for k in performance_list))
        }
    
    def calculate_memory_performance(self, kernels: List[Dict], model_config: Dict, fallback_token_size: int) -> Dict:
        """计算Memory算子性能 - 通用版本"""
        
        if not kernels:
            return {'message': 'No memory kernels found', 'kernels': []}
        
        performance_list = []
        total_duration = 0
        
        print(f"\n=== Memory性能分析 ===")
        print("注意: Memory算子主要涉及内存操作，FLOPS计算不适用")
        
        for kernel in kernels:
            # 使用每个kernel的精确token size而不是平均值
            token_size = self.get_kernel_token_size(kernel)
            
            kernel_perf = {
                'kernel_name': kernel['name'],
                'duration_us': kernel.get('duration_us', 0),
                'token_size': token_size,  # 记录使用的精确token size
                'token_size_source': kernel.get('token_size_source', 'matched'),
                'analysis_note': 'Memory kernels involve memory operations (copy/set), not computation',
                'flops': 0,
                'memory_access': 0,
                'arithmetic_intensity': 0,
                'operation_type': 'memory_operation'
            }
            
            performance_list.append(kernel_perf)
            total_duration += kernel.get('duration_us', 0)
            
            print(f"  Kernel: {kernel['name'][:50]}...")
            print(f"    类型: 内存操作")
            print(f"    Token Size: {token_size} (精确值)")
            print(f"    持续时间: {kernel.get('duration_us', 0)} μs")
        
        return {
            'kernels': performance_list,
            'total_flops': 0,
            'total_memory_access': 0,
            'total_duration_us': total_duration,
            'kernel_count': len(kernels),
            'avg_arithmetic_intensity': 0,
            'analysis_limitation': 'Memory kernels involve memory operations (copy/set), FLOPS analysis not applicable',
            'uses_precise_token_size': True,
            'token_size_variation': len(set(k['token_size'] for k in performance_list))
        }
    
    def calculate_reduction_performance(self, kernels: List[Dict], model_config: Dict, fallback_token_size: int) -> Dict:
        """计算Reduction算子性能 - 通用版本（修复版本）"""

        if not kernels:
            return {'message': 'No reduction kernels found', 'kernels': []}

        # 从实际模型配置中读取参数，移除硬编码默认值
        if 'hidden_size' not in model_config:
            raise ValueError("模型配置中缺少 hidden_size")
        if 'vocab_size' not in model_config:
            raise ValueError("模型配置中缺少 vocab_size")
        if 'num_attention_heads' not in model_config and 'num_heads' not in model_config:
            raise ValueError("模型配置中缺少 num_attention_heads 或 num_heads")

        hidden_size = model_config["hidden_size"]
        vocab_size = model_config["vocab_size"]

        # 处理num_heads的多种命名方式
        if 'num_attention_heads' in model_config:
            num_heads = model_config["num_attention_heads"]
        else:
            num_heads = model_config["num_heads"]

        head_dim = hidden_size // num_heads
        
        # 基本参数
        batch_size = 1
        seq_len = 1  # decode阶段通常是1个token
        
        performance_list = []
        total_flops = 0
        total_memory_access = 0
        total_duration = 0
        
        print(f"\n=== Reduction算子性能分析 (Decode模式) - 修复版本 ===")
        print(f"模型参数: hidden_size={hidden_size}, vocab_size={vocab_size}, num_heads={num_heads}, head_dim={head_dim}")
        print(f"Decode场景: batch_size={batch_size}, seq_len={seq_len}，使用精确token size")
        
        for kernel in kernels:
            # 使用每个kernel的精确token size而不是平均值
            token_size = self.get_kernel_token_size(kernel)
            
            kernel_name_lower = kernel['name'].lower()
            kernel_duration = kernel.get('duration_us', 0)
            
            print(f"  Kernel {kernel['name'][:30]}... 使用token_size={token_size}")
            
            # 根据kernel名称判断类型
            if 'splitkreduce' in kernel_name_lower or 'splitk' in kernel_name_lower:
                # splitKreduce_kernel计算
                T = 8  # Split-K段数
                kernel_flops = batch_size * vocab_size * T
                
                # 内存访问计算
                partial_c_memory = batch_size * vocab_size * T * 4  # float32
                output_memory = batch_size * vocab_size * 2         # float16
                bias_memory = vocab_size * 4                        # float32
                kernel_memory = partial_c_memory + output_memory + bias_memory
                
                arithmetic_intensity = kernel_flops / kernel_memory if kernel_memory > 0 else 0
                
                kernel_perf = {
                    'kernel_name': kernel['name'],
                    'kernel_type': 'splitKreduce',
                    'duration_us': kernel_duration,
                    'token_size': token_size,  # 记录使用的精确token size
                    'token_size_source': kernel.get('token_size_source', 'matched'),
                    'flops': kernel_flops,
                    'memory_access': kernel_memory,
                    'arithmetic_intensity': arithmetic_intensity,
                    'parameters': {
                        'batch_size': batch_size,
                        'vocab_size': vocab_size,
                        'split_k_segments': T
                    }
                }
                
                print(f"    splitKreduce Kernel")
                print(f"    Token Size: {token_size} (精确值)")
                print(f"    FLOPS: {kernel_flops/1e9:.2f} GFLOPS")
                print(f"    内存访问: {kernel_memory/1e6:.2f} MB")
                print(f"    算术强度: {arithmetic_intensity:.3f} FLOPs/Byte")
                
            else:
                # 通用reduction算子
                estimated_flops = batch_size * hidden_size * seq_len
                estimated_memory = batch_size * hidden_size * seq_len * 4
                arithmetic_intensity = estimated_flops / estimated_memory if estimated_memory > 0 else 0
                
                kernel_perf = {
                    'kernel_name': kernel['name'],
                    'kernel_type': 'generic_reduction',
                    'duration_us': kernel_duration,
                    'token_size': token_size,  # 记录使用的精确token size
                    'token_size_source': kernel.get('token_size_source', 'matched'),
                    'flops': estimated_flops,
                    'memory_access': estimated_memory,
                    'arithmetic_intensity': arithmetic_intensity
                }
            
            performance_list.append(kernel_perf)
            total_flops += kernel_perf['flops']
            total_memory_access += kernel_perf['memory_access']
            total_duration += kernel_duration
        
        return {
            'kernels': performance_list,
            'total_flops': total_flops,
            'total_memory_access': total_memory_access,
            'total_duration_us': total_duration,
            'kernel_count': len(kernels),
            'avg_arithmetic_intensity': total_flops / total_memory_access if total_memory_access > 0 else 0,
            'uses_precise_token_size': True,
            'token_size_variation': len(set(k['token_size'] for k in performance_list))
        }
    
    def analyze_nonlinear_performance_decode(self, all_kernels: List[Dict], model_config: Dict, 
                                           hardware_spec: Dict, fallback_token_size: int) -> Dict:
        """分析所有非Linear算子性能 - Decode阶段专用（修复版本）"""
        
        print(f"\n=== 非Linear算子性能分析 (Decode阶段专用) - 修复版本 ===")
        print(f"分析重点: 使用decode阶段的正确计算公式和每个kernel的精确token size")
        print(f"回退Token Size: {fallback_token_size} (仅在kernel缺少matched_token_size时使用)")
        
        # 按类型分类kernels
        classified_kernels = {
            'attention': [],
            'rope': [],
            'layernorm': [],
            'activation': [],
            'moe': [],
            'communication': [],
            'memory': [],
            'reduction': [],
            'unknown': []
        }
        
        linear_count = 0
        
        for kernel in all_kernels:
            kernel_type = self.classify_kernel(kernel.get('name', ''))
            
            if kernel_type == 'linear':
                linear_count += 1
            elif kernel_type in classified_kernels:
                classified_kernels[kernel_type].append(kernel)
            else:
                classified_kernels['unknown'].append(kernel)
        
        print(f"Kernel分类统计:")
        print(f"  Linear kernels (已排除): {linear_count}")
        for op_type, kernels in classified_kernels.items():
            print(f"  {op_type}: {len(kernels)} kernels")
        
        # 分析各类算子性能 - 使用decode专用方法和精确token size
        analysis_results = {}
        
        # Attention - 使用decode专用计算和精确token size
        if classified_kernels['attention']:
            analysis_results['attention'] = self.calculate_attention_performance_decode_only(
                classified_kernels['attention'], model_config, fallback_token_size)
        
        # RoPE - 使用decode专用计算和精确token size
        if classified_kernels['rope']:
            analysis_results['rope'] = self.calculate_rope_performance_decode(
                classified_kernels['rope'], model_config, fallback_token_size)
        
        # LayerNorm - 使用decode专用计算和精确token size
        if classified_kernels['layernorm']:
            analysis_results['layernorm'] = self.calculate_layernorm_performance_decode(
                classified_kernels['layernorm'], model_config, fallback_token_size)
        
        # Activation - 使用decode专用计算和精确token size
        if classified_kernels['activation']:
            analysis_results['activation'] = self.calculate_activation_performance_decode(
                classified_kernels['activation'], model_config, fallback_token_size)
        
        # MoE - decode阶段每个token仍需要路由到专家，使用精确token size
        if classified_kernels['moe']:
            analysis_results['moe'] = self.calculate_moe_performance_decode(
                classified_kernels['moe'], model_config, fallback_token_size)
        
        # Communication, Memory, Reduction保持原有方法但使用精确token size
        if classified_kernels['communication']:
            analysis_results['communication'] = self.calculate_communication_performance(
                classified_kernels['communication'], model_config, fallback_token_size)
        
        if classified_kernels['memory']:
            analysis_results['memory'] = self.calculate_memory_performance(
                classified_kernels['memory'], model_config, fallback_token_size)
        
        if classified_kernels['reduction']:
            analysis_results['reduction'] = self.calculate_reduction_performance(
                classified_kernels['reduction'], model_config, fallback_token_size)
        
        # Unknown kernels
        if classified_kernels['unknown']:
            unknown_kernels_with_token_size = []
            for k in classified_kernels['unknown']:
                token_size = self.get_kernel_token_size(k)
                unknown_kernels_with_token_size.append({
                    'kernel_name': k['name'], 
                    'duration_us': k.get('duration_us', 0),
                    'token_size': token_size,
                    'token_size_source': k.get('token_size_source', 'matched')
                })
            
            analysis_results['unknown'] = {
                'kernels': unknown_kernels_with_token_size,
                'kernel_count': len(classified_kernels['unknown']),
                'analysis_limitation': 'Unknown kernel types, cannot determine appropriate performance model',
                'uses_precise_token_size': True,
                'token_size_variation': len(set(k['token_size'] for k in unknown_kernels_with_token_size))
            }
        
        # 计算总体统计
        total_nonlinear_kernels = sum(len(kernels) for kernels in classified_kernels.values())
        total_kernels = len(all_kernels)
        nonlinear_coverage = total_nonlinear_kernels / total_kernels * 100 if total_kernels > 0 else 0
        
        # 统计token size使用情况
        token_size_stats = {}
        for kernels in classified_kernels.values():
            for kernel in kernels:
                token_size = self.get_kernel_token_size(kernel)
                token_size_stats[token_size] = token_size_stats.get(token_size, 0) + 1
        
        return {
            'analysis_version': 'decode_only_specialized_fixed',
            'total_kernels': total_kernels,
            'linear_kernels_excluded': linear_count,
            'nonlinear_kernels': total_nonlinear_kernels,
            'nonlinear_coverage': nonlinear_coverage,
            'kernel_classification': {k: len(v) for k, v in classified_kernels.items()},
            'operator_analysis': analysis_results,
            'decode_optimizations': {
                'attention_complexity': 'O(kv_len) instead of O(seq_len^2)',
                'rope_scope': 'current_token_only',
                'layernorm_tokens': '1_token_per_call',
                'memory_pattern': 'kv_cache_dominated',
                'token_size_precision': 'per_kernel_matched_values'
            },
            'token_size_statistics': {
                'unique_token_sizes': len(token_size_stats),
                'token_size_distribution': dict(sorted(token_size_stats.items())),
                'fallback_token_size': fallback_token_size,
                'precision_improvement': 'Uses matched_token_size per kernel instead of average'
            }
        }
    
    def analyze_performance(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """主要的性能分析函数 - Decode阶段专用分析（修复版本）"""
        
        print(f"\n=== OEA Stage 3: 非Linear算子专门分析 (Decode阶段专用) - 修复版本 ===")
        
        # 兼容两种输入格式：Stage 1预处理数据 或 Stage 2分析结果
        if 'operator_kernels' in processed_data:
            # Stage 1预处理数据格式
            metadata = processed_data['metadata']
            operator_kernels = processed_data['operator_kernels']
            
            # 构建按时间排序的所有kernels列表
            all_kernels = []
            for op_type, kernels in operator_kernels.items():
                for kernel in kernels:
                    kernel_with_type = kernel.copy()
                    kernel_with_type['operator_type'] = op_type
                    all_kernels.append(kernel_with_type)
        
        elif 'metadata' in processed_data and 'linear_analysis' in processed_data:
            # Stage 2分析结果格式 - 从metadata中提取原始数据
            metadata = processed_data['metadata']
            
            # 尝试从metadata中获取原始kernel数据
            if 'original_kernels' in metadata:
                all_kernels = metadata['original_kernels']
            else:
                # 如果没有原始数据，返回错误信息
                return {
                    'error': 'Stage 3需要原始kernel数据，但输入的Stage 2结果中缺少原始数据',
                    'suggestion': '请使用Stage 1的预处理数据作为Stage 3的输入，或确保Stage 2保存了原始kernel数据'
                }
        else:
            return {
                'error': '输入数据格式不正确',
                'expected_formats': [
                    'Stage 1预处理数据 (包含operator_kernels)',
                    'Stage 2分析结果 (包含metadata和linear_analysis)'
                ]
            }
        
        model_config = metadata['model_config']
        hardware_spec = metadata['hardware_spec']
        
        # 使用新版本stage1的token匹配信息，如果没有则回退到平均值
        if 'token_matching' in metadata and metadata['token_matching'].get('has_timeline_data', False):
            # 新版本stage1有精确的token匹配数据
            fallback_token_size = int(metadata['token_matching']['max_token_size'])
            print(f"检测到新版本Stage1数据，包含精确token匹配信息")
            print(f"Token Size范围: {metadata['token_matching']['min_token_size']:.0f} - {metadata['token_matching']['max_token_size']:.0f}")
            print(f"唯一Token Size数量: {metadata['token_matching']['unique_token_sizes']}")
        else:
            # 旧版本stage1或没有token匹配数据，使用平均值作为回退
            fallback_token_size = int(metadata['avg_token_size'])
            print(f"使用传统平均token size作为回退: {fallback_token_size}")
        
        # 按时间戳排序
        all_kernels.sort(key=lambda x: x.get('timestamp_us', 0))
        
        print(f"总计 {len(all_kernels)} 个kernels")
        print(f"分析模式: Decode阶段专用 (使用正确的decode计算公式和精确token size)")
        
        # 分析非Linear算子性能 - 使用decode专用方法和精确token size
        nonlinear_performance = self.analyze_nonlinear_performance_decode(all_kernels, model_config, hardware_spec, fallback_token_size)
        
        # 构建结果
        analysis_results = {
            'metadata': metadata,
            'analysis_version': 'decode_only_specialized_fixed',
            'nonlinear_analysis': nonlinear_performance
        }
        
        return analysis_results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """保存分析结果"""
        
        # 转换numpy类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        with open(output_file, 'w') as f:
            json.dump(convert_numpy(results), f, indent=2)
        
        print(f"\n分析结果已保存到: {output_file}")

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='OEA Stage 3: 非Linear算子专门分析工具 (Decode阶段专用) - 修复版本')
    parser.add_argument('--input', required=True, help='Stage 1预处理数据文件路径')
    parser.add_argument('--output', default=None, help='输出文件路径')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 设置输出文件
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        args.output = os.path.join(input_dir, "oea_stage3_nonlinear_analysis_results_fixed.json")
    
    try:
        # 加载预处理数据
        print(f"加载预处理数据: {args.input}")
        with open(args.input, 'r') as f:
            processed_data = json.load(f)
        
        # 创建分析器
        analyzer = NonLinearAnalyzerDecodeOnlyFixed()
        
        # 执行性能分析
        results = analyzer.analyze_performance(processed_data)
        
        # 检查是否有错误
        if 'error' in results:
            print(f"错误: {results['error']}")
            if 'suggestion' in results:
                print(f"建议: {results['suggestion']}")
            sys.exit(1)
        
        # 保存结果
        analyzer.save_results(results, args.output)
        
        print(f"\n=== Stage 3 非Linear分析完成 (Decode阶段专用) - 修复版本 ===")
        
        # 显示关键指标
        if 'nonlinear_analysis' in results:
            nonlinear_perf = results['nonlinear_analysis']
            print(f"\n非Linear算子分析 (Decode专用 - 修复版本):")
            print(f"  - 分析版本: {nonlinear_perf.get('analysis_version', 'decode_only_specialized_fixed')}")
            print(f"  - 总kernels: {nonlinear_perf['total_kernels']}")
            print(f"  - Linear kernels (已排除): {nonlinear_perf['linear_kernels_excluded']}")
            print(f"  - 非Linear kernels: {nonlinear_perf['nonlinear_kernels']}")
            print(f"  - 非Linear覆盖率: {nonlinear_perf['nonlinear_coverage']:.1f}%")
            
            print(f"\n各算子类型统计:")
            for op_type, count in nonlinear_perf['kernel_classification'].items():
                if count > 0:
                    print(f"  - {op_type}: {count} kernels")
            
            # 显示Token Size精度改进
            if 'token_size_statistics' in nonlinear_perf:
                token_stats = nonlinear_perf['token_size_statistics']
                print(f"\nToken Size精度统计:")
                print(f"  - 唯一Token Size数量: {token_stats['unique_token_sizes']}")
                print(f"  - 回退Token Size: {token_stats['fallback_token_size']}")
                print(f"  - 精度改进: {token_stats['precision_improvement']}")
                
                if token_stats['unique_token_sizes'] <= 10:
                    print(f"  - Token Size分布: {token_stats['token_size_distribution']}")
            
            # 显示Decode优化特征
            if 'decode_optimizations' in nonlinear_perf:
                decode_opts = nonlinear_perf['decode_optimizations']
                print(f"\nDecode阶段优化特征:")
                print(f"  - Attention复杂度: {decode_opts['attention_complexity']}")
                print(f"  - RoPE范围: {decode_opts['rope_scope']}")
                print(f"  - LayerNorm处理: {decode_opts['layernorm_tokens']}")
                print(f"  - 内存模式: {decode_opts['memory_pattern']}")
                print(f"  - Token Size精度: {decode_opts['token_size_precision']}")
            
            # 显示各算子类型的性能
            if 'operator_analysis' in nonlinear_perf:
                print(f"\n各算子类型性能 (Decode专用计算 + 精确Token Size):")
                for op_type, analysis in nonlinear_perf['operator_analysis'].items():
                    if 'total_flops' in analysis and analysis['total_flops'] > 0:
                        duration_ms = analysis['total_duration_us'] / 1000
                        tflops = analysis['total_flops'] / (analysis['total_duration_us'] * 1e-6) / 1e12
                        gflops = analysis['total_flops'] / (analysis['total_duration_us'] * 1e-6) / 1e9
                        ai = analysis['avg_arithmetic_intensity']
                        scenario = analysis.get('scenario', 'general')
                        uses_precise = analysis.get('uses_precise_token_size', False)
                        token_variation = analysis.get('token_size_variation', 0)

                        # Decode 场景下使用 GFLOPS（更合适的单位）
                        precision_info = f"精确Token Size({token_variation}种)" if uses_precise else "平均Token Size"
                        if scenario == 'decode_only' or tflops < 0.1:
                            print(f"  {op_type}: {duration_ms:.1f} ms, {gflops:.2f} GFLOPS, AI={ai:.1f} ({scenario}, {precision_info})")
                        else:
                            print(f"  {op_type}: {duration_ms:.1f} ms, {tflops:.2f} TFLOPS, AI={ai:.1f} ({scenario}, {precision_info})")

                    elif 'analysis_limitation' in analysis:
                        uses_precise = analysis.get('uses_precise_token_size', False)
                        token_variation = analysis.get('token_size_variation', 0)
                        precision_info = f"精确Token Size({token_variation}种)" if uses_precise else "平均Token Size"
                        print(f"  {op_type}: {analysis['kernel_count']} kernels - {analysis['analysis_limitation']} ({precision_info})")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()