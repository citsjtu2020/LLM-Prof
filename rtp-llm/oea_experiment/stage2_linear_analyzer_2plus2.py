#!/usr/bin/env python3
"""
OEA Stage 2: Linear算子专门分析脚本 (2+2模式版本)
基于原有stage2脚本，支持2+2模式：qkv_proj, o_proj, gate_proj+up_proj(融合), down_proj
当gate_proj和up_proj融合到一个算子中时，时间平分，FLOPS分别计算

主要改进:
1. 支持2+2模式检测：Norm → 2个Linear → Norm → 2个Linear → AddBias
2. 处理gate_proj和up_proj融合的情况，时间平分
3. 保持原有的输入输出格式完全兼容
4. 使用每个kernel的精确token size值

使用方法:
python stage2_linear_analyzer_2plus2.py --input stage1_processed_data_fixed.json
"""

import json
import numpy as np
import argparse
import os
import sys
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple

class LinearAnalyzer2Plus2:
    def __init__(self):
        """初始化Linear算子分析器（2+2模式版本）"""
        
        # Linear算子匹配模式 - 排除splitKreduce
        self.linear_patterns = [
            r'nvjet.*',
            r'.*gemm.*',
            r'.*sgemm.*',
            r'.*hgemm.*',
            r'.*cutlass.*gemm.*',
            r'.*acext.*',
            r'.*cublas.*'
        ]
        
        # 排除的模式 - 这些不应该被识别为Linear
        self.linear_exclude_patterns = [
            r'.*splitkreduce.*',
            r'.*splitK.*reduce.*',
            r'.*reduce.*'
        ]
        
        # Normalization算子匹配模式 - 更精确地匹配，排除fused类型
        self.norm_patterns = [
            r'^.*generalRmsNorm.*$',  # 只匹配 generalRmsNorm
            r'^.*general_rms_norm.*$',  # 匹配 general_rms_norm 变体
            r'^.*layernorm.*$',  # 匹配 layernorm
            r'^.*layer_norm.*$'   # 匹配 layer_norm
        ]
        
        # 排除的norm模式 - 这些不应该被识别为标准的norm
        self.norm_exclude_patterns = [
            r'.*fused.*',  # 排除所有fused类型的norm
            r'.*fusedQk.*',  # 排除fusedQkRmsNorm等
            r'.*qk.*norm.*',  # 排除qk相关的norm
            r'.*attention.*norm.*'  # 排除attention相关的norm
        ]
        
        # add_bias算子匹配模式 - 扩展以包含具体的kernel名称
        self.add_bias_patterns = [
            # 通用模式
            r'.*add.*bias.*',
            r'.*bias.*add.*',
            r'.*addBias.*',
            r'.*bias_add.*',
            r'.*elementwise_add.*bias.*',
            r'.*bias.*elementwise.*',
            r'.*elementwise.*add.*',
            r'.*add_.*',
            r'.*_add_.*',
            r'.*residual.*add.*',
            r'.*add.*residual.*',
            # 具体的kernel名称模式
            r'.*addBiasResidual.*',  # 匹配 addBiasResidual 函数
            r'.*rtp_llm::addBiasResidual.*',  # 匹配 rtp_llm 命名空间
            r'.*fastertransformer::addBiasResidual.*',  # 匹配 fastertransformer 命名空间
            r'.*::addBiasResidual.*',  # 匹配任何命名空间的 addBiasResidual
            r'void.*addBiasResidual.*',  # 匹配函数签名形式
        ]
        
        # 预编译正则表达式以提高性能
        self.compiled_linear_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.linear_patterns]
        self.compiled_linear_exclude_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.linear_exclude_patterns]
        self.compiled_norm_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.norm_patterns]
        self.compiled_norm_exclude_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.norm_exclude_patterns]
        self.compiled_add_bias_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.add_bias_patterns]
        
        # 缓存分类结果
        self.classification_cache = {}
       
    def is_linear_kernel(self, kernel_name: str) -> bool:
        """判断是否为Linear kernel - 排除splitKreduce等，优化性能版本"""
        # 使用缓存避免重复计算
        if kernel_name in self.classification_cache:
            return self.classification_cache[kernel_name].get('is_linear', False)
        
        kernel_name_lower = kernel_name.lower()
        
        # 首先检查排除模式 - 使用预编译的正则表达式
        for exclude_pattern in self.compiled_linear_exclude_patterns:
            if exclude_pattern.search(kernel_name_lower):
                self.classification_cache[kernel_name] = {'is_linear': False}
                return False
        
        # 然后检查匹配模式 - 使用预编译的正则表达式
        is_linear = any(pattern.search(kernel_name_lower) for pattern in self.compiled_linear_patterns)
        self.classification_cache[kernel_name] = {'is_linear': is_linear}
        return is_linear
    
    def is_norm_kernel(self, kernel_name: str) -> bool:
        """判断是否为Normalization kernel"""
        if kernel_name in self.classification_cache:
            return self.classification_cache[kernel_name].get('is_norm', False)
        
        kernel_name_lower = kernel_name.lower()
        
        # 首先检查排除模式
        for exclude_pattern in self.compiled_norm_exclude_patterns:
            if exclude_pattern.search(kernel_name_lower):
                self.classification_cache[kernel_name] = {'is_norm': False}
                return False
        
        # 然后检查匹配模式
        is_norm = any(pattern.search(kernel_name_lower) for pattern in self.compiled_norm_patterns)
        self.classification_cache[kernel_name] = {'is_norm': is_norm}
        return is_norm
    
    def is_add_bias_kernel(self, kernel_name: str) -> bool:
        """判断是否为add_bias kernel"""
        if kernel_name in self.classification_cache:
            return self.classification_cache[kernel_name].get('is_add_bias', False)
        
        kernel_name_lower = kernel_name.lower()
        is_add_bias = any(pattern.search(kernel_name_lower) for pattern in self.compiled_add_bias_patterns)
        self.classification_cache[kernel_name] = {'is_add_bias': is_add_bias}
        return is_add_bias
    
    def get_kernel_token_size(self, kernel: Dict) -> int:
        """获取kernel的token size - 优先使用matched_token_size，否则使用默认值"""
        # 新版本stage1输出包含matched_token_size字段
        if 'matched_token_size' in kernel:
            return int(kernel['matched_token_size'])
        
        # 兼容旧版本：如果没有matched_token_size，使用平均值
        # 这个值应该从metadata中获取，但这里先用一个合理的默认值
        return 2048
    
    def find_transformer_layers_2plus2(self, all_kernels: List[Dict]) -> Tuple[List[Dict], List[int]]:
        """
        寻找符合Transformer层模式的kernel组合 (2+2模式)
        模式: RmsNorm → 2个Linear → RmsNorm → 2个Linear → add_bias
        """
        
        # 按类型分类所有kernels - 使用双重检测
        norm_indices = []
        linear_indices = []
        add_bias_indices = []
        
        for i, kernel in enumerate(all_kernels):
            kernel_name = kernel.get('name', '')
            operator_type = kernel.get('operator_type', '')
            
            # 双重检测：operator_type 和 kernel 名称匹配
            is_norm_by_name = self.is_norm_kernel(kernel_name)
            is_norm_by_type = (operator_type == 'layernorm')
            
            is_linear_by_name = self.is_linear_kernel(kernel_name)
            is_linear_by_type = (operator_type == 'linear')
            
            is_add_bias_by_name = self.is_add_bias_kernel(kernel_name)
            is_add_bias_by_type = (operator_type == 'add_bias')
            
            # 首先检查是否应该被排除
            should_exclude_as_norm = any(pattern.search(kernel_name.lower()) 
                                        for pattern in self.compiled_norm_exclude_patterns)

            if (is_norm_by_name or is_norm_by_type) and not should_exclude_as_norm:
                norm_indices.append(i)
         
            elif is_linear_by_name or is_linear_by_type:
                linear_indices.append(i)
            elif is_add_bias_by_name or is_add_bias_by_type:
                add_bias_indices.append(i)
        
        print(f"找到 {len(norm_indices)} 个Norm kernels")
        print(f"找到 {len(linear_indices)} 个Linear kernels (排除splitKreduce)")
        print(f"找到 {len(add_bias_indices)} 个add_bias kernels")
        
        if len(norm_indices) < 2 or len(add_bias_indices) < 1 or len(linear_indices) < 4:
            print("算子数量不足，无法进行精确分组 (2+2模式需要至少4个Linear)")
            return [], linear_indices
        
        # 寻找符合模式的层
        valid_layers = []
        used_linear_indices = set()
        search_start_idx = 0  # 从这个索引开始搜索下一个层
        
        print(f"\n=== 寻找Transformer层模式 (2+2模式) ===")
        
        while search_start_idx < len(all_kernels):
            # 找到搜索起始位置之后的第一个norm
            norm1_candidates = [idx for idx in norm_indices if idx >= search_start_idx]
            if len(norm1_candidates) < 2:
                break
            
            norm1_idx = norm1_candidates[0]
            
            # 找到第一个norm之后的第二个norm
            norm2_candidates = [idx for idx in norm_indices if idx > norm1_idx]
            if not norm2_candidates:
                break
            
            norm2_idx = norm2_candidates[0]
            
            # 找到第二个norm之后最近的add_bias
            add_bias_candidates = [idx for idx in add_bias_indices if idx > norm2_idx]
            if not add_bias_candidates:
                # 如果没有add_bias，跳到下一个norm继续搜索
                search_start_idx = norm2_idx + 1
                continue
            
            add_bias_idx = min(add_bias_candidates)
            
            # 找到两个区间的Linear kernels（只考虑未使用的）
            attention_linear = [idx for idx in linear_indices 
                              if norm1_idx < idx < norm2_idx and idx not in used_linear_indices]
            mlp_linear = [idx for idx in linear_indices 
                         if norm2_idx < idx < add_bias_idx and idx not in used_linear_indices]
            
            # 检查是否符合2+2模式
            if len(attention_linear) == 2 and len(mlp_linear) == 2:
                layer_info = {
                    'layer_id': len(valid_layers),
                    'norm1_idx': norm1_idx,
                    'norm2_idx': norm2_idx,
                    'add_bias_idx': add_bias_idx,
                    'attention_linear': attention_linear,
                    'mlp_linear': mlp_linear,
                    'pattern': f"Norm[{norm1_idx}] → Linear{attention_linear} → Norm[{norm2_idx}] → Linear{mlp_linear} → AddBias[{add_bias_idx}]",
                    'mode': '2plus2'
                }
                
                valid_layers.append(layer_info)
                used_linear_indices.update(attention_linear + mlp_linear)
                
                print(f"  层 {len(valid_layers)}: {layer_info['pattern']}")
                
                # 从add_bias之后开始搜索下一个层
                search_start_idx = add_bias_idx + 1
            else:
                print(f"  候选层不匹配: Attention Linear={len(attention_linear)}, MLP Linear={len(mlp_linear)} (期望2+2)")
                # 不符合模式，从第二个norm重新开始搜索（第二个norm作为新的第一个norm）
                search_start_idx = norm2_idx
                print(f"  不匹配，下次搜索从第二个norm索引 {search_start_idx} 重新开始")
        
        print(f"\n总共找到 {len(valid_layers)} 个有效的Transformer层 (2+2模式)")
        
        # 处理剩余未分配的Linear kernels
        remaining_linear = [idx for idx in linear_indices if idx not in used_linear_indices]
        print(f"剩余未分配的Linear kernels: {len(remaining_linear)} 个")
        
        # 记录剩余Linear kernels的详细信息（只显示前10个）
        if remaining_linear:
            print("剩余Linear kernels详情 (前10个):")
            for idx in remaining_linear[:10]:
                kernel = all_kernels[idx]
                print(f"  [{idx}] {kernel['name']}")
        
        return valid_layers, remaining_linear
    
    def calculate_projection_performance_2plus2(self, layer_info: Dict, all_kernels: List[Dict], 
                                               model_config: Dict) -> Dict:
        """计算单个Transformer层中各projection的性能 - 2+2模式，处理融合的gate+up"""
        
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
        else:
            num_heads = model_config["num_heads"]
        
        # 处理GQA/MQA配置
        # 支持两种参数命名：num_key_value_heads 和 num_kv_heads
        if 'num_key_value_heads' in model_config:
            num_kv_heads = model_config["num_key_value_heads"]
        elif 'num_kv_heads' in model_config:
            num_kv_heads = model_config["num_kv_heads"]
        else:
            num_kv_heads = num_heads
        
        head_dim = hidden_size // num_heads
        dtype_bytes = 2  # bfloat16
        
        # 获取Tensor Parallelism配置
        tp_size = getattr(self, 'tp_size', 1)
        
        # Projection配置 - 考虑TP的影响
        # 在TP模式下，某些维度会被分割
        proj_configs = {
            'qkv_proj': {
                'N': (num_heads + 2 * num_kv_heads) * head_dim // tp_size,  # TP分割注意力头
                'K': hidden_size
            },
            'o_proj': {
                'N': hidden_size, 
                'K': num_heads * head_dim // tp_size  # TP分割输入维度
            },
            'gate_proj': {
                'N': intermediate_size // tp_size,  # TP分割MLP维度
                'K': hidden_size
            },
            'up_proj': {
                'N': intermediate_size // tp_size,   # TP分割MLP维度
                'K': hidden_size
            },
            'down_proj': {
                'N': hidden_size,
                'K': intermediate_size // tp_size    # TP分割输入维度
            }
        }
        
        # 分配kernels到projections - 2+2模式
        attention_linear_indices = layer_info['attention_linear']  # qkv_proj, o_proj
        mlp_linear_indices = layer_info['mlp_linear']  # gate_proj+up_proj(融合), down_proj
        
        layer_performance = {}
        
        print(f"  实际配置验证: hidden_size={hidden_size}, intermediate_size={intermediate_size}")
        print(f"  注意力配置: num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
        print(f"  TP配置: tp_size={tp_size}")
        print(f"  2+2模式: Attention={len(attention_linear_indices)}, MLP={len(mlp_linear_indices)}")
        
        # 处理Attention部分：qkv_proj, o_proj
        attention_proj_names = ['qkv_proj', 'o_proj']
        for i, proj_name in enumerate(attention_proj_names):
            if i < len(attention_linear_indices):
                kernel_idx = attention_linear_indices[i]
                kernel = all_kernels[kernel_idx]
                
                # 使用每个kernel的精确token size
                token_size = self.get_kernel_token_size(kernel)
                M = token_size
                
                N = proj_configs[proj_name]['N']
                K = proj_configs[proj_name]['K']
                
                flops = 2 * M * N * K
                memory_access = (M * K + N * K + M * N) * dtype_bytes
                arithmetic_intensity = flops / memory_access if memory_access > 0 else 0
                
                layer_performance[proj_name] = {
                    'kernel_idx': kernel_idx,
                    'kernel_name': kernel['name'],
                    'duration_us': kernel.get('duration_us', 0),
                    'token_size': token_size,
                    'matrix_dims': [M, N, K],
                    'flops': flops,
                    'memory_access': memory_access,
                    'arithmetic_intensity': arithmetic_intensity,
                    'group_id': layer_info['layer_id'],
                    'tp_size': tp_size,
                    'projection_type': 'attention',
                    'config_validation': {
                        'hidden_size': hidden_size,
                        'intermediate_size': intermediate_size,
                        'vocab_size': vocab_size,
                        'num_heads': num_heads,
                        'num_kv_heads': num_kv_heads,
                        'head_dim': head_dim
                    }
                }
                
                print(f"    {proj_name}: token_size={token_size}, dims=[{M}, {N}, {K}], FLOPS={flops/1e9:.2f}G")
        
        # 处理MLP部分：gate_proj+up_proj(融合), down_proj
        if len(mlp_linear_indices) >= 1:
            # 第一个MLP kernel：gate_proj+up_proj融合
            fused_kernel_idx = mlp_linear_indices[0]
            fused_kernel = all_kernels[fused_kernel_idx]
            
            # 使用精确token size
            token_size = self.get_kernel_token_size(fused_kernel)
            M = token_size
            
            # 融合kernel的总时间平分给gate_proj和up_proj
            total_duration = fused_kernel.get('duration_us', 0)
            split_duration = total_duration / 2.0
            
            # gate_proj和up_proj有相同的矩阵维度
            gate_N = proj_configs['gate_proj']['N']
            gate_K = proj_configs['gate_proj']['K']
            up_N = proj_configs['up_proj']['N']
            up_K = proj_configs['up_proj']['K']
            
            # 分别计算gate_proj和up_proj的性能
            for proj_name, N, K in [('gate_proj', gate_N, gate_K), ('up_proj', up_N, up_K)]:
                flops = 2 * M * N * K
                memory_access = (M * K + N * K + M * N) * dtype_bytes
                arithmetic_intensity = flops / memory_access if memory_access > 0 else 0
                
                layer_performance[proj_name] = {
                    'kernel_idx': fused_kernel_idx,
                    'kernel_name': fused_kernel['name'],
                    'duration_us': split_duration,  # 时间平分
                    'original_duration_us': total_duration,  # 保留原始时间
                    'token_size': token_size,
                    'matrix_dims': [M, N, K],
                    'flops': flops,
                    'memory_access': memory_access,
                    'arithmetic_intensity': arithmetic_intensity,
                    'group_id': layer_info['layer_id'],
                    'tp_size': tp_size,
                    'projection_type': 'mlp_fused',
                    'fused_with': 'gate_proj+up_proj' if proj_name == 'gate_proj' else 'gate_proj+up_proj',
                    'config_validation': {
                        'hidden_size': hidden_size,
                        'intermediate_size': intermediate_size,
                        'vocab_size': vocab_size,
                        'num_heads': num_heads,
                        'num_kv_heads': num_kv_heads,
                        'head_dim': head_dim
                    }
                }
                
                print(f"    {proj_name}: token_size={token_size}, dims=[{M}, {N}, {K}], FLOPS={flops/1e9:.2f}G, duration={split_duration:.1f}us (融合平分)")
        
        # 处理down_proj
        if len(mlp_linear_indices) >= 2:
            down_kernel_idx = mlp_linear_indices[1]
            down_kernel = all_kernels[down_kernel_idx]
            
            token_size = self.get_kernel_token_size(down_kernel)
            M = token_size
            
            N = proj_configs['down_proj']['N']
            K = proj_configs['down_proj']['K']
            
            flops = 2 * M * N * K
            memory_access = (M * K + N * K + M * N) * dtype_bytes
            arithmetic_intensity = flops / memory_access if memory_access > 0 else 0
            
            layer_performance['down_proj'] = {
                'kernel_idx': down_kernel_idx,
                'kernel_name': down_kernel['name'],
                'duration_us': down_kernel.get('duration_us', 0),
                'token_size': token_size,
                'matrix_dims': [M, N, K],
                'flops': flops,
                'memory_access': memory_access,
                'arithmetic_intensity': arithmetic_intensity,
                'group_id': layer_info['layer_id'],
                'tp_size': tp_size,
                'projection_type': 'mlp',
                'config_validation': {
                    'hidden_size': hidden_size,
                    'intermediate_size': intermediate_size,
                    'vocab_size': vocab_size,
                    'num_heads': num_heads,
                    'num_kv_heads': num_kv_heads,
                    'head_dim': head_dim
                }
            }
            
            print(f"    down_proj: token_size={token_size}, dims=[{M}, {N}, {K}], FLOPS={flops/1e9:.2f}G")
        
        return layer_performance
    
    def calculate_lm_head_performance(self, remaining_indices: List[int], 
                                     all_kernels: List[Dict], 
                                     model_config: Dict) -> List[Dict]:
        """计算LM Head kernels的性能（使用精确token size）"""
        
        if not remaining_indices:
            return []
        
        # 从实际模型配置中读取参数，移除硬编码默认值
        if 'hidden_size' not in model_config:
            raise ValueError("模型配置中缺少 hidden_size")
        if 'vocab_size' not in model_config:
            raise ValueError("模型配置中缺少 vocab_size")
            
        hidden_size = model_config["hidden_size"]  # H
        vocab_size = model_config["vocab_size"]   # V - 使用实际值
        dtype_bytes = 2  # bfloat16
        
        # 获取Tensor Parallelism配置
        tp_size = getattr(self, 'tp_size', 1)  # 从类属性获取
        
        # LM Head计算参数
        B = 1  # batch_size，通常为1
        H = hidden_size
        V = vocab_size // tp_size  # TP分割词汇表维度
        
        print(f"\n=== LM Head性能计算 (使用精确token size) ===")
        print(f"模型配置: hidden_size={H}, vocab_size={vocab_size}, tp_size={tp_size}")
        print(f"注意: 在TP={tp_size}模式下，词汇表维度被分割为 {V}")
        
        lm_head_performance = []
        
        for kernel_idx in remaining_indices:
            kernel = all_kernels[kernel_idx]
            
            # 关键修改：使用每个kernel的精确token size
            token_size = self.get_kernel_token_size(kernel)
            L = token_size  # 使用精确的token size
            
            print(f"  Kernel[{kernel_idx}]: token_size={token_size}")
            print(f"    矩阵维度: hidden_states[{B}, {L}, {H}] × W_lm[{H}, {V}] → logits[{B}, {L}, {V}]")
            
            # LM Head的FLOPS计算: logits = hidden_states × W_lm + b_lm
            # 矩阵乘法FLOPS: 2 * B * L * H * V (每个输出元素需要H次乘法和H次加法)
            # 偏置加法FLOPS: B * L * V (每个输出元素加一次偏置)
            matmul_flops = 2 * B * L * H * V
            bias_flops = B * L * V
            total_flops = matmul_flops + bias_flops
            
            # 内存访问计算
            # 读取: hidden_states[B*L*H] + W_lm[H*V] + b_lm[V]
            # 写入: logits[B*L*V]
            memory_read = (B * L * H + H * V + V) * dtype_bytes
            memory_write = B * L * V * dtype_bytes
            total_memory_access = memory_read + memory_write
            
            # 算术强度
            arithmetic_intensity = total_flops / total_memory_access if total_memory_access > 0 else 0
            
            lm_head_performance.append({
                'kernel_idx': kernel_idx,
                'kernel_name': kernel['name'],
                'duration_us': kernel.get('duration_us', 0),
                'token_size': token_size,  # 新增：记录使用的精确token size
                'matrix_dims': [B, L, H, V],  # [batch, seq_len, hidden, vocab_per_tp]
                'flops': total_flops,
                'matmul_flops': matmul_flops,
                'bias_flops': bias_flops,
                'memory_access': total_memory_access,
                'memory_read': memory_read,
                'memory_write': memory_write,
                'arithmetic_intensity': arithmetic_intensity,
                'group_id': 'lm_head',
                'operation_type': 'lm_head_projection',
                'tp_size': tp_size,
                'original_vocab_size': vocab_size,
                'config_validation': {
                    'hidden_size': hidden_size,
                    'vocab_size': vocab_size,
                    'tp_size': tp_size
                }
            })
            
            print(f"    FLOPS: {total_flops/1e9:.2f} GFLOPS (矩阵乘法: {matmul_flops/1e9:.2f}, 偏置: {bias_flops/1e9:.2f})")
            print(f"    内存访问: {total_memory_access/1e6:.2f} MB")
            print(f"    算术强度: {arithmetic_intensity:.2f}")
        
        return lm_head_performance
    
    def calculate_remaining_linear_performance(self, remaining_indices: List[int], 
                                             all_kernels: List[Dict], 
                                             model_config: Dict) -> Dict:
        """计算剩余Linear kernels的性能（现在专门处理LM Head，使用精确token size）"""
        
        if not remaining_indices:
            return {}
        
        print(f"\n=== 剩余Linear算子分析 (使用精确token size) ===")
        print(f"发现 {len(remaining_indices)} 个剩余Linear kernels")
        
        # 检查kernel类型分布
        kernel_names = [all_kernels[idx]['name'] for idx in remaining_indices]
        unique_names = set(kernel_names)
        
        print(f"剩余Linear kernels类型:")
        for name in unique_names:
            count = kernel_names.count(name)
            print(f"  - {name}: {count} 次")
        
        # 修改逻辑：所有剩余的Linear kernels都按LM Head方式计算
        # 因为在Transformer架构中，除了5种标准projection之外，剩余的Linear操作主要是LM Head
        print(f"将所有剩余Linear kernels按LM Head方式计算（使用精确token size）")
        lm_head_performance = self.calculate_lm_head_performance(remaining_indices, all_kernels, model_config)
        
        return {
            'lm_head': lm_head_performance
        }
    
    def analyze_linear_performance(self, all_kernels: List[Dict], model_config: Dict, 
                                 hardware_spec: Dict) -> Dict:
        """分析Linear算子性能 - 2+2模式，使用精确token size"""
        
        print(f"\n=== Linear算子性能分析 (2+2模式，使用精确token size) ===")
        
        # 从hardware_spec中获取TP配置并设置为类属性
        self.tp_size = hardware_spec.get('tensor_parallelism', 1)
        self.hardware_spec = hardware_spec
        
        # 验证必需的模型配置存在
        required_configs = ['hidden_size', 'intermediate_size', 'vocab_size']
        for config_key in required_configs:
            if config_key not in model_config:
                raise ValueError(f"模型配置中缺少必需参数: {config_key}")
        
        # 验证注意力头数参数存在（支持两种命名）
        if 'num_attention_heads' not in model_config and 'num_heads' not in model_config:
            raise ValueError("模型配置中缺少 num_attention_heads 或 num_heads")
        
        # 显示实际使用的配置
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
        
        print(f"实际模型配置:")
        print(f"  - hidden_size: {hidden_size}")
        print(f"  - intermediate_size: {intermediate_size}")
        print(f"  - vocab_size: {vocab_size}")
        print(f"  - tensor_parallelism: {self.tp_size}")
        
        # 处理GQA/MQA配置
        if 'num_key_value_heads' in model_config:
            num_kv_heads = model_config["num_key_value_heads"]
            print(f"  - num_key_value_heads: {num_kv_heads} (使用GQA/MQA)")
        elif 'num_kv_heads' in model_config:
            num_kv_heads = model_config["num_kv_heads"]
            print(f"  - num_kv_heads: {num_kv_heads} (使用GQA/MQA)")
        else:
            print(f"  - 未使用GQA/MQA，num_kv_heads = num_heads = {num_heads}")
        
        # 寻找Transformer层 - 2+2模式
        valid_layers, remaining_linear = self.find_transformer_layers_2plus2(all_kernels)
        
        # 初始化projection性能统计
        proj_names = ['qkv_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        projection_performance = {name: {'executions': [], 'total_flops': 0, 'total_memory_access': 0} 
                                for name in proj_names}
        
        # 处理精确分组的层
        for layer_info in valid_layers:
            layer_perf = self.calculate_projection_performance_2plus2(layer_info, all_kernels, model_config)
            
            for proj_name, perf in layer_perf.items():
                projection_performance[proj_name]['executions'].append(perf)
                projection_performance[proj_name]['total_flops'] += perf['flops']
                projection_performance[proj_name]['total_memory_access'] += perf['memory_access']
        
        # 处理剩余的Linear kernels（现在专门处理LM Head）
        remaining_perf = self.calculate_remaining_linear_performance(remaining_linear, all_kernels, model_config)
        
        # 更新projection_performance以包含LM Head结果
        if 'lm_head' in remaining_perf:
            # LM Head kernels
            lm_head_executions = remaining_perf['lm_head']
            projection_performance['lm_head'] = {
                'executions': lm_head_executions,
                'total_flops': sum(ex['flops'] for ex in lm_head_executions),
                'total_memory_access': sum(ex['memory_access'] for ex in lm_head_executions),
                'execution_count': len(lm_head_executions),
                'precise_group_count': 0,
                'average_group_count': 0,
                'lm_head_count': len(lm_head_executions),
                'total_duration_us': sum(ex['duration_us'] for ex in lm_head_executions),
                'valid_groups': 0
            }
            print(f"lm_head: {len(lm_head_executions)} 次执行 (LM Head专用计算，使用精确token size)")
        else:
            # 传统的平均分配结果
            for proj_name, executions in remaining_perf.items():
                for exec_info in executions:
                    projection_performance[proj_name]['executions'].append(exec_info)
                    projection_performance[proj_name]['total_flops'] += exec_info['flops']
                    projection_performance[proj_name]['total_memory_access'] += exec_info['memory_access']
        
        # 计算汇总性能指标
        total_duration = 0
        total_flops = 0
        total_memory_access = 0
        
        proj_names = ['qkv_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        if 'lm_head' in projection_performance:
            proj_names.append('lm_head')
        
        for proj_name in proj_names:
            if proj_name in projection_performance:
                executions = projection_performance[proj_name]['executions']
                if executions:
                    proj_duration = sum(ex['duration_us'] for ex in executions)
                    total_duration += proj_duration
                    
                    if proj_name == 'lm_head':
                        lm_head_count = len(executions)
                        projection_performance[proj_name].update({
                            'execution_count': lm_head_count,
                            'precise_group_count': 0,
                            'average_group_count': 0,
                            'lm_head_count': lm_head_count,
                            'total_duration_us': proj_duration,
                            'valid_groups': 0
                        })
                        print(f"lm_head: {lm_head_count} 次执行 (LM Head专用计算，使用精确token size)")
                    else:
                        precise_count = len([ex for ex in executions if ex['group_id'] != 'average'])
                        average_count = len([ex for ex in executions if ex['group_id'] == 'average'])
                        
                        projection_performance[proj_name].update({
                            'execution_count': len(executions),
                            'precise_group_count': precise_count,
                            'average_group_count': average_count,
                            'total_duration_us': proj_duration,
                            'valid_groups': len(valid_layers)
                        })
                        
                        # 特殊处理融合的gate_proj和up_proj
                        if proj_name in ['gate_proj', 'up_proj']:
                            fused_count = len([ex for ex in executions if ex.get('projection_type') == 'mlp_fused'])
                            if fused_count > 0:
                                print(f"{proj_name}: {len(executions)} 次执行 (精确:{precise_count}, 平均:{average_count}, 融合:{fused_count})")
                            else:
                                print(f"{proj_name}: {len(executions)} 次执行 (精确:{precise_count}, 平均:{average_count})")
                        else:
                            print(f"{proj_name}: {len(executions)} 次执行 (精确:{precise_count}, 平均:{average_count})")
        
        total_flops = sum(proj['total_flops'] for proj in projection_performance.values())
        total_memory_access = sum(proj['total_memory_access'] for proj in projection_performance.values())
        
        # 计算整体性能指标
        flops_per_second = total_flops / (total_duration * 1e-6) if total_duration > 0 else 0
        memory_bandwidth = total_memory_access / (total_duration * 1e-6) if total_duration > 0 else 0
        arithmetic_intensity = total_flops / total_memory_access if total_memory_access > 0 else 0
        
        return {
            'kernel_count': len([k for k in all_kernels if self.is_linear_kernel(k.get('name', ''))]),
            'total_duration_us': total_duration,
            'total_flops': total_flops,
            'flops_per_second': flops_per_second,
            'tflops_per_second': flops_per_second / 1e12,
            'memory_bandwidth_gb_s': memory_bandwidth / 1e9,
            'arithmetic_intensity': arithmetic_intensity,
            'efficiency_compute': min(1.0, flops_per_second / hardware_spec.get('peak_flops_fp16', 1e15)),
            'efficiency_memory': min(1.0, memory_bandwidth / hardware_spec.get('memory_bandwidth', 1e12)),
            'projection_analysis': "precise_transformer_layer_based_2plus2_mode_with_exact_token_size",
            'projection_breakdown': projection_performance,
            'valid_transformer_layers': len(valid_layers),
            'remaining_linear_kernels': len(remaining_linear),
            'model_config_used': {
                'hidden_size': hidden_size,
                'intermediate_size': intermediate_size,
                'vocab_size': vocab_size,
                'tensor_parallelism': self.tp_size
            },
            'token_size_method': 'exact_per_kernel_matched_token_size',
            'layer_detection_mode': '2plus2_with_fused_gate_up'
        }
    
    def analyze_performance(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """主要的性能分析函数 - 专门分析Linear算子，2+2模式，使用精确token size"""
        
        print(f"\n=== OEA Stage 2: Linear算子专门分析 (2+2模式，使用精确token size) ===")
        
        # 提取数据
        metadata = processed_data['metadata']
        operator_kernels = processed_data['operator_kernels']
        
        model_config = metadata['model_config']
        hardware_spec = metadata['hardware_spec']
        
        # 检查是否有token匹配信息
        has_token_matching = 'token_matching' in metadata
        if has_token_matching:
            token_method = metadata['token_matching']['method']
            print(f"检测到token匹配信息: {token_method}")
        else:
            print("未检测到token匹配信息，将使用兼容模式")
        
        # 构建按时间排序的所有kernels列表
        all_kernels = []
        for op_type, kernels in operator_kernels.items():
            for kernel in kernels:
                kernel_with_type = kernel.copy()
                kernel_with_type['operator_type'] = op_type
                all_kernels.append(kernel_with_type)
        
        # 按时间戳排序
        all_kernels.sort(key=lambda x: x.get('timestamp_us', 0))
        
        print(f"总计 {len(all_kernels)} 个kernels")
        
        # 筛选出所有Linear kernels
        linear_kernels = [k for k in all_kernels if self.is_linear_kernel(k.get('name', ''))]
        print(f"识别出 {len(linear_kernels)} 个Linear kernels")
        
        if not linear_kernels:
            print("未找到Linear kernels，无法进行分析")
            return {
                'metadata': metadata,
                'linear_analysis': {
                    'kernel_count': 0,
                    'message': 'No linear kernels found'
                }
            }
        
        # 检查token size使用情况
        kernels_with_matched_token = [k for k in linear_kernels if 'matched_token_size' in k]
        print(f"其中 {len(kernels_with_matched_token)} 个kernels包含精确token size信息")
        
        if kernels_with_matched_token:
            token_sizes = [k['matched_token_size'] for k in kernels_with_matched_token]
            print(f"Token size范围: {min(token_sizes):.0f} - {max(token_sizes):.0f}")
            print(f"平均token size: {np.mean(token_sizes):.1f}")
        
        # 分析Linear算子性能
        linear_analysis = self.analyze_linear_performance(all_kernels, model_config, hardware_spec)
        
        # 构建输出结果
        result = {
            'metadata': metadata,
            'linear_analysis': linear_analysis,
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_method': 'precise_token_size_based_2plus2_mode'
        }
        
        # 性能总结
        print(f"\n=== Linear算子性能总结 (2+2模式) ===")
        print(f"总计算量: {linear_analysis['total_flops']/1e12:.2f} TFLOPS")
        print(f"实际算力: {linear_analysis['tflops_per_second']:.2f} TFLOPS/s")
        print(f"内存带宽: {linear_analysis['memory_bandwidth_gb_s']:.2f} GB/s")
        print(f"算术强度: {linear_analysis['arithmetic_intensity']:.2f}")
        print(f"计算效率: {linear_analysis['efficiency_compute']*100:.1f}%")
        print(f"内存效率: {linear_analysis['efficiency_memory']*100:.1f}%")
        
        return result
    
    def save_analysis_results(self, results: Dict[str, Any], output_file: str):
        """保存分析结果"""
        
        # 转换numpy类型为Python原生类型
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
        
        print(f"\nLinear算子分析结果已保存到: {output_file}")

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='OEA Stage 2 Linear算子分析器 (2+2模式版本)')
    parser.add_argument('--input', required=True,
                       help='Stage 1处理数据文件路径')
    parser.add_argument('--output', default=None,
                       help='输出文件路径 (默认: 自动生成)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 设置输出文件
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"oea_stage2_linear_analysis_2plus2_results.json"
    
    try:
        # 读取Stage 1数据
        with open(args.input, 'r') as f:
            processed_data = json.load(f)
        
        # 创建分析器
        analyzer = LinearAnalyzer2Plus2()
        
        # 执行分析
        results = analyzer.analyze_performance(processed_data)
        
        # 保存结果
        analyzer.save_analysis_results(results, args.output)
        
        print(f"\n=== Stage 2 Linear算子分析完成 (2+2模式版本) ===")
        print(f"输出文件: {args.output}")
        print(f"请使用以下命令进行 Stage 3 分析:")
        print(f"python3 stage3_nonlinear_analyzer.py --input {args.output}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()