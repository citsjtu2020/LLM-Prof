#!/usr/bin/env python3
"""
OEA Stage 2: SGLang Linear算子专门分析脚本 (基于act_and_mul的3+1模式)
基于 flashinfer::activation::act_and_mul_kernel 算子进行检测和分组

检测规则:
1. 找到所有名称包含 act_and_mul 的算子作为 anchor
2. 在每个 anchor 前找最近的3个 linear 算子: qkv_proj, o_proj, gate_proj+up_proj(融合)
3. 在每个 anchor 后找最近的1个 linear 算子: down_proj
4. 剩余的所有 linear 算子都是 lm_head (应该总共10个)

使用方法:
python stage2_linear_analyzer_sglang_fixed.py --input stage1_processed_data.json
"""

import json
import numpy as np
import argparse
import os
import sys
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple

class SGLangLinearAnalyzer:
    def __init__(self):
        """初始化SGLang Linear算子分析器（基于act_and_mul的3+1模式）"""
        
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
        
        # act_and_mul算子匹配模式 - 更精确的匹配
        self.act_and_mul_patterns = [
            r'.*flashinfer::activation::act_and_mul_kernel.*',
            r'.*act_and_mul_kernel.*',
            r'.*act_and_mul.*'
        ]
        
        # 预编译正则表达式以提高性能
        self.compiled_linear_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.linear_patterns]
        self.compiled_linear_exclude_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.linear_exclude_patterns]
        self.compiled_act_and_mul_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.act_and_mul_patterns]
        
        # 缓存分类结果
        self.classification_cache = {}
        
        # 模型配置数据库
        self.model_configs = {
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
            'Qwen2.5-3B': {
                'hidden_size': 2048,
                'intermediate_size': 11008,
                'vocab_size': 151936,
                'num_attention_heads': 16,
                'num_key_value_heads': 2,
                'num_hidden_layers': 36
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
                'num_hidden_layers': 48
            },
            'Qwen2.5-7B': {
                'hidden_size': 3584,
                'intermediate_size': 18944,
                'vocab_size': 152064,
                'num_attention_heads': 28,
                'num_key_value_heads': 4,
                'num_hidden_layers': 28
            },
            'Qwen2.5-32B': {
                'hidden_size': 5120,
                'intermediate_size': 27648,
                'vocab_size': 152064,
                'num_attention_heads': 40,
                'num_key_value_heads': 8,
                'num_hidden_layers': 64
            },
            'Llama-3.1-8B': {
                'hidden_size': 4096,
                'intermediate_size': 14336,
                'vocab_size': 128256,
                'num_attention_heads': 32,
                'num_key_value_heads': 8,
                'num_hidden_layers': 32
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
       
    def get_model_config_from_name(self, model_name: str) -> Dict[str, Any]:
        """根据模型名称获取配置信息"""
        
        print(f"=== 解析模型配置 ===")
        print(f"模型名称: {model_name}")
        
        # 直接匹配
        if model_name in self.model_configs:
            config = self.model_configs[model_name].copy()
            print(f"找到精确匹配的配置: {model_name}")
            return config
        
        # 模糊匹配
        for config_name, config_data in self.model_configs.items():
            if config_name.lower() in model_name.lower() or model_name.lower() in config_name.lower():
                config = config_data.copy()
                print(f"找到模糊匹配的配置: {config_name} -> {model_name}")
                return config
        
        # 默认配置（Qwen3-14B）
        print(f"警告: 未找到匹配的模型配置 '{model_name}'，使用默认Qwen3-14B配置")
        return self.model_configs['Qwen3-14B'].copy()

    def is_linear_kernel(self, kernel_name: str) -> bool:
        """判断是否为Linear kernel - 排除splitKreduce等"""
        if kernel_name in self.classification_cache:
            return self.classification_cache[kernel_name].get('is_linear', False)
        
        kernel_name_lower = kernel_name.lower()
        
        # 首先检查排除模式
        for exclude_pattern in self.compiled_linear_exclude_patterns:
            if exclude_pattern.search(kernel_name_lower):
                self.classification_cache[kernel_name] = {'is_linear': False}
                return False
        
        # 然后检查匹配模式
        is_linear = any(pattern.search(kernel_name_lower) for pattern in self.compiled_linear_patterns)
        self.classification_cache[kernel_name] = {'is_linear': is_linear}
        return is_linear
    
    def is_act_and_mul_kernel(self, kernel_name: str) -> bool:
        """判断是否为act_and_mul kernel"""
        if kernel_name in self.classification_cache:
            return self.classification_cache[kernel_name].get('is_act_and_mul', False)
        
        kernel_name_lower = kernel_name.lower()
        is_act_and_mul = any(pattern.search(kernel_name_lower) for pattern in self.compiled_act_and_mul_patterns)
        self.classification_cache[kernel_name] = {'is_act_and_mul': is_act_and_mul}
        return is_act_and_mul
    
    def get_kernel_token_size(self, kernel: Dict) -> int:
        """
        获取kernel的token size - 根据phase返回正确的值
        
        Args:
            kernel: kernel字典，必须包含'phase'字段
            
        Returns:
            token_size: 
                - prefill阶段返回seq_len（输入序列长度）
                - decode阶段返回1（每次生成1个token）
                - unknown阶段返回seq_len（保守估计）
        """
        phase = kernel.get('phase', 'unknown')
        
        if phase == 'prefill':
            # Prefill阶段：处理完整的输入序列
            return self.seq_len
        elif phase == 'decode':
            # Decode阶段：每次只生成1个token
            return 1
        else:
            # Unknown阶段：使用seq_len作为保守估计
            return self.seq_len
    
    def determine_kernel_phase(self, kernel: Dict, iteration_data: Dict) -> str:
        """根据kernel的时间戳确定它属于prefill还是decode阶段"""
        kernel_ts = kernel.get('timestamp', 0)
        
        # 从iteration_data中获取各个iteration的时间范围
        iterations = iteration_data.get('iterations', [])
        
        for iteration in iterations:
            start_ts = iteration.get('start_ts', 0)
            end_ts = iteration.get('end_ts', 0)
            phase = iteration.get('phase', 'unknown')
            
            if start_ts <= kernel_ts <= end_ts:
                return phase
        
        return 'unknown'
    
    def find_transformer_layers_3plus1(self, all_kernels: List[Dict], iteration_data: Dict) -> Tuple[List[Dict], List[int]]:
        """
        寻找符合SGLang Transformer层模式的kernel组合 (基于实际数据的精确3+1模式)
        模式: 基于 act_and_mul 算子作为 anchor，根据实际execution pattern进行分配
        """
        
        print(f"=== 寻找SGLang Transformer层模式 (精确3+1模式) ===")
        
        # 找到所有 act_and_mul 算子作为 anchor
        act_and_mul_indices = []
        linear_indices = []
        
        for i, kernel in enumerate(all_kernels):
            kernel_name = kernel.get('name', '')
            operator_type = kernel.get('operator_type', '')
            
            # 检测 act_and_mul 算子
            is_act_and_mul_by_name = self.is_act_and_mul_kernel(kernel_name)
            is_act_and_mul_by_type = (operator_type == 'activation' and 'act_and_mul' in kernel_name.lower())
            
            if is_act_and_mul_by_name or is_act_and_mul_by_type:
                act_and_mul_indices.append(i)
            
            # 检测 linear 算子
            is_linear_by_name = self.is_linear_kernel(kernel_name)
            is_linear_by_type = (operator_type == 'linear')
            
            if is_linear_by_name or is_linear_by_type:
                linear_indices.append(i)
        
        print(f"找到 {len(act_and_mul_indices)} 个 act_and_mul anchors")
        print(f"找到 {len(linear_indices)} 个 Linear kernels")
        
        if len(act_and_mul_indices) == 0:
            print("错误: 没有找到 act_and_mul anchor 算子")
            return [], linear_indices
        
        # 按阶段分组处理anchors - 直接使用kernel中的phase信息
        prefill_anchors = []
        decode_anchors = []
        prefill_linear = []
        decode_linear = []
        
        for anchor_idx in act_and_mul_indices:
            anchor_kernel = all_kernels[anchor_idx]
            # 直接使用kernel中已有的phase信息，而不是重新判断
            phase = anchor_kernel.get('phase', 'unknown')
            
            if phase == 'prefill':
                prefill_anchors.append(anchor_idx)
            elif phase == 'decode':
                decode_anchors.append(anchor_idx)
        
        for linear_idx in linear_indices:
            linear_kernel = all_kernels[linear_idx]
            # 直接使用kernel中已有的phase信息
            phase = linear_kernel.get('phase', 'unknown')
            
            if phase == 'prefill':
                prefill_linear.append(linear_idx)
            elif phase == 'decode':
                decode_linear.append(linear_idx)
        
        print(f"Prefill: {len(prefill_anchors)} anchors, {len(prefill_linear)} linear")
        print(f"Decode: {len(decode_anchors)} anchors, {len(decode_linear)} linear")
        
        # 获取iteration信息进行精确分配
        iterations = iteration_data.get('iterations', [])
        prefill_iterations = [it for it in iterations if it.get('phase') == 'prefill']
        decode_iterations = [it for it in iterations if it.get('phase') == 'decode']
        
        print(f"Iterations: {len(prefill_iterations)} prefill, {len(decode_iterations)} decode")
        
        # 计算实际的层数
        num_layers = len(prefill_anchors)  # prefill阶段每层1个anchor
        anchors_per_decode_iter = len(decode_anchors) // len(decode_iterations) if decode_iterations else 0
        
        print(f"推断模型层数: {num_layers}")
        print(f"每个decode iteration的anchors: {anchors_per_decode_iter}")
        
        # 基于实际模式进行精确分配
        valid_layers = []
        used_linear_indices = set()
        
        # 处理Prefill阶段 - 严格的3+1模式
        print(f"\n--- 处理Prefill阶段 (每层1个anchor) ---")
        prefill_anchors_sorted = sorted(prefill_anchors)
        prefill_linear_sorted = sorted(prefill_linear)
        
        for i, anchor_idx in enumerate(prefill_anchors_sorted):
            anchor_kernel = all_kernels[anchor_idx]
            
            # 为每个prefill anchor分配4个linear kernels
            # 找到这个anchor前面最近的3个和后面最近的1个linear
            available_before = [idx for idx in prefill_linear_sorted 
                              if idx < anchor_idx and idx not in used_linear_indices]
            available_after = [idx for idx in prefill_linear_sorted 
                             if idx > anchor_idx and idx not in used_linear_indices]
            
            # 取最近的3个前置和1个后置
            before_linear = sorted(available_before)[-3:] if len(available_before) >= 3 else available_before
            after_linear = sorted(available_after)[:1] if len(available_after) >= 1 else []
            
            # 如果不够4个，从整体pool中补充
            total_needed = 4
            current_assigned = len(before_linear) + len(after_linear)
            
            if current_assigned < total_needed:
                # 从剩余的prefill linear中补充
                remaining_prefill = [idx for idx in prefill_linear_sorted 
                                   if idx not in used_linear_indices and 
                                   idx not in before_linear and idx not in after_linear]
                
                needed_more = total_needed - current_assigned
                additional = remaining_prefill[:needed_more]
                
                # 按位置分配到before或after
                for add_idx in additional:
                    if add_idx < anchor_idx and len(before_linear) < 3:
                        before_linear.append(add_idx)
                    else:
                        after_linear.append(add_idx)
            
            if len(before_linear) + len(after_linear) >= 3:  # 至少需要3个linear
                layer_info = {
                    'layer_id': len(valid_layers),
                    'anchor_idx': anchor_idx,
                    'anchor_kernel': anchor_kernel,
                    'phase': 'prefill',
                    'before_linear': sorted(before_linear),
                    'after_linear': sorted(after_linear),
                    'pattern': f"Linear{sorted(before_linear)} → ActAndMul[{anchor_idx}] → Linear{sorted(after_linear)}",
                    'mode': 'prefill_3plus1'
                }
                
                valid_layers.append(layer_info)
                used_linear_indices.update(before_linear + after_linear)
                
                print(f"  ✓ Prefill层 {len(valid_layers)}: 使用了{len(before_linear + after_linear)}个linear")
        
        # 处理Decode阶段 - 基于iteration分组
        print(f"\n--- 处理Decode阶段 (按iteration分组) ---")
        decode_anchors_sorted = sorted(decode_anchors)
        decode_linear_sorted = sorted(decode_linear)
        
        # 按iteration分组decode anchors
        anchors_per_iter = len(decode_anchors) // len(decode_iterations) if decode_iterations else len(decode_anchors)
        
        for iter_idx in range(len(decode_iterations)):
            iter_start = iter_idx * anchors_per_iter
            iter_end = min((iter_idx + 1) * anchors_per_iter, len(decode_anchors_sorted))
            iter_anchors = decode_anchors_sorted[iter_start:iter_end]
            
            print(f"  Decode iteration {iter_idx + 1}: {len(iter_anchors)} anchors")
            
            # 为这个iteration的每个anchor分配linear kernels
            # 每个anchor需要4个linear：前3个 + 后1个（严格的3+1模式）
            for anchor_idx in iter_anchors:
                anchor_kernel = all_kernels[anchor_idx]
                anchor_ts = anchor_kernel.get('ts', 0)
                
                # 使用时间戳而不是索引来判断before/after
                # 找到这个anchor前面和后面可用的linear（基于时间戳）
                available_before = []
                available_after = []
                
                for linear_idx in decode_linear_sorted:
                    if linear_idx in used_linear_indices:
                        continue
                    linear_kernel = all_kernels[linear_idx]
                    linear_ts = linear_kernel.get('ts', 0)
                    
                    if linear_ts < anchor_ts:
                        available_before.append(linear_idx)
                    elif linear_ts > anchor_ts:
                        available_after.append(linear_idx)
                
                # 按时间戳排序，取最近的3个before和1个after
                available_before.sort(key=lambda idx: all_kernels[idx].get('ts', 0))
                available_after.sort(key=lambda idx: all_kernels[idx].get('ts', 0))
                
                # 严格要求：必须有至少3个before和至少1个after
                if len(available_before) >= 3 and len(available_after) >= 1:
                    # 取最近的3个前置linear和1个后置linear
                    before_linear = available_before[-3:]
                    after_linear = available_after[:1]
                    
                    layer_info = {
                        'layer_id': len(valid_layers),
                        'anchor_idx': anchor_idx,
                        'anchor_kernel': anchor_kernel,
                        'phase': 'decode',
                        'iteration': iter_idx + 1,
                        'before_linear': sorted(before_linear),
                        'after_linear': sorted(after_linear),
                        'pattern': f"Linear{sorted(before_linear)} → ActAndMul[{anchor_idx}] → Linear{sorted(after_linear)}",
                        'mode': 'decode_3plus1'
                    }
                    
                    valid_layers.append(layer_info)
                    
                    # 标记已使用的linear
                    used_linear_indices.update(before_linear + after_linear)
                else:
                    print(f"    警告: anchor {anchor_idx} 的linear不足，before={len(available_before)}, after={len(available_after)}")
        
        print(f"\n总共找到 {len(valid_layers)} 个有效的SGLang Transformer层")
        
        # 处理剩余未分配的Linear kernels
        remaining_linear_final = [idx for idx in linear_indices if idx not in used_linear_indices]
        print(f"剩余未分配的Linear kernels: {len(remaining_linear_final)} 个 (lm_head)")
        
        # 验证分配结果
        total_used_linear = len(used_linear_indices)
        expected_used = len(act_and_mul_indices) * 4  # 每个anchor对应4个linear
        print(f"验证: 使用了{total_used_linear}个linear, 预期{expected_used}个")
        print(f"验证: 剩余{len(remaining_linear_final)}个linear作为lm_head")
        
        return valid_layers, remaining_linear_final
    
    def calculate_projection_performance_3plus1(self, layer_info: Dict, all_kernels: List[Dict], 
                                               model_config: Dict) -> Dict:
        """计算单个SGLang Transformer层中各projection的性能 - 3+1模式"""
        
        # 验证必需的模型配置
        required_configs = ['hidden_size', 'intermediate_size', 'vocab_size']
        for config_key in required_configs:
            if config_key not in model_config:
                raise ValueError(f"模型配置中缺少必需参数: {config_key}")
        
        # 验证注意力头数参数存在（支持两种命名）
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
        proj_configs = {
            'qkv_proj': {
                'N': (num_heads + 2 * num_kv_heads) * head_dim // tp_size,
                'K': hidden_size
            },
            'o_proj': {
                'N': hidden_size, 
                'K': num_heads * head_dim // tp_size
            },
            'gate_proj': {
                'N': intermediate_size // tp_size,
                'K': hidden_size
            },
            'up_proj': {
                'N': intermediate_size // tp_size,
                'K': hidden_size
            },
            'down_proj': {
                'N': hidden_size,
                'K': intermediate_size // tp_size
            }
        }
        
        # 分配kernels到projections - 3+1模式
        before_linear_indices = layer_info['before_linear']  # [qkv_proj, o_proj, gate_proj+up_proj]
        after_linear_indices = layer_info['after_linear']    # [down_proj]
        
        layer_performance = {}
        
        print(f"  SGLang 3+1模式分析 - 层 {layer_info['layer_id']} (phase: {layer_info['phase']})")
        print(f"  模型配置: hidden_size={hidden_size}, intermediate_size={intermediate_size}")
        print(f"  注意力配置: num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
        print(f"  TP配置: tp_size={tp_size}")
        
        # 处理前3个Linear: qkv_proj, o_proj, gate_proj+up_proj(融合)
        proj_names = ['qkv_proj', 'o_proj', 'gate_proj+up_proj']
        for i, proj_name in enumerate(proj_names):
            if i < len(before_linear_indices):
                kernel_idx = before_linear_indices[i]
                kernel = all_kernels[kernel_idx]
                
                token_size = self.get_kernel_token_size(kernel)
                M = token_size
                
                if proj_name == 'gate_proj+up_proj':
                    # 融合的gate_proj+up_proj，需要特殊处理
                    # 修复：使用'dur'字段而不是'duration_us'
                    total_duration = kernel.get('dur', 0)
                    split_duration = total_duration / 2.0
                    
                    # 分别计算gate_proj和up_proj
                    # 修复FLOPS计算：由于kernel是sliced1x2，每个projection只完成一半计算
                    for sub_proj in ['gate_proj', 'up_proj']:
                        N = proj_configs[sub_proj]['N']
                        K = proj_configs[sub_proj]['K']
                        
                        # 修复：FLOPS需要除以2，因为是sliced1x2的kernel
                        # 原来：flops = 2 * M * N * K
                        # 修正：每个projection只完成一半计算
                        flops = M * N * K
                        memory_access = (M * K + N * K + M * N) * dtype_bytes
                        arithmetic_intensity = flops / memory_access if memory_access > 0 else 0
                        
                        layer_performance[sub_proj] = {
                            'kernel_idx': kernel_idx,
                            'kernel_name': kernel['name'],
                            'duration_us': split_duration,
                            'original_duration_us': total_duration,
                            'token_size': token_size,
                            'matrix_dims': [M, N, K],
                            'flops': flops,
                            'memory_access': memory_access,
                            'arithmetic_intensity': arithmetic_intensity,
                            'group_id': layer_info['layer_id'],
                            'tp_size': tp_size,
                            'projection_type': 'mlp_fused',
                            'phase': layer_info['phase'],
                            'fused_with': 'gate_proj+up_proj'
                        }
                        
                        print(f"    {sub_proj}: token_size={token_size}, dims=[{M}, {N}, {K}], FLOPS={flops/1e9:.2f}G (融合平分)")
                else:
                    # 单独的projection
                    N = proj_configs[proj_name]['N']
                    K = proj_configs[proj_name]['K']
                    
                    flops = 2 * M * N * K
                    memory_access = (M * K + N * K + M * N) * dtype_bytes
                    arithmetic_intensity = flops / memory_access if memory_access > 0 else 0
                    
                    layer_performance[proj_name] = {
                        'kernel_idx': kernel_idx,
                        'kernel_name': kernel['name'],
                        # 修复：使用'dur'字段而不是'duration_us'
                        'duration_us': kernel.get('dur', 0),
                        'token_size': token_size,
                        'matrix_dims': [M, N, K],
                        'flops': flops,
                        'memory_access': memory_access,
                        'arithmetic_intensity': arithmetic_intensity,
                        'group_id': layer_info['layer_id'],
                        'tp_size': tp_size,
                        'projection_type': 'attention',
                        'phase': layer_info['phase']
                    }
                    
                    print(f"    {proj_name}: token_size={token_size}, dims=[{M}, {N}, {K}], FLOPS={flops/1e9:.2f}G")
        
        # 处理后1个Linear: down_proj
        if len(after_linear_indices) >= 1:
            down_kernel_idx = after_linear_indices[0]
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
                # 修复：使用'dur'字段而不是'duration_us'
                'duration_us': down_kernel.get('dur', 0),
                'token_size': token_size,
                'matrix_dims': [M, N, K],
                'flops': flops,
                'memory_access': memory_access,
                'arithmetic_intensity': arithmetic_intensity,
                'group_id': layer_info['layer_id'],
                'tp_size': tp_size,
                'projection_type': 'mlp',
                'phase': layer_info['phase']
            }
            
            print(f"    down_proj: token_size={token_size}, dims=[{M}, {N}, {K}], FLOPS={flops/1e9:.2f}G")
        
        return layer_performance
    
    def calculate_lm_head_performance(self, remaining_indices: List[int], 
                                     all_kernels: List[Dict], 
                                     model_config: Dict,
                                     iteration_data: Dict) -> List[Dict]:
        """计算LM Head kernels的性能，并标记所属的iteration阶段"""
        
        if not remaining_indices:
            return []
        
        # 验证模型配置
        if 'hidden_size' not in model_config:
            raise ValueError("模型配置中缺少 hidden_size")
        if 'vocab_size' not in model_config:
            raise ValueError("模型配置中缺少 vocab_size")
            
        hidden_size = model_config["hidden_size"]
        vocab_size = model_config["vocab_size"]
        dtype_bytes = 2  # bfloat16
        
        # 获取Tensor Parallelism配置
        tp_size = getattr(self, 'tp_size', 1)
        
        # LM Head计算参数
        B = 1  # batch_size
        H = hidden_size
        V = vocab_size // tp_size  # TP分割词汇表维度
        
        print(f"\n=== LM Head性能计算 (SGLang框架) ===")
        print(f"模型配置: hidden_size={H}, vocab_size={vocab_size}, tp_size={tp_size}")
        print(f"预期lm_head数量: 10 (1 prefill + 9 decode)")
        
        lm_head_performance = []
        prefill_count = 0
        decode_count = 0
        
        for kernel_idx in remaining_indices:
            kernel = all_kernels[kernel_idx]
            
            # 确定这个lm_head属于哪个阶段
            phase = self.determine_kernel_phase(kernel, iteration_data)
            if phase == 'prefill':
                prefill_count += 1
            elif phase == 'decode':
                decode_count += 1
            
            token_size = self.get_kernel_token_size(kernel)
            L = token_size
            
            # LM Head的FLOPS计算
            matmul_flops = 2 * B * L * H * V
            bias_flops = B * L * V
            total_flops = matmul_flops + bias_flops
            
            # 内存访问计算
            memory_read = (B * L * H + H * V + V) * dtype_bytes
            memory_write = B * L * V * dtype_bytes
            total_memory_access = memory_read + memory_write
            
            # 算术强度
            arithmetic_intensity = total_flops / total_memory_access if total_memory_access > 0 else 0
            
            lm_head_performance.append({
                'kernel_idx': kernel_idx,
                'kernel_name': kernel['name'],
                # 修复：使用'dur'字段而不是'duration_us'
                'duration_us': kernel.get('dur', 0),
                'token_size': token_size,
                'matrix_dims': [B, L, H, V],
                'flops': total_flops,
                'matmul_flops': matmul_flops,
                'bias_flops': bias_flops,
                'memory_access': total_memory_access,
                'arithmetic_intensity': arithmetic_intensity,
                'group_id': 'lm_head',
                'operation_type': 'lm_head_projection',
                'tp_size': tp_size,
                'phase': phase,
                'original_vocab_size': vocab_size
            })
            
            print(f"  LM Head[{kernel_idx}] ({phase}): token_size={token_size}, FLOPS={total_flops/1e9:.2f}G")
        
        print(f"LM Head阶段分布: prefill={prefill_count}, decode={decode_count}")
        
        return lm_head_performance
    
    def analyze_linear_performance(self, all_kernels: List[Dict], model_config: Dict, 
                                 hardware_spec: Dict, iteration_data: Dict) -> Dict:
        """分析SGLang Linear算子性能 - 基于act_and_mul的3+1模式"""
        
        print(f"\n=== SGLang Linear算子性能分析 (基于act_and_mul的3+1模式) ===")
        
        # 从hardware_spec中获取TP配置并设置为类属性
        self.tp_size = hardware_spec.get('tensor_parallelism', 1)
        self.hardware_spec = hardware_spec
        
        # 验证必需的模型配置
        required_configs = ['hidden_size', 'intermediate_size', 'vocab_size']
        for config_key in required_configs:
            if config_key not in model_config:
                raise ValueError(f"模型配置中缺少必需参数: {config_key}")
        
        # 验证注意力头数参数
        if 'num_attention_heads' not in model_config and 'num_heads' not in model_config:
            raise ValueError("模型配置中缺少 num_attention_heads 或 num_heads")
        
        print(f"SGLang框架配置:")
        print(f"  - hidden_size: {model_config['hidden_size']}")
        print(f"  - intermediate_size: {model_config['intermediate_size']}")
        print(f"  - vocab_size: {model_config['vocab_size']}")
        print(f"  - tensor_parallelism: {self.tp_size}")
        
        # 1. 寻找Transformer层模式 (基于act_and_mul的3+1模式)
        valid_layers, remaining_linear = self.find_transformer_layers_3plus1(all_kernels, iteration_data)
        
        if not valid_layers:
            print("警告: 没有找到有效的SGLang Transformer层")
            return {
                'layers': [],
                'remaining_linear': {
                    'lm_head': self.calculate_lm_head_performance(remaining_linear, all_kernels, model_config, iteration_data)
                },
                'summary': {
                    'total_layers': 0,
                    'total_lm_head': len(remaining_linear),
                    'framework': 'sglang',
                    'detection_mode': '3plus1_act_and_mul',
                    'act_and_mul_anchors': 0,  # 修复KeyError
                    'phase_distribution': {
                        'prefill_layers': 0,
                        'decode_layers': 0
                    }
                }
            }
        
        # 2. 计算每个层的projection性能
        layer_results = []
        for layer_info in valid_layers:
            print(f"\n--- 计算层 {layer_info['layer_id']} 性能 ---")
            layer_performance = self.calculate_projection_performance_3plus1(layer_info, all_kernels, model_config)
            
            layer_result = {
                'layer_info': layer_info,
                'projections': layer_performance
            }
            layer_results.append(layer_result)
        
        # 3. 计算剩余Linear kernels性能 (LM Head)
        remaining_performance = {
            'lm_head': self.calculate_lm_head_performance(remaining_linear, all_kernels, model_config, iteration_data)
        }
        
        # 新增：计算projection性能汇总统计
        print(f"\n=== 计算Projection性能汇总 ===")
        projection_summary = {}
        proj_names = ['qkv_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head']
        
        for proj_name in proj_names:
            executions = []
            
            # 收集该projection的所有执行实例
            if proj_name == 'lm_head':
                executions = remaining_performance['lm_head']
            else:
                for layer_result in layer_results:
                    if proj_name in layer_result['projections']:
                        executions.append(layer_result['projections'][proj_name])
            
            if executions:
                total_flops = sum(ex['flops'] for ex in executions)
                total_memory_access = sum(ex['memory_access'] for ex in executions)
                total_duration_us = sum(ex['duration_us'] for ex in executions)
                avg_ai = total_flops / total_memory_access if total_memory_access > 0 else 0
                
                # 按阶段统计
                prefill_execs = [ex for ex in executions if ex.get('phase') == 'prefill']
                decode_execs = [ex for ex in executions if ex.get('phase') == 'decode']
                
                projection_summary[proj_name] = {
                    'execution_count': len(executions),
                    'total_flops': total_flops,
                    'total_memory_access': total_memory_access,
                    'total_duration_us': total_duration_us,
                    'average_arithmetic_intensity': avg_ai,
                    'tflops': total_flops / 1e12,
                    'phase_distribution': {
                        'prefill_count': len(prefill_execs),
                        'decode_count': len(decode_execs),
                        'prefill_flops': sum(ex['flops'] for ex in prefill_execs),
                        'decode_flops': sum(ex['flops'] for ex in decode_execs)
                    }
                }
                
                print(f"{proj_name}: {len(executions)} 次执行, "
                      f"总FLOPS={total_flops/1e12:.2f}T, "
                      f"平均AI={avg_ai:.2f}, "
                      f"prefill={len(prefill_execs)}, decode={len(decode_execs)}")
        
        # 计算整体性能指标
        total_flops = sum(proj['total_flops'] for proj in projection_summary.values())
        total_memory_access = sum(proj['total_memory_access'] for proj in projection_summary.values())
        total_duration_us = sum(proj['total_duration_us'] for proj in projection_summary.values())
        
        overall_ai = total_flops / total_memory_access if total_memory_access > 0 else 0
        flops_per_second = total_flops / (total_duration_us * 1e-6) if total_duration_us > 0 else 0
        memory_bandwidth = total_memory_access / (total_duration_us * 1e-6) if total_duration_us > 0 else 0
        
        # 计算效率（相对于硬件峰值）
        peak_flops = hardware_spec.get('peak_flops_fp16', 312e12)  # 默认A100的312 TFLOPs
        peak_bandwidth = hardware_spec.get('memory_bandwidth', 2e12)  # 默认2TB/s
        
        efficiency_compute = min(1.0, flops_per_second / peak_flops) if peak_flops > 0 else 0
        efficiency_memory = min(1.0, memory_bandwidth / peak_bandwidth) if peak_bandwidth > 0 else 0
        
        print(f"\n=== 整体性能指标 ===")
        print(f"总FLOPS: {total_flops/1e12:.2f} TFLOPs")
        print(f"实际算力: {flops_per_second/1e12:.2f} TFLOPs/s")
        print(f"内存带宽: {memory_bandwidth/1e9:.2f} GB/s")
        print(f"整体算术强度: {overall_ai:.2f}")
        print(f"计算效率: {efficiency_compute*100:.1f}%")
        print(f"内存效率: {efficiency_memory*100:.1f}%")
        
        # 4. 生成分析结果
        analysis_result = {
            'layers': layer_results,
            'remaining_linear': remaining_performance,
            'projection_summary': projection_summary,
            'overall_performance': {
                'total_flops': total_flops,
                'total_memory_access': total_memory_access,
                'total_duration_us': total_duration_us,
                'overall_arithmetic_intensity': overall_ai,
                'tflops_per_second': flops_per_second / 1e12,
                'memory_bandwidth_gb_s': memory_bandwidth / 1e9,
                'efficiency_compute': efficiency_compute,
                'efficiency_memory': efficiency_memory
            },
            'summary': {
                'total_layers': len(valid_layers),
                'total_lm_head': len(remaining_linear),
                'framework': 'sglang',
                'detection_mode': '3plus1_act_and_mul',
                'act_and_mul_anchors': len([layer['anchor_idx'] for layer in valid_layers]),
                'phase_distribution': {
                    'prefill_layers': len([layer for layer in valid_layers if layer['phase'] == 'prefill']),
                    'decode_layers': len([layer for layer in valid_layers if layer['phase'] == 'decode'])
                }
            }
        }
        
        print(f"\n=== SGLang分析完成 ===")
        print(f"检测到 {len(valid_layers)} 个Transformer层")
        print(f"检测到 {len(remaining_linear)} 个LM Head")
        print(f"阶段分布: prefill层={analysis_result['summary']['phase_distribution']['prefill_layers']}, "
              f"decode层={analysis_result['summary']['phase_distribution']['decode_layers']}")
        
        return analysis_result
    
    def load_stage1_data(self, input_file: str) -> Dict:
        """加载stage1处理的数据"""
        
        print(f"=== 加载Stage1数据 ===")
        print(f"输入文件: {input_file}")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"找不到输入文件: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"成功加载Stage1数据")
        
        # 验证数据结构
        required_keys = ['case_info', 'hardware_spec', 'phase_analysis', 'iteration_analysis']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Stage1数据缺少必需字段: {key}")
        
        return data
    
    def process_stage1_data(self, input_file: str) -> Dict:
        """处理stage1数据，进行Linear算子分析"""
        
        # 1. 加载stage1数据
        stage1_data = self.load_stage1_data(input_file)
        
        # 2. 提取必要信息
        case_info = stage1_data['case_info']
        hardware_spec = stage1_data['hardware_spec']
        phase_analysis = stage1_data['phase_analysis']
        iteration_analysis = stage1_data['iteration_analysis']
        
        # 3. 合并所有阶段的kernels - 修复数据结构
        all_kernels = []
        for phase, analysis in phase_analysis.items():
            if 'operator_stats' in analysis:  # 修复：使用operator_stats而不是operators
                for op_type, op_data in analysis['operator_stats'].items():
                    if 'kernels' in op_data:
                        for kernel in op_data['kernels']:
                            kernel['phase'] = phase
                            kernel['operator_type'] = op_type
                            all_kernels.append(kernel)
        
        print(f"合并得到 {len(all_kernels)} 个kernels")
        
        # 4. 从case_info中提取seq_len并设置为实例属性
        self.seq_len = case_info.get('seq_len', 2048)  # 默认2048
        print(f"设置seq_len: {self.seq_len}")
        
        # 5. 构建模型配置 - 从模型名称解析
        model_name = case_info.get('model_name', 'Unknown')
        model_config = self.get_model_config_from_name(model_name)
        
        print(f"使用模型配置: {model_config}")
        
        # 6. 进行Linear算子分析
        linear_analysis = self.analyze_linear_performance(
            all_kernels, model_config, hardware_spec, iteration_analysis
        )
        
        # 7. 构建完整结果
        result = {
            'stage1_data': stage1_data,
            'linear_analysis': linear_analysis,
            'model_config': model_config,
            'analysis_metadata': {
                'framework': 'sglang',
                'detection_method': 'act_and_mul_3plus1',
                'total_kernels_analyzed': len(all_kernels),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return result
    
    def save_analysis_result(self, result: Dict, output_file: str):
        """保存分析结果"""
        
        print(f"=== 保存分析结果 ===")
        print(f"输出文件: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"SGLang Linear分析结果已保存")

def main():
    parser = argparse.ArgumentParser(description='SGLang Linear算子分析器 (基于act_and_mul的3+1模式)')
    parser.add_argument('--input', required=True, help='Stage1处理的JSON文件路径')
    parser.add_argument('--output', help='输出文件路径（可选）')
    
    args = parser.parse_args()
    
    try:
        # 创建分析器
        analyzer = SGLangLinearAnalyzer()
        
        # 处理数据
        result = analyzer.process_stage1_data(args.input)
        
        # 生成输出文件名和路径
        if args.output:
            output_file = args.output
        else:
            # 修复：从result['stage1_data']中获取pod_name
            pod_name = result.get('stage1_data', {}).get('case_info', {}).get('pod_name', 'unknown')
            # 获取输入文件所在目录
            input_dir = os.path.dirname(args.input)
            # 生成输出文件路径：输入文件所在文件夹/oea_stage2_pod_name_processed.json
            output_file = os.path.join(input_dir, f"oea_stage2_{pod_name}_processed.json")
        
        # 保存结果
        analyzer.save_analysis_result(result, output_file)
        
        print(f"\n=== SGLang Linear分析完成 ===")
        print(f"输入文件: {args.input}")
        print(f"输出文件: {output_file}")
        
        # 打印简要统计
        linear_analysis = result['linear_analysis']
        summary = linear_analysis['summary']
        
        print(f"检测结果:")
        print(f"  - 框架: {summary['framework']}")
        print(f"  - 检测模式: {summary['detection_mode']}")
        print(f"  - Transformer层: {summary['total_layers']}")
        print(f"  - LM Head: {summary['total_lm_head']}")
        print(f"  - act_and_mul anchors: {summary['act_and_mul_anchors']}")
        
        if 'phase_distribution' in summary:
            phase_dist = summary['phase_distribution']
            print(f"  - 阶段分布: prefill={phase_dist['prefill_layers']}, decode={phase_dist['decode_layers']}")
        
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()