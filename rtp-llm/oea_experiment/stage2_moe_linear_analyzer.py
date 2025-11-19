#!/usr/bin/env python3
"""
OEA Stage 2: MoE Linear算子专门分析脚本（时间范围修复版本）
基于原有stage2_moe_linear_analyzer.py，专门负责MoE模型Linear算子的精确分析
使用每个kernel的精确token size值而不是平均值，并修复时间计算逻辑

主要改进:
1. 使用每个kernel的matched_token_size而不是平均token_size
2. 修复时间计算：使用正确的时间范围而不是单个kernel时间
3. qkv_proj: 从整个层的norm算子开始到第1个linear算子结束
4. o_proj: 第2个linear开始 → 第3个linear开始
5. router: 第3个linear开始 → 第4个linear开始
6. up_proj: 第4个linear开始 → 第5个linear开始
7. down_proj: 第5个linear开始 → 最终add_bias开始
8. 剩余linear: 时间保持不变

使用方法:
python stage2_moe_linear_analyzer_time_range_fixed.py --input stage1_processed_data_fixed.json
"""

import json
import numpy as np
import argparse
import os
import sys
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple

class MoELinearAnalyzerTimeRangeFixed:
    def __init__(self):
        """初始化MoE Linear算子分析器（时间范围修复版本）"""
        
        # Linear算子匹配模式 - 排除splitKreduce
        self.linear_patterns = [
            r'nvjet.*',
            r'.*gemm.*',
            r'.*sgemm.*',
            r'.*hgemm.*',
            r'.*gemvt.*',
            r'.*gemvn.*',
            r'.*cutlass.*gemm.*',
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
    
    def get_kernel_token_size(self, kernel: Dict) -> int:
        """获取kernel的token size - 严格使用matched_token_size"""
        # 严格使用matched_token_size字段
        if 'matched_token_size' in kernel:
            return int(kernel['matched_token_size'])
        else:
            raise ValueError(f"Kernel '{kernel.get('name', 'unknown')}' 缺少 'matched_token_size' 字段，无法计算性能")

    
    def is_linear_kernel(self, kernel_name: str) -> bool:
        """判断是否为Linear kernel - 排除splitKreduce等，使用预编译正则表达式和缓存"""
        if kernel_name in self.classification_cache:
            return self.classification_cache[kernel_name] == 'linear'
        
        # 首先检查排除模式
        for pattern in self.compiled_linear_exclude_patterns:
            if pattern.search(kernel_name):
                self.classification_cache[kernel_name] = 'excluded'
                return False
        
        # 然后检查匹配模式
        for pattern in self.compiled_linear_patterns:
            if pattern.search(kernel_name):
                self.classification_cache[kernel_name] = 'linear'
                return True
        
        self.classification_cache[kernel_name] = 'other'
        return False
    
    def is_norm_kernel(self, kernel_name: str) -> bool:
        """判断是否为Normalization kernel，使用预编译正则表达式和缓存"""
        if kernel_name in self.classification_cache:
            return self.classification_cache[kernel_name] == 'norm'
        
        # 首先检查排除模式 - 如果匹配排除模式，直接返回False
        for pattern in self.compiled_norm_exclude_patterns:
            if pattern.search(kernel_name):
                self.classification_cache[kernel_name] = 'excluded'
                return False
        
        # 然后检查匹配模式
        for pattern in self.compiled_norm_patterns:
            if pattern.search(kernel_name):
                self.classification_cache[kernel_name] = 'norm'
                return True
        
        self.classification_cache[kernel_name] = 'other'
        return False
    
    def is_add_bias_kernel(self, kernel_name: str) -> bool:
        """判断是否为add_bias kernel，使用预编译正则表达式和缓存"""
        if kernel_name in self.classification_cache:
            return self.classification_cache[kernel_name] == 'add_bias'
        
        for pattern in self.compiled_add_bias_patterns:
            if pattern.search(kernel_name):
                self.classification_cache[kernel_name] = 'add_bias'
                return True
        
        return False
    
    def is_generalrms_norm_kernel(self, kernel_name: str) -> bool:
        """严格判断是否为generalRmsNorm kernel，排除fusedQkRmsNorm等fused类型"""
        if kernel_name in self.classification_cache:
            return self.classification_cache[kernel_name] == 'generalrms_norm'
        
        kernel_name_lower = kernel_name.lower()
        
        # 首先检查排除模式 - 如果匹配排除模式，直接返回False
        exclude_patterns = [
            r'.*fused.*',  # 排除所有fused类型
            r'.*fusedqk.*',  # 排除fusedQkRmsNorm等
            r'.*qk.*norm.*',  # 排除qk相关的norm
            r'.*attention.*norm.*'  # 排除attention相关的norm
        ]
        
        for pattern in exclude_patterns:
            if re.search(pattern, kernel_name_lower):
                self.classification_cache[kernel_name] = 'excluded_norm'
                return False
        
        # 然后检查是否为generalRmsNorm
        generalrms_patterns = [
            r'^.*generalrmsnorm.*$',  # 严格匹配 generalRmsNorm
            r'^.*general_rms_norm.*$'  # 匹配 general_rms_norm 变体
        ]
        
        for pattern in generalrms_patterns:
            if re.search(pattern, kernel_name_lower):
                self.classification_cache[kernel_name] = 'generalrms_norm'
                return True
        
        self.classification_cache[kernel_name] = 'other_norm'
        return False
    
    def find_transformer_layers(self, all_kernels: List[Dict]) -> Tuple[List[Dict], List[int]]:
        """
        寻找符合Transformer层模式的kernel组合
        模式: generalRmsNorm → 2个Linear → generalRmsNorm → 3个Linear(MoE专家) → add_bias
        强调：只有连续的两个generalRmsNorm和add_bias才能作为层分隔
        """
        
        # 直接使用Stage 1的分类结果，避免重复的正则表达式匹配
        norm_indices = []
        linear_indices = []
        add_bias_indices = []
        
        print("正在分类kernels...")
        for i, kernel in enumerate(all_kernels):
            if i % 10000 == 0:  # 每处理10000个kernel显示进度
                print(f"  处理进度: {i}/{len(all_kernels)} ({i/len(all_kernels)*100:.1f}%)")
            
            # 优先使用Stage 1的分类结果
            kernel_type = kernel.get('operator_type', '')
            kernel_name = kernel.get('name', '')
            
            if kernel_type == 'layernorm':
                # 对于layernorm类型，进一步严格检查是否为generalRmsNorm
                if self.is_generalrms_norm_kernel(kernel_name):
                    norm_indices.append(i)
                    print(f"  找到generalRmsNorm[{i}]: {kernel_name}")
            elif kernel_type == 'linear':
                linear_indices.append(i)
            elif kernel_type == 'add_bias':
                add_bias_indices.append(i)
                print(f"  找到add_bias[{i}]: {kernel_name}")
            else:
                # 只有当Stage 1没有分类时才使用正则表达式（作为备用）
                if self.is_generalrms_norm_kernel(kernel_name):
                    norm_indices.append(i)
                    print(f"  找到generalRmsNorm[{i}] (备用匹配): {kernel_name}")
                elif self.is_linear_kernel(kernel_name):
                    linear_indices.append(i)
                elif self.is_add_bias_kernel(kernel_name):
                    add_bias_indices.append(i)
                    print(f"  找到add_bias[{i}] (备用匹配): {kernel_name}")
        
        print(f"分类完成!")
        print(f"找到 {len(norm_indices)} 个generalRmsNorm kernels (严格匹配)")
        print(f"找到 {len(linear_indices)} 个Linear kernels (排除splitKreduce)")
        print(f"找到 {len(add_bias_indices)} 个add_bias kernels")
        
        if len(norm_indices) < 2 or len(add_bias_indices) < 1 or len(linear_indices) < 5:
            print("算子数量不足，无法进行精确分组")
            return [], linear_indices
        
        # 寻找符合模式的层
        valid_layers = []
        used_linear_indices = set()
        search_start_idx = 0  # 从这个索引开始搜索下一个层
        
        print(f"\n=== 寻找MoE Transformer层模式 (严格generalRmsNorm分隔) ===")
        
        while search_start_idx < len(all_kernels):
            # 找到搜索起始位置之后的第一个generalRmsNorm
            norm1_candidates = [idx for idx in norm_indices if idx >= search_start_idx]
            if len(norm1_candidates) < 2:
                break
            
            norm1_idx = norm1_candidates[0]
            
            # 找到第一个generalRmsNorm之后的第二个generalRmsNorm
            norm2_candidates = [idx for idx in norm_indices if idx > norm1_idx]
            if not norm2_candidates:
                break
            
            norm2_idx = norm2_candidates[0]
            
            # 找到第二个generalRmsNorm之后最近的add_bias
            add_bias_candidates = [idx for idx in add_bias_indices if idx > norm2_idx]
            if not add_bias_candidates:
                # 如果没有add_bias，跳到下一个norm继续搜索
                search_start_idx = norm2_idx + 1
                continue
            
            add_bias_idx = min(add_bias_candidates)
            
            # 找到两个区间的Linear kernels（只考虑未使用的）
            linear_segment1 = [idx for idx in linear_indices 
                              if norm1_idx < idx < norm2_idx and idx not in used_linear_indices]
            linear_segment2 = [idx for idx in linear_indices 
                              if norm2_idx < idx < add_bias_idx and idx not in used_linear_indices]
            
            # 检查是否符合2+3模式
            if len(linear_segment1) == 2 and len(linear_segment2) == 3:
                layer_info = {
                    'layer_id': len(valid_layers),
                    'norm1_idx': norm1_idx,
                    'norm2_idx': norm2_idx,
                    'add_bias_idx': add_bias_idx,
                    'attention_linear': linear_segment1,
                    'mlp_linear': linear_segment2,
                    'start_time': all_kernels[norm1_idx]['timestamp_us'],
                    'end_time': all_kernels[add_bias_idx]['timestamp_us'] + all_kernels[add_bias_idx]['duration_us'],
                    'pattern': f"generalRmsNorm[{norm1_idx}] → Linear{linear_segment1} → generalRmsNorm[{norm2_idx}] → Linear{linear_segment2} → AddBias[{add_bias_idx}]"
                }
                
                valid_layers.append(layer_info)
                print(f"✓ 层 {len(valid_layers)}: {layer_info['pattern']}")
                
                # 标记已使用的Linear kernels
                for idx in linear_segment1:
                    used_linear_indices.add(idx)
                for idx in linear_segment2:
                    used_linear_indices.add(idx)
                
                # 下次搜索从add_bias之后的第一个generalRmsNorm开始
                next_norm_candidates = [idx for idx in norm_indices if idx > add_bias_idx]
                if next_norm_candidates:
                    search_start_idx = next_norm_candidates[0]
                    print(f"  成功匹配，下次搜索从下一个generalRmsNorm索引 {search_start_idx} 开始")
                else:
                    search_start_idx = len(all_kernels)  # 没有更多norm，结束搜索
                    print(f"  成功匹配，没有更多generalRmsNorm，搜索结束")
            else:
                print(f"× 不符合模式 (需要2+3，实际{len(linear_segment1)}+{len(linear_segment2)})")
                # 不符合模式，从第二个generalRmsNorm重新开始搜索（第二个norm作为新的第一个norm）
                search_start_idx = norm2_idx
                print(f"  不匹配，下次搜索从第二个generalRmsNorm索引 {search_start_idx} 重新开始")
        
        print(f"\n总共找到 {len(valid_layers)} 个有效的MoE Transformer层")
        
        # 处理剩余未分配的Linear kernels
        remaining_linear = [idx for idx in linear_indices if idx not in used_linear_indices]
        print(f"剩余未分配的Linear kernels: {len(remaining_linear)} 个")
        
        # 记录剩余Linear kernels的详细信息
        if remaining_linear:
            print("剩余Linear kernels详情:")
            for idx in remaining_linear:
                kernel = all_kernels[idx]
                print(f"  [{idx}] {kernel['name']}")
        
        return valid_layers, remaining_linear
    
    def calculate_projection_performance(self, layer_info: Dict, all_kernels: List[Dict], 
                                       model_config: Dict) -> Dict:
        """计算单个MoE Transformer层中各projection的性能 - 修复：使用精确的时间范围计算"""
        
        # 从实际模型配置中读取参数
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
        
        # MoE特有参数
        num_experts = model_config.get("num_experts", 128)
        num_experts_per_tok = model_config.get("num_experts_per_tok", 8)
        moe_intermediate_size = model_config.get("moe_intermediate_size", intermediate_size)
        
        print(f"  MoE配置: num_experts={num_experts}, num_experts_per_tok={num_experts_per_tok}")
        print(f"  MoE中间层大小: {moe_intermediate_size}")
        
        head_dim = hidden_size // num_heads
        dtype_bytes = 2  # bfloat16
        
        print(f"  模型参数: hidden_size={hidden_size}, intermediate_size={intermediate_size}")
        print(f"  注意力参数: num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
        
        # 获取关键时间点
        norm1_idx = layer_info['norm1_idx']
        attention_linear_indices = layer_info['attention_linear']  # 2个kernels
        mlp_linear_indices = layer_info['mlp_linear']  # 3个kernels
        add_bias_idx = layer_info['add_bias_idx']
        
        # 确保有5个Linear kernels
        all_linear_indices = attention_linear_indices + mlp_linear_indices
        if len(all_linear_indices) != 5:
            raise ValueError(f"期望5个Linear kernels，实际找到{len(all_linear_indices)}个")
        
        # 获取各个关键时间点
        norm1_start = all_kernels[norm1_idx]['timestamp_us']  # 层开始的norm算子开始
        linear1_start = all_kernels[all_linear_indices[0]]['timestamp_us']  # 第1个linear开始
        linear1_end = linear1_start + all_kernels[all_linear_indices[0]]['duration_us']  # 第1个linear结束
        linear2_start = all_kernels[all_linear_indices[1]]['timestamp_us']  # 第2个linear开始
        linear3_start = all_kernels[all_linear_indices[2]]['timestamp_us']  # 第3个linear开始
        linear4_start = all_kernels[all_linear_indices[3]]['timestamp_us']  # 第4个linear开始
        linear5_start = all_kernels[all_linear_indices[4]]['timestamp_us']  # 第5个linear开始
        add_bias_start = all_kernels[add_bias_idx]['timestamp_us']  # add_bias开始
        
        # 计算各projection的时间范围（按用户要求修改）
        qkv_duration_us = linear1_end - norm1_start  # 修复：整个层的norm算子开始 → 第1个linear算子结束
        o_duration_us = linear3_start - linear2_start    # 第2个linear开始 → 第3个linear开始
        router_duration_us = linear4_start - linear3_start  # 第3个linear开始 → 第4个linear开始
        up_duration_us = linear5_start - linear4_start   # 第4个linear开始 → 第5个linear开始
        down_duration_us = add_bias_start - linear5_start  # 第5个linear开始 → 最终add_bias开始
        
        print(f"  时间范围计算:")
        print(f"    qkv_proj: {qkv_duration_us}μs (Norm1开始 → Linear1结束)")
        print(f"    o_proj: {o_duration_us}μs (Linear2开始 → Linear3开始)")
        print(f"    router: {router_duration_us}μs (Linear3开始 → Linear4开始)")
        print(f"    up_proj: {up_duration_us}μs (Linear4开始 → Linear5开始)")
        print(f"    down_proj: {down_duration_us}μs (Linear5开始 → AddBias开始)")
        
        # 分析各个Linear projection
        projections = {}
        
        # 1. QKV projection - 修复：从norm开始到第1个linear结束
        qkv_kernel = all_kernels[all_linear_indices[0]]
        token_size = self.get_kernel_token_size(qkv_kernel)
        
        qkv_flops = 2 * token_size * hidden_size * (num_heads * head_dim + 2 * num_kv_heads * head_dim)
        qkv_memory = token_size * hidden_size * dtype_bytes + (num_heads * head_dim + 2 * num_kv_heads * head_dim) * hidden_size * dtype_bytes + token_size * (num_heads * head_dim + 2 * num_kv_heads * head_dim) * dtype_bytes
        
        projections['qkv_proj'] = {
            'kernel_name': qkv_kernel['name'],
            'kernel_idx': all_linear_indices[0],
            'duration_us': qkv_duration_us,  # 修复：从norm开始到linear1结束
            'token_size': token_size,
            'flops': qkv_flops,
            'memory_bytes': qkv_memory,
            'arithmetic_intensity': qkv_flops / qkv_memory,
            'tflops': qkv_flops / (qkv_duration_us * 1e6) if qkv_duration_us > 0 else 0,
            'memory_bandwidth_gbps': qkv_memory / (qkv_duration_us * 1e3) if qkv_duration_us > 0 else 0,
            'time_range': f"{norm1_start}-{linear1_end}μs"
        }
        
        # 2. O projection - 使用精确时间范围
        o_kernel = all_kernels[all_linear_indices[1]]
        token_size = self.get_kernel_token_size(o_kernel)
        
        o_flops = 2 * token_size * num_heads * head_dim * hidden_size
        o_memory = token_size * num_heads * head_dim * dtype_bytes + num_heads * head_dim * hidden_size * dtype_bytes + token_size * hidden_size * dtype_bytes
        
        projections['o_proj'] = {
            'kernel_name': o_kernel['name'],
            'kernel_idx': all_linear_indices[1],
            'duration_us': o_duration_us,
            'token_size': token_size,
            'flops': o_flops,
            'memory_bytes': o_memory,
            'arithmetic_intensity': o_flops / o_memory,
            'tflops': o_flops / (o_duration_us * 1e6) if o_duration_us > 0 else 0,
            'memory_bandwidth_gbps': o_memory / (o_duration_us * 1e3) if o_duration_us > 0 else 0,
            'time_range': f"{linear2_start}-{linear3_start}μs"
        }
        
        # 3. Router projection - 使用精确时间范围
        router_kernel = all_kernels[all_linear_indices[2]]
        token_size = self.get_kernel_token_size(router_kernel)
        
        router_flops = 2 * token_size * hidden_size * num_experts
        router_memory = (
            token_size * hidden_size * dtype_bytes +
            num_experts * hidden_size * dtype_bytes +
            token_size * num_experts * dtype_bytes
        )
        
        projections['router'] = {
            'kernel_name': router_kernel['name'],
            'kernel_idx': all_linear_indices[2],
            'duration_us': router_duration_us,
            'token_size': token_size,
            'flops': router_flops,
            'memory_bytes': router_memory,
            'arithmetic_intensity': router_flops / router_memory,
            'tflops': router_flops / (router_duration_us * 1e6) if router_duration_us > 0 else 0,
            'memory_bandwidth_gbps': router_memory / (router_duration_us * 1e3) if router_duration_us > 0 else 0,
            'time_range': f"{linear3_start}-{linear4_start}μs"
        }
        
        # 4. Up projection - 使用精确时间范围
        up_kernel = all_kernels[all_linear_indices[3]]
        token_size = self.get_kernel_token_size(up_kernel)
        total_active_tokens = token_size * num_experts_per_tok
        
        up_flops = 2 * total_active_tokens * hidden_size * moe_intermediate_size
        up_memory = (
            token_size * hidden_size * dtype_bytes +
            num_experts * hidden_size * moe_intermediate_size * dtype_bytes +
            token_size * num_experts_per_tok * moe_intermediate_size * dtype_bytes
        )
        
        projections['up_proj'] = {
            'kernel_name': up_kernel['name'],
            'kernel_idx': all_linear_indices[3],
            'duration_us': up_duration_us,
            'token_size': token_size,
            'flops': up_flops,
            'memory_bytes': up_memory,
            'arithmetic_intensity': up_flops / up_memory,
            'tflops': up_flops / (up_duration_us * 1e6) if up_duration_us > 0 else 0,
            'memory_bandwidth_gbps': up_memory / (up_duration_us * 1e3) if up_duration_us > 0 else 0,
            'time_range': f"{linear4_start}-{linear5_start}μs"
        }
        
        # 5. Down projection - 使用精确时间范围
        down_kernel = all_kernels[all_linear_indices[4]]
        token_size = self.get_kernel_token_size(down_kernel)
        total_active_tokens = token_size * num_experts_per_tok
        
        down_flops = 2 * total_active_tokens * moe_intermediate_size * hidden_size
        down_memory = (
            total_active_tokens * moe_intermediate_size * dtype_bytes +
            num_experts_per_tok * moe_intermediate_size * hidden_size * dtype_bytes +
            token_size * hidden_size * dtype_bytes
        )
        
        projections['down_proj'] = {
            'kernel_name': down_kernel['name'],
            'kernel_idx': all_linear_indices[4],
            'duration_us': down_duration_us,
            'token_size': token_size,
            'flops': down_flops,
            'memory_bytes': down_memory,
            'arithmetic_intensity': down_flops / down_memory,
            'tflops': down_flops / (down_duration_us * 1e6) if down_duration_us > 0 else 0,
            'memory_bandwidth_gbps': down_memory / (down_duration_us * 1e3) if down_duration_us > 0 else 0,
            'time_range': f"{linear5_start}-{add_bias_start}μs"
        }
        
        print(f"    qkv_proj: {qkv_duration_us}μs, token_size={token_size}, FLOPS={qkv_flops/1e9:.2f}G, TFLOPS={projections['qkv_proj']['tflops']:.2f}")
        print(f"    o_proj: {o_duration_us}μs, token_size={token_size}, FLOPS={o_flops/1e9:.2f}G, TFLOPS={projections['o_proj']['tflops']:.2f}")
        print(f"    router: {router_duration_us}μs, token_size={token_size}, FLOPS={router_flops/1e9:.2f}G, TFLOPS={projections['router']['tflops']:.2f}")
        print(f"    up_proj: {up_duration_us}μs, token_size={token_size}, FLOPS={up_flops/1e9:.2f}G, TFLOPS={projections['up_proj']['tflops']:.2f}")
        print(f"    down_proj: {down_duration_us}μs, token_size={token_size}, FLOPS={down_flops/1e9:.2f}G, TFLOPS={projections['down_proj']['tflops']:.2f}")
        
        return projections

    def calculate_lm_head_performance(self, remaining_indices: List[int], 
                                     all_kernels: List[Dict], 
                                     model_config: Dict) -> List[Dict]:
        """计算LM Head kernels的性能（使用精确token size，时间保持不变）"""
        
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
        
        print(f"\n=== LM Head性能计算 (使用精确token size，时间保持不变) ===")
        print(f"模型配置: hidden_size={H}, vocab_size={vocab_size}, tp_size={tp_size}")
        print(f"注意: 在TP={tp_size}模式下，词汇表维度被分割为 {V}")
        
        lm_head_performance = []
        
        for kernel_idx in remaining_indices:
            kernel = all_kernels[kernel_idx]
            num_experts_per_tok = model_config.get("num_experts_per_tok", 8)

            # 关键修改：使用每个kernel的精确token size
            token_size = self.get_kernel_token_size(kernel)
            L = token_size  # 使用精确的token size
            
            # 剩余linear时间保持不变 - 使用kernel自身的duration
            kernel_duration_us = kernel.get('duration_us', 0)
            
            print(f"  Kernel[{kernel_idx}]: token_size={token_size}, duration={kernel_duration_us}μs")
            print(f"    矩阵维度: hidden_states[{B}, {L}, {H}] × W_lm[{H}, {V}] → logits[{B}, {L}, {V}]")
            
            # LM Head的FLOPS计算: logits = hidden_states × W_lm + b_lm
            # 矩阵乘法FLOPS: 2 * B * L * H * V (每个输出元素需要H次乘法和H次加法)
            # 偏置加法FLOPS: B * L * V (每个输出元素加一次偏置)
            matmul_flops = 2 * B * L * H * V / num_experts_per_tok
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
                'duration_us': kernel_duration_us,  # 剩余linear时间保持不变
                'token_size': token_size,  # 新增：记录使用的精确token size
                'matrix_dims': [B, L, H, V],  # [batch, seq_len, hidden, vocab_per_tp]
                'flops': total_flops,
                'matmul_flops': matmul_flops,
                'bias_flops': bias_flops,
                'memory_access': total_memory_access,
                'memory_read': memory_read,
                'memory_write': memory_write,
                'arithmetic_intensity': arithmetic_intensity,
                'tflops': total_flops / (kernel_duration_us * 1e6) if kernel_duration_us > 0 else 0,
                'memory_bandwidth_gbps': total_memory_access / (kernel_duration_us * 1e3) if kernel_duration_us > 0 else 0,
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
            print(f"    TFLOPS: {total_flops / (kernel_duration_us * 1e6) / 1e12:.2f}")
        
        return lm_head_performance
    
    def calculate_remaining_linear_performance(self, remaining_indices: List[int], 
                                             all_kernels: List[Dict], 
                                             model_config: Dict) -> Dict:
        """计算剩余Linear kernels的性能（现在专门处理LM Head，使用精确token size，时间保持不变）"""
        
        if not remaining_indices:
            return {}
        
        print(f"\n=== 剩余Linear算子分析 (使用精确token size，时间保持不变) ===")
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
        print(f"将所有剩余Linear kernels按LM Head方式计算（使用精确token size，时间保持不变）")
        lm_head_performance = self.calculate_lm_head_performance(remaining_indices, all_kernels, model_config)
        
        return {
            'lm_head': lm_head_performance
        }

    def analyze_linear_performance(self, processed_data: Dict) -> Dict:
        """分析Linear算子性能 - 使用精确token size和修复的时间范围计算"""
        
        print("=== OEA Stage 2: MoE Linear算子性能分析 (时间范围修复版本) ===")
        
        # 获取数据
        metadata = processed_data['metadata']
        operator_kernels = processed_data['operator_kernels']
        
        # 使用metadata中的配置
        model_config = metadata['model_config']
        hardware_spec = metadata['hardware_spec']
        
        # 从hardware_spec中获取TP配置并设置为类属性
        self.tp_size = hardware_spec.get('tensor_parallelism', 1)
        self.hardware_spec = hardware_spec
        
        # 检查是否有token匹配信息
        has_token_matching = 'token_matching' in metadata
        if has_token_matching:
            token_method = metadata['token_matching']['method']
            print(f"检测到token匹配信息: {token_method}")
        else:
            print("未检测到token匹配信息，将使用兼容模式")
        
        print(f"模型: {model_config.get('_name_or_path', 'Unknown')}")
        print(f"架构: {model_config.get('architectures', ['Unknown'])[0]}")
        print(f"Tensor Parallelism: {self.tp_size}")
        
        # 直接使用Stage 1分类好的kernels，并添加算子类型标记
        all_kernels = []
        for op_type, kernels in operator_kernels.items():
            for kernel in kernels:
                kernel['operator_type'] = op_type  # 添加算子类型标记
                all_kernels.append(kernel)
        
        all_kernels.sort(key=lambda x: x['timestamp_us'])
        
        print(f"总kernels数量: {len(all_kernels)}")
        print(f"其中 add_bias kernels: {len(operator_kernels.get('add_bias', []))}")
        print(f"其中 layernorm kernels: {len(operator_kernels.get('layernorm', []))}")
        print(f"其中 linear kernels: {len(operator_kernels.get('linear', []))}")
        
        # 检查token size使用情况
        linear_kernels = [k for k in all_kernels if k.get('operator_type') == 'linear']
        kernels_with_matched_token = [k for k in linear_kernels if 'matched_token_size' in k]
        print(f"其中 {len(kernels_with_matched_token)} 个linear kernels包含精确token size信息")
        
        if kernels_with_matched_token:
            token_sizes = [k['matched_token_size'] for k in kernels_with_matched_token]
            print(f"Token size范围: {min(token_sizes):.0f} - {max(token_sizes):.0f}")
            print(f"平均token size: {np.mean(token_sizes):.1f}")
        
        # 寻找Transformer层
        valid_layers, remaining_linear = self.find_transformer_layers(all_kernels)
        
        if not valid_layers:
            print("未找到有效的MoE Transformer层")
            return {'error': 'No valid MoE transformer layers found'}
        
        # 分析每个层的性能
        layer_analysis = []
        projection_stats = {
            'qkv_proj': {'total_duration_us': 0, 'total_flops': 0, 'total_memory_access': 0, 'executions': 0},
            'o_proj': {'total_duration_us': 0, 'total_flops': 0, 'total_memory_access': 0, 'executions': 0},
            'router': {'total_duration_us': 0, 'total_flops': 0, 'total_memory_access': 0, 'executions': 0},
            'up_proj': {'total_duration_us': 0, 'total_flops': 0, 'total_memory_access': 0, 'executions': 0},
            'down_proj': {'total_duration_us': 0, 'total_flops': 0, 'total_memory_access': 0, 'executions': 0}
        }
        
        for layer_info in valid_layers:
            print(f"\n=== 分析MoE层 {layer_info['layer_id']} ===")
            
            # 计算各projection性能
            projections = self.calculate_projection_performance(
                layer_info, all_kernels, model_config
            )
            
            # 累计各projection统计
            for proj_name, proj_data in projections.items():
                if proj_name in projection_stats:
                    projection_stats[proj_name]['total_duration_us'] += proj_data['duration_us']
                    projection_stats[proj_name]['total_flops'] += proj_data['flops']
                    projection_stats[proj_name]['total_memory_access'] += proj_data['memory_bytes']
                    projection_stats[proj_name]['executions'] += 1
            
            # 计算层级统计
            layer_duration = layer_info['end_time'] - layer_info['start_time']
            total_flops = sum(p.get('flops', 0) for p in projections.values() if isinstance(p, dict))
            total_memory = sum(p.get('memory_bytes', 0) for p in projections.values() if isinstance(p, dict))
            
            layer_stats = {
                'layer_id': layer_info['layer_id'],
                'duration_us': layer_duration,
                'total_flops': total_flops,
                'total_memory_bytes': total_memory,
                'average_arithmetic_intensity': total_flops / total_memory if total_memory > 0 else 0,
                'layer_tflops': total_flops / (layer_duration * 1e6) if layer_duration > 0 else 0,
                'projections': projections
            }
            
            layer_analysis.append(layer_stats)
            
            # 打印层级摘要
            print(f"层 {layer_info['layer_id']} 摘要:")
            print(f"  持续时间: {layer_duration:.0f} μs")
            print(f"  总计算量: {total_flops/1e12:.2f} TFLOPS")
            print(f"  总内存访问: {total_memory/1e9:.2f} GB")
            print(f"  平均算术强度: {total_flops/total_memory:.2f} FLOPS/Byte")
            print(f"  层级性能: {total_flops/(layer_duration*1e6):.2f} TFLOPS")
        
        # 处理剩余的Linear kernels（现在专门处理LM Head）
        remaining_perf = self.calculate_remaining_linear_performance(remaining_linear, all_kernels, model_config)
        
        # 计算整体统计
        total_layers = len(layer_analysis)
        avg_layer_duration = np.mean([l['duration_us'] for l in layer_analysis])
        avg_layer_tflops = np.mean([l['layer_tflops'] for l in layer_analysis])
        avg_arithmetic_intensity = np.mean([l['average_arithmetic_intensity'] for l in layer_analysis])
        
        # 计算总体Linear性能统计
        total_linear_duration = sum(projection_stats[proj]['total_duration_us'] for proj in projection_stats)
        total_linear_flops = sum(projection_stats[proj]['total_flops'] for proj in projection_stats)
        total_linear_memory = sum(projection_stats[proj]['total_memory_access'] for proj in projection_stats)
        total_linear_kernels = sum(projection_stats[proj]['executions'] for proj in projection_stats)
        
        # 添加LM Head统计
        if 'lm_head' in remaining_perf:
            lm_head_executions = remaining_perf['lm_head']
            lm_head_duration = sum(ex['duration_us'] for ex in lm_head_executions)
            lm_head_flops = sum(ex['flops'] for ex in lm_head_executions)
            lm_head_memory = sum(ex['memory_access'] for ex in lm_head_executions)
            
            projection_stats['lm_head'] = {
                'total_duration_us': lm_head_duration,
                'total_flops': lm_head_flops,
                'total_memory_access': lm_head_memory,
                'executions': len(lm_head_executions)
            }
            
            total_linear_duration += lm_head_duration
            total_linear_flops += lm_head_flops
            total_linear_memory += lm_head_memory
            total_linear_kernels += len(lm_head_executions)
            
            print(f"LM Head统计: {len(lm_head_executions)} 次执行 (使用精确token size，时间保持不变)")
        
        # 计算效率指标
        peak_flops = hardware_spec.get('peak_flops_fp16', 148e12)  # 默认H20的FP16算力
        peak_memory_bandwidth = hardware_spec.get('memory_bandwidth', 4022e9)  # 默认H20的内存带宽
        
        compute_efficiency = (total_linear_flops / (total_linear_duration * 1e-6)) / peak_flops if total_linear_duration > 0 else 0
        memory_efficiency = (total_linear_memory / (total_linear_duration * 1e-6)) / peak_memory_bandwidth if total_linear_duration > 0 else 0
        
        linear_analysis = {
            'total_duration_us': total_linear_duration,
            'total_flops': total_linear_flops,
            'total_memory_access': total_linear_memory,
            'tflops_per_second': total_linear_flops / (total_linear_duration * 1e-6) / 1e12 if total_linear_duration > 0 else 0,
            'arithmetic_intensity': total_linear_flops / total_linear_memory if total_linear_memory > 0 else 0,
            'efficiency_compute': compute_efficiency,
            'efficiency_memory': memory_efficiency,
            'valid_transformer_layers': total_layers,
            'remaining_linear_kernels': len(remaining_linear),
            'projection_breakdown': projection_stats,
            'token_size_method': 'exact_per_kernel_matched_token_size',
            'time_calculation_method': 'time_range_based_fixed'
        }
        
        overall_stats = {
            'total_layers_analyzed': total_layers,
            'average_layer_duration_us': avg_layer_duration,
            'average_layer_tflops': avg_layer_tflops,
            'average_arithmetic_intensity': avg_arithmetic_intensity,
            'remaining_linear_kernels': len(remaining_linear)
        }
        
        print(f"\n=== 整体统计 (时间范围修复版本) ===")
        print(f"分析的MoE层数: {total_layers}")
        print(f"平均层持续时间: {avg_layer_duration:.0f} μs")
        print(f"平均层性能: {avg_layer_tflops:.2f} TFLOPS")
        print(f"平均算术强度: {avg_arithmetic_intensity:.2f} FLOPS/Byte")
        print(f"剩余未分配Linear kernels: {len(remaining_linear)}")
        
        # 计算覆盖率
        total_kernels = len(all_kernels)
        linear_coverage = (total_linear_kernels / total_kernels * 100) if total_kernels > 0 else 0
        
        return {
            'metadata': metadata,
            'model_config': model_config,
            'layer_analysis': layer_analysis,
            'overall_stats': overall_stats,
            'remaining_kernels': remaining_linear,
            'remaining_performance': remaining_perf,
            'linear_analysis': linear_analysis,
            'linear_coverage': linear_coverage,
            'linear_kernels': total_linear_kernels,
            'total_kernels': total_kernels,
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_method': 'time_range_based_precise_token_size'
        }

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='OEA Stage 2: MoE Linear算子性能分析工具（时间范围修复版本）')
    parser.add_argument('--input', required=True, 
                       help='Stage 1预处理数据文件路径')
    parser.add_argument('--output', default=None,
                       help='输出文件路径 (默认: 自动生成)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 设置输出文件
    if args.output is None:
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"oea_stage2_linear_analysis_results.json"
    
    try:
        # 读取预处理数据
        with open(args.input, 'r') as f:
            processed_data = json.load(f)
        
        # 创建分析器
        analyzer = MoELinearAnalyzerTimeRangeFixed()
        
        # 执行分析
        results = analyzer.analyze_linear_performance(processed_data)
        
        # 保存结果
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n=== MoE Linear分析完成 (时间范围修复版本) ===")
        print(f"结果已保存到: {args.output}")
        
        # 显示关键指标 - 类似stage2_linear_analyzer的输出格式
        if 'linear_analysis' in results and 'total_duration_us' in results['linear_analysis']:
            linear_perf = results['linear_analysis']
            print(f"\nMoE Linear算子性能 (时间范围修复版本):")
            print(f"  - Linear覆盖率: {results['linear_coverage']:.1f}% ({results['linear_kernels']}/{results['total_kernels']})")
            print(f"  - 总执行时间: {linear_perf['total_duration_us']/1000:.1f} ms")
            print(f"  - 计算性能: {linear_perf['tflops_per_second']:.2f} TFLOPS")
            print(f"  - 算术强度: {linear_perf['arithmetic_intensity']:.2f} OPs/Byte")
            print(f"  - 计算效率: {linear_perf['efficiency_compute']*100:.1f}%")
            print(f"  - 内存效率: {linear_perf['efficiency_memory']*100:.1f}%")
            print(f"  - 精确分组层数: {linear_perf['valid_transformer_layers']}")
            print(f"  - 剩余未分组kernels: {linear_perf['remaining_linear_kernels']}")
            print(f"  - Token Size方法: {linear_perf['token_size_method']}")
            print(f"  - 时间计算方法: {linear_perf['time_calculation_method']}")
            
            # 显示各projection的性能
            proj_breakdown = linear_perf['projection_breakdown']
            print(f"\n各Projection性能 (时间范围修复版本):")
            for proj_name, proj_data in proj_breakdown.items():
                if proj_data['executions']:
                    duration_ms = proj_data['total_duration_us'] / 1000
                    tflops = proj_data['total_flops'] / (proj_data['total_duration_us'] * 1e-6) / 1e12
                    ai = proj_data['total_flops'] / proj_data['total_memory_access'] if proj_data['total_memory_access'] > 0 else 0
                    print(f"  {proj_name}: {duration_ms:.1f} ms, {tflops:.2f} TFLOPS, AI={ai:.1f}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()