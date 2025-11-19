#!/usr/bin/env python3
"""
OEA Stage 2: vLLM Linear算子专门分析脚本 (基于fwd_splitkv_kernel的模式检测)

vLLM框架Linear算子检测规则:
1. Prefill阶段:
   - fwd_splitkv_kernel前1个gemm + 后3个gemm = 1个Transformer层
   - 剩余gemm为lm_head

2. Decode阶段 - 模式1 (3-gemm模式):
   - 连续两个fwd_splitkv_kernel之间有3个gemm
   - 对应: o_proj, gate_proj+up_proj, down_proj
   - 剩余gemm为lm_head

3. Decode阶段 - 模式2 (4-gemm模式):
   - 连续两个fwd_splitkv_kernel之间有4个gemm
   - fwd_splitkv_kernel前1个gemm为qkv_proj
   - fwd_splitkv_kernel后3个gemm为: o_proj, gate_proj+up_proj, down_proj
   - 剩余gemm为lm_head

使用方法:
python stage2_linear_analyzer_vllm.py --input stage1_processed_data.json
"""

import json
import numpy as np
import argparse
import os
import sys
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict

class VLLMLinearAnalyzer:
    def __init__(self):
        """初始化vLLM Linear算子分析器"""
        
        # Linear算子匹配模式 (GEMM kernels)
        self.linear_patterns = [
            r'.*gemm.*',
            r'.*sgemm.*',
            r'.*hgemm.*',
            r'.*cutlass.*gemm.*',
            r'nvjet.*',
            r'.*acext.*',
            r'.*cublas.*'
        ]
        
        # 排除的模式
        self.linear_exclude_patterns = [
            r'.*splitkreduce.*',
            r'.*splitK.*reduce.*',
            r'.*reduce.*'
        ]
        
        # fwd_splitkv_kernel算子匹配模式 (vLLM的attention kernel)
        self.fwd_splitkv_patterns = [
            r'.*fwd_splitkv_kernel.*',
            r'.*splitkv.*'
        ]
        
        # 预编译正则表达式
        self.compiled_linear_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.linear_patterns]
        self.compiled_linear_exclude_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.linear_exclude_patterns]
        self.compiled_fwd_splitkv_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.fwd_splitkv_patterns]
        
        # 缓存分类结果
        self.classification_cache = {}
        
        # 添加模型配置数据库
        self.model_configs = {
            'Qwen2.5-32B': {
                'hidden_size': 5120,
                'intermediate_size': 27648,
                'vocab_size': 152064,
                'num_attention_heads': 40,
                'num_key_value_heads': 8,
                'num_hidden_layers': 64
            },
            'Qwen2.5-7B': {
                'hidden_size': 3584,
                'intermediate_size': 18944,
                'vocab_size': 152064,
                'num_attention_heads': 28,
                'num_key_value_heads': 4,
                'num_hidden_layers': 28
            },
            'Qwen2.5-14B': {
                'hidden_size': 5120,
                'intermediate_size': 13824,
                'vocab_size': 152064,
                'num_attention_heads': 40,
                'num_key_value_heads': 8,
                'num_hidden_layers': 48
            },
            'Qwen2.5-3B': {
                'hidden_size': 2048,
                'intermediate_size': 11008,
                'vocab_size': 151936,
                'num_attention_heads': 16,
                'num_key_value_heads': 2,
                'num_hidden_layers': 36
            },
            'Llama-3.1-8B': {
                'hidden_size': 4096,
                'intermediate_size': 14336,
                'vocab_size': 128256,
                'num_attention_heads': 32,
                'num_key_value_heads': 8,
                'num_hidden_layers': 32
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
        
        # 默认配置（Qwen2.5-32B）
        print(f"警告: 未找到匹配的模型配置 '{model_name}'，使用默认Qwen2.5-32B配置")
        return self.model_configs['Qwen2.5-32B'].copy()
    
    def get_kernel_token_size(self, kernel: Dict, phase: str) -> int:
        """获取kernel的token size - 根据phase返回正确的值"""
        if phase == 'prefill':
            # Prefill阶段：处理完整的输入序列
            return kernel.get('token_size', 128)
        elif phase == 'decode':
            # Decode阶段：每次只生成1个token
            return 1
        else:
            # Unknown阶段：使用token_size作为保守估计
            return kernel.get('token_size', 128)
    
    def is_linear_kernel(self, kernel_name: str) -> bool:
        """判断是否为Linear kernel (GEMM)"""
        if kernel_name in self.classification_cache:
            return self.classification_cache[kernel_name]
        
        # 先检查排除模式
        for pattern in self.compiled_linear_exclude_patterns:
            if pattern.match(kernel_name):
                self.classification_cache[kernel_name] = False
                return False
        
        # 检查是否匹配Linear模式
        for pattern in self.compiled_linear_patterns:
            if pattern.match(kernel_name):
                self.classification_cache[kernel_name] = True
                return True
        
        self.classification_cache[kernel_name] = False
        return False
    
    def is_fwd_splitkv_kernel(self, kernel_name: str) -> bool:
        """
        判断是否为attention kernel (非combine类型)
        支持两种类型:
        1. Ampere系列: fwd_splitkv_kernel
        2. Hopper系列: FlashAttnFwdSm90
        只统计实际的attention kernel，不包括combine kernel
        """
        # 排除combine类型
        if 'combine' in kernel_name.lower():
            return False
        
        # 检查Hopper系列的FlashAttnFwdSm90算子
        if 'flashattnfwdsm90' in kernel_name.lower():
            return True
            
        # 检查Ampere系列的fwd_splitkv算子
        for pattern in self.compiled_fwd_splitkv_patterns:
            if pattern.match(kernel_name):
                return True
        return False
    
    def load_stage1_data(self, input_file: str) -> Dict:
        """加载Stage 1的分析结果"""
        print(f"=== 加载Stage 1数据 ===")
        print(f"输入文件: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ 成功加载Stage 1数据")
        return data
    
    def extract_kernels_from_iterations(self, stage1_data: Dict) -> Tuple[List[Dict], Dict]:
        """从iterations中提取所有kernels"""
        print(f"\n=== 从iterations中提取kernels ===")
        
        all_kernels = []
        iteration_info = {}
        
        iterations = stage1_data.get('iteration_analysis', {}).get('iterations', [])
        
        for iteration in iterations:
            iteration_id = iteration['iteration_id']
            phase = iteration['phase']
            kernels = iteration.get('kernels', [])
            
            iteration_info[iteration_id] = {
                'phase': phase,
                'start_ts': iteration['start_ts'],
                'end_ts': iteration['end_ts'],
                'kernel_count': len(kernels)
            }
            
            # 为每个kernel添加iteration信息
            for kernel in kernels:
                kernel['iteration_id'] = iteration_id
                kernel['iteration_phase'] = phase
                all_kernels.append(kernel)
        
        print(f"总共提取了 {len(all_kernels)} 个kernels")
        print(f"来自 {len(iterations)} 个iterations")
        
        return all_kernels, iteration_info
    
    def calculate_projection_performance(self, layer: Dict, model_config: Dict, 
                                        hardware_spec: Dict, case_info: Dict) -> Dict:
        """计算单个Transformer层中各projection的性能"""
        
        hidden_size = model_config["hidden_size"]
        intermediate_size = model_config["intermediate_size"]
        num_heads = model_config.get("num_attention_heads", model_config.get("num_heads", 40))
        num_kv_heads = model_config.get("num_key_value_heads", model_config.get("num_kv_heads", num_heads))
        
        head_dim = hidden_size // num_heads
        dtype_bytes = 2  # bfloat16/float16
        
        # 获取Tensor Parallelism配置
        tp_size = hardware_spec.get('tensor_parallelism', 1)
        
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
            'gate_up_proj': {
                'N': 2 * intermediate_size // tp_size,  # gate和up融合
                'K': hidden_size
            },
            'down_proj': {
                'N': hidden_size,
                'K': intermediate_size // tp_size
            }
        }
        
        layer_performance = {}
        phase = layer.get('phase', 'unknown')
        
        print(f"  计算层 {layer.get('layer_id')} 性能 (phase: {phase})")
        
        # 计算每个projection的性能
        for proj_name in ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj']:
            if proj_name not in layer:
                continue
            
            kernel = layer[proj_name]
            token_size = self.get_kernel_token_size(kernel, phase)
            M = token_size
            N = proj_configs[proj_name]['N']
            K = proj_configs[proj_name]['K']
            
            # GEMM FLOPS: 2*M*N*K (乘法和加法)
            flops = 2 * M * N * K
            
            # 内存访问: 读A(M*K) + 读B(K*N) + 写C(M*N)
            memory_access = (M * K + K * N + M * N) * dtype_bytes
            
            # 算术强度
            arithmetic_intensity = flops / memory_access if memory_access > 0 else 0
            
            layer_performance[proj_name] = {
                'kernel_name': kernel.get('name', ''),
                'duration_us': kernel.get('dur', 0),
                'token_size': token_size,
                'matrix_dims': [M, N, K],
                'flops': flops,
                'memory_access': memory_access,
                'arithmetic_intensity': arithmetic_intensity,
                'tp_size': tp_size,
                'phase': phase
            }
            
            print(f"    {proj_name}: token_size={token_size}, dims=[{M},{N},{K}], "
                  f"FLOPS={flops/1e9:.2f}G, AI={arithmetic_intensity:.2f}")
        
        return layer_performance
    
    def calculate_lm_head_performance(self, lm_head_kernels: List[Dict], 
                                     model_config: Dict,
                                     hardware_spec: Dict, case_info: Dict) -> List[Dict]:
        """计算LM Head kernels的性能"""
        
        if not lm_head_kernels:
            return []
        
        hidden_size = model_config["hidden_size"]
        vocab_size = model_config["vocab_size"]
        dtype_bytes = 2  # bfloat16/float16
        
        # 获取Tensor Parallelism配置
        tp_size = hardware_spec.get('tensor_parallelism', 1)
        
        # LM Head计算参数
        B = 1  # batch_size
        H = hidden_size
        V = vocab_size // tp_size  # TP分割词汇表维度
        
        print(f"\n=== LM Head性能计算 ===")
        print(f"模型配置: hidden_size={H}, vocab_size={vocab_size}, tp_size={tp_size}")
        
        lm_head_performance = []
        
        for lm_head in lm_head_kernels:
            kernel = lm_head.get('kernel', lm_head)
            phase = kernel.get('phase', 'unknown')
            token_size = self.get_kernel_token_size(kernel, phase)
            L = token_size
            
            # LM Head的FLOPS计算: matmul + bias
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
                'kernel_name': kernel.get('name', ''),
                'duration_us': kernel.get('dur', 0),
                'token_size': token_size,
                'matrix_dims': [B, L, H, V],
                'flops': total_flops,
                'matmul_flops': matmul_flops,
                'bias_flops': bias_flops,
                'memory_access': total_memory_access,
                'arithmetic_intensity': arithmetic_intensity,
                'tp_size': tp_size,
                'phase': phase,
                'original_vocab_size': vocab_size
            })
            
            print(f"  LM Head ({phase}): token_size={token_size}, FLOPS={total_flops/1e9:.2f}G, AI={arithmetic_intensity:.2f}")
        
        return lm_head_performance
    
    def analyze_vllm_linear_patterns(self, all_kernels: List[Dict], iteration_info: Dict) -> Dict:
        """分析vLLM框架的Linear算子模式"""
        
        print(f"\n=== 分析vLLM Linear算子模式 ===")
        
        results = {
            'framework': 'vllm',
            'detection_mode': 'fwd_splitkv_based',
            'transformer_layers': [],
            'lm_head_kernels': [],
            'statistics': {
                'prefill': {'layers': 0, 'lm_head': 0},
                'decode': {'layers': 0, 'lm_head': 0, 'mode': None}
            }
        }
        
        # 按iteration分组处理
        iterations_dict = defaultdict(list)
        for kernel in all_kernels:
            iteration_id = kernel.get('iteration_id')
            if iteration_id is not None:
                iterations_dict[iteration_id].append(kernel)
        
        # 处理每个iteration
        for iteration_id in sorted(iterations_dict.keys()):
            kernels = iterations_dict[iteration_id]
            phase = iteration_info[iteration_id]['phase']
            
            print(f"\n--- 处理Iteration {iteration_id} ({phase}阶段) ---")
            print(f"  总kernels: {len(kernels)}")
            
            # 提取Linear和fwd_splitkv kernels
            linear_kernels = []
            fwd_splitkv_kernels = []
            
            for i, kernel in enumerate(kernels):
                kernel_name = kernel.get('name', '')
                
                if self.is_fwd_splitkv_kernel(kernel_name):
                    fwd_splitkv_kernels.append({
                        'index': i,
                        'kernel': kernel,
                        'name': kernel_name
                    })
                elif self.is_linear_kernel(kernel_name):
                    linear_kernels.append({
                        'index': i,
                        'kernel': kernel,
                        'name': kernel_name
                    })
            
            print(f"  Linear kernels: {len(linear_kernels)}")
            print(f"  fwd_splitkv kernels: {len(fwd_splitkv_kernels)}")
            
            if phase == 'prefill':
                # Prefill阶段: fwd_splitkv前1个 + 后3个 = 1层
                layers, lm_heads = self._analyze_prefill_pattern(
                    linear_kernels, fwd_splitkv_kernels, iteration_id
                )
                results['transformer_layers'].extend(layers)
                results['lm_head_kernels'].extend(lm_heads)
                results['statistics']['prefill']['layers'] += len(layers)
                results['statistics']['prefill']['lm_head'] += len(lm_heads)
                
            elif phase == 'decode':
                # Decode阶段: 检测3-gemm或4-gemm模式
                layers, lm_heads, mode = self._analyze_decode_pattern(
                    linear_kernels, fwd_splitkv_kernels, iteration_id
                )
                results['transformer_layers'].extend(layers)
                results['lm_head_kernels'].extend(lm_heads)
                results['statistics']['decode']['layers'] += len(layers)
                results['statistics']['decode']['lm_head'] += len(lm_heads)
                
                if results['statistics']['decode']['mode'] is None:
                    results['statistics']['decode']['mode'] = mode
        
        # 打印统计信息
        print(f"\n=== 检测结果统计 ===")
        print(f"框架: {results['framework']}")
        print(f"检测模式: {results['detection_mode']}")
        print(f"Prefill阶段:")
        print(f"  - Transformer层: {results['statistics']['prefill']['layers']}")
        print(f"  - LM Head: {results['statistics']['prefill']['lm_head']}")
        print(f"Decode阶段:")
        print(f"  - Transformer层: {results['statistics']['decode']['layers']}")
        print(f"  - LM Head: {results['statistics']['decode']['lm_head']}")
        print(f"  - 模式: {results['statistics']['decode']['mode']}")
        
        return results
    
    def _analyze_prefill_pattern(self, linear_kernels: List[Dict], 
                                 fwd_splitkv_kernels: List[Dict],
                                 iteration_id: int) -> Tuple[List[Dict], List[Dict]]:
        """
        分析Prefill阶段的模式
        规则: fwd_splitkv前1个gemm + 后3个gemm = 1个Transformer层
        剩余的Linear kernels为LM Head（每个iteration通常1个）
        """
        print(f"  分析Prefill模式...")
        
        layers = []
        used_linear_indices = set()
        
        for splitkv_info in fwd_splitkv_kernels:
            splitkv_idx = splitkv_info['index']
            
            # 找到这个fwd_splitkv前面的1个linear
            before_linear = [lk for lk in linear_kernels 
                           if lk['index'] < splitkv_idx and lk['index'] not in used_linear_indices]
            before_linear.sort(key=lambda x: x['index'])
            
            # 找到这个fwd_splitkv后面的3个linear
            after_linear = [lk for lk in linear_kernels 
                          if lk['index'] > splitkv_idx and lk['index'] not in used_linear_indices]
            after_linear.sort(key=lambda x: x['index'])
            
            if len(before_linear) >= 1 and len(after_linear) >= 3:
                # 取最近的1个前置和3个后置
                qkv = before_linear[-1]
                o_proj = after_linear[0]
                gate_up = after_linear[1]
                down = after_linear[2]
                
                layer = {
                    'layer_id': len(layers),
                    'iteration_id': iteration_id,
                    'phase': 'prefill',
                    'pattern': 'prefill_1+3',
                    'qkv_proj': qkv['kernel'],
                    'o_proj': o_proj['kernel'],
                    'gate_up_proj': gate_up['kernel'],
                    'down_proj': down['kernel'],
                    'anchor_kernel': splitkv_info['kernel']
                }
                
                layers.append(layer)
                used_linear_indices.update([qkv['index'], o_proj['index'], 
                                          gate_up['index'], down['index']])
                
                print(f"    ✓ 检测到Prefill层 {len(layers)}")
        
        # 处理完所有Transformer层后，剩余的linear为lm_head（每个iteration通常1个）
        lm_heads = []
        for lk in linear_kernels:
            if lk['index'] not in used_linear_indices:
                lm_head = {
                    'iteration_id': iteration_id,
                    'phase': 'prefill',
                    'kernel': lk['kernel'],
                    'type': 'lm_head'
                }
                lm_heads.append(lm_head)
        
        print(f"    检测到 {len(layers)} 个Transformer层, {len(lm_heads)} 个LM Head")
        
        return layers, lm_heads
    
    def _analyze_decode_pattern(self, linear_kernels: List[Dict],
                                fwd_splitkv_kernels: List[Dict],
                                iteration_id: int) -> Tuple[List[Dict], List[Dict], str]:
        """
        分析Decode阶段的模式
        模式1: 连续两个fwd_splitkv之间有3个gemm (o_proj, gate_up, down)
        模式2: 连续两个fwd_splitkv之间有4个gemm (qkv在前, o_proj, gate_up, down在后)
        """
        print(f"  分析Decode模式...")
        
        if len(fwd_splitkv_kernels) < 2:
            print(f"    警告: fwd_splitkv数量不足，无法检测模式")
            return [], [], 'unknown'
        
        # 检测模式: 统计连续两个fwd_splitkv之间的linear数量
        gemm_counts = []
        for i in range(len(fwd_splitkv_kernels) - 1):
            start_idx = fwd_splitkv_kernels[i]['index']
            end_idx = fwd_splitkv_kernels[i + 1]['index']
            
            between_linear = [lk for lk in linear_kernels 
                            if start_idx < lk['index'] < end_idx]
            gemm_counts.append(len(between_linear))
        
        # 判断模式
        avg_gemm_count = np.mean(gemm_counts) if gemm_counts else 0
        
        if avg_gemm_count >= 3.5:  # 接近4
            mode = '4-gemm'
            print(f"    检测到模式: 4-gemm (qkv在前)")
            return self._analyze_decode_4gemm_pattern(linear_kernels, fwd_splitkv_kernels, iteration_id)
        else:  # 接近3
            mode = '3-gemm'
            print(f"    检测到模式: 3-gemm (无qkv)")
            return self._analyze_decode_3gemm_pattern(linear_kernels, fwd_splitkv_kernels, iteration_id)
    
    def _analyze_decode_3gemm_pattern(self, linear_kernels: List[Dict],
                                     fwd_splitkv_kernels: List[Dict],
                                     iteration_id: int) -> Tuple[List[Dict], List[Dict], str]:
        """
        Decode 3-gemm模式: 连续两个fwd_splitkv之间有3个gemm
        对应: o_proj, gate_proj+up_proj, down_proj
        剩余的Linear kernels为LM Head（每个iteration通常1个）
        """
        layers = []
        used_linear_indices = set()
        
        for i in range(len(fwd_splitkv_kernels) - 1):
            start_idx = fwd_splitkv_kernels[i]['index']
            end_idx = fwd_splitkv_kernels[i + 1]['index']
            
            between_linear = [lk for lk in linear_kernels 
                            if start_idx < lk['index'] < end_idx 
                            and lk['index'] not in used_linear_indices]
            between_linear.sort(key=lambda x: x['index'])
            
            if len(between_linear) >= 3:
                o_proj = between_linear[0]
                gate_up = between_linear[1]
                down = between_linear[2]
                
                layer = {
                    'layer_id': len(layers),
                    'iteration_id': iteration_id,
                    'phase': 'decode',
                    'pattern': 'decode_3gemm',
                    'o_proj': o_proj['kernel'],
                    'gate_up_proj': gate_up['kernel'],
                    'down_proj': down['kernel'],
                    'anchor_kernel': fwd_splitkv_kernels[i]['kernel']
                }
                
                layers.append(layer)
                used_linear_indices.update([o_proj['index'], gate_up['index'], down['index']])
        
        # 处理完所有Transformer层后，剩余的linear为lm_head（每个iteration通常1个）
        lm_heads = []
        for lk in linear_kernels:
            if lk['index'] not in used_linear_indices:
                lm_head = {
                    'iteration_id': iteration_id,
                    'phase': 'decode',
                    'kernel': lk['kernel'],
                    'type': 'lm_head'
                }
                lm_heads.append(lm_head)
        
        print(f"    检测到 {len(layers)} 个Transformer层, {len(lm_heads)} 个LM Head")
        
        return layers, lm_heads, '3-gemm'
    
    def _analyze_decode_4gemm_pattern(self, linear_kernels: List[Dict],
                                     fwd_splitkv_kernels: List[Dict],
                                     iteration_id: int) -> Tuple[List[Dict], List[Dict], str]:
        """
        Decode 4-gemm模式: 连续两个fwd_splitkv之间有4个gemm
        fwd_splitkv前1个为qkv_proj, 后3个为o_proj, gate_up, down
        剩余的Linear kernels为LM Head（每个iteration通常1个）
        """
        layers = []
        used_linear_indices = set()
        
        for splitkv_info in fwd_splitkv_kernels:
            splitkv_idx = splitkv_info['index']
            
            # 找到这个fwd_splitkv前面的1个linear (qkv_proj)
            before_linear = [lk for lk in linear_kernels 
                           if lk['index'] < splitkv_idx and lk['index'] not in used_linear_indices]
            before_linear.sort(key=lambda x: x['index'])
            
            # 找到这个fwd_splitkv后面的3个linear (o, gate_up, down)
            after_linear = [lk for lk in linear_kernels 
                          if lk['index'] > splitkv_idx and lk['index'] not in used_linear_indices]
            after_linear.sort(key=lambda x: x['index'])
            
            if len(before_linear) >= 1 and len(after_linear) >= 3:
                qkv = before_linear[-1]
                o_proj = after_linear[0]
                gate_up = after_linear[1]
                down = after_linear[2]
                
                layer = {
                    'layer_id': len(layers),
                    'iteration_id': iteration_id,
                    'phase': 'decode',
                    'pattern': 'decode_4gemm',
                    'qkv_proj': qkv['kernel'],
                    'o_proj': o_proj['kernel'],
                    'gate_up_proj': gate_up['kernel'],
                    'down_proj': down['kernel'],
                    'anchor_kernel': splitkv_info['kernel']
                }
                
                layers.append(layer)
                used_linear_indices.update([qkv['index'], o_proj['index'], 
                                          gate_up['index'], down['index']])
        
        # 处理完所有Transformer层后，剩余的linear为lm_head（每个iteration通常1个）
        lm_heads = []
        for lk in linear_kernels:
            if lk['index'] not in used_linear_indices:
                lm_head = {
                    'iteration_id': iteration_id,
                    'phase': 'decode',
                    'kernel': lk['kernel'],
                    'type': 'lm_head'
                }
                lm_heads.append(lm_head)
        
        print(f"    检测到 {len(layers)} 个Transformer层, {len(lm_heads)} 个LM Head")
        
        return layers, lm_heads, '4-gemm'
    
    def generate_output_data(self, stage1_data: Dict, analysis_results: Dict) -> Dict:
        """生成输出数据结构"""
        
        print(f"\n=== 生成输出数据 ===")
        
        output_data = {
            'case_info': stage1_data.get('case_info', {}),
            'hardware_spec': stage1_data.get('hardware_spec', {}),
            'iteration_analysis': stage1_data.get('iteration_analysis', {}),
            'linear_analysis': analysis_results,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return output_data
    
    def save_results(self, output_data: Dict, input_file: str, output_file: str = None):
        """保存分析结果"""
        
        if output_file is None:
            # 根据输入文件名生成输出文件名
            input_dir = os.path.dirname(input_file)
            input_basename = os.path.basename(input_file)
            
            # 提取pod_name (从stage1文件名中提取)
            # 格式: oea_stage1_<pod_name>_processed.json
            if 'oea_stage1_' in input_basename:
                pod_name = input_basename.replace('oea_stage1_', '').replace('_processed.json', '')
                output_file = os.path.join(input_dir, f'oea_stage2_{pod_name}_processed.json')
            else:
                output_file = input_file.replace('stage1', 'stage2')
        
        print(f"\n=== 保存分析结果 ===")
        print(f"输出文件: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"vLLM Linear分析结果已保存")
        
        return output_file

def main():
    parser = argparse.ArgumentParser(description='vLLM OEA Stage 2: Linear算子分析')
    parser.add_argument('--input', required=True, help='Stage 1处理后的JSON文件路径')
    parser.add_argument('--output', help='输出文件路径（可选）')
    
    args = parser.parse_args()
    
    try:
        print("=" * 80)
        print("vLLM Linear算子分析 (Stage 2)")
        print("=" * 80)
        
        # 创建分析器
        analyzer = VLLMLinearAnalyzer()
        
        # 加载Stage 1数据
        stage1_data = analyzer.load_stage1_data(args.input)
        
        # 提取kernels
        all_kernels, iteration_info = analyzer.extract_kernels_from_iterations(stage1_data)
        
        # 分析vLLM Linear模式
        analysis_results = analyzer.analyze_vllm_linear_patterns(all_kernels, iteration_info)
        
        # ===== 新增：计算性能指标 =====
        print(f"\n=== 计算性能指标 (FLOPS, Memory Access, AI) ===")
        
        # 提取必要信息
        case_info = stage1_data.get('case_info', {})
        hardware_spec = stage1_data.get('hardware_spec', {})
        model_name = case_info.get('model_name', 'Unknown')
        
        # 获取模型配置
        model_config = analyzer.get_model_config_from_name(model_name)
        
        # 为每个 transformer layer 计算性能
        print(f"为 {len(analysis_results['transformer_layers'])} 个 Transformer Layers 计算性能...")
        for i, layer in enumerate(analysis_results['transformer_layers']):
            layer['performance'] = analyzer.calculate_projection_performance(
                layer, model_config, hardware_spec, case_info
            )
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1} 个 layers...")
        
        # 为 lm_head 计算性能
        if analysis_results['lm_head_kernels']:
            print(f"为 {len(analysis_results['lm_head_kernels'])} 个 LM Head Kernels 计算性能...")
            lm_head_performance = analyzer.calculate_lm_head_performance(
                analysis_results['lm_head_kernels'], model_config, hardware_spec, case_info
            )
            
            # 更新 lm_head_kernels 添加性能信息
            for i, lm_head in enumerate(analysis_results['lm_head_kernels']):
                if i < len(lm_head_performance):
                    lm_head['performance'] = lm_head_performance[i]
        
        # 添加模型配置到结果中
        analysis_results['model_config'] = model_config
        
        print("✓ 性能计算完成")
        # ===== 性能计算结束 =====
        
        # 生成输出数据
        output_data = analyzer.generate_output_data(stage1_data, analysis_results)
        
        # 保存结果
        output_file = analyzer.save_results(output_data, args.input, args.output)
        
        print("\n" + "=" * 80)
        print("=== vLLM Linear分析完成 ===")
        print("=" * 80)
        print(f"输入文件: {args.input}")
        print(f"输出文件: {output_file}")
        print(f"检测结果:")
        print(f"  - 框架: {analysis_results['framework']}")
        print(f"  - 检测模式: {analysis_results['detection_mode']}")
        print(f"  - Transformer层: {len(analysis_results['transformer_layers'])}")
        print(f"  - LM Head: {len(analysis_results['lm_head_kernels'])}")
        print(f"  - Prefill层: {analysis_results['statistics']['prefill']['layers']}")
        print(f"  - Decode层: {analysis_results['statistics']['decode']['layers']}")
        print(f"  - Decode模式: {analysis_results['statistics']['decode']['mode']}")
        
    except Exception as e:
        print(f"\n错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()