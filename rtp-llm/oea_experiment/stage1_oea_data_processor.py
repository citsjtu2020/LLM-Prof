#!/usr/bin/env python3
"""
OEA Stage 1: 数据处理和预分析脚本 (修复版本)
负责加载配置、读取trace数据、算子分类等基础处理
修复了时间戳匹配问题：将配置文件的绝对时间戳转换为相对时间戳

主要修复:
1. 修复时间基准不一致问题：配置文件使用绝对时间戳，kernel使用相对时间戳
2. 正确的时间戳转换：绝对时间 -> 相对时间
3. 为每个kernel匹配最接近时间戳的精确token size
4. 保持与原始stage1完全兼容的输出格式

使用方法:
python stage1_oea_data_processor_fixed.py --case_path traces_after_sea_section_part2/mkt-qwen3-30b-a3b-feeds-na175-custom accelerator.inference-part0-3cc70f0e-a-d735
"""

import json
import numpy as np
import argparse
import os
import sys
from datetime import datetime
import pytz
from typing import Dict, List, Any, Tuple
import re
import glob
import bisect

class TokenSizeMatcher:
    """Token Size匹配器 - 修复时间戳匹配问题"""
    
    def __init__(self):
        """初始化Token Size匹配器"""
        self.token_size_timeline = []  # [(relative_timestamp_us, token_size), ...]
        self.base_timestamp_us = None
        
    def parse_execute_token_size_timeline(self, content: str, first_kernel_timestamp_us: float = 0) -> List[Tuple[int, float]]:
        """
        从配置文件内容中解析execute_token_size时间线，并转换为相对时间戳
        
        Args:
            content: 配置文件内容
            first_kernel_timestamp_us: 第一个kernel的时间戳（用于时间基准对齐）
        """
        
        print(f"=== 解析execute_token_size时间线 ===")
        
        # 提取execute_token_size部分
        execute_token_pattern = r'"execute_token_size":\s*(.*?)\s*,'
        match = re.search(execute_token_pattern, content, re.DOTALL)
        
        if not match:
            print("警告: 未找到execute_token_size数据，将使用固定token size")
            return []
        
        token_data = match.group(1).strip()
        print(f"找到execute_token_size数据段")
        
        # 解析时间戳和token size
        lines = token_data.split('\n')
        absolute_timeline_data = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
                
            # 匹配时间戳格式: 2025-10-12 23:50:06.000
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})', line)
            if timestamp_match and i + 1 < len(lines):
                timestamp_str = timestamp_match.group(1)
                
                # 下一行是token size值
                i += 1
                token_line = lines[i].strip()
                
                # 解析token size值，处理K后缀
                token_size = self._parse_token_size(token_line)
                
                # 转换时间戳为微秒（绝对时间戳）
                dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                absolute_timestamp_us = int(dt.timestamp() * 1_000_000)
                
                absolute_timeline_data.append((absolute_timestamp_us, token_size))
                print(f"  {timestamp_str} -> {token_size} tokens (绝对时间戳: {absolute_timestamp_us})")
            
            i += 1
        
        if not absolute_timeline_data:
            print("警告: 未解析到有效的execute_token_size数据")
            return []
        
        # 按时间戳排序
        absolute_timeline_data.sort(key=lambda x: x[0])
        
        # 转换为相对时间戳
        base_absolute_timestamp = absolute_timeline_data[0][0]
        relative_timeline_data = []
        
        print(f"\n=== 时间戳转换：绝对时间 -> 相对时间 ===")
        print(f"基准绝对时间戳: {base_absolute_timestamp} ({datetime.fromtimestamp(base_absolute_timestamp/1_000_000)})")
        print(f"第一个kernel时间戳: {first_kernel_timestamp_us} 微秒")
        
        # 转换为相对时间戳：假设第一个配置时间戳对应第一个kernel的时间戳
        for abs_ts, token_size in absolute_timeline_data:
            # 计算相对时间戳：(当前绝对时间 - 基准绝对时间) + 第一个kernel时间戳
            relative_ts = (abs_ts - base_absolute_timestamp) + first_kernel_timestamp_us
            relative_timeline_data.append((relative_ts, token_size))
            
            abs_dt = datetime.fromtimestamp(abs_ts/1_000_000)
            print(f"  {abs_dt} -> 相对时间: {relative_ts} 微秒 -> {token_size} tokens")
        
        self.token_size_timeline = relative_timeline_data
        self.base_timestamp_us = relative_timeline_data[0][0] if relative_timeline_data else None
        
        print(f"\n解析完成: 共 {len(relative_timeline_data)} 个时间点")
        if relative_timeline_data:
            print(f"相对时间范围: {relative_timeline_data[0][0]} - {relative_timeline_data[-1][0]} (微秒)")
            print(f"Token size范围: {min(x[1] for x in relative_timeline_data)} - {max(x[1] for x in relative_timeline_data)}")
        
        return relative_timeline_data
    
    def _parse_token_size(self, token_str: str) -> float:
        """解析token size字符串，处理K、M、G等后缀"""
        token_str = token_str.strip().rstrip(',').strip()
        
        # 定义单位映射
        unit_multipliers = {
            'K': 1000, 'k': 1000,
            'M': 1000000, 'm': 1000000,
            'G': 1000000000, 'g': 1000000000,
            'B': 1000000000, 'b': 1000000000
        }
        
        # 使用正则表达式匹配数值和单位
        pattern = r'^([0-9]*\.?[0-9]+)\s*([KkMmGgBb]?)\s*$'
        match = re.match(pattern, token_str)
        
        if match:
            number_str = match.group(1)
            unit = match.group(2).strip()
            
            try:
                base_value = float(number_str)
                
                # 如果有单位，应用乘数
                if unit and unit in unit_multipliers:
                    final_value = base_value * unit_multipliers[unit]
                    print(f"Token size解析: '{token_str}' -> {base_value} × {unit_multipliers[unit]} = {final_value}")
                    return final_value
                else:
                    return base_value
                    
            except ValueError:
                print(f"警告: 无法解析token size数值部分: '{number_str}'")
                return 0.0
        else:
            # 如果正则匹配失败，尝试直接转换为浮点数
            try:
                return float(token_str)
            except ValueError:
                print(f"警告: 无法解析token size: '{token_str}'，使用默认值0")
                return 0.0
    
    def find_closest_token_size(self, target_timestamp_us: int) -> Tuple[float, int, float]:
        """
        找到最接近目标时间戳的token size
        
        Returns:
            (token_size, closest_timestamp_us, time_diff_ms)
        """
        if not self.token_size_timeline:
            # 如果没有时间线数据，返回默认值
            return 2048.0, target_timestamp_us, 0.0
        
        # 使用二分查找找到最接近的时间戳
        timestamps = [x[0] for x in self.token_size_timeline]
        
        # 找到插入位置
        pos = bisect.bisect_left(timestamps, target_timestamp_us)
        
        # 确定最接近的点
        candidates = []
        
        if pos > 0:
            candidates.append(pos - 1)
        if pos < len(timestamps):
            candidates.append(pos)
        
        if not candidates:
            candidates = [0]
        
        # 找到时间差最小的点
        best_idx = min(candidates, key=lambda i: abs(timestamps[i] - target_timestamp_us))
        
        closest_timestamp_us = self.token_size_timeline[best_idx][0]
        token_size = self.token_size_timeline[best_idx][1]
        time_diff_ms = abs(target_timestamp_us - closest_timestamp_us) / 1000
        
        return token_size, closest_timestamp_us, time_diff_ms

class OEADataProcessorFixed:
    def __init__(self):
        """初始化OEA数据处理器（修复版本）"""
        
        # 集成Token Size匹配器
        self.token_matcher = TokenSizeMatcher()
        
        # 硬件规格数据库
        self.hardware_specs = {
            "NVIDIA_A10": {
                "mem_bandwidth": 600 * (1024**3), 
                "FP16": 125e12, 
                "INT8": 250e12, 
                "memsize": 24 * (1024**3), 
                "onchip_buffer": 0, 
                "interconnect_bandwidth": 64 * (1024**3)
            },
            "NVIDIA_L20": {
                "mem_bandwidth": 864 * (1024**3), 
                "FP16": 119.5e12, 
                "INT8": 239e12, 
                "memsize": 48 * (1024**3), 
                "onchip_buffer": 0, 
                "interconnect_bandwidth": None
            },
            "NVIDIA_H20_SXM5_96GB": {
                "mem_bandwidth": 4022 * (1024**3), 
                "FP16": 148e12, 
                "INT8": 296e12, 
                "memsize": 96 * (1024**3), 
                "onchip_buffer": 0, 
                "interconnect_bandwidth": 900 * (1024**3)
            },
            "NVIDIA_H20_SXM5_141GB": {
                "mem_bandwidth": 4800 * (1024**3), 
                "FP16": 148e12, 
                "INT8": 296e12, 
                "memsize": 141 * (1024**3), 
                "onchip_buffer": 0, 
                "interconnect_bandwidth": 900 * (1024**3)
            },
            "NVIDIA_A100_SXM4_80GB": {
                "mem_bandwidth": 2039 * (1024**3), 
                "FP16": 312e12, 
                "INT8": 624e12, 
                "memsize": 80 * (1024**3), 
                "onchip_buffer": 0, 
                "interconnect_bandwidth": 600 * (1024**3)
            },
            "NVIDIA_A800_SXM4_80GB": {
                "mem_bandwidth": 2039 * (1024**3), 
                "FP16": 312e12, 
                "INT8": 624e12, 
                "memsize": 80 * (1024**3), 
                "onchip_buffer": 0, 
                "interconnect_bandwidth": 400 * (1024**3)
            },
            "NVIDIA_H800": {
                "mem_bandwidth": 3350 * (1024**3), 
                "FP16": 989e12, 
                "INT8": 1979e12, 
                "memsize": 80 * (1024**3), 
                "onchip_buffer": 0, 
                "interconnect_bandwidth": 400 * (1024**3)
            },
            "NVIDIA_H100": {
                "mem_bandwidth": 3350 * (1024**3), 
                "FP16": 989e12, 
                "INT8": 1979e12, 
                "memsize": 80 * (1024**3), 
                "onchip_buffer": 0, 
                "interconnect_bandwidth": 900 * (1024**3)
            },
            "NVIDIA_H200": {
                "mem_bandwidth": 4800 * (1024**3), 
                "FP16": 989e12, 
                "INT8": 1979e12, 
                "memsize": 141 * (1024**3), 
                "onchip_buffer": 0, 
                "interconnect_bandwidth": 900 * (1024**3)
            },
            "NVIDIA_B200": {
                "mem_bandwidth": 8000 * (1024**3), 
                "FP16": 2250e12, 
                "INT8": 4500e12, 
                "memsize": 192 * (1024**3), 
                "onchip_buffer": 0, 
                "interconnect_bandwidth": 1800 * (1024**3)
            },
            "AMD_MI308X": {
                "mem_bandwidth": 4000 * (1024**3), 
                "FP16": 115e12, 
                "INT8": 230e12, 
                "memsize": 192 * (1024**3), 
                "onchip_buffer": 0, 
                "interconnect_bandwidth": 300 * (1024**3)
            }
        }

        # 硬件类型映射
        self.hardware_name_mapping = {
            "H20": "NVIDIA_H20_SXM5_96GB",
            "A100": "NVIDIA_A100_SXM4_80GB", 
            "A800": "NVIDIA_A800_SXM4_80GB",
            "H800": "NVIDIA_H800",
            "H100": "NVIDIA_H100",
            "H200": "NVIDIA_H200",
            "B200": "NVIDIA_B200",
            "L20": "NVIDIA_L20",
            "A10": "NVIDIA_A10",
            "MI308X": "AMD_MI308X"
        }
        
        # 算子分类规则
        self.operator_patterns = {
            'reduction': [
                r'.*splitkreduce.*',
                r'.*splitK.*reduce.*',
                r'.*split.*k.*reduce.*',
                r'.*cublaslt.*splitkreduce.*',
                r'.*cublas.*splitk.*',
                r'.*splitk.*kernel.*',
                r'.*mergestates.*',
                r'.*merge.*states.*',
                r'.*reduction.*',
                r'.*reduce.*',
            ],
            'linear': [
                r'nvjet.*',
                r'.*gemm.*',
                r'.*sgemm.*',
                r'.*hgemm.*',
                r'.*cutlass.*gemm.*',
                r'.*gemvn.*',
                r'.*gemvt.*',
                r'(?!.*splitkreduce).*cublas.*'
            ],
            'attention': [
                r'.*flash.*attn.*',
                r'.*attention.*',
                r'.*softmax.*',
                r'.*scaled_dot_product.*'
            ],
            'rope': [
                r'.*rope.*',
                r'.*rotary.*',
                r'.*position.*embedding.*'
            ],
            'layernorm': [
                r'.*rms.*norm.*',
                r'.*layer.*norm.*',
                r'.*norm.*',
                r'.*generalRmsNorm.*'
            ],
            'activation': [
                r'.*silu.*',
                r'.*gelu.*',
                r'.*relu.*',
                r'.*swish.*',
                r'.*silu_and_mul.*'
            ],
            'add_bias': [
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
                r'.*add.*residual.*'
            ],
            'moe': [
                r'.*moe.*',
                r'.*expert.*',
                r'.*gate.*',
                r'.*router.*'
            ],
            'communication': [
                r'.*nccl.*',
                r'.*all_reduce.*',
                r'.*all_gather.*'
            ],
            'memory': [
                r'.*copy.*',
                r'.*memcpy.*',
                r'.*transpose.*'
            ]
        }
        
        # 预编译正则表达式
        self.compiled_patterns = {}
        for op_type, patterns in self.operator_patterns.items():
            self.compiled_patterns[op_type] = [re.compile(pattern) for pattern in patterns]
        
        # 缓存分类结果
        self.classification_cache = {}

    def get_hardware_spec(self, gpu_type: str, memory_size: float = None, f_peak: float = None) -> Dict[str, Any]:
        """根据GPU类型获取硬件规格"""
        
        # 标准化GPU类型名称
        normalized_gpu_type = gpu_type.upper().strip()
        
        # 直接匹配完整名称
        if normalized_gpu_type in self.hardware_specs:
            spec = self.hardware_specs[normalized_gpu_type].copy()
        # 通过简化名称映射
        elif normalized_gpu_type in self.hardware_name_mapping:
            full_name = self.hardware_name_mapping[normalized_gpu_type]
            spec = self.hardware_specs[full_name].copy()
        else:
            # 模糊匹配
            matched_spec = None
            for spec_name, spec_data in self.hardware_specs.items():
                if normalized_gpu_type in spec_name.upper():
                    matched_spec = spec_data.copy()
                    break
            
            if matched_spec:
                spec = matched_spec
            else:
                print(f"警告: 未识别的GPU类型 '{gpu_type}'，使用默认H20规格")
                spec = self.hardware_specs["NVIDIA_H20_SXM5_96GB"].copy()
        
        # 根据显存大小进一步细化识别（特别是H20的不同型号）
        if memory_size and "H20" in normalized_gpu_type:
            if memory_size > 100:  # 141GB型号
                spec = self.hardware_specs["NVIDIA_H20_SXM5_141GB"].copy()
            else:  # 96GB型号
                spec = self.hardware_specs["NVIDIA_H20_SXM5_96GB"].copy()
        
        # 转换为标准格式
        return {
            "memory_bandwidth": spec["mem_bandwidth"],
            "peak_flops_fp16": spec["FP16"], 
            "peak_flops_int8": spec["INT8"],
            "memory_size": spec["memsize"],
            "name": gpu_type,
            "interconnect_bandwidth": spec.get("interconnect_bandwidth", 0),
            "onchip_buffer": spec.get("onchip_buffer", 0)
        }
    
    def parse_hardware_config_json(self, hw_json_str: str) -> Dict[str, Any]:
        """解析硬件配置JSON"""
        try:
            # 首先尝试直接解析
            return json.loads(hw_json_str)
        except json.JSONDecodeError:
            # 如果直接解析失败，进行预处理
            print("JSON格式不规范，尝试修复...")
            
            # 处理Python表达式
            processed_str = hw_json_str
            
            # 替换数学表达式
            math_expressions = [
                (r'2765 \* \(1024\*\*3\)', str(2765 * (1024**3))),
                (r'118e12', str(118e12)),
                (r'236e12', str(236e12)),
                (r'96 \* \(1024\*\*3\)', str(96 * (1024**3))),
                (r'700 \* \(1024\*\*3\)', str(700 * (1024**3))),
                (r'4022 \* \(1024\*\*3\)', str(4022 * (1024**3))),
                (r'148e12', str(148e12)),
                (r'296e12', str(296e12)),
                (r'900 \* \(1024\*\*3\)', str(900 * (1024**3))),
            ]
            
            for pattern, replacement in math_expressions:
                processed_str = re.sub(pattern, replacement, processed_str)
            
            # 再次尝试解析
            try:
                return json.loads(processed_str)
            except json.JSONDecodeError as e:
                print(f"JSON修复后仍然解析失败: {e}")
                
                # 最后的备用方案：手动提取关键字段
                hw_config = {}
                
                # 提取gpu_type
                gpu_type_match = re.search(r'"gpu_type":\s*"([^"]+)"', hw_json_str)
                if gpu_type_match:
                    hw_config["gpu_type"] = gpu_type_match.group(1)
                
                # 提取f_peak
                f_peak_match = re.search(r'"f_peak":\s*([0-9.]+)', hw_json_str)
                if f_peak_match:
                    hw_config["f_peak"] = float(f_peak_match.group(1))
                
                # 提取memory_bandwidth
                mem_bw_match = re.search(r'"memory_bandwidth":\s*([0-9.]+)', hw_json_str)
                if mem_bw_match:
                    hw_config["memory_bandwidth"] = float(mem_bw_match.group(1))
                
                # 提取memory_size
                mem_size_match = re.search(r'"memory_size":\s*([0-9.]+)', hw_json_str)
                if mem_size_match:
                    hw_config["memory_size"] = float(mem_size_match.group(1))
                
                # 提取n_gpu
                n_gpu_match = re.search(r'"n_gpu":\s*([0-9]+)', hw_json_str)
                if n_gpu_match:
                    hw_config["n_gpu"] = int(n_gpu_match.group(1))
                
                # 提取tensor_parallelism
                tp_match = re.search(r'"tensor_parallelism":\s*([0-9]+)', hw_json_str)
                if tp_match:
                    hw_config["tensor_parallelism"] = int(tp_match.group(1))
                
                print(f"使用正则表达式提取的硬件配置: {hw_config}")
                return hw_config

    def _get_first_kernel_timestamp(self, case_path: str) -> float:
        """获取第一个kernel的时间戳，用于时间基准对齐"""
        
        try:
            # 查找trace文件
            trace_file = self.find_trace_file(case_path)
            
            # 读取trace数据
            with open(trace_file, 'r') as f:
                data = json.load(f)
            
            events = data.get('traceEvents', data) if isinstance(data, dict) else data
            kernel_events = [e for e in events if isinstance(e, dict) and 'name' in e and 'ts' in e and e.get('cat') == 'kernel']
            
            if kernel_events:
                # 按时间戳排序，获取第一个kernel的时间戳
                sorted_events = sorted(kernel_events, key=lambda x: x['ts'])
                first_timestamp = sorted_events[0]['ts']
                print(f"第一个kernel时间戳: {first_timestamp} 微秒")
                return first_timestamp
            else:
                print("警告: 未找到kernel事件，使用默认时间基准0")
                return 0.0
                
        except Exception as e:
            print(f"警告: 获取第一个kernel时间戳失败: {e}，使用默认时间基准0")
            return 0.0

    def load_case_config(self, case_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], List[Tuple[int, float]]]:
        """从案例路径加载配置信息，并解析execute_token_size时间线"""
        
        print(f"=== 加载案例配置 ===")
        print(f"案例路径: {case_path}")
        
        # 查找配置文件
        config_file = os.path.join(case_path, "prefill_metrics_with_config.txt")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"找不到配置文件: {config_file}")
        
        # 解析配置文件
        model_config = {}
        hardware_spec = {}
        
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 先获取第一个kernel的时间戳用于时间基准对齐
        first_kernel_timestamp_us = self._get_first_kernel_timestamp(case_path)
        
        # 使用TokenMatcher解析execute_token_size时间线，传入第一个kernel时间戳
        execute_token_timeline = self.token_matcher.parse_execute_token_size_timeline(
            content, first_kernel_timestamp_us
        )
        
        # 提取模型配置 JSON
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_pattern, content, re.DOTALL)
        
        if len(json_matches) >= 1:
            # 第一个JSON是模型配置
            try:
                model_config = json.loads(json_matches[0])
                print(f"模型: {model_config.get('_name_or_path', 'Unknown')}")
                print(f"架构: {model_config.get('architectures', ['Unknown'])[0]}")
                print(f"隐藏层数: {model_config.get('num_hidden_layers', 'Unknown')}")
                print(f"隐藏维度: {model_config.get('hidden_size', 'Unknown')}")
            except json.JSONDecodeError as e:
                print(f"警告: 模型配置JSON解析失败: {e}")
        
        if len(json_matches) >= 2:
            # 第二个JSON是硬件配置
            hw_json_str = json_matches[1]
            
            try:
                hw_config = self.parse_hardware_config_json(hw_json_str)
                
                # 提取GPU类型和基本参数
                gpu_type = hw_config.get("gpu_type", "ZW810e")
                memory_size = hw_config.get("memory_size", 96.0)
                f_peak = hw_config.get("f_peak", 118.0)
                
                # 使用智能硬件识别获取完整规格
                hardware_spec = self.get_hardware_spec(gpu_type, memory_size, f_peak)
                
                # 补充配置文件中的其他参数
                hardware_spec.update({
                    "n_gpu": hw_config.get("n_gpu", 1),
                    "tensor_parallelism": hw_config.get("tensor_parallelism", 1)
                })
                
                # 如果配置文件中有具体的数值，优先使用配置文件的值
                if "memory_bandwidth" in hw_config:
                    hardware_spec["memory_bandwidth"] = hw_config["memory_bandwidth"] * 1e9
                if "f_peak" in hw_config:
                    hardware_spec["peak_flops_fp16"] = hw_config["f_peak"] * 1e12
                if "memory_size" in hw_config:
                    hardware_spec["memory_size"] = hw_config["memory_size"] * 1e9
                
                print(f"硬件: {hardware_spec['name']}")
                print(f"显存: {hardware_spec['memory_size']/1e9:.0f} GB")
                print(f"FP16算力: {hardware_spec['peak_flops_fp16']/1e12:.0f} TFLOPS")
                print(f"内存带宽: {hardware_spec['memory_bandwidth']/1e12:.1f} TB/s")
                
            except Exception as e:
                print(f"警告: 硬件配置解析失败: {e}")
                print(f"使用默认ZW810e硬件配置")
                # 使用默认的ZW810e配置
                hardware_spec = self.get_hardware_spec("ZW810e")
                hardware_spec.update({
                    "n_gpu": 1,
                    "tensor_parallelism": 1
                })
        else:
            # 如果没有硬件配置，使用默认值
            print("未找到硬件配置，使用默认ZW810e配置")
            hardware_spec = self.get_hardware_spec("ZW810e")
            hardware_spec.update({
                "n_gpu": 1,
                "tensor_parallelism": 1
            })
        
        return model_config, hardware_spec, execute_token_timeline
            
    def find_trace_file(self, case_path: str) -> str:
        """查找案例中的trace文件"""
        
        # 查找可能的trace文件
        trace_patterns = [
            "**/*.json",
            "**/cupti_*.json", 
            "**/ecos-trace-*.json"
        ]
        
        trace_files = []
        for pattern in trace_patterns:
            files = glob.glob(os.path.join(case_path, pattern), recursive=True)
            trace_files.extend(files)
        
        # 过滤掉小文件和非trace文件
        valid_traces = []
        for file_path in trace_files:
            if os.path.getsize(file_path) > 1024 * 1024:  # 大于1MB
                filename = os.path.basename(file_path).lower()
                if any(keyword in filename for keyword in ['trace', 'cupti', 'ecos']):
                    valid_traces.append(file_path)
        
        if not valid_traces:
            raise FileNotFoundError(f"在 {case_path} 中找不到有效的trace文件")
        
        # 选择最大的trace文件
        trace_file = max(valid_traces, key=os.path.getsize)
        print(f"Trace文件: {trace_file}")
        print(f"文件大小: {os.path.getsize(trace_file)/1024/1024:.1f} MB")
        
        return trace_file
    
    def classify_kernel(self, kernel_name: str) -> str:
        """根据kernel名称分类算子类型"""
        # 检查缓存
        if kernel_name in self.classification_cache:
            return self.classification_cache[kernel_name]
        
        kernel_name_lower = kernel_name.lower()
        
        # 特殊处理：确保gemm_ktype开头的算子被分类为linear
        if kernel_name_lower.startswith('gemm_ktype'):
            self.classification_cache[kernel_name] = 'linear'
            return 'linear'
        
        # 使用预编译的正则表达式
        for op_type, compiled_patterns in self.compiled_patterns.items():
            for pattern in compiled_patterns:
                if pattern.search(kernel_name_lower):
                    # 缓存结果
                    self.classification_cache[kernel_name] = op_type
                    return op_type
        
        # 缓存未分类结果
        self.classification_cache[kernel_name] = 'other'
        return 'other'
    
    def enrich_kernels_with_matched_token_size(self, kernels: List[Dict]) -> List[Dict]:
        """为每个kernel匹配最接近时间戳的token size"""
        
        print(f"\n=== 为 {len(kernels)} 个kernels匹配精确token size ===")
        
        enriched_kernels = []
        token_size_stats = {}
        time_diff_stats = []
        
        for i, kernel in enumerate(kernels):
            if i % 10000 == 0:
                print(f"  处理进度: {i}/{len(kernels)} ({i/len(kernels)*100:.1f}%)")
            
            target_timestamp = kernel.get('timestamp_us', kernel.get('ts', 0))
            
            # 找到最接近的token size
            token_size, closest_timestamp, time_diff_ms = self.token_matcher.find_closest_token_size(target_timestamp)
            
            # 创建enriched kernel
            enriched_kernel = kernel.copy()
            enriched_kernel.update({
                'matched_token_size': token_size,
                'matched_timestamp_us': closest_timestamp,
                'token_size_time_diff_ms': time_diff_ms,
                'token_size_source': 'timeline_matched'
            })
            
            enriched_kernels.append(enriched_kernel)
            
            # 统计token size分布
            token_size_key = int(token_size)
            token_size_stats[token_size_key] = token_size_stats.get(token_size_key, 0) + 1
            time_diff_stats.append(time_diff_ms)
        
        print(f"匹配完成!")
        
        # 统计信息
        print(f"\nToken Size分布:")
        for token_size, count in sorted(token_size_stats.items()):
            print(f"  {token_size} tokens: {count} kernels ({count/len(kernels)*100:.1f}%)")
        
        if time_diff_stats:
            print(f"\n时间差统计:")
            print(f"  平均时间差: {sum(time_diff_stats)/len(time_diff_stats):.1f} ms")
            print(f"  最大时间差: {max(time_diff_stats):.1f} ms")
            print(f"  最小时间差: {min(time_diff_stats):.1f} ms")
        
        return enriched_kernels
    
    def process_case(self, case_path: str) -> Dict[str, Any]:
        """处理指定案例的数据，返回预处理结果（修复版本）"""
        
        print(f"\n=== OEA Stage 1: 数据处理 (修复版本) ===")
        
        # 1. 加载案例配置和execute_token_size时间线
        model_config, hardware_spec, execute_token_timeline = self.load_case_config(case_path)
        
        # 2. 查找trace文件
        trace_file = self.find_trace_file(case_path)
        
        # 3. 读取trace数据
        print(f"\n=== 读取Trace数据 ===")
        try:
            with open(trace_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            raise Exception(f"读取trace文件失败: {e}")
        
        events = data.get('traceEvents', data) if isinstance(data, dict) else data
        kernel_events = [e for e in events if isinstance(e, dict) and 'name' in e and 'ts' in e and e.get('cat') == 'kernel']
        sorted_events = sorted(kernel_events, key=lambda x: x['ts'])
        
        print(f"总事件数: {len(events)}")
        print(f"Kernel事件数: {len(kernel_events)}")
        
        # 4. 为每个kernel匹配精确的token size
        enriched_events = self.enrich_kernels_with_matched_token_size(sorted_events)
        
        # 5. 按算子类型分类kernels
        print(f"\n=== 算子分类 ===")
        operator_kernels = {
            'linear': [],
            'attention': [],
            'rope': [],
            'layernorm': [],
            'activation': [],
            'add_bias': [],
            'moe': [],
            'communication': [],
            'memory': [],
            'reduction': [],
            'other': []
        }
        
        # 添加进度显示
        total_kernels = len(enriched_events)
        processed_count = 0
        
        for event in enriched_events:
            kernel_name = event['name']
            op_type = self.classify_kernel(kernel_name)
            
            # 标准化kernel数据 - 保持与原始stage1兼容的格式
            kernel_data = {
                'name': kernel_name,
                'duration_us': event.get('dur', 0),
                'timestamp_us': event['ts'],
                'matched_token_size': event['matched_token_size'],
                'matched_timestamp_us': event['matched_timestamp_us'],
                'token_size_time_diff_ms': event['token_size_time_diff_ms'],
                'token_size_source': event['token_size_source'],
                'args': event.get('args', {})
            }
            
            operator_kernels[op_type].append(kernel_data)
            
            # 显示进度
            processed_count += 1
            if processed_count % 10000 == 0 or processed_count == total_kernels:
                progress = (processed_count / total_kernels) * 100
                print(f"处理进度: {processed_count}/{total_kernels} ({progress:.1f}%)")
        
        # 打印分类结果
        for op_type, kernels in operator_kernels.items():
            if kernels:
                print(f"{op_type:12s}: {len(kernels):6d} kernels")
        
        # 6. 计算token size统计（基于匹配的值，而不是平均值）
        all_matched_token_sizes = []
        for kernels in operator_kernels.values():
            for kernel in kernels:
                all_matched_token_sizes.append(kernel['matched_token_size'])
        
        if all_matched_token_sizes:
            avg_token_size = np.mean(all_matched_token_sizes)
            min_token_size = min(all_matched_token_sizes)
            max_token_size = max(all_matched_token_sizes)
            unique_token_sizes = len(set(all_matched_token_sizes))
        else:
            avg_token_size = 100  # 默认值
            min_token_size = 100
            max_token_size = 100
            unique_token_sizes = 1
        
        print(f"\n精确Token Size统计:")
        print(f"  平均Token Size: {avg_token_size:.1f}")
        print(f"  Token Size范围: {min_token_size:.0f} - {max_token_size:.0f}")
        print(f"  唯一Token Size数量: {unique_token_sizes}")
        
        # 7. 构建预处理结果 - 保持与原始stage1兼容的格式
        processed_data = {
            'metadata': {
                'case_path': case_path,
                'model_config': model_config,
                'hardware_spec': hardware_spec,
                'avg_token_size': avg_token_size,  # 保持兼容性，但现在基于匹配的值计算
                'total_kernels': len(sorted_events),
                'processing_timestamp': datetime.now().isoformat(),
                # 新增的token匹配相关信息
                'token_matching': {
                    'method': 'timeline_matched_fixed',
                    'timeline_points': len(execute_token_timeline),
                    'has_timeline_data': len(execute_token_timeline) > 0,
                    'min_token_size': min_token_size,
                    'max_token_size': max_token_size,
                    'unique_token_sizes': unique_token_sizes,
                    'time_alignment': 'relative_timestamp_corrected'
                }
            },
            'operator_kernels': operator_kernels,
            'execute_token_timeline': execute_token_timeline,
            'operator_distribution': {op: len(kernels) for op, kernels in operator_kernels.items()}
        }
        
        return processed_data
    
    def save_processed_data(self, processed_data: Dict[str, Any], output_file: str):
        """保存预处理结果"""
        
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
            json.dump(convert_numpy(processed_data), f, indent=2)
        
        print(f"\n预处理结果已保存到: {output_file}")

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='OEA Stage 1数据处理器 (修复版本)')
    parser.add_argument('--case_path', required=True,
                       help='案例路径，包含配置文件和trace数据')
    parser.add_argument('--output', default=None,
                       help='输出文件路径 (默认: 自动生成)')
    
    args = parser.parse_args()
    
    # 检查案例路径
    if not os.path.exists(args.case_path):
        print(f"错误: 案例路径不存在: {args.case_path}")
        sys.exit(1)
    
    # 设置输出文件
    if args.output is None:
        case_name = os.path.basename(args.case_path.rstrip('/'))
        args.output = f"oea_stage1_processed_data.json"
    
    try:
        # 创建处理器
        processor = OEADataProcessorFixed()
        
        # 处理案例
        result = processor.process_case(args.case_path)
        
        # 保存结果
        processor.save_processed_data(result, args.output)
        
        print(f"\n=== Stage 1处理完成 (修复版本) ===")
        print(f"输出文件: {args.output}")
        print(f"请使用以下命令进行 Stage 2 性能分析:")
        print(f"python3 stage2_linear_analyzer.py --input {args.output}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()