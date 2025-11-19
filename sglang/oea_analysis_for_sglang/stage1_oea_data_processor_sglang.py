#!/usr/bin/env python3
"""
OEA Stage 1: SGLang 数据处理和预分析脚本
基于 iteration 检测结果进行 OEA 分析
支持从 CSV 文件和 iteration 检测 JSON 结果进行分析

主要功能:
1. 加载 cases_after_sea_with_mea.csv 中的案例信息
2. 读取 iteration 检测 JSON 结果
3. 按照 prefill/decode 阶段分类算子
4. 生成 OEA 分析所需的数据结构

使用方法:
python stage1_oea_data_processor_sglang.py --csv_file cases_after_sea_with_mea.csv --iteration_json dtoh_iteration_analysis_H800_Qwen3-14B_batch4_input2048_output10.trace.json
"""

import json
import numpy as np
import pandas as pd
import argparse
import os
import sys
from typing import Dict, List, Any, Tuple
import re
import gzip

class SGLangOEADataProcessor:
    def __init__(self):
        """初始化 SGLang OEA 数据处理器"""
        
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
            }
        }

        # 硬件类型映射
        self.hardware_name_mapping = {
            "H20": "NVIDIA_H20_SXM5_96GB",
            "A100": "NVIDIA_A100_SXM4_80GB", 
            "A800": "NVIDIA_A800_SXM4_80GB",
            "H800": "NVIDIA_H800",
            "H100": "NVIDIA_H100",
            "L20": "NVIDIA_L20",
            "A10": "NVIDIA_A10"
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
            self.compiled_patterns[op_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        # 缓存分类结果
        self.classification_cache = {}

    def get_hardware_spec(self, gpu_type: str) -> Dict[str, Any]:
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
                print(f"警告: 未识别的GPU类型 '{gpu_type}'，使用默认H800规格")
                spec = self.hardware_specs["NVIDIA_H800"].copy()
        
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

    def classify_operator(self, kernel_name: str) -> str:
        """对算子进行分类"""
        
        # 检查缓存
        if kernel_name in self.classification_cache:
            return self.classification_cache[kernel_name]
        
        # 按优先级顺序检查各类算子
        for op_type, compiled_patterns in self.compiled_patterns.items():
            for pattern in compiled_patterns:
                if pattern.search(kernel_name):
                    self.classification_cache[kernel_name] = op_type
                    return op_type
        
        # 如果没有匹配到，归类为其他
        self.classification_cache[kernel_name] = 'other'
        return 'other'

    def load_csv_cases(self, csv_file: str) -> pd.DataFrame:
        """加载 CSV 文件中的案例信息"""
        
        print(f"=== 加载 CSV 案例文件 ===")
        print(f"文件路径: {csv_file}")
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"找不到 CSV 文件: {csv_file}")
        
        df = pd.read_csv(csv_file)
        print(f"成功加载 {len(df)} 个案例")
        print(f"列名: {list(df.columns)}")
        
        # 验证必要的列是否存在
        required_columns = ['pod_name', 'model_name', 'GPU_type', 'batch_size', 'token_size']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"警告: 缺少必要的列: {missing_columns}")
        
        return df

    def load_iteration_analysis(self, json_file: str) -> Dict[str, Any]:
        """加载 iteration 检测分析结果"""
        
        print(f"=== 加载 Iteration 分析结果 ===")
        print(f"文件路径: {json_file}")
        
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"找不到 iteration 分析文件: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"成功加载 iteration 分析结果")
        print(f"总 iterations: {data['kernel_analysis']['total_iterations']}")
        print(f"Prefill iterations: {data['kernel_analysis']['prefill_iterations']}")
        print(f"Decode iterations: {data['kernel_analysis']['decode_iterations']}")
        
        return data

    def load_trace_file(self, trace_file: str) -> List[Dict]:
        """加载 trace 文件"""
        
        print(f"=== 加载 Trace 文件 ===")
        print(f"文件路径: {trace_file}")
        
        if not os.path.exists(trace_file):
            raise FileNotFoundError(f"找不到 trace 文件: {trace_file}")
        
        try:
            if trace_file.endswith('.gz'):
                with gzip.open(trace_file, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(trace_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Chrome Trace Format 通常有 'traceEvents' 字段
            if 'traceEvents' in data:
                events = data['traceEvents']
            elif isinstance(data, list):
                events = data
            else:
                print("无法识别的 trace 格式")
                return []
            
            print(f"成功加载 {len(events)} 个 trace events")
            return events
            
        except Exception as e:
            print(f"加载 trace 文件失败: {e}")
            raise

    def extract_kernels_by_iteration(self, trace_events: List[Dict], iteration_data: Dict) -> Dict[str, List[Dict]]:
        """根据 iteration 信息提取各阶段的 kernels"""
        
        print(f"=== 按 Iteration 提取 Kernels ===")
        
        # 提取所有 GPU kernel 事件
        gpu_kernels = []
        for event in trace_events:
            if (event.get('ph') == 'X' and  # Complete events
                event.get('cat', '').lower() in ['kernel', 'gpu', 'cuda'] and
                'name' in event and 'ts' in event and 'dur' in event):
                gpu_kernels.append(event)
        
        print(f"总 GPU kernels: {len(gpu_kernels)}")
        
        # 按 iteration 分类 kernels
        iteration_kernels = {
            'prefill': [],
            'decode': []
        }
        
        iterations = iteration_data.get('iterations', [])
        
        for iteration in iterations:
            iteration_id = iteration['iteration_id']
            phase = iteration['phase']
            start_ts = iteration['start_ts']
            end_ts = iteration['end_ts']
            
            # 找到这个 iteration 时间范围内的 kernels
            iteration_gpu_kernels = []
            for kernel in gpu_kernels:
                kernel_ts = kernel['ts']
                if start_ts <= kernel_ts <= end_ts:
                    iteration_gpu_kernels.append(kernel)
            
            iteration_kernels[phase].extend(iteration_gpu_kernels)
            print(f"Iteration {iteration_id} ({phase}): {len(iteration_gpu_kernels)} kernels")
        
        print(f"Prefill 总 kernels: {len(iteration_kernels['prefill'])}")
        print(f"Decode 总 kernels: {len(iteration_kernels['decode'])}")
        
        return iteration_kernels

    def analyze_operators_by_phase(self, iteration_kernels: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """按阶段分析算子分布"""
        
        print(f"=== 按阶段分析算子分布 ===")
        
        phase_analysis = {}
        
        for phase, kernels in iteration_kernels.items():
            print(f"\n--- 分析 {phase.upper()} 阶段 ---")
            
            # 算子分类统计
            operator_stats = {}
            total_duration = 0
            
            for kernel in kernels:
                kernel_name = kernel['name']
                duration = kernel.get('dur', 0)
                
                # 分类算子
                op_type = self.classify_operator(kernel_name)
                
                if op_type not in operator_stats:
                    operator_stats[op_type] = {
                        'count': 0,
                        'total_duration': 0,
                        'kernels': []
                    }
                
                operator_stats[op_type]['count'] += 1
                operator_stats[op_type]['total_duration'] += duration
                operator_stats[op_type]['kernels'].append(kernel)
                
                total_duration += duration
            
            # 计算比例
            for op_type, stats in operator_stats.items():
                stats['duration_ratio'] = stats['total_duration'] / total_duration if total_duration > 0 else 0
                stats['avg_duration'] = stats['total_duration'] / stats['count'] if stats['count'] > 0 else 0
            
            phase_analysis[phase] = {
                'total_kernels': len(kernels),
                'total_duration': total_duration,
                'operator_stats': operator_stats
            }
            
            # 打印统计信息
            print(f"总 kernels: {len(kernels)}")
            print(f"总持续时间: {total_duration:.1f} μs")
            print("算子分布:")
            
            # 按持续时间排序
            sorted_ops = sorted(operator_stats.items(), 
                              key=lambda x: x[1]['total_duration'], 
                              reverse=True)
            
            for op_type, stats in sorted_ops:
                print(f"  {op_type}: {stats['count']} kernels, "
                      f"{stats['total_duration']:.1f} μs ({stats['duration_ratio']*100:.1f}%)")
        
        return phase_analysis

    def generate_oea_data_structure(self, case_info: Dict, hardware_spec: Dict, 
                                   phase_analysis: Dict, iteration_data: Dict) -> Dict:
        """生成 OEA 分析所需的数据结构"""
        
        print(f"=== 生成 OEA 数据结构 ===")
        
        # 基本案例信息
        oea_data = {
            'case_info': case_info,
            'hardware_spec': hardware_spec,
            'iteration_analysis': iteration_data,
            'phase_analysis': phase_analysis,
            'model_config': {
                'model_name': case_info.get('model_name', 'Unknown'),
                'batch_size': case_info.get('batch_size', 1),
                'input_length': case_info.get('input_length', 2048),
                'output_length': case_info.get('output_length', 10)
            }
        }
        
        # 为每个阶段生成算子效率分析数据
        oea_data['operator_efficiency'] = {}
        
        for phase, analysis in phase_analysis.items():
            phase_data = {
                'phase': phase,
                'total_duration': analysis['total_duration'],
                'operators': {}
            }
            
            for op_type, stats in analysis['operator_stats'].items():
                # 计算算子的基本统计信息
                operator_data = {
                    'type': op_type,
                    'count': stats['count'],
                    'total_duration': stats['total_duration'],
                    'avg_duration': stats['avg_duration'],
                    'duration_ratio': stats['duration_ratio'],
                    'kernels': []
                }
                
                # 添加每个 kernel 的详细信息
                for kernel in stats['kernels']:
                    kernel_data = {
                        'name': kernel['name'],
                        'duration': kernel.get('dur', 0),
                        'timestamp': kernel.get('ts', 0),
                        'classified_type': op_type
                    }
                    operator_data['kernels'].append(kernel_data)
                
                phase_data['operators'][op_type] = operator_data
            
            oea_data['operator_efficiency'][phase] = phase_data
        
        return oea_data

    def save_analysis_result(self, oea_data: Dict, output_file: str):
        """保存分析结果"""
        
        print(f"=== 保存分析结果 ===")
        print(f"输出文件: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(oea_data, f, indent=2, ensure_ascii=False)
        
        print(f"分析结果已保存")

    def find_matching_trace_files(self, pod_name: str, analysis_dir: str) -> Tuple[str, str]:
        """
        根据 pod_name 查找匹配的 trace 文件和 iteration 分析文件
        
        Args:
            pod_name: CSV 中的 pod_name，如 'Qwen3-14B_batch4_input2048_output10'
            analysis_dir: 分析文件夹路径
            
        Returns:
            (trace_file_path, iteration_json_path)
        """
        print(f"=== 查找匹配的文件 ===")
        print(f"Pod name: {pod_name}")
        print(f"分析目录: {analysis_dir}")
        
        # 查找 iteration 分析 JSON 文件
        iteration_json = None
        trace_file = None
        
        # 遍历分析目录中的文件
        for filename in os.listdir(analysis_dir):
            if filename.startswith('dtoh_iteration_analysis_') and filename.endswith('.json'):
                # 提取文件名中的关键信息
                # 例如: dtoh_iteration_analysis_H800_Qwen3-14B_batch4_input2048_output10.trace.json
                base_name = filename.replace('dtoh_iteration_analysis_', '').replace('.json', '')
                
                # 检查是否匹配 pod_name
                if pod_name in base_name or self._fuzzy_match_pod_name(pod_name, base_name):
                    iteration_json = os.path.join(analysis_dir, filename)
                    
                    # 推断对应的 trace 文件名
                    trace_filename = base_name
                    if not trace_filename.endswith('.trace'):
                        trace_filename += '.trace.json'
                    else:
                        trace_filename += '.json'
                    
                    # 查找 trace 文件
                    possible_trace_paths = [
                        os.path.join(analysis_dir, trace_filename),
                        os.path.join(analysis_dir, base_name + '.json'),
                        os.path.join(analysis_dir, base_name),
                        # 也在上级目录查找
                        os.path.join(os.path.dirname(analysis_dir), trace_filename),
                        os.path.join(os.path.dirname(analysis_dir), 'H800', trace_filename.replace('H800_', '')),
                    ]
                    
                    for trace_path in possible_trace_paths:
                        if os.path.exists(trace_path):
                            trace_file = trace_path
                            break
                    
                    if trace_file:
                        break
        
        if not iteration_json:
            raise FileNotFoundError(f"找不到匹配 pod_name '{pod_name}' 的 iteration 分析文件")
        
        if not trace_file:
            print(f"警告: 找不到匹配的 trace 文件，将从 iteration JSON 中获取路径")
            trace_file = None
        
        print(f"找到 iteration 分析文件: {iteration_json}")
        print(f"找到 trace 文件: {trace_file}")
        
        return trace_file, iteration_json

    def _fuzzy_match_pod_name(self, pod_name: str, filename: str) -> bool:
        """模糊匹配 pod_name 和文件名"""
        
        # 提取关键信息进行匹配
        pod_parts = pod_name.replace('-', '_').split('_')
        file_parts = filename.replace('-', '_').split('_')
        
        # 检查模型名、batch size、input/output 等关键信息
        key_matches = 0
        for part in pod_parts:
            if part.lower() in [fp.lower() for fp in file_parts]:
                key_matches += 1
        
        # 如果匹配度超过 50%，认为是匹配的
        return key_matches >= len(pod_parts) * 0.5

    def extract_case_info_from_csv_row(self, row: pd.Series) -> Dict[str, Any]:
        """从 CSV 行中提取案例信息"""
        
        return {
            'pod_name': row.get('pod_name', 'Unknown'),
            'model_name': row.get('model_name', 'Unknown'),
            'gpu_type': row.get('GPU_type', 'H800'),
            'batch_size': int(row.get('batch_size', 1)),
            'token_size': int(row.get('token_size', 2048)),
            'seq_len': int(row.get('seq_len', 2048)),
            'iteration': int(row.get('iteration', 10)),
            'gpu_util': float(row.get('GPU_util', 0.0)),
            'qps': float(row.get('qps', 0.0)),
            'fpr': float(row.get('FPR', 0.0)),
            'iips': float(row.get('IIPS', 0.0)),
            'mie': float(row.get('MIE', 0.0))
        }

    def process_case_from_csv(self, csv_file: str, pod_name: str = None, analysis_dir: str = None) -> Dict:
        """从 CSV 文件处理指定案例"""
        
        print(f"=== 从 CSV 处理案例 ===")
        
        if analysis_dir is None:
            analysis_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1. 加载 CSV 案例信息
        df = self.load_csv_cases(csv_file)
        
        # 2. 选择要处理的案例
        if pod_name:
            # 查找指定的 pod_name
            matching_rows = df[df['pod_name'] == pod_name]
            if len(matching_rows) == 0:
                raise ValueError(f"在 CSV 中找不到 pod_name '{pod_name}'")
            case_row = matching_rows.iloc[0]
        else:
            # 使用第一个案例
            if len(df) == 0:
                raise ValueError("CSV 文件为空")
            case_row = df.iloc[0]
            pod_name = case_row['pod_name']
        
        print(f"处理案例: {pod_name}")
        
        # 3. 提取案例信息
        case_info = self.extract_case_info_from_csv_row(case_row)
        
        # 4. 查找匹配的文件
        trace_file, iteration_json = self.find_matching_trace_files(pod_name, analysis_dir)
        
        # 5. 加载 iteration 分析结果
        iteration_data = self.load_iteration_analysis(iteration_json)
        
        # 6. 从 iteration 数据中获取 trace 文件路径（如果之前没找到）
        if trace_file is None:
            trace_file = iteration_data.get('trace_file', '')
            if not trace_file or not os.path.exists(trace_file):
                raise FileNotFoundError(f"无法找到对应的 trace 文件")
        
        # 7. 加载 trace 数据
        trace_events = self.load_trace_file(trace_file)
        
        # 8. 按 iteration 提取 kernels
        iteration_kernels = self.extract_kernels_by_iteration(trace_events, iteration_data)
        
        # 9. 按阶段分析算子
        phase_analysis = self.analyze_operators_by_phase(iteration_kernels)
        
        # 10. 获取硬件规格
        hardware_spec = self.get_hardware_spec(case_info['gpu_type'])
        
        # 11. 生成 OEA 数据结构
        oea_data = self.generate_oea_data_structure(
            case_info, hardware_spec, phase_analysis, iteration_data
        )
        
        return oea_data

    def process_case(self, csv_file: str, iteration_json: str, trace_file: str = None) -> Dict:
        """处理单个案例（保持向后兼容）"""
        
        print(f"=== 开始处理案例（兼容模式）===")
        
        # 从 iteration_json 文件名推断 pod_name
        # 文件名格式: dtoh_iteration_analysis__Users_glenn_Downloads_sglang_cases_after_sea_A800_Qwen2.5-3B_batch1_input1024_output10_Qwen2.5-3B_batch1_input1024_output10.trace.json
        # 需要提取: Qwen2.5-3B_batch1_input1024_output10
        base_name = os.path.basename(iteration_json)
        if base_name.startswith('dtoh_iteration_analysis_'):
            # 去掉前缀和后缀
            temp_name = base_name.replace('dtoh_iteration_analysis_', '').replace('.trace.json', '').replace('.json', '')
            
            # 从路径中提取最后一个重复的部分作为 pod_name
            # 格式: _Users_glenn_Downloads_sglang_cases_after_sea_A800_Qwen2.5-3B_batch1_input1024_output10_Qwen2.5-3B_batch1_input1024_output10
            # 分割并找到重复的部分
            parts = temp_name.split('_')
            
            # 尝试从目录路径中提取 pod_name
            # 目录格式: /Users/glenn/Downloads/sglang/cases_after_sea/A800/Qwen2.5-3B_batch1_input1024_output10/
            analysis_dir = os.path.dirname(iteration_json)
            dir_name = os.path.basename(analysis_dir)
            
            # 如果目录名匹配 pod_name 格式，使用目录名
            if dir_name and '_batch' in dir_name:
                inferred_pod_name = dir_name
                print(f"从目录名推断 pod_name: {inferred_pod_name}")
            else:
                # 否则尝试从文件名末尾提取
                # 查找最后一个看起来像 pod_name 的部分 (包含 _batch)
                for i in range(len(parts) - 1, -1, -1):
                    if '_batch' in '_'.join(parts[i:]):
                        potential_name = '_'.join(parts[i:])
                        # 检查是否包含必要的组件
                        if '_batch' in potential_name and '_input' in potential_name and '_output' in potential_name:
                            inferred_pod_name = potential_name
                            print(f"从文件名推断 pod_name: {inferred_pod_name}")
                            break
                else:
                    inferred_pod_name = None
                    print(f"警告: 无法从文件名推断 pod_name")
        else:
            inferred_pod_name = None
            print(f"警告: 文件名格式不符合预期")
        
        analysis_dir = os.path.dirname(iteration_json)
        
        return self.process_case_from_csv(csv_file, inferred_pod_name, analysis_dir)

def main():
    parser = argparse.ArgumentParser(description='SGLang OEA Stage 1 数据处理器')
    parser.add_argument('--csv_file', required=True, help='CSV 案例文件路径')
    parser.add_argument('--pod_name', help='指定要处理的 pod_name（可选，默认处理第一个）')
    parser.add_argument('--analysis_dir', help='分析文件夹路径（可选，默认为脚本所在目录）')
    parser.add_argument('--output', help='输出文件路径（可选）')
    
    # 兼容旧版本参数
    parser.add_argument('--iteration_json', help='Iteration 分析 JSON 文件路径（兼容模式）')
    parser.add_argument('--trace_file', help='Trace 文件路径（兼容模式）')
    
    args = parser.parse_args()
    
    try:
        # 创建处理器
        processor = SGLangOEADataProcessor()
        
        # 判断使用新模式还是兼容模式
        if args.iteration_json:
            # 兼容模式：使用旧的参数格式
            print("使用兼容模式处理...")
            oea_data = processor.process_case(
                args.csv_file, 
                args.iteration_json, 
                args.trace_file
            )
            
            # 生成输出文件名和路径
            if args.output:
                output_file = args.output
            else:
                # 获取pod_name
                pod_name = oea_data['case_info'].get('pod_name', 'unknown')
                # 获取iteration_json所在目录
                analysis_dir = os.path.dirname(args.iteration_json)
                # 生成输出文件路径：案例json所在文件夹/oea_stage1_pod_name_processed.json
                output_file = os.path.join(analysis_dir, f"oea_stage1_{pod_name}_processed.json")
        else:
            # 新模式：从 CSV 和分析目录处理
            print("使用新模式处理...")
            oea_data = processor.process_case_from_csv(
                args.csv_file,
                args.pod_name,
                args.analysis_dir
            )
            
            # 生成输出文件名和路径
            if args.output:
                output_file = args.output
            else:
                # 获取pod_name
                pod_name = args.pod_name or oea_data['case_info'].get('pod_name', 'unknown')
                # 获取分析目录
                analysis_dir = args.analysis_dir or os.getcwd()
                # 生成输出文件路径：案例json所在文件夹/oea_stage1_pod_name_processed.json
                output_file = os.path.join(analysis_dir, f"oea_stage1_{pod_name}_processed.json")
        
        # 保存结果
        processor.save_analysis_result(oea_data, output_file)
        
        print(f"\n=== 处理完成 ===")
        print(f"案例信息:")
        print(f"  Pod Name: {oea_data['case_info'].get('pod_name', 'Unknown')}")
        print(f"  Model: {oea_data['case_info'].get('model_name', 'Unknown')}")
        print(f"  GPU Type: {oea_data['case_info'].get('gpu_type', 'Unknown')}")
        print(f"  Batch Size: {oea_data['case_info'].get('batch_size', 'Unknown')}")
        print(f"输出文件: {output_file}")
        
        # 打印简要统计
        if 'phase_analysis' in oea_data:
            for phase, analysis in oea_data['phase_analysis'].items():
                total_kernels = analysis.get('total_kernels', 0)
                total_duration = analysis.get('total_duration', 0)
                print(f"  {phase.upper()}: {total_kernels} kernels, {total_duration:.1f} μs")
        
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()