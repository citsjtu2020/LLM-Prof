#!/usr/bin/env python3
"""
OEA Stage 4: 增强版算子效率分析器 (修复Linear算子识别)
正确处理Stage 2的projection_breakdown数据结构

主要改进:
1. 正确识别Stage 2中的Linear算子projection类型
2. 支持MoE和Linear两种分析模式
3. 从projection_breakdown中提取准确的Linear算子数据
4. 完整整合所有三个stage的分析结果
5. 准确计算时间占比，包括所有Linear算子的贡献

Stage 2 Linear算子类型:
- MoE模式: qkv_proj, o_proj, router, up_proj, down_proj, lm_head
- Linear模式: qkv_proj, o_proj, gate_proj, up_proj, down_proj, lm_head

使用方法:
python stage4_oea_efficiency_analyzer_enhanced_fixed.py --case_path traces_after_sea_section_part1/case_name
"""

import json
import numpy as np
import argparse
import os
import sys
import math
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

class OEAEfficiencyAnalyzerEnhancedFixed:
    def __init__(self):
        """初始化增强版OEA效率分析器"""
        
        # Linear算子的projection类型 - 支持两种模式
        self.moe_linear_projections = [
            'qkv_proj', 'o_proj', 'router', 'up_proj', 'down_proj', 'lm_head'
        ]
        
        self.standard_linear_projections = [
            'qkv_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head'
        ]
        
        # 当前使用的Linear projection类型（运行时确定）
        self.current_linear_projections = []
        
        # 非Linear算子类型 - 从Stage 3获取
        self.nonlinear_types = [
            'attention', 'layernorm', 'activation', 'rope', 'reduction', 'memory', 'moe'
        ]
        
        # 其他算子类型 - 从Stage 1获取
        self.other_types = [
            'add_bias', 'other', 'communication'
        ]
        
        # 算子类别分类 - 基于计算特性（动态更新）
        self.operator_categories = {
            # 计算密集型: 高FLOPS，主要受计算能力限制
            'compute_intensive': set(['moe', 'attention']),  # Linear projections会动态添加
            # 内存密集型: 低FLOPS，主要受内存带宽限制  
            'memory_intensive': {'layernorm', 'rope', 'reduction', 'activation'},
            # 非计算类开销: 辅助操作，不涉及主要计算
            'overhead': {'memory', 'add_bias', 'other', 'communication'}
        }
        
        print(f"=== 初始化增强版OEA效率分析器 (修复Linear算子识别) ===")
        print(f"支持的Linear模式:")
        print(f"  MoE模式: {self.moe_linear_projections}")
        print(f"  Linear模式: {self.standard_linear_projections}")
        print(f"非Linear算子类型: {self.nonlinear_types}")
        print(f"其他算子类型: {self.other_types}")
    
    def detect_linear_analysis_mode(self, stage2_data: Optional[Dict]) -> str:
        """检测Stage 2的分析模式（MoE或Linear）"""
        
        if stage2_data is None:
            return 'unknown'
        
        if 'linear_analysis' not in stage2_data:
            return 'unknown'
        
        linear_analysis = stage2_data['linear_analysis']
        
        if 'projection_breakdown' not in linear_analysis:
            return 'unknown'
        
        projection_breakdown = linear_analysis['projection_breakdown']
        available_projections = set(projection_breakdown.keys())
        
        # 检查是否包含router（MoE模式特有）
        if 'router' in available_projections:
            mode = 'moe'
            self.current_linear_projections = self.moe_linear_projections
        elif 'gate_proj' in available_projections:
            mode = 'linear'
            self.current_linear_projections = self.standard_linear_projections
        else:
            # 根据可用的projections推断
            if len(available_projections & set(self.moe_linear_projections)) > len(available_projections & set(self.standard_linear_projections)):
                mode = 'moe'
                self.current_linear_projections = self.moe_linear_projections
            else:
                mode = 'linear'
                self.current_linear_projections = self.standard_linear_projections
        
        # 更新算子类别分类
        self.operator_categories['compute_intensive'].update(set(self.current_linear_projections))
        
        print(f"\n=== 检测到Stage 2分析模式: {mode.upper()} ===")
        print(f"当前Linear projections: {self.current_linear_projections}")
        print(f"可用projections: {list(available_projections)}")
        
        return mode
    
    def get_all_operator_types(self) -> List[str]:
        """获取所有算子类型（包括当前模式的Linear projections）"""
        return self.current_linear_projections + self.nonlinear_types + self.other_types
    
    def get_kernel_token_size(self, kernel: Dict) -> int:
        """获取kernel的token size - 优先使用matched_token_size，否则使用默认值"""
        if 'matched_token_size' in kernel:
            return int(kernel['matched_token_size'])
        elif 'token_size' in kernel:
            return int(kernel['token_size'])
        return 2048
    
    def extract_end_to_end_time(self, stage1_data: Dict) -> Dict:
        """提取端侧时间信息"""
        
        metadata = stage1_data.get('metadata', {})
        
        end_to_end_info = {
            'total_end_to_end_us': 0,
            'inference_start_time': 0,
            'inference_end_time': 0,
            'data_source': 'unknown'
        }
        
        # 尝试从不同字段获取端侧时间
        if 'inference_time_us' in metadata:
            end_to_end_info['total_end_to_end_us'] = metadata['inference_time_us']
            end_to_end_info['data_source'] = 'metadata.inference_time_us'
        elif 'total_time_us' in metadata:
            end_to_end_info['total_end_to_end_us'] = metadata['total_time_us']
            end_to_end_info['data_source'] = 'metadata.total_time_us'
        elif 'timing' in metadata:
            timing = metadata['timing']
            if 'inference_start_time' in timing and 'inference_end_time' in timing:
                start_time = timing['inference_start_time']
                end_time = timing['inference_end_time']
                end_to_end_info['total_end_to_end_us'] = (end_time - start_time) * 1e6
                end_to_end_info['inference_start_time'] = start_time
                end_to_end_info['inference_end_time'] = end_time
                end_to_end_info['data_source'] = 'metadata.timing'
        
        # 如果没有找到端侧时间，从kernel时间估算
        if end_to_end_info['total_end_to_end_us'] == 0:
            total_kernel_time = 0
            if 'operator_kernels' in stage1_data:
                for op_type, kernels in stage1_data['operator_kernels'].items():
                    if isinstance(kernels, list):
                        for kernel in kernels:
                            total_kernel_time += kernel.get('duration_us', 0)
            
            # 估算端侧时间为kernel时间的1.3倍（考虑调度开销）
            end_to_end_info['total_end_to_end_us'] = total_kernel_time * 1.3
            end_to_end_info['data_source'] = 'estimated_from_kernels'
        
        return end_to_end_info
    
    def load_all_stage_results(self, case_path: str) -> Tuple[Dict, Optional[Dict], Optional[Dict]]:
        """加载所有stage的分析结果"""
        
        print(f"\n=== 加载所有Stage分析结果 (增强版本) ===")
        print(f"案例路径: {case_path}")
        
        # Stage 1: 预处理数据 (必需)
        stage1_files = [
            "oea_stage1_processed_data_fixed.json",
            "oea_stage1_processed_data.json"
        ]
        
        stage1_data = None
        for stage1_file in stage1_files:
            full_path = os.path.join(case_path, stage1_file)
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    stage1_data = json.load(f)
                print(f"✓ Stage 1数据加载完成: {stage1_file}")
                print(f"  算子类型数: {len(stage1_data.get('operator_kernels', {}))}")
                break
        
        if stage1_data is None:
            raise FileNotFoundError(f"Stage 1结果文件不存在，尝试了: {stage1_files}")
        
        # Stage 2: Linear算子分析 (可选，但重要)
        stage2_files = [
            "oea_stage2_linear_analysis_results_fixed.json",
            "oea_stage2_linear_analysis_results.json",
            "oea_stage2_moe_linear_analysis_results_fixed.json",
            "oea_stage2_moe_linear_analysis_results.json"
        ]
        
        stage2_data = None
        for stage2_file in stage2_files:
            full_path = os.path.join(case_path, stage2_file)
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    stage2_data = json.load(f)
                print(f"✓ Stage 2数据加载完成: {stage2_file}")
                print(f"  Linear算子覆盖率: {stage2_data.get('linear_coverage', 0):.1f}%")
                break
        
        if stage2_data is None:
            print("⚠ Stage 2结果文件不存在，将从Stage 1提取Linear算子数据")
        
        # Stage 3: 非Linear算子分析 (可选)
        stage3_files = [
            "oea_stage3_nonlinear_analysis_results_fixed.json",
            "oea_stage3_nonlinear_analysis_results.json"
        ]
        
        stage3_data = None
        for stage3_file in stage3_files:
            full_path = os.path.join(case_path, stage3_file)
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    stage3_data = json.load(f)
                print(f"✓ Stage 3数据加载完成: {stage3_file}")
                nonlinear_coverage = stage3_data.get('nonlinear_analysis', {}).get('nonlinear_coverage', 0)
                print(f"  非Linear算子覆盖率: {nonlinear_coverage:.1f}%")
                break
        
        if stage3_data is None:
            print("⚠ Stage 3结果文件不存在，将从Stage 1提取非Linear算子数据")
        
        return stage1_data, stage2_data, stage3_data
    
    def extract_hardware_specs(self, stage1_data: Dict) -> Dict:
        """提取硬件规格信息"""
        
        hardware_spec = stage1_data['metadata']['hardware_spec']
        
        specs = {
            'phi': hardware_spec['peak_flops_fp16'] / 1e12,  # φ: 峰值计算吞吐量 (TFLOPs/s)
            'pi': hardware_spec['memory_bandwidth'] / 1e9,   # π: 峰值内存带宽 (GB/s)
            'memory_size': hardware_spec['memory_size'] / 1e9,  # 显存大小 (GB)
            'gpu_name': hardware_spec['name'],
            'n_gpu': hardware_spec.get('n_gpu', 1),
            'tensor_parallelism': hardware_spec.get('tensor_parallelism', 1)
        }
        
        print(f"\n=== 硬件规格 ===")
        print(f"GPU: {specs['gpu_name']}")
        print(f"峰值计算能力 φ: {specs['phi']:.1f} TFLOPs/s")
        print(f"峰值内存带宽 π: {specs['pi']:.1f} GB/s")
        print(f"显存大小: {specs['memory_size']:.0f} GB")
        print(f"GPU数量: {specs['n_gpu']}")
        
        return specs
    
    def extract_linear_operator_data_from_stage2(self, operator_type: str, stage2_data: Dict) -> Dict:
        """从Stage 2中提取Linear算子数据（修复版本）"""
        
        operator_data = {
            'total_flops': 0,
            'total_memory_access': 0,
            'total_duration_us': 0,
            'kernel_count': 0,
            'executions': [],
            'data_source': 'stage2_not_found',
            'uses_precise_token_size': False,
            'token_size_variation': 0
        }
        
        if stage2_data is None:
            return operator_data
        
        # 检查linear_analysis结构
        if 'linear_analysis' not in stage2_data:
            return operator_data
        
        linear_analysis = stage2_data['linear_analysis']
        
        # 主要方法: 从projection_breakdown获取
        if 'projection_breakdown' in linear_analysis and operator_type in linear_analysis['projection_breakdown']:
            proj_data = linear_analysis['projection_breakdown'][operator_type]
            operator_data['total_flops'] = proj_data.get('total_flops', 0)
            operator_data['total_duration_us'] = proj_data.get('total_duration_us', 0)
            operator_data['kernel_count'] = proj_data.get('executions', 0)  # executions就是kernel数量
            operator_data['data_source'] = 'stage2_projection_breakdown'
            
            # 计算内存访问量 (从projection_breakdown直接获取)
            operator_data['total_memory_access'] = proj_data.get('total_memory_access', 0) / 1e9  # 转换为GB
            
            # Stage 2使用精确token size
            token_size_method = linear_analysis.get('token_size_method', 'unknown')
            operator_data['uses_precise_token_size'] = 'exact' in token_size_method or 'matched' in token_size_method
            operator_data['token_size_variation'] = 1  # projection级别的统计
            
            return operator_data
        
        # 备选方法: 从operator_breakdown获取
        if 'operator_breakdown' in linear_analysis and operator_type in linear_analysis['operator_breakdown']:
            op_data = linear_analysis['operator_breakdown'][operator_type]
            operator_data['total_flops'] = op_data.get('total_flops', 0)
            operator_data['total_duration_us'] = op_data.get('total_duration_us', 0)
            operator_data['kernel_count'] = op_data.get('kernel_count', 0)
            operator_data['data_source'] = 'stage2_operator_breakdown'
            
            # 检查是否使用了精确token size
            operator_data['uses_precise_token_size'] = op_data.get('uses_precise_token_size', False)
            operator_data['token_size_variation'] = op_data.get('token_size_variation', 0)
            
            # 计算内存访问量
            if operator_data['total_flops'] > 0:
                operator_data['total_memory_access'] = operator_data['total_flops'] * 40 / 1e12  # 转换为GB
            
            return operator_data
        
        return operator_data
    
    def extract_nonlinear_operator_data_from_stage3(self, operator_type: str, stage3_data: Dict) -> Dict:
        """从Stage 3中提取非Linear算子数据"""
        
        operator_data = {
            'total_flops': 0,
            'total_memory_access': 0,
            'total_duration_us': 0,
            'kernel_count': 0,
            'executions': [],
            'data_source': 'stage3_not_found',
            'uses_precise_token_size': False,
            'token_size_variation': 0
        }
        
        if stage3_data is None:
            return operator_data
        
        # 从stage3获取非Linear算子数据
        if 'nonlinear_analysis' in stage3_data:
            nonlinear_analysis = stage3_data['nonlinear_analysis']
            if 'operator_analysis' in nonlinear_analysis:
                op_analysis = nonlinear_analysis['operator_analysis']
                if operator_type in op_analysis:
                    op_data = op_analysis[operator_type]
                    operator_data['total_flops'] = op_data.get('total_flops', 0)
                    operator_data['total_memory_access'] = op_data.get('total_memory_access', 0) / 1e9  # 转换为GB
                    operator_data['total_duration_us'] = op_data.get('total_duration_us', 0)
                    operator_data['kernel_count'] = op_data.get('kernel_count', 0)
                    operator_data['data_source'] = 'stage3_operator_analysis'
                    
                    # 检查是否使用了精确token size
                    operator_data['uses_precise_token_size'] = op_data.get('uses_precise_token_size', False)
                    operator_data['token_size_variation'] = op_data.get('token_size_variation', 0)
        
        return operator_data
    
    def extract_other_operator_data_from_stage1(self, operator_type: str, stage1_data: Dict) -> Dict:
        """从Stage 1中提取其他算子数据"""
        
        operator_data = {
            'total_flops': 0,
            'total_memory_access': 0,
            'total_duration_us': 0,
            'kernel_count': 0,
            'executions': [],
            'data_source': 'stage1_not_found',
            'uses_precise_token_size': False,
            'token_size_variation': 0
        }
        
        # 从stage1获取其他算子数据（add_bias, other等）
        if 'operator_kernels' in stage1_data:
            op_kernels = stage1_data['operator_kernels']
            if operator_type in op_kernels:
                kernels = op_kernels[operator_type]
                if isinstance(kernels, list) and len(kernels) > 0:
                    operator_data['kernel_count'] = len(kernels)
                    operator_data['total_duration_us'] = sum(k.get('duration_us', 0) for k in kernels)
                    operator_data['data_source'] = 'stage1_operator_kernels'
                    
                    # 检查是否使用了精确token size
                    token_sizes = set()
                    for kernel in kernels:
                        token_size = self.get_kernel_token_size(kernel)
                        token_sizes.add(token_size)
                    
                    operator_data['uses_precise_token_size'] = any('matched_token_size' in k for k in kernels)
                    operator_data['token_size_variation'] = len(token_sizes)
                    
                    # 对于辅助算子，FLOPS通常为0或很小
                    operator_data['total_flops'] = 0
                    operator_data['total_memory_access'] = operator_data['total_duration_us'] * 0.1 / 1e6  # 估算内存访问
        
        return operator_data
    
    def extract_comprehensive_operator_data(self, operator_type: str, stage1_data: Dict, 
                                          stage2_data: Optional[Dict], stage3_data: Optional[Dict]) -> Dict:
        """从所有stage中提取完整的算子数据（增强版本）"""
        
        # Linear算子projection类型 - 优先从Stage 2获取详细数据
        if operator_type in self.current_linear_projections:
            operator_data = self.extract_linear_operator_data_from_stage2(operator_type, stage2_data)
            
            # 如果Stage 2没有数据，尝试从Stage 1获取
            if operator_data['data_source'] == 'stage2_not_found':
                operator_data = self.extract_other_operator_data_from_stage1(operator_type, stage1_data)
                operator_data['data_source'] = 'stage1_fallback_for_linear'
        
        # 非Linear算子 - 优先从Stage 3获取
        elif operator_type in self.nonlinear_types:
            operator_data = self.extract_nonlinear_operator_data_from_stage3(operator_type, stage3_data)
            
            # 如果Stage 3没有数据，尝试从Stage 1获取
            if operator_data['data_source'] == 'stage3_not_found':
                operator_data = self.extract_other_operator_data_from_stage1(operator_type, stage1_data)
                operator_data['data_source'] = 'stage1_fallback_for_nonlinear'
        
        # 其他算子 - 从Stage 1获取
        else:
            operator_data = self.extract_other_operator_data_from_stage1(operator_type, stage1_data)
        
        return operator_data
    
    def calculate_dynamic_roofline(self, operator_type: str, flops: float, memory_access: float,
                                 hardware_specs: Dict) -> Dict:
        """计算动态Roofline模型参数"""
        
        # 算术强度 AI_O = R̂_O / (M̂_O / 1000)
        flops_tflops = flops / 1e12  # 转换为TFLOPs
        memory_gb = memory_access  # 已经是GB
        
        if memory_gb > 0:
            arithmetic_intensity = flops_tflops / memory_gb
        else:
            arithmetic_intensity = float('inf') if flops_tflops > 0 else 0
        
        # 硬件参数
        phi = hardware_specs['phi']  # 峰值计算吞吐量 (TFLOPs/s)
        pi = hardware_specs['pi']    # 峰值内存带宽 (GB/s)
        
        # 计算性能上界 r*_O
        memory_bound_threshold = phi / pi  # φ/π
        
        if arithmetic_intensity < memory_bound_threshold:
            # Memory-bound: r*_O = AI_O × π
            performance_bound = arithmetic_intensity * pi
            bound_type = "Memory-bound"
        else:
            # Compute-bound: r*_O = φ
            performance_bound = phi
            bound_type = "Compute-bound"
        
        return {
            'arithmetic_intensity': arithmetic_intensity,
            'performance_bound_tflops': performance_bound,
            'bound_type': bound_type,
            'memory_bound_threshold': memory_bound_threshold,
            'flops_tflops': flops_tflops,
            'memory_gb': memory_gb
        }
    
    def calculate_operator_efficiency(self, operator_type: str, operator_data: Dict,
                                    roofline_params: Dict) -> Dict:
        """计算算子效率度 δ_O = r̂_O / r*_O"""
        
        # 获取实际执行时间
        total_duration_us = operator_data.get('total_duration_us', 0)
        
        # 计算实际吞吐量 r̂_O = R̂_O / dur̂_O
        if total_duration_us > 0:
            duration_seconds = total_duration_us * 1e-6
            actual_throughput_tflops = roofline_params['flops_tflops'] / duration_seconds
        else:
            actual_throughput_tflops = 0
        
        # 计算效率度 δ_O = r̂_O / r*_O
        performance_bound = roofline_params['performance_bound_tflops']
        if performance_bound > 0:
            efficiency_degree = actual_throughput_tflops / performance_bound
        else:
            efficiency_degree = 0
        
        # 限制效率度在合理范围内
        efficiency_degree = min(efficiency_degree, 1.0)
        
        return {
            'total_duration_us': total_duration_us,
            'actual_throughput_tflops': actual_throughput_tflops,
            'efficiency_degree': efficiency_degree,
            'kernel_count': operator_data.get('kernel_count', 0),
            'uses_precise_token_size': operator_data.get('uses_precise_token_size', False),
            'token_size_variation': operator_data.get('token_size_variation', 0)
        }
    
    def calculate_time_proportions(self, operator_data: Dict, total_kernel_time_us: float, 
                                 total_end_to_end_us: float) -> Dict:
        """计算时间占比（相对于kernel时间和端侧时间）"""
        
        operator_time_us = operator_data.get('total_duration_us', 0)
        
        # 相对于总kernel时间的占比（论文指标）
        kernel_proportion = operator_time_us / total_kernel_time_us if total_kernel_time_us > 0 else 0
        
        # 相对于端侧时间的占比（新增指标）
        end_to_end_proportion = operator_time_us / total_end_to_end_us if total_end_to_end_us > 0 else 0
        
        return {
            'kernel_time_proportion': min(kernel_proportion, 1.0),
            'end_to_end_proportion': min(end_to_end_proportion, 1.0)
        }
    
    def calculate_bottleneck_score(self, efficiency_degree: float, time_proportion: float) -> float:
        """计算瓶颈评分 β_O = (1 - δ_O) × p_O"""
        
        bottleneck_score = (1 - efficiency_degree) * time_proportion
        return max(0, bottleneck_score)  # 确保非负
    
    def safe_get_kernel_count(self, kernel_count_value) -> int:
        """安全获取kernel_count的整数值，处理可能是列表的情况"""
        if isinstance(kernel_count_value, list):
            return len(kernel_count_value)
        elif isinstance(kernel_count_value, (int, float)):
            return int(kernel_count_value)
        else:
            return 0

    def analyze_comprehensive_efficiency(self, stage1_data: Dict, stage2_data: Optional[Dict], 
                                       stage3_data: Optional[Dict]) -> Dict:
        """综合效率分析（增强版本）"""
        
        print(f"\n=== 综合效率分析 (增强版本 - 修复Linear算子识别) ===")
        
        # 检测Stage 2分析模式并设置Linear projections
        analysis_mode = self.detect_linear_analysis_mode(stage2_data)
        
        # 提取端侧时间
        end_to_end_info = self.extract_end_to_end_time(stage1_data)
        total_end_to_end_us = end_to_end_info['total_end_to_end_us']
        
        print(f"\n端侧时间信息:")
        print(f"  总端侧时间: {total_end_to_end_us/1000:.1f} ms")
        print(f"  数据源: {end_to_end_info['data_source']}")
        
        # 提取硬件规格
        hardware_specs = self.extract_hardware_specs(stage1_data)
        
        # 获取所有算子类型（包括当前模式的Linear projections）
        all_operator_types = self.get_all_operator_types()
        
        # 分析所有算子类型
        operator_results = {}
        total_compute_time_us = 0
        total_flops = 0
        total_memory_access = 0
        
        # 按类别统计
        category_results = {
            'compute_intensive': {},
            'memory_intensive': {},
            'overhead': {}
        }
        category_times = {'compute_intensive': 0, 'memory_intensive': 0, 'overhead': 0}
        
        print(f"\n=== 算子数据提取 (增强版本 - 包括{analysis_mode.upper()}模式Linear projections) ===")
        
        # 统计数据源
        data_source_stats = {}
        linear_projection_stats = {}
        
        for operator_type in all_operator_types:
            # 提取算子数据
            operator_data = self.extract_comprehensive_operator_data(
                operator_type, stage1_data, stage2_data, stage3_data
            )
            
            # 统计数据源
            data_source = operator_data['data_source']
            data_source_stats[data_source] = data_source_stats.get(data_source, 0) + 1
            
            # 统计Linear projection情况
            if operator_type in self.current_linear_projections:
                linear_projection_stats[operator_type] = {
                    'found': self.safe_get_kernel_count(operator_data['kernel_count']) > 0,
                    'data_source': data_source,
                    'duration_ms': operator_data['total_duration_us'] / 1000
                }
            kernel_count_value = self.safe_get_kernel_count(operator_data['kernel_count'])
            if kernel_count_value > 0:
                print(f"{operator_type:12s}: {kernel_count_value:4d} kernels, "
                      f"{operator_data['total_duration_us']/1000:6.1f} ms, "
                      f"数据源: {operator_data['data_source']}")
                
                if operator_data['uses_precise_token_size']:
                    print(f"              使用精确Token Size ({operator_data['token_size_variation']}种不同值)")
                else:
                    print(f"              使用平均Token Size")
                
                # 累计统计
                total_compute_time_us += operator_data['total_duration_us']
                total_flops += operator_data['total_flops']
                total_memory_access += operator_data['total_memory_access']
                
                # 按类别统计
                for category, operators in self.operator_categories.items():
                    if operator_type in operators:
                        category_times[category] += operator_data['total_duration_us']
                        break
                
                # 计算Roofline参数
                roofline_params = self.calculate_dynamic_roofline(
                    operator_type, operator_data['total_flops'], 
                    operator_data['total_memory_access'], hardware_specs
                )
                
                # 计算效率度
                efficiency_metrics = self.calculate_operator_efficiency(
                    operator_type, operator_data, roofline_params
                )
                
                # 存储结果
                operator_results[operator_type] = {
                    'operator_data': operator_data,
                    'roofline_params': roofline_params,
                    'efficiency_metrics': efficiency_metrics
                }
                
                # 按类别存储
                for category, operators in self.operator_categories.items():
                    if operator_type in operators:
                        category_results[category][operator_type] = operator_results[operator_type]
                        break
            else:
                print(f"{operator_type:12s}: 未找到数据 (数据源: {operator_data['data_source']})")
        
        print(f"\n数据源统计:")
        for source, count in data_source_stats.items():
            print(f"  {source}: {count} 个算子")
        
        print(f"\nLinear Projection统计 ({analysis_mode.upper()}模式):")
        linear_found = 0
        linear_total_time = 0
        for proj_type, stats in linear_projection_stats.items():
            status = "✓" if stats['found'] else "✗"
            print(f"  {status} {proj_type}: {stats['duration_ms']:.1f}ms, 数据源: {stats['data_source']}")
            if stats['found']:
                linear_found += 1
                linear_total_time += stats['duration_ms']
        
        print(f"Linear覆盖率: {linear_found}/{len(self.current_linear_projections)} ({linear_found/len(self.current_linear_projections)*100:.1f}%)")
        print(f"Linear总时间: {linear_total_time:.1f} ms")
        
        print(f"\n总计算时间: {total_compute_time_us/1000:.1f} ms")
        print(f"总FLOPS: {total_flops/1e12:.2f} TFLOPs")
        print(f"总内存访问: {total_memory_access:.2f} GB")
        
        # 端侧时间分析
        kernel_utilization = total_compute_time_us / total_end_to_end_us if total_end_to_end_us > 0 else 0
        idle_time_us = total_end_to_end_us - total_compute_time_us
        idle_proportion = idle_time_us / total_end_to_end_us if total_end_to_end_us > 0 else 0
        
        print(f"\n=== 端侧时间分析 ===")
        print(f"总端侧时间: {total_end_to_end_us/1000:.1f} ms")
        print(f"Kernel时间利用率: {kernel_utilization:.3f} ({kernel_utilization*100:.1f}%)")
        print(f"空闲/调度时间: {idle_time_us/1000:.1f} ms ({idle_proportion*100:.1f}%)")
        
        # 按类别的时间占比
        print(f"\n按算子类别的时间分布:")
        print(f"{'类别':15s} {'时间(ms)':>10s} {'占Kernel%':>10s} {'占端侧%':>10s} {'包含算子':>30s}")
        print("-" * 80)
        
        category_descriptions = {
            'compute_intensive': '计算密集型',
            'memory_intensive': '内存密集型', 
            'overhead': '非计算类开销'
        }
        
        for category, time_us in category_times.items():
            kernel_pct = (time_us / total_compute_time_us * 100) if total_compute_time_us > 0 else 0
            end_to_end_pct = (time_us / total_end_to_end_us * 100) if total_end_to_end_us > 0 else 0
            desc = category_descriptions.get(category, category)
            
            # 获取该类别中有数据的算子
            category_operators = []
            for op_type in self.operator_categories[category]:
                if op_type in operator_results:
                    category_operators.append(op_type)
            
            operators_str = ','.join(category_operators[:3])  # 只显示前3个
            if len(category_operators) > 3:
                operators_str += f"+{len(category_operators)-3}more"
            
            print(f"{desc:15s} {time_us/1000:10.1f} {kernel_pct:10.1f} {end_to_end_pct:10.1f} {operators_str:>30s}")
        
        # 空闲时间
        print(f"{'空闲/调度时间':15s} {idle_time_us/1000:10.1f} {'N/A':>10s} {idle_proportion*100:10.1f} {'系统开销':>30s}")
        
        # 计算时间占比和瓶颈评分
        print(f"\n=== 效率分析结果 ===")
        
        bottleneck_ranking = []
        
        for operator_type, results in operator_results.items():
            # 计算时间占比
            time_proportions = self.calculate_time_proportions(
                results['operator_data'], total_compute_time_us, total_end_to_end_us
            )
            
            # 计算瓶颈评分
            efficiency_degree = results['efficiency_metrics']['efficiency_degree']
            bottleneck_score = self.calculate_bottleneck_score(
                efficiency_degree, time_proportions['kernel_time_proportion']
            )
            
            # 更新结果
            results['time_proportions'] = time_proportions
            results['bottleneck_score'] = bottleneck_score
            
            # 确定算子类别
            operator_category = 'other'
            for category, operators in self.operator_categories.items():
                if operator_type in operators:
                    operator_category = category
                    break
            
            # 添加到排名列表
            kernel_count_safe = self.safe_get_kernel_count(results['operator_data']['kernel_count'])
            if kernel_count_safe > 0:
                bottleneck_ranking.append({
                    'operator_type': operator_type,
                    'category': operator_category,
                    'bottleneck_score': bottleneck_score,
                    'efficiency_degree': efficiency_degree,
                    'kernel_time_proportion': time_proportions['kernel_time_proportion'],
                    'end_to_end_proportion': time_proportions['end_to_end_proportion'],
                    'bound_type': results['roofline_params']['bound_type'],
                    'arithmetic_intensity': results['roofline_params']['arithmetic_intensity'],
                    'duration_ms': results['operator_data']['total_duration_us'] / 1000,
                    'uses_precise_token_size': results['efficiency_metrics']['uses_precise_token_size'],
                    'token_size_variation': results['efficiency_metrics']['token_size_variation'],
                    'data_source': results['operator_data']['data_source'],
                    'is_linear_projection': operator_type in self.current_linear_projections
                })
        
        # 按瓶颈评分排序
        bottleneck_ranking.sort(key=lambda x: x['bottleneck_score'], reverse=True)
        
        # 显示结果
        print(f"{'算子类型':12s} {'类别':10s} {'瓶颈评分':>8s} {'效率度':>8s} {'时间占比':>8s} {'端侧占比':>8s} {'瓶颈类型':>12s} {'Linear?':>8s}")
        print("-" * 100)
        
        for item in bottleneck_ranking:
            category_short = {
                'compute_intensive': '计算密集',
                'memory_intensive': '内存密集',
                'overhead': '非计算开销'
            }.get(item['category'], item['category'][:8])
            
            linear_mark = "Linear" if item['is_linear_projection'] else "NonLin"
            
            print(f"{item['operator_type']:12s} "
                  f"{category_short:10s} "
                  f"{item['bottleneck_score']:8.3f} "
                  f"{item['efficiency_degree']:8.3f} "
                  f"{item['kernel_time_proportion']:8.3f} "
                  f"{item['end_to_end_proportion']:8.3f} "
                  f"{item['bound_type']:>12s} "
                  f"{linear_mark:>8s}")
        
        # 计算整体效率指标
        overall_efficiency = total_flops / (total_compute_time_us * 1e-6) / hardware_specs['phi'] / 1e12 if total_compute_time_us > 0 else 0
        overall_memory_utilization = total_memory_access / (total_compute_time_us * 1e-6) / hardware_specs['pi'] if total_compute_time_us > 0 else 0
        
        print(f"\n=== 整体效率指标 ===")
        print(f"整体计算效率: {overall_efficiency:.3f}")
        print(f"整体内存利用率: {overall_memory_utilization:.3f}")
        
        return {
            'hardware_specs': hardware_specs,
            'end_to_end_info': end_to_end_info,
            'operator_results': operator_results,
            'category_results': category_results,
            'bottleneck_ranking': bottleneck_ranking,
            'time_breakdown': {
                'total_end_to_end_us': total_end_to_end_us,
                'total_kernel_time_us': total_compute_time_us,
                'idle_time_us': idle_time_us,
                'kernel_utilization': kernel_utilization,
                'idle_proportion': idle_proportion,
                'category_times': category_times
            },
            'overall_metrics': {
                'total_compute_time_us': total_compute_time_us,
                'total_flops': total_flops,
                'total_memory_access': total_memory_access,
                'overall_efficiency': overall_efficiency,
                'overall_memory_utilization': overall_memory_utilization
            },
            'linear_analysis': {
                'analysis_mode': analysis_mode,
                'linear_projections': self.current_linear_projections,
                'linear_projection_stats': linear_projection_stats,
                'linear_coverage': linear_found / len(self.current_linear_projections) if self.current_linear_projections else 0,
                'linear_total_time_ms': linear_total_time
            },
            'coverage_analysis': {
                'data_source_stats': data_source_stats,
                'total_operators_analyzed': len(operator_results),
                'total_operators_expected': len(all_operator_types)
            },
            'analysis_version': 'oea_efficiency_enhanced_fixed_linear'
        }
    
    def generate_comprehensive_report(self, analysis_results: Dict) -> str:
        """生成综合分析报告（增强版本）"""
        
        report = []
        report.append("=" * 80)
        report.append("OEA 算子效率分析报告 (增强版本 - 修复Linear算子识别)")
        report.append("=" * 80)
        
        # 硬件信息
        hw_specs = analysis_results['hardware_specs']
        report.append(f"\n硬件配置:")
        report.append(f"  GPU: {hw_specs['gpu_name']}")
        report.append(f"  峰值计算能力: {hw_specs['phi']:.1f} TFLOPs/s")
        report.append(f"  峰值内存带宽: {hw_specs['pi']:.1f} GB/s")
        report.append(f"  显存大小: {hw_specs['memory_size']:.0f} GB")
        
        # Linear分析情况
        linear_analysis = analysis_results['linear_analysis']
        report.append(f"\nLinear算子分析:")
        report.append(f"  分析模式: {linear_analysis['analysis_mode'].upper()}")
        report.append(f"  Linear projections: {', '.join(linear_analysis['linear_projections'])}")
        report.append(f"  Linear覆盖率: {linear_analysis['linear_coverage']:.1%}")
        report.append(f"  Linear总时间: {linear_analysis['linear_total_time_ms']:.1f} ms")
        
        # 覆盖情况分析
        coverage = analysis_results['coverage_analysis']
        report.append(f"\n算子覆盖情况:")
        report.append(f"  分析的算子: {coverage['total_operators_analyzed']}/{coverage['total_operators_expected']}")
        
        report.append(f"  数据源分布:")
        for source, count in coverage['data_source_stats'].items():
            report.append(f"    {source}: {count} 个算子")
        
        # 端侧时间分析
        time_breakdown = analysis_results['time_breakdown']
        report.append(f"\n端侧时间分析:")
        report.append(f"  总端侧时间: {time_breakdown['total_end_to_end_us']/1000:.1f} ms")
        report.append(f"  总Kernel时间: {time_breakdown['total_kernel_time_us']/1000:.1f} ms")
        report.append(f"  空闲/调度时间: {time_breakdown['idle_time_us']/1000:.1f} ms")
        report.append(f"  Kernel时间利用率: {time_breakdown['kernel_utilization']:.3f} ({time_breakdown['kernel_utilization']*100:.1f}%)")
        report.append(f"  空闲时间占比: {time_breakdown['idle_proportion']:.3f} ({time_breakdown['idle_proportion']*100:.1f}%)")
        
        # 整体指标
        overall = analysis_results['overall_metrics']
        report.append(f"\n整体性能指标:")
        report.append(f"  总计算时间: {overall['total_compute_time_us']/1000:.1f} ms")
        report.append(f"  总FLOPS: {overall['total_flops']/1e12:.2f} TFLOPs")
        report.append(f"  总内存访问: {overall['total_memory_access']:.2f} GB")
        report.append(f"  整体计算效率: {overall['overall_efficiency']:.3f}")
        report.append(f"  整体内存利用率: {overall['overall_memory_utilization']:.3f}")
        
        # 按类别的时间分布
        report.append(f"\n按算子类别的时间分布:")
        category_times = time_breakdown['category_times']
        total_kernel_time = time_breakdown['total_kernel_time_us']
        total_end_to_end_time = time_breakdown['total_end_to_end_us']
        
        category_descriptions = {
            'compute_intensive': '计算密集型 (包括Linear projections)',
            'memory_intensive': '内存密集型', 
            'overhead': '非计算类开销'
        }
        
        for category, time_us in category_times.items():
            kernel_pct = (time_us / total_kernel_time * 100) if total_kernel_time > 0 else 0
            end_to_end_pct = (time_us / total_end_to_end_time * 100) if total_end_to_end_time > 0 else 0
            desc = category_descriptions.get(category, category)
            
            report.append(f"  {desc}")
            report.append(f"    时间: {time_us/1000:.1f} ms")
            report.append(f"    占Kernel时间: {kernel_pct:.1f}%")
            report.append(f"    占端侧时间: {end_to_end_pct:.1f}%")
        
        # Linear projection详情
        report.append(f"\nLinear Projection详情:")
        linear_proj_stats = linear_analysis['linear_projection_stats']
        for proj_type, stats in linear_proj_stats.items():
            status = "✓" if stats['found'] else "✗"
            report.append(f"  {status} {proj_type}: {stats['duration_ms']:.1f}ms, 数据源: {stats['data_source']}")
        
        # 瓶颈分析
        report.append(f"\n瓶颈分析 (按瓶颈评分排序):")
        report.append(f"{'算子类型':12s} {'类别':10s} {'瓶颈评分':>8s} {'效率度':>8s} {'时间占比':>8s} {'端侧占比':>8s} {'瓶颈类型':>12s} {'Linear?':>8s}")
        report.append("-" * 100)
        
        for item in analysis_results['bottleneck_ranking']:
            category_short = {
                'compute_intensive': '计算密集',
                'memory_intensive': '内存密集',
                'overhead': '非计算开销'
            }.get(item['category'], item['category'][:8])
            
            linear_mark = "Linear" if item['is_linear_projection'] else "NonLin"
            
            report.append(f"{item['operator_type']:12s} "
                         f"{category_short:10s} "
                         f"{item['bottleneck_score']:8.3f} "
                         f"{item['efficiency_degree']:8.3f} "
                         f"{item['kernel_time_proportion']:8.3f} "
                         f"{item['end_to_end_proportion']:8.3f} "
                         f"{item['bound_type']:>12s} "
                         f"{linear_mark:>8s}")
        
        # 优化建议
        report.append(f"\n优化建议:")
        
        # Linear覆盖建议
        if linear_analysis['linear_coverage'] < 1.0:
            missing_count = len(linear_analysis['linear_projections']) - len([s for s in linear_proj_stats.values() if s['found']])
            report.append(f"  1. Linear算子覆盖不完整 ({linear_analysis['linear_coverage']:.1%})")
            report.append(f"     - 缺失 {missing_count} 个Linear projection")
            report.append(f"     - 建议检查Stage 2的分析结果")
        
        # 分析空闲时间
        if time_breakdown['idle_proportion'] > 0.2:
            report.append(f"  2. 空闲时间过高 ({time_breakdown['idle_proportion']*100:.1f}%)")
            report.append(f"     - 考虑优化调度策略，减少kernel启动开销")
            report.append(f"     - 检查是否存在同步点导致的等待")
        
        # Top瓶颈算子建议
        top_bottlenecks = analysis_results['bottleneck_ranking'][:3]
        for i, item in enumerate(top_bottlenecks, 3):
            report.append(f"  {i}. {item['operator_type']} (瓶颈评分: {item['bottleneck_score']:.3f})")
            if item['bound_type'] == 'Memory-bound':
                report.append(f"     - 内存带宽受限，考虑优化数据访问模式")
            else:
                report.append(f"     - 计算受限，考虑优化算法实现")
            
            if item['is_linear_projection']:
                report.append(f"     - Linear projection，数据源: {item['data_source']}")
            else:
                report.append(f"     - 非Linear算子，数据源: {item['data_source']}")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, analysis_results: Dict, output_file: str):
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
            json.dump(convert_numpy(analysis_results), f, indent=2)
        
        print(f"\n分析结果已保存到: {output_file}")
    
    def analyze_case(self, case_path: str) -> Dict:
        """分析指定案例的效率（增强版本）"""
        
        print(f"\n=== OEA Stage 4: 增强版算子效率分析 (修复Linear算子识别) ===")
        print(f"案例路径: {case_path}")
        
        # 加载所有stage的结果
        stage1_data, stage2_data, stage3_data = self.load_all_stage_results(case_path)
        
        # 执行综合效率分析
        analysis_results = self.analyze_comprehensive_efficiency(stage1_data, stage2_data, stage3_data)
        
        # 生成报告
        report = self.generate_comprehensive_report(analysis_results)
        print(f"\n{report}")
        
        return analysis_results

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='OEA Stage 4: 增强版算子效率分析器 (修复Linear算子识别)')
    parser.add_argument('--case_path', required=True,
                       help='案例路径，包含所有stage的分析结果')
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
        args.output = os.path.join(args.case_path, f"oea_stage4_efficiency_analysis_results.json")
    
    try:
        # 创建分析器
        analyzer = OEAEfficiencyAnalyzerEnhancedFixed()
        
        # 分析案例
        results = analyzer.analyze_case(args.case_path)
        
        # 保存结果
        analyzer.save_results(results, args.output)
        
        print(f"\n=== Stage 4 增强版效率分析完成 (修复Linear算子识别) ===")
        print(f"输出文件: {args.output}")
        
        # 显示关键指标
        overall = results['overall_metrics']
        linear_analysis = results['linear_analysis']
        time_breakdown = results['time_breakdown']
        
        print(f"\n关键指标:")
        print(f"  - 总端侧时间: {time_breakdown['total_end_to_end_us']/1000:.1f} ms")
        print(f"  - 总计算时间: {overall['total_compute_time_us']/1000:.1f} ms")
        print(f"  - Kernel时间利用率: {time_breakdown['kernel_utilization']:.3f}")
        print(f"  - 整体计算效率: {overall['overall_efficiency']:.3f}")
        print(f"  - 整体内存利用率: {overall['overall_memory_utilization']:.3f}")
        print(f"  - Linear分析模式: {linear_analysis['analysis_mode'].upper()}")
        print(f"  - Linear覆盖率: {linear_analysis['linear_coverage']:.1%}")
        print(f"  - Linear总时间: {linear_analysis['linear_total_time_ms']:.1f} ms")
        
        # 显示各类别时间
        category_times = time_breakdown['category_times']
        total_kernel_time = time_breakdown['total_kernel_time_us']
        
        print(f"\n算子类别时间分布:")
        for category, time_us in category_times.items():
            pct = (time_us / total_kernel_time * 100) if total_kernel_time > 0 else 0
            category_name = {
                'compute_intensive': '计算密集型',
                'memory_intensive': '内存密集型',
                'overhead': '非计算类开销'
            }.get(category, category)
            print(f"  - {category_name}: {time_us/1000:.1f} ms ({pct:.1f}%)")
        
        # 显示Top 5瓶颈（包括Linear projections）
        print(f"\nTop 5 瓶颈算子:")
        for i, item in enumerate(results['bottleneck_ranking'][:5], 1):
            linear_mark = " [Linear]" if item['is_linear_projection'] else ""
            print(f"  {i}. {item['operator_type']}{linear_mark} ({item['category']}): "
                  f"瓶颈评分={item['bottleneck_score']:.3f}, "
                  f"效率度={item['efficiency_degree']:.3f}, "
                  f"时间={item['duration_ms']:.1f}ms")
        
        # 显示Linear projection情况
        linear_proj_stats = linear_analysis['linear_projection_stats']
        linear_ops = [proj for proj, stats in linear_proj_stats.items() if stats['found']]
        if linear_ops:
            print(f"\nLinear Projection详情:")
            for proj_type in linear_ops:
                stats = linear_proj_stats[proj_type]
                print(f"  - {proj_type}: {stats['duration_ms']:.1f}ms, 数据源: {stats['data_source']}")
        else:
            print(f"\n⚠ 警告: 未找到任何Linear projection数据!")
            print(f"  预期的Linear projections: {linear_analysis['linear_projections']}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()