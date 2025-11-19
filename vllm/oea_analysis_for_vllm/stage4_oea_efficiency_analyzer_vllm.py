#!/usr/bin/env python3
"""
OEA Stage 4: vLLM框架算子效率分析器
整合Stage 1/2/3的分析结果，计算算子效率和瓶颈评分

主要功能:
1. 从Stage 2提取Linear算子数据（transformer_layers + lm_head_kernels）
2. 从Stage 3提取非Linear算子数据（prefill_analysis + decode_analysis）
3. 计算动态Roofline模型参数（算术强度、性能上界）
4. 计算算子效率度 δ_O = r̂_O / φ
5. 计算瓶颈评分 β_O = (1 - δ_O) × p_O
6. 生成综合效率分析报告

使用方法:
python stage4_oea_efficiency_analyzer_vllm.py \
    --stage1 oea_stage1_xxx_processed.json \
    --stage2 oea_stage2_xxx_processed.json \
    --stage3 oea_stage3_xxx_processed.json \
    --output oea_stage4_efficiency_analysis.json
"""

import json
import numpy as np
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

class OEAEfficiencyAnalyzerVLLM:
    def __init__(self):
        """初始化vLLM框架的OEA效率分析器"""
        
        # vLLM的Linear算子类型（从Stage 2的transformer_layers获取）
        self.linear_projections = [
            'qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj', 'lm_head'
        ]
        
        # 非Linear算子类型（从Stage 3获取）
        self.nonlinear_types = [
            'attention', 'rope', 'layernorm', 'activation', 'moe', 
            'memory', 'reduction', 'communication'
        ]
        
        # 算子类别分类
        self.operator_categories = {
            'compute_intensive': set(self.linear_projections + ['attention', 'moe']),
            'memory_intensive': {'rope', 'layernorm', 'activation', 'reduction'},
            'overhead': {'memory', 'communication'}
        }
        
        print(f"=== 初始化vLLM OEA效率分析器 ===")
        print(f"Linear projections: {self.linear_projections}")
        print(f"非Linear算子: {self.nonlinear_types}")
    
    def load_stage_results(self, stage1_file: str, stage2_file: Optional[str], 
                          stage3_file: Optional[str]) -> Tuple[Dict, Optional[Dict], Optional[Dict]]:
        """加载所有stage的分析结果"""
        
        print(f"\n=== 加载Stage分析结果 ===")
        
        # Stage 1: 必需
        if not os.path.exists(stage1_file):
            raise FileNotFoundError(f"Stage 1文件不存在: {stage1_file}")
        
        with open(stage1_file, 'r') as f:
            stage1_data = json.load(f)
        print(f"✓ Stage 1加载完成: {stage1_file}")
        
        # Stage 2: 可选
        stage2_data = None
        if stage2_file and os.path.exists(stage2_file):
            with open(stage2_file, 'r') as f:
                stage2_data = json.load(f)
            print(f"✓ Stage 2加载完成: {stage2_file}")
        else:
            print(f"⚠ Stage 2文件不存在，将从Stage 1提取Linear数据")
        
        # Stage 3: 可选
        stage3_data = None
        if stage3_file and os.path.exists(stage3_file):
            with open(stage3_file, 'r') as f:
                stage3_data = json.load(f)
            print(f"✓ Stage 3加载完成: {stage3_file}")
        else:
            print(f"⚠ Stage 3文件不存在，将从Stage 1提取非Linear数据")
        
        return stage1_data, stage2_data, stage3_data
    
    def extract_hardware_specs(self, stage1_data: Dict) -> Dict:
        """提取硬件规格信息"""
        
        hardware_spec = stage1_data.get('hardware_spec', {})
        
        specs = {
            'phi': hardware_spec.get('peak_flops_fp16', 312e12) / 1e12,  # TFLOPs/s
            'pi': hardware_spec.get('memory_bandwidth', 2e12) / 1e9,      # GB/s
            'memory_size': hardware_spec.get('memory_size', 80e9) / 1e9,  # GB
            'gpu_name': hardware_spec.get('name', 'Unknown'),
            'n_gpu': hardware_spec.get('n_gpu', 1)
        }
        
        print(f"\n=== 硬件规格 ===")
        print(f"GPU: {specs['gpu_name']}")
        print(f"峰值计算能力 φ: {specs['phi']:.1f} TFLOPs/s")
        print(f"峰值内存带宽 π: {specs['pi']:.1f} GB/s")
        print(f"显存大小: {specs['memory_size']:.0f} GB")
        
        return specs
    
    def extract_linear_data_from_stage2(self, proj_name: str, stage2_data: Dict) -> Dict:
        """从Stage 2的transformer_layers中提取Linear算子数据
        
        vLLM Stage2 结构:
        linear_analysis.transformer_layers[i].performance[proj_name] = {
            'flops', 'memory_access', 'duration_us', ...
        }
        """
        
        operator_data = {
            'total_flops': 0,
            'total_memory_access': 0,
            'total_duration_us': 0,
            'kernel_count': 0,
            'data_source': 'not_found'
        }
        
        if stage2_data is None:
            return operator_data
        
        linear_analysis = stage2_data.get('linear_analysis', {})
        transformer_layers = linear_analysis.get('transformer_layers', [])
        lm_head_kernels = linear_analysis.get('lm_head_kernels', [])
        
        # 处理 transformer layers
        if proj_name != 'lm_head':
            for layer in transformer_layers:
                performance = layer.get('performance', {})
                if proj_name in performance:
                    proj_data = performance[proj_name]
                    operator_data['total_flops'] += proj_data.get('flops', 0)
                    operator_data['total_memory_access'] += proj_data.get('memory_access', 0) / 1e9  # 转GB
                    operator_data['total_duration_us'] += proj_data.get('duration_us', 0)
                    operator_data['kernel_count'] += 1
            
            if operator_data['kernel_count'] > 0:
                operator_data['data_source'] = 'stage2_transformer_layers'
        
        # 处理 lm_head
        else:
            for lm_head in lm_head_kernels:
                performance = lm_head.get('performance', {})
                if performance:
                    operator_data['total_flops'] += performance.get('flops', 0)
                    operator_data['total_memory_access'] += performance.get('memory_access', 0) / 1e9  # 转GB
                    operator_data['total_duration_us'] += performance.get('duration_us', 0)
                    operator_data['kernel_count'] += 1
            
            if operator_data['kernel_count'] > 0:
                operator_data['data_source'] = 'stage2_lm_head_kernels'
        
        return operator_data
    
    def extract_nonlinear_data_from_stage3(self, op_type: str, stage3_data: Dict) -> Dict:
        """从Stage 3的prefill_analysis和decode_analysis中提取非Linear算子数据"""
        
        operator_data = {
            'total_flops': 0,
            'total_memory_access': 0,
            'total_duration_us': 0,
            'kernel_count': 0,
            'data_source': 'not_found',
            'prefill_flops': 0,
            'decode_flops': 0
        }
        
        if stage3_data is None:
            return operator_data
        
        # 合并prefill和decode的数据
        prefill_analysis = stage3_data.get('prefill_analysis', {})
        decode_analysis = stage3_data.get('decode_analysis', {})
        
        prefill_data = prefill_analysis.get(op_type, {})
        decode_data = decode_analysis.get(op_type, {})
        
        if prefill_data or decode_data:
            operator_data['total_flops'] = prefill_data.get('total_flops', 0) + decode_data.get('total_flops', 0)
            operator_data['total_memory_access'] = (
                prefill_data.get('total_memory_access', 0) + 
                decode_data.get('total_memory_access', 0)
            ) / 1e9  # 转GB
            operator_data['total_duration_us'] = (
                prefill_data.get('total_duration_us', 0) + 
                decode_data.get('total_duration_us', 0)
            )
            operator_data['kernel_count'] = (
                prefill_data.get('kernel_count', 0) + 
                decode_data.get('kernel_count', 0)
            )
            operator_data['prefill_flops'] = prefill_data.get('total_flops', 0)
            operator_data['decode_flops'] = decode_data.get('total_flops', 0)
            operator_data['data_source'] = 'stage3_combined'
        
        return operator_data
    
    def extract_operator_data_from_stage1(self, op_type: str, stage1_data: Dict) -> Dict:
        """从Stage 1中提取算子数据（fallback）"""
        
        operator_data = {
            'total_flops': 0,
            'total_memory_access': 0,
            'total_duration_us': 0,
            'kernel_count': 0,
            'data_source': 'not_found'
        }
        
        # 从 iterations 中提取
        iterations = stage1_data.get('iteration_analysis', {}).get('iterations', [])
        
        for iteration in iterations:
            kernels = iteration.get('kernels', [])
            for kernel in kernels:
                kernel_name = kernel.get('name', '').lower()
                # 简单的关键词匹配
                if op_type.lower() in kernel_name:
                    operator_data['kernel_count'] += 1
                    operator_data['total_duration_us'] += kernel.get('dur', 0)
        
        if operator_data['kernel_count'] > 0:
            operator_data['data_source'] = 'stage1_fallback'
            # 估算内存访问
            operator_data['total_memory_access'] = operator_data['total_duration_us'] * 0.1 / 1e6
        
        return operator_data
    
    def calculate_dynamic_roofline(self, flops: float, memory_access: float,
                                   hardware_specs: Dict) -> Dict:
        """计算动态Roofline模型参数
        
        算术强度: AI_O = R̂_O / M̂_O
        性能上界: r*_O = min(AI_O × π, φ)
        
        注意：由于实际GEMM库的优化（tile、cache、Tensor Core等），
        简单的内存访问量估算往往不准确，导致算术强度被高估。
        因此，我们保留Roofline分析用于理论参考，但效率计算直接使用硬件峰值。
        """
        
        flops_tflops = flops / 1e12
        memory_gb = memory_access
        
        # 算术强度
        if memory_gb > 0:
            arithmetic_intensity = flops_tflops / memory_gb
        else:
            arithmetic_intensity = float('inf') if flops_tflops > 0 else 0
        
        phi = hardware_specs['phi']
        pi = hardware_specs['pi']
        
        # 计算理论性能上界（仅用于参考）
        memory_bound_threshold = phi / pi
        
        if arithmetic_intensity < memory_bound_threshold:
            theoretical_bound = arithmetic_intensity * pi
            bound_type = "Memory-bound"
        else:
            theoretical_bound = phi
            bound_type = "Compute-bound"
        
        return {
            'arithmetic_intensity': arithmetic_intensity,
            'theoretical_bound_tflops': theoretical_bound,  # 理论上界（仅参考）
            'performance_bound_tflops': phi,  # 实际使用硬件峰值作为上界
            'bound_type': bound_type,
            'memory_bound_threshold': memory_bound_threshold,
            'flops_tflops': flops_tflops,
            'memory_gb': memory_gb
        }
    
    def calculate_operator_efficiency(self, operator_data: Dict, roofline_params: Dict, 
                                     hardware_specs: Dict) -> Dict:
        """计算算子效率度 δ_O = r̂_O / φ
        
        修改说明：
        - 原来：δ_O = r̂_O / r*_O，其中 r*_O 来自 Roofline 模型
        - 现在：δ_O = r̂_O / φ，直接使用硬件峰值作为上界
        
        原因：
        1. 简单的内存访问量公式 (M×K + K×N + M×N) 无法准确反映实际情况
        2. 现代GEMM库通过tile优化、cache、Tensor Core等技术，实际有效AI远高于理论值
        3. 使用硬件峰值更直观，避免效率超过100%的异常情况
        """
        
        total_duration_us = operator_data.get('total_duration_us', 0)
        
        # 实际吞吐量
        if total_duration_us > 0:
            duration_seconds = total_duration_us * 1e-6
            actual_throughput_tflops = roofline_params['flops_tflops'] / duration_seconds
        else:
            actual_throughput_tflops = 0
        
        # 效率度：直接使用硬件峰值
        peak_compute_tflops = hardware_specs['phi']
        if peak_compute_tflops > 0:
            efficiency_degree = min(actual_throughput_tflops / peak_compute_tflops, 1.0)
        else:
            efficiency_degree = 0
        
        # 保留理论上界用于对比分析
        theoretical_bound = roofline_params.get('theoretical_bound_tflops', peak_compute_tflops)
        if theoretical_bound > 0:
            theoretical_efficiency = actual_throughput_tflops / theoretical_bound
        else:
            theoretical_efficiency = 0
        
        return {
            'total_duration_us': total_duration_us,
            'actual_throughput_tflops': actual_throughput_tflops,
            'efficiency_degree': efficiency_degree,  # 基于硬件峰值的效率
            'theoretical_efficiency': theoretical_efficiency,  # 基于Roofline的理论效率（仅参考）
            'peak_compute_tflops': peak_compute_tflops,
            'theoretical_bound_tflops': theoretical_bound,
            'kernel_count': operator_data.get('kernel_count', 0)
        }
    
    def calculate_time_proportions(self, operator_time_us: float, 
                                   total_kernel_time_us: float) -> Dict:
        """计算时间占比"""
        
        kernel_proportion = operator_time_us / total_kernel_time_us if total_kernel_time_us > 0 else 0
        
        return {
            'kernel_time_proportion': min(kernel_proportion, 1.0)
        }
    
    def calculate_bottleneck_score(self, efficiency_degree: float, time_proportion: float) -> float:
        """计算瓶颈评分 β_O = (1 - δ_O) × p_O"""
        
        return max(0, (1 - efficiency_degree) * time_proportion)
    
    def analyze_comprehensive_efficiency(self, stage1_data: Dict, stage2_data: Optional[Dict],
                                        stage3_data: Optional[Dict]) -> Dict:
        """综合效率分析"""
        
        print(f"\n=== 综合效率分析 ===")
        
        # 提取硬件规格
        hardware_specs = self.extract_hardware_specs(stage1_data)
        
        # 分析所有算子
        operator_results = {}
        total_kernel_time_us = 0
        total_flops = 0
        total_memory_access = 0
        
        # 统计各类别
        category_times = {'compute_intensive': 0, 'memory_intensive': 0, 'overhead': 0}
        
        print(f"\n=== 算子数据提取 ===")
        
        # 1. Linear算子（从Stage 2）
        for proj_name in self.linear_projections:
            operator_data = self.extract_linear_data_from_stage2(proj_name, stage2_data)
            
            # Fallback到Stage 1
            if operator_data['data_source'] == 'not_found':
                operator_data = self.extract_operator_data_from_stage1(proj_name, stage1_data)
            
            if operator_data['kernel_count'] > 0:
                print(f"{proj_name:15s}: {operator_data['kernel_count']:4d} kernels, "
                      f"{operator_data['total_duration_us']/1000:6.1f} ms, "
                      f"数据源: {operator_data['data_source']}")
                
                # 计算Roofline参数
                roofline_params = self.calculate_dynamic_roofline(
                    operator_data['total_flops'],
                    operator_data['total_memory_access'],
                    hardware_specs
                )
                
                # 计算效率指标
                efficiency_metrics = self.calculate_operator_efficiency(
                    operator_data, roofline_params, hardware_specs
                )
                
                operator_results[proj_name] = {
                    'operator_data': operator_data,
                    'roofline_params': roofline_params,
                    'efficiency_metrics': efficiency_metrics
                }
                
                # 累计统计
                total_kernel_time_us += operator_data['total_duration_us']
                total_flops += operator_data['total_flops']
                total_memory_access += operator_data['total_memory_access']
                category_times['compute_intensive'] += operator_data['total_duration_us']
        
        # 2. 非Linear算子（从Stage 3）
        for op_type in self.nonlinear_types:
            operator_data = self.extract_nonlinear_data_from_stage3(op_type, stage3_data)
            
            # Fallback到Stage 1
            if operator_data['data_source'] == 'not_found':
                operator_data = self.extract_operator_data_from_stage1(op_type, stage1_data)
            
            if operator_data['kernel_count'] > 0:
                print(f"{op_type:15s}: {operator_data['kernel_count']:4d} kernels, "
                      f"{operator_data['total_duration_us']/1000:6.1f} ms, "
                      f"数据源: {operator_data['data_source']}")
                
                # 计算Roofline参数
                roofline_params = self.calculate_dynamic_roofline(
                    operator_data['total_flops'],
                    operator_data['total_memory_access'],
                    hardware_specs
                )
                
                # 计算效率指标
                efficiency_metrics = self.calculate_operator_efficiency(
                    operator_data, roofline_params, hardware_specs
                )
                
                operator_results[op_type] = {
                    'operator_data': operator_data,
                    'roofline_params': roofline_params,
                    'efficiency_metrics': efficiency_metrics
                }
                
                # 累计统计
                total_kernel_time_us += operator_data['total_duration_us']
                total_flops += operator_data['total_flops']
                total_memory_access += operator_data['total_memory_access']
                
                # 分类统计
                if op_type in self.operator_categories['compute_intensive']:
                    category_times['compute_intensive'] += operator_data['total_duration_us']
                elif op_type in self.operator_categories['memory_intensive']:
                    category_times['memory_intensive'] += operator_data['total_duration_us']
                else:
                    category_times['overhead'] += operator_data['total_duration_us']
        
        # 计算瓶颈评分
        print(f"\n=== 效率分析结果 ===")
        
        bottleneck_ranking = []
        
        for op_type, results in operator_results.items():
            time_proportions = self.calculate_time_proportions(
                results['operator_data']['total_duration_us'],
                total_kernel_time_us
            )
            
            efficiency_degree = results['efficiency_metrics']['efficiency_degree']
            bottleneck_score = self.calculate_bottleneck_score(
                efficiency_degree,
                time_proportions['kernel_time_proportion']
            )
            
            results['time_proportions'] = time_proportions
            results['bottleneck_score'] = bottleneck_score
            
            # 确定类别
            op_category = 'other'
            for category, operators in self.operator_categories.items():
                if op_type in operators:
                    op_category = category
                    break
            
            bottleneck_ranking.append({
                'operator_type': op_type,
                'category': op_category,
                'bottleneck_score': bottleneck_score,
                'efficiency_degree': efficiency_degree,
                'kernel_time_proportion': time_proportions['kernel_time_proportion'],
                'bound_type': results['roofline_params']['bound_type'],
                'arithmetic_intensity': results['roofline_params']['arithmetic_intensity'],
                'duration_ms': results['operator_data']['total_duration_us'] / 1000,
                'data_source': results['operator_data']['data_source'],
                'is_linear': op_type in self.linear_projections
            })
        
        # 排序
        bottleneck_ranking.sort(key=lambda x: x['bottleneck_score'], reverse=True)
        
        # 显示结果
        print(f"{'算子类型':15s} {'类别':10s} {'瓶颈评分':>8s} {'效率度':>8s} "
              f"{'时间占比':>8s} {'瓶颈类型':>12s} {'类型':>8s}")
        print("-" * 95)
        
        for item in bottleneck_ranking:
            category_short = {
                'compute_intensive': '计算密集',
                'memory_intensive': '内存密集',
                'overhead': '开销'
            }.get(item['category'], item['category'][:8])
            
            op_mark = "Linear" if item['is_linear'] else "NonLin"
            
            print(f"{item['operator_type']:15s} {category_short:10s} "
                  f"{item['bottleneck_score']:8.3f} {item['efficiency_degree']:8.3f} "
                  f"{item['kernel_time_proportion']:8.3f} {item['bound_type']:>12s} "
                  f"{op_mark:>8s}")
        
        # 整体效率
        overall_efficiency = total_flops / (total_kernel_time_us * 1e-6) / hardware_specs['phi'] / 1e12 if total_kernel_time_us > 0 else 0
        
        print(f"\n=== 整体效率指标 ===")
        print(f"总计算时间: {total_kernel_time_us/1000:.1f} ms")
        print(f"总FLOPS: {total_flops/1e12:.2f} TFLOPs")
        print(f"整体计算效率: {overall_efficiency:.3f}")
        
        return {
            'hardware_specs': hardware_specs,
            'operator_results': operator_results,
            'bottleneck_ranking': bottleneck_ranking,
            'overall_metrics': {
                'total_kernel_time_us': total_kernel_time_us,
                'total_flops': total_flops,
                'total_memory_access': total_memory_access,
                'overall_efficiency': overall_efficiency
            },
            'category_times': category_times,
            'analysis_version': 'oea_vllm_stage4'
        }
    
    def generate_report(self, analysis_results: Dict) -> str:
        """生成分析报告"""
        
        report = []
        report.append("=" * 80)
        report.append("OEA Stage 4: vLLM算子效率分析报告")
        report.append("=" * 80)
        
        # 硬件信息
        hw = analysis_results['hardware_specs']
        report.append(f"\n硬件配置:")
        report.append(f"  GPU: {hw['gpu_name']}")
        report.append(f"  峰值计算: {hw['phi']:.1f} TFLOPs/s")
        report.append(f"  峰值带宽: {hw['pi']:.1f} GB/s")
        
        # 整体指标
        overall = analysis_results['overall_metrics']
        report.append(f"\n整体性能:")
        report.append(f"  总计算时间: {overall['total_kernel_time_us']/1000:.1f} ms")
        report.append(f"  总FLOPS: {overall['total_flops']/1e12:.2f} TFLOPs")
        report.append(f"  整体效率: {overall['overall_efficiency']:.3f}")
        
        # 类别时间分布
        category_times = analysis_results['category_times']
        total_time = overall['total_kernel_time_us']
        
        report.append(f"\n按类别时间分布:")
        for category, time_us in category_times.items():
            pct = (time_us / total_time * 100) if total_time > 0 else 0
            category_name = {
                'compute_intensive': '计算密集型',
                'memory_intensive': '内存密集型',
                'overhead': '开销'
            }.get(category, category)
            report.append(f"  {category_name}: {time_us/1000:.1f} ms ({pct:.1f}%)")
        
        # 瓶颈分析
        report.append(f"\n瓶颈分析 (Top 10):")
        report.append(f"{'算子':15s} {'类别':10s} {'瓶颈评分':>8s} {'效率度':>8s} "
                     f"{'时间占比':>8s} {'类型':>8s}")
        report.append("-" * 75)
        
        for item in analysis_results['bottleneck_ranking'][:10]:
            category_short = {
                'compute_intensive': '计算密集',
                'memory_intensive': '内存密集',
                'overhead': '开销'
            }.get(item['category'], item['category'][:8])
            
            op_mark = "Linear" if item['is_linear'] else "NonLin"
            
            report.append(f"{item['operator_type']:15s} {category_short:10s} "
                         f"{item['bottleneck_score']:8.3f} {item['efficiency_degree']:8.3f} "
                         f"{item['kernel_time_proportion']:8.3f} {op_mark:>8s}")
        
        # 优化建议
        report.append(f"\n优化建议:")
        top_bottlenecks = analysis_results['bottleneck_ranking'][:3]
        for i, item in enumerate(top_bottlenecks, 1):
            report.append(f"  {i}. {item['operator_type']} (瓶颈评分: {item['bottleneck_score']:.3f})")
            if item['bound_type'] == 'Memory-bound':
                report.append(f"     - 内存带宽受限，优化数据访问模式")
            else:
                report.append(f"     - 计算受限，优化算法实现")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, analysis_results: Dict, output_file: str):
        """保存结果"""
        
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
        
        print(f"\n结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='OEA Stage 4: vLLM算子效率分析器')
    parser.add_argument('--stage1', required=True, help='Stage 1结果文件')
    parser.add_argument('--stage2', default=None, help='Stage 2结果文件')
    parser.add_argument('--stage3', default=None, help='Stage 3结果文件')
    parser.add_argument('--output', help='输出文件路径（可选）')
    
    args = parser.parse_args()
    
    try:
        # 创建分析器
        analyzer = OEAEfficiencyAnalyzerVLLM()
        
        # 加载数据
        stage1_data, stage2_data, stage3_data = analyzer.load_stage_results(
            args.stage1, args.stage2, args.stage3
        )
        
        # 生成输出文件名和路径
        if args.output:
            output_file = args.output
        else:
            # 从stage1_data中获取pod_name
            pod_name = stage1_data.get('case_info', {}).get('pod_name', 'unknown')
            # 获取stage1文件所在目录
            input_dir = os.path.dirname(args.stage1)
            # 生成输出文件路径：输入文件所在文件夹/oea_stage4_pod_name_processed.json
            output_file = os.path.join(input_dir, f'oea_stage4_{pod_name}_processed.json')
        
        # 分析
        results = analyzer.analyze_comprehensive_efficiency(
            stage1_data, stage2_data, stage3_data
        )
        
        # 生成报告
        report = analyzer.generate_report(results)
        print(f"\n{report}")
        
        # 保存结果
        analyzer.save_results(results, output_file)
        
        print(f"\n=== Stage 4分析完成 ===")
        
        # 显示关键指标
        overall = results['overall_metrics']
        print(f"\n关键指标:")
        print(f"  - 总计算时间: {overall['total_kernel_time_us']/1000:.1f} ms")
        print(f"  - 整体效率: {overall['overall_efficiency']:.3f}")
        
        print(f"\nTop 5瓶颈算子:")
        for i, item in enumerate(results['bottleneck_ranking'][:5], 1):
            print(f"  {i}. {item['operator_type']}: 瓶颈评分={item['bottleneck_score']:.3f}, "
                  f"效率度={item['efficiency_degree']:.3f}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()