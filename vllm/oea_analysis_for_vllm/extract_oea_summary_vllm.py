#!/usr/bin/env python3
"""
OEA结果数据提取器 - vLLM版本
从完整的OEA Stage 4结果中提取汇总信息，去除详细的kernel级别数据

主要功能:
1. 保留hardware_specs、bottleneck_ranking、overall_metrics等核心字段
2. 添加end_to_end_info、time_breakdown、linear_analysis、coverage_analysis、category_results
3. 简化operator_results，去除详细的kernel信息
4. 生成适合横向对比的精简数据结构，与原OEA版本保持一致

使用方法:
python extract_oea_summary_vllm.py \
    --input oea_stage4_Qwen3-14B_batch4_input2048_output10_processed.json \
    --output oea_stage4_Qwen3-14B_batch4_input2048_output10_summary.json
"""

import json
import argparse
import os
from typing import Dict, Any
from datetime import datetime

class OEASummaryExtractorVLLM:
    def __init__(self):
        """初始化vLLM OEA结果提取器"""
        print("=== vLLM OEA结果数据提取器 ===")
        
        # vLLM的Linear算子列表
        self.linear_projections = ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj', 'lm_head']
        
    def extract_summary_data(self, full_results: Dict[str, Any]) -> Dict[str, Any]:
        """从完整的OEA结果中提取汇总数据"""
        
        summary_data = {}
        
        # 1. 直接保留的字段
        preserve_fields = [
            'hardware_specs',
            'bottleneck_ranking',
            'overall_metrics',
            'analysis_version'
        ]
        
        for field in preserve_fields:
            if field in full_results:
                summary_data[field] = full_results[field]
                print(f"✓ 保留字段: {field}")
        
        # 2. 构建end_to_end_info
        summary_data['end_to_end_info'] = self._build_end_to_end_info(full_results)
        print(f"✓ 构建字段: end_to_end_info")
        
        # 3. 构建time_breakdown
        summary_data['time_breakdown'] = self._build_time_breakdown(full_results)
        print(f"✓ 构建字段: time_breakdown")
        
        # 4. 构建linear_analysis
        summary_data['linear_analysis'] = self._build_linear_analysis(full_results)
        print(f"✓ 构建字段: linear_analysis")
        
        # 5. 构建coverage_analysis
        summary_data['coverage_analysis'] = self._build_coverage_analysis(full_results)
        print(f"✓ 构建字段: coverage_analysis")
        
        # 6. 构建category_results (替换category_times)
        summary_data['category_results'] = self._build_category_results(full_results)
        print(f"✓ 构建字段: category_results")
        
        # 7. 简化operator_results
        if 'operator_results' in full_results:
            summary_data['operator_results'] = {}
            
            for operator_type, operator_data in full_results['operator_results'].items():
                simplified_operator = {}
                
                if 'operator_data' in operator_data:
                    op_data = operator_data['operator_data']
                    simplified_op_data = {
                        'total_flops': op_data.get('total_flops', 0),
                        'total_memory_access': op_data.get('total_memory_access', 0),
                        'total_duration_us': op_data.get('total_duration_us', 0),
                        'kernel_count': op_data.get('kernel_count', 0),
                        'data_source': op_data.get('data_source', 'unknown')
                    }
                    
                    if 'prefill_flops' in op_data:
                        simplified_op_data['prefill_flops'] = op_data['prefill_flops']
                    if 'decode_flops' in op_data:
                        simplified_op_data['decode_flops'] = op_data['decode_flops']
                    
                    simplified_operator['operator_data'] = simplified_op_data
                
                for key in ['roofline_params', 'efficiency_metrics', 'time_proportions', 'bottleneck_score']:
                    if key in operator_data:
                        simplified_operator[key] = operator_data[key]
                
                summary_data['operator_results'][operator_type] = simplified_operator
            
            print(f"✓ 简化operator_results: {len(summary_data['operator_results'])} 个算子")
        
        return summary_data
    
    def _build_end_to_end_info(self, full_results: Dict[str, Any]) -> Dict[str, Any]:
        """构建end_to_end_info字段"""
        overall = full_results.get('overall_metrics', {})
        
        return {
            'total_end_to_end_us': overall.get('total_kernel_time_us', 0),
            'inference_start_time': 0,  # vLLM Stage4没有这个信息
            'inference_end_time': overall.get('total_kernel_time_us', 0),
            'data_source': 'stage4_vllm'
        }
    
    def _build_time_breakdown(self, full_results: Dict[str, Any]) -> Dict[str, Any]:
        """构建time_breakdown字段"""
        overall = full_results.get('overall_metrics', {})
        category_times = full_results.get('category_times', {})
        
        total_kernel_time = overall.get('total_kernel_time_us', 0)
        total_end_to_end = total_kernel_time  # vLLM没有idle time信息
        
        idle_time = 0
        kernel_utilization = 1.0 if total_end_to_end > 0 else 0
        idle_proportion = 0.0
        
        return {
            'total_end_to_end_us': total_end_to_end,
            'total_kernel_time_us': total_kernel_time,
            'idle_time_us': idle_time,
            'kernel_utilization': kernel_utilization,
            'idle_proportion': idle_proportion,
            'category_times': category_times
        }
    
    def _build_linear_analysis(self, full_results: Dict[str, Any]) -> Dict[str, Any]:
        """构建linear_analysis字段"""
        operator_results = full_results.get('operator_results', {})
        
        # 提取Linear算子统计
        linear_stats = {}
        linear_total_time = 0
        
        for proj_name in self.linear_projections:
            if proj_name in operator_results:
                op_data = operator_results[proj_name].get('operator_data', {})
                duration_us = op_data.get('total_duration_us', 0)
                linear_total_time += duration_us
                
                linear_stats[proj_name] = {
                    'total_duration_us': duration_us,
                    'total_flops': op_data.get('total_flops', 0),
                    'total_memory_access': op_data.get('total_memory_access', 0),
                    'kernel_count': op_data.get('kernel_count', 0),
                    'data_source': op_data.get('data_source', 'unknown')
                }
        
        # 计算覆盖率
        total_time = full_results.get('overall_metrics', {}).get('total_kernel_time_us', 0)
        linear_coverage = (linear_total_time / total_time) if total_time > 0 else 0
        
        return {
            'analysis_mode': 'vllm_stage4',
            'linear_projections': self.linear_projections,
            'linear_projection_stats': linear_stats,
            'linear_coverage': linear_coverage,
            'linear_total_time_ms': linear_total_time / 1000
        }
    
    def _build_coverage_analysis(self, full_results: Dict[str, Any]) -> Dict[str, Any]:
        """构建coverage_analysis字段"""
        operator_results = full_results.get('operator_results', {})
        
        # 统计数据源
        data_source_stats = {}
        for op_type, op_data in operator_results.items():
            source = op_data.get('operator_data', {}).get('data_source', 'unknown')
            data_source_stats[source] = data_source_stats.get(source, 0) + 1
        
        return {
            'data_source_stats': data_source_stats,
            'total_operators_analyzed': len(operator_results),
            'total_operators_expected': len(self.linear_projections) + 8  # 6个linear + 8个非linear
        }
    
    def _build_category_results(self, full_results: Dict[str, Any]) -> Dict[str, Any]:
        """构建category_results字段（替换category_times）"""
        category_times = full_results.get('category_times', {})
        operator_results = full_results.get('operator_results', {})
        
        # 算子分类
        operator_categories = {
            'compute_intensive': set(self.linear_projections + ['attention', 'moe']),
            'memory_intensive': {'rope', 'layernorm', 'activation', 'reduction'},
            'overhead': {'memory', 'communication'}
        }
        
        category_results = {}
        
        for category, operators in operator_categories.items():
            total_time = category_times.get(category, 0)
            total_flops = 0
            total_memory = 0
            operator_count = 0
            operators_in_category = []
            
            for op_type, op_data in operator_results.items():
                if op_type in operators:
                    op_info = op_data.get('operator_data', {})
                    total_flops += op_info.get('total_flops', 0)
                    total_memory += op_info.get('total_memory_access', 0)
                    operator_count += 1
                    operators_in_category.append(op_type)
            
            category_results[category] = {
                'total_time_us': total_time,
                'total_flops': total_flops,
                'total_memory_access': total_memory,
                'operator_count': operator_count,
                'operators_count': len(operators_in_category)
            }
        
        return category_results
    
    def _estimate_data_size(self, data: Dict[str, Any]) -> int:
        """估算数据大小（字符数）"""
        try:
            return len(json.dumps(data, ensure_ascii=False))
        except:
            return 0
    
    def extract_from_file(self, input_file: str, output_file: str) -> bool:
        """从文件提取OEA汇总数据"""
        
        print(f"\n=== 开始提取OEA汇总数据 ===")
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
        
        if not os.path.exists(input_file):
            print(f"❌ 输入文件不存在: {input_file}")
            return False
        
        try:
            print(f"\n📖 读取完整OEA结果...")
            with open(input_file, 'r', encoding='utf-8') as f:
                full_results = json.load(f)
            
            print(f"✓ 成功读取OEA结果文件")
            print(f"  原始数据包含字段: {list(full_results.keys())}")
            
            print(f"\n🔄 提取汇总数据...")
            summary_data = self.extract_summary_data(full_results)
            
            print(f"\n💾 保存汇总数据...")
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 成功保存OEA汇总数据到: {output_file}")
            
            self._print_extraction_summary(summary_data, full_results)
            
            return True
            
        except Exception as e:
            print(f"❌ 提取过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _print_extraction_summary(self, summary_data: Dict[str, Any], full_results: Dict[str, Any]):
        """打印提取结果统计"""
        
        print(f"\n=== 提取结果统计 ===")
        
        # 数据压缩效果
        original_size = self._estimate_data_size(full_results)
        summary_size = self._estimate_data_size(summary_data)
        if original_size > 0:
            compression_ratio = summary_size / original_size
            print(f"数据压缩比: {compression_ratio:.1%}")
            print(f"原始大小: ~{original_size:,} 字符")
            print(f"简化大小: ~{summary_size:,} 字符")
            print(f"减少数据: {(1-compression_ratio)*100:.1f}%")
        
        # 各部分数据统计
        if 'operator_results' in summary_data:
            print(f"\n算子分析结果: {len(summary_data['operator_results'])} 个算子")
        
        if 'bottleneck_ranking' in summary_data:
            print(f"瓶颈排名: {len(summary_data['bottleneck_ranking'])} 个算子")
        
        if 'category_results' in summary_data:
            print(f"类别分析结果: {len(summary_data['category_results'])} 个类别")
        
        if 'linear_analysis' in summary_data:
            linear = summary_data['linear_analysis']
            print(f"Linear分析: {len(linear.get('linear_projection_stats', {}))} 个projection")
        
        # 硬件信息
        if 'hardware_specs' in summary_data:
            hw = summary_data['hardware_specs']
            print(f"\n硬件信息: {hw.get('gpu_name', 'Unknown')} ({hw.get('n_gpu', 1)} GPU)")
            print(f"峰值计算: {hw.get('phi', 0):.1f} TFLOPs/s")
            print(f"峰值带宽: {hw.get('pi', 0):.1f} GB/s")
        
        # 整体指标
        if 'overall_metrics' in summary_data:
            metrics = summary_data['overall_metrics']
            print(f"\n整体效率: {metrics.get('overall_efficiency', 0):.3f}")
            print(f"总计算时间: {metrics.get('total_kernel_time_us', 0)/1000:.1f} ms")
            print(f"总FLOPS: {metrics.get('total_flops', 0)/1e12:.2f} TFLOPs")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='vLLM OEA结果数据提取器')
    parser.add_argument('--input', required=True, help='输入的完整OEA Stage 4结果文件')
    parser.add_argument('--output', help='输出的汇总数据文件（可选）')
    
    args = parser.parse_args()
    
    try:
        # 生成输出文件名和路径
        if args.output:
            output_file = args.output
        else:
            # 从输入文件名提取pod_name
            input_basename = os.path.basename(args.input)
            if input_basename.startswith('oea_stage4_') and input_basename.endswith('_processed.json'):
                pod_name = input_basename[len('oea_stage4_'):-len('_processed.json')]
            else:
                pod_name = 'unknown'
            
            # 获取输入文件所在目录
            input_dir = os.path.dirname(args.input)
            # 生成输出文件路径：输入文件所在文件夹/oea_summary_pod_name.json
            output_file = os.path.join(input_dir, f'oea_summary_{pod_name}.json')
        
        extractor = OEASummaryExtractorVLLM()
        success = extractor.extract_from_file(args.input, output_file)
        
        if success:
            print(f"\n✅ 提取完成！")
            return 0
        else:
            print(f"\n❌ 提取失败！")
            return 1
            
    except Exception as e:
        print(f"\n❌ 程序异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())