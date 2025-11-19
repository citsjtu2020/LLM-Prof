#!/usr/bin/env python3
"""
OEA结果数据提取器 - 用于横向对比分析
从完整的OEA Stage 4结果中提取汇总信息，去除详细的kernel级别数据

主要功能:
1. 保留hardware_specs、end_to_end_info、bottleneck_ranking、time_breakdown、overall_metrics、linear_analysis、coverage_analysis
2. 简化operator_results和category_results，去除详细的kernel信息
3. 生成适合横向对比的精简数据结构

使用方法:
python extract_oea_summary.py --input_file path/to/oea_stage4_efficiency_analysis_results.json --output_file path/to/oea_summary.json
"""

import json
import argparse
import os
from typing import Dict, Any
from datetime import datetime

class OEASummaryExtractor:
    def __init__(self):
        """初始化OEA结果提取器"""
        print("=== OEA结果数据提取器 ===")
        
    def extract_summary_data(self, full_results: Dict[str, Any]) -> Dict[str, Any]:
        """从完整的OEA结果中提取汇总数据"""
        
        summary_data = {}
        
        # 1. 直接保留的字段 (完全不变)
        preserve_fields = [
            'hardware_specs',
            'end_to_end_info', 
            'bottleneck_ranking',
            'time_breakdown',
            'overall_metrics',
            'linear_analysis',
            'coverage_analysis',
            'analysis_version'
        ]
        
        for field in preserve_fields:
            if field in full_results:
                summary_data[field] = full_results[field]
                print(f"✓ 保留字段: {field}")
        
        # 2. 简化operator_results - 只保留汇总统计，去除所有详细kernel信息
        if 'operator_results' in full_results:
            summary_data['operator_results'] = {}
            
            for operator_type, operator_data in full_results['operator_results'].items():
                # 只保留汇总信息，完全去除详细数据
                simplified_operator = {}
                
                if 'operator_data' in operator_data:
                    op_data = operator_data['operator_data']
                    # 只保留汇总统计，完全去除kernel相关的所有信息
                    simplified_op_data = {
                        'total_flops': op_data.get('total_flops', 0),
                        'total_memory_access': op_data.get('total_memory_access', 0),
                        'total_duration_us': op_data.get('total_duration_us', 0),
                        'data_source': op_data.get('data_source', 'unknown'),
                        'uses_precise_token_size': op_data.get('uses_precise_token_size', False),
                        'token_size_variation': op_data.get('token_size_variation', 0)
                    }
                    # 完全不保留kernel_count、executions列表和其他详细数据
                    simplified_operator['operator_data'] = simplified_op_data
                
                # 保留其他分析结果（这些通常很小）
                for key in ['roofline_params', 'efficiency_metrics', 'time_proportions', 'bottleneck_score']:
                    if key in operator_data:
                        if key == 'efficiency_metrics':
                            # 简化efficiency_metrics，去除kernel相关的详细信息
                            original_metrics = operator_data[key]
                            simplified_metrics = {
                                'efficiency_degree': original_metrics.get('efficiency_degree', 0),
                                'uses_precise_token_size': original_metrics.get('uses_precise_token_size', False),
                                'token_size_variation': original_metrics.get('token_size_variation', 0)
                            }
                            # 完全不保留kernel_count和其他详细kernel信息
                            simplified_operator[key] = simplified_metrics
                        else:
                            # 其他字段直接保留
                            simplified_operator[key] = operator_data[key]
                
                summary_data['operator_results'][operator_type] = simplified_operator
            
            print(f"✓ 简化operator_results: {len(summary_data['operator_results'])} 个算子")
        
        # 3. 简化category_results - 只保留统计数据，去除详细列表
        if 'category_results' in full_results:
            summary_data['category_results'] = {}
            
            for category, category_data in full_results['category_results'].items():
                # 只保留统计信息，完全去除详细的operators列表
                simplified_category = {
                    'total_time_us': category_data.get('total_time_us', 0),
                    'total_flops': category_data.get('total_flops', 0),
                    'total_memory_access': category_data.get('total_memory_access', 0),
                    'operator_count': category_data.get('operator_count', 0)
                }
                
                # 如果有详细的operators列表，只保留数量统计
                if 'operators' in category_data:
                    operators_list = category_data['operators']
                    simplified_category['operators_count'] = len(operators_list) if isinstance(operators_list, list) else 0
                else:
                    simplified_category['operators_count'] = 0
                
                # 完全不保留operators详细列表
                summary_data['category_results'][category] = simplified_category
            
            print(f"✓ 简化category_results: {len(summary_data['category_results'])} 个类别")
        
        # 4. 添加提取元信息
        # summary_data['extraction_info'] = {
        #     'extracted_at': datetime.now().isoformat(),
        #     'extraction_version': 'oea_summary_v1.0',
        #     'original_data_size_estimation': self._estimate_data_size(full_results),
        #     'summary_data_size_estimation': self._estimate_data_size(summary_data),
        #     'compression_ratio': None  # 将在后面计算
        # }
        
        # # 计算压缩比
        # original_size = summary_data['extraction_info']['original_data_size_estimation']
        # summary_size = summary_data['extraction_info']['summary_data_size_estimation']
        # if original_size > 0:
        #     compression_ratio = summary_size / original_size
        #     summary_data['extraction_info']['compression_ratio'] = compression_ratio
        #     print(f"✓ 数据压缩比: {compression_ratio:.3f} (原始: ~{original_size:,} 字符, 简化: ~{summary_size:,} 字符)")
        #     print(f"✓ 压缩效果: 减少了 {(1-compression_ratio)*100:.1f}% 的数据量")
        
        return summary_data
    
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
        
        # 检查输入文件
        if not os.path.exists(input_file):
            print(f"❌ 输入文件不存在: {input_file}")
            return False
        
        try:
            # 读取完整的OEA结果
            print(f"\n📖 读取完整OEA结果...")
            with open(input_file, 'r', encoding='utf-8') as f:
                full_results = json.load(f)
            
            print(f"✓ 成功读取OEA结果文件")
            print(f"  原始数据包含字段: {list(full_results.keys())}")
            
            # 提取汇总数据
            print(f"\n🔄 提取汇总数据...")
            summary_data = self.extract_summary_data(full_results)
            
            # 保存汇总数据
            print(f"\n💾 保存汇总数据...")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 成功保存OEA汇总数据到: {output_file}")
            
            # 显示提取结果统计
            self._print_extraction_summary(summary_data)
            
            return True
            
        except Exception as e:
            print(f"❌ 提取过程出错: {str(e)}")
            return False
    
    def _print_extraction_summary(self, summary_data: Dict[str, Any]):
        """打印提取结果统计"""
        
        print(f"\n=== 提取结果统计 ===")
        
        # 基本信息
        # if 'extraction_info' in summary_data:
        #     info = summary_data['extraction_info']
        #     print(f"提取时间: {info.get('extracted_at', 'Unknown')}")
        #     print(f"提取版本: {info.get('extraction_version', 'Unknown')}")
        #     if info.get('compression_ratio'):
        #         print(f"数据压缩比: {info['compression_ratio']:.1%}")
        
        # 各部分数据统计
        if 'operator_results' in summary_data:
            print(f"算子分析结果: {len(summary_data['operator_results'])} 个算子")
        
        if 'category_results' in summary_data:
            print(f"类别分析结果: {len(summary_data['category_results'])} 个类别")
        
        if 'bottleneck_ranking' in summary_data:
            print(f"瓶颈排名: {len(summary_data['bottleneck_ranking'])} 个算子")
        
        if 'linear_analysis' in summary_data:
            linear_analysis = summary_data['linear_analysis']
            if 'linear_projections' in linear_analysis:
                print(f"Linear分析: {len(linear_analysis['linear_projections'])} 个projection类型")
        
        # 硬件信息
        if 'hardware_specs' in summary_data:
            hw = summary_data['hardware_specs']
            print(f"硬件信息: {hw.get('gpu_name', 'Unknown')} ({hw.get('n_gpu', 1)} GPU)")
        
        # 整体指标
        if 'overall_metrics' in summary_data:
            metrics = summary_data['overall_metrics']
            print(f"整体效率: {metrics.get('overall_efficiency', 0):.3f}")
            print(f"内存利用率: {metrics.get('overall_memory_utilization', 0):.3f}")

def batch_extract_summaries(input_dir: str, output_dir: str = None, pattern: str = "oea_stage4_efficiency_analysis_results.json", in_place: bool = False):
    """批量提取多个案例的OEA汇总数据"""
    
    print(f"\n=== 批量提取OEA汇总数据 ===")
    print(f"输入目录: {input_dir}")
    if in_place:
        print(f"输出模式: 原地输出 (与原文件同目录)")
    else:
        print(f"输出目录: {output_dir}")
    print(f"文件模式: {pattern}")
    
    extractor = OEASummaryExtractor()
    
    # 如果不是原地输出，创建输出目录
    if not in_place and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    # 遍历输入目录
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file == pattern:
                total_count += 1
                
                input_file = os.path.join(root, file)
                
                # 构造输出文件路径
                if in_place:
                    # 原地输出：与原文件同目录
                    output_file = os.path.join(root, "oea_summary.json")
                else:
                    # 输出到指定目录
                    rel_path = os.path.relpath(root, input_dir)
                    output_subdir = os.path.join(output_dir, rel_path)
                    output_file = os.path.join(output_subdir, "oea_summary.json")
                
                print(f"\n--- 处理案例 {total_count} ---")
                rel_path = os.path.relpath(root, input_dir)
                print(f"案例路径: {rel_path}")
                
                if extractor.extract_from_file(input_file, output_file):
                    success_count += 1
                    print(f"✅ 案例 {total_count} 处理成功")
                else:
                    print(f"❌ 案例 {total_count} 处理失败")
    
    print(f"\n=== 批量提取完成 ===")
    print(f"总案例数: {total_count}")
    print(f"成功提取: {success_count}")
    print(f"失败数量: {total_count - success_count}")
    print(f"成功率: {success_count/total_count*100:.1f}%" if total_count > 0 else "N/A")

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='OEA结果数据提取器 - 用于横向对比分析')
    parser.add_argument('--input_file', type=str, help='输入的完整OEA结果文件路径')
    parser.add_argument('--output_file', type=str, help='输出的OEA汇总文件路径')
    parser.add_argument('--batch_input_dir', type=str, help='批量处理的输入目录')
    parser.add_argument('--batch_output_dir', type=str, help='批量处理的输出目录')
    parser.add_argument('--in_place', action='store_true', help='原地输出：将oea_summary.json保存在与原文件相同的目录下')
    parser.add_argument('--pattern', type=str, default='oea_stage4_efficiency_analysis_results.json',
                       help='批量处理时的文件名模式')
    
    args = parser.parse_args()
    
    if args.input_file and args.output_file:
        # 单文件处理
        extractor = OEASummaryExtractor()
        success = extractor.extract_from_file(args.input_file, args.output_file)
        exit(0 if success else 1)
        
    elif args.batch_input_dir:
        # 批量处理
        if args.in_place:
            # 原地输出模式
            batch_extract_summaries(args.batch_input_dir, in_place=True, pattern=args.pattern)
        elif args.batch_output_dir:
            # 指定输出目录模式
            batch_extract_summaries(args.batch_input_dir, args.batch_output_dir, args.pattern)
        else:
            print("❌ 批量处理需要指定输出目录 (--batch_output_dir) 或使用原地输出模式 (--in_place)")
            exit(1)
        
    else:
        print("请提供输入和输出参数:")
        print("单文件处理: --input_file <input> --output_file <output>")
        print("批量处理: --batch_input_dir <input_dir> --batch_output_dir <output_dir>")
        print("原地批量处理: --batch_input_dir <input_dir> --in_place")
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()