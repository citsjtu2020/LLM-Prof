#!/usr/bin/env python3
"""
SGLang框架 LLM-Prof 数据整合脚本
整合SGLang框架案例的SEA、MEA、OEA三层数据，用于横向对比分析
数据源：
1. SEA+MEA: oea_analysis_for_sglang/cases_after_sea_with_mea.csv
2. OEA: cases_after_sea/{GPU_TYPE}/{CASE_NAME}/oea_summary_*.json
"""

import json
import os
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from glob import glob

class SGLangDataIntegrator:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.hardware_specs = {}
        self.load_hardware_specs()
        
    def load_hardware_specs(self):
        """加载硬件规格数据"""
        hardware_specs_raw = {
            "NVIDIA_A10": {"mem_bandwidth": 600 * (1024**3), "FP16": 125e12, "INT8": 250e12, "memsize": 24 * (1024**3)},
            "NVIDIA_L20": {"mem_bandwidth": 864 * (1024**3), "FP16": 119.5e12, "INT8": 239e12, "memsize": 48 * (1024**3)},
            "NVIDIA_H20_SXM5_96GB": {"mem_bandwidth": 4022 * (1024**3), "FP16": 148e12, "INT8": 296e12, "memsize": 96 * (1024**3)},
            "NVIDIA_H20_SXM5_141GB": {"mem_bandwidth": 4800 * (1024**3), "FP16": 148e12, "INT8": 296e12, "memsize": 141 * (1024**3)},
            "NVIDIA_A100_SXM4_80GB": {"mem_bandwidth": 2039 * (1024**3), "FP16": 312e12, "INT8": 624e12, "memsize": 80 * (1024**3)},
            "NVIDIA_A800_SXM4_80GB": {"mem_bandwidth": 2039 * (1024**3), "FP16": 312e12, "INT8": 624e12, "memsize": 80 * (1024**3)},
            "NVIDIA_H800": {"mem_bandwidth": 3350 * (1024**3), "FP16": 989e12, "INT8": 1979e12, "memsize": 80 * (1024**3)},
            "NVIDIA_H100": {"mem_bandwidth": 3350 * (1024**3), "FP16": 989e12, "INT8": 1979e12, "memsize": 80 * (1024**3)},
            "NVIDIA_H200": {"mem_bandwidth": 4800 * (1024**3), "FP16": 989e12, "INT8": 1979e12, "memsize": 141 * (1024**3)},
            "NVIDIA_B200": {"mem_bandwidth": 8000 * (1024**3), "FP16": 2250e12, "INT8": 4500e12, "memsize": 192 * (1024**3)},
            "AMD_MI308X": {"mem_bandwidth": 4000 * (1024**3), "FP16": 115e12, "INT8": 230e12, "memsize": 192 * (1024**3)},
        }
        
        # 转换为TFLOPs和GB/s单位
        for gpu_name, specs in hardware_specs_raw.items():
            self.hardware_specs[gpu_name] = {
                "mem_bandwidth_gbps": specs["mem_bandwidth"] / (1024**3),
                "fp16_tflops": specs["FP16"] / 1e12,
                "int8_tops": specs["INT8"] / 1e12,
                "memory_size_gb": specs["memsize"] / (1024**3)
            }
    
    def load_sea_mea_data(self) -> pd.DataFrame:
        """加载SEA+MEA数据"""
        # 修改路径：先尝试 oea_analysis_for_sglang 目录
        csv_file = self.base_dir / "oea_analysis_for_sglang" / "cases_after_sea_with_mea.csv"
        
        # 如果不存在，尝试当前目录
        if not csv_file.exists():
            csv_file = self.base_dir / "cases_after_sea_with_mea.csv"
        
        if not csv_file.exists():
            print(f"❌ 未找到SEA+MEA数据文件: {csv_file}")
            return pd.DataFrame()
        
        print(f"📊 加载SEA+MEA数据: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"   找到 {len(df)} 个案例的SEA+MEA数据")
        
        return df
    
    def find_oea_summary_files(self) -> Dict[str, Path]:
        """查找所有OEA summary文件"""
        oea_files = {}
        
        cases_dir = self.base_dir / "cases_after_sea"
        if not cases_dir.exists():
            print(f"❌ 未找到案例目录: {cases_dir}")
            return oea_files
        
        # 遍历GPU类型目录
        for gpu_dir in cases_dir.iterdir():
            if not gpu_dir.is_dir():
                continue
            
            # 遍历案例目录
            for case_dir in gpu_dir.iterdir():
                if not case_dir.is_dir():
                    continue
                
                # 查找oea_summary_*.json文件
                summary_files = list(case_dir.glob("oea_summary_*.json"))
                if summary_files:
                    case_name = case_dir.name
                    oea_files[case_name] = summary_files[0]
        
        print(f"📁 找到 {len(oea_files)} 个OEA summary文件")
        return oea_files
    
    def load_oea_data(self, oea_file: Path) -> Optional[Dict]:
        """加载OEA数据"""
        try:
            with open(oea_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 加载OEA数据失败 {oea_file}: {e}")
            return None
    
    def parse_case_name(self, case_name: str) -> Dict[str, Any]:
        """从案例名称解析信息
        格式: ModelName_batchX_inputY_outputZ
        例如: Qwen3-32B_batch8_input1024_output10
        """
        parts = case_name.split('_')
        
        info = {
            'model_name': parts[0] if parts else None,
            'batch_size': None,
            'input_size': None,
            'output_size': None
        }
        
        for part in parts:
            if part.startswith('batch'):
                info['batch_size'] = int(part.replace('batch', ''))
            elif part.startswith('input'):
                info['input_size'] = int(part.replace('input', ''))
            elif part.startswith('output'):
                info['output_size'] = int(part.replace('output', ''))
        
        return info
    
    def map_gpu_type_to_spec_key(self, gpu_type: str) -> str:
        """将GPU类型映射到硬件规格键"""
        mapping = {
            'H20': 'NVIDIA_H20_SXM5_96GB',
            'H800': 'NVIDIA_H800',
            'A100': 'NVIDIA_A100_SXM4_80GB',
            'A800': 'NVIDIA_A800_SXM4_80GB',
            'L20': 'NVIDIA_L20'
        }
        return mapping.get(gpu_type, gpu_type)
    
    def integrate_all_data(self) -> List[Dict]:
        """整合所有数据"""
        print("🚀 开始SGLang框架数据整合...")
        print("=" * 60)
        
        # 1. 加载SEA+MEA数据
        sea_mea_df = self.load_sea_mea_data()
        if sea_mea_df.empty:
            print("❌ SEA+MEA数据为空，无法继续")
            return []
        
        # 2. 查找OEA summary文件
        print("\n📁 查找OEA数据...")
        oea_files = self.find_oea_summary_files()
        
        # 3. 整合数据
        print("\n🔄 整合数据...")
        integrated_data = []
        
        for idx, row in sea_mea_df.iterrows():
            case_name = row['pod_name']
            print(f"\n处理案例 [{idx+1}/{len(sea_mea_df)}]: {case_name}")
            
            # 解析案例名称
            case_info = self.parse_case_name(case_name)
            
            # 构建案例数据
            case_data = {
                'case_name': case_name,
                'sea_mea_data': row.to_dict(),
                'case_info': case_info
            }
            
            # 查找对应的OEA数据
            if case_name in oea_files:
                oea_file = oea_files[case_name]
                oea_data = self.load_oea_data(oea_file)
                if oea_data:
                    case_data['oea_data'] = oea_data
                    case_data['oea_file'] = str(oea_file)
                    print(f"   ✅ OEA数据加载成功")
                else:
                    print(f"   ❌ OEA数据加载失败")
            else:
                print(f"   ⚠️  未找到OEA数据")
            
            integrated_data.append(case_data)
        
        print(f"\n✅ 数据整合完成，共处理 {len(integrated_data)} 个案例")
        return integrated_data
    
    def extract_key_metrics(self, case_data: Dict) -> Dict:
        """提取关键指标，保持与vLLM框架一致的列结构"""
        metrics = {}
        
        # SEA+MEA层数据
        sea_mea = case_data.get('sea_mea_data', {})
        case_info = case_data.get('case_info', {})
        
        # 基本信息 - 生成case_id（使用索引）
        metrics['case_id'] = None  # 后续在导出时填充
        metrics['pod_name'] = sea_mea.get('pod_name')
        metrics['model_name'] = sea_mea.get('model_name')
        metrics['gpu_type'] = sea_mea.get('GPU_type')
        metrics['gpu_num'] = 1  # SGLang案例默认为1
        
        # 分组信息 - SGLang案例暂时不分组
        metrics['group_name'] = 'SGLang Framework'
        metrics['group_id'] = 0
        
        # SEA层指标
        metrics['sea_qps'] = sea_mea.get('qps')
        metrics['sea_fpr'] = sea_mea.get('FPR')
        metrics['sea_token_size'] = sea_mea.get('token_size')
        
        # MEA层指标
        metrics['mea_iips'] = sea_mea.get('IIPS')
        metrics['mea_total_iterations'] = sea_mea.get('iteration')
        metrics['mea_avg_iteration_duration_us'] = None  # 可以从IIPS计算
        if metrics['mea_iips'] and metrics['mea_iips'] > 0:
            metrics['mea_avg_iteration_duration_us'] = 1_000_000 / metrics['mea_iips']
        metrics['mea_std_iteration_duration_us'] = None  # CSV中没有此数据
        metrics['mea_mie'] = sea_mea.get('MIE')
        
        # GPU利用率
        metrics['gpu_utilization'] = sea_mea.get('GPU_util')
        
        # OEA层指标
        oea_data = case_data.get('oea_data', {})
        if oea_data:
            overall_metrics = oea_data.get('overall_metrics', {})
            bottleneck_ranking = oea_data.get('bottleneck_ranking', [])
            
            # 整体效率指标
            metrics['oea_overall_efficiency'] = overall_metrics.get('overall_efficiency')
            metrics['oea_total_compute_time_us'] = overall_metrics.get('total_kernel_time_us')
            metrics['oea_total_flops'] = overall_metrics.get('total_flops')
            
            # 内存利用率 - 从hardware_specs计算
            hw_specs = oea_data.get('hardware_specs', {})
            total_memory_access = overall_metrics.get('total_memory_access', 0)
            total_time_s = overall_metrics.get('total_kernel_time_us', 0) / 1_000_000
            peak_bandwidth_gbps = hw_specs.get('pi', 0)  # SGLang使用pi字段表示带宽(GB/s)
            
            if total_time_s > 0 and peak_bandwidth_gbps > 0:
                actual_bandwidth_gbps = total_memory_access / total_time_s
                metrics['oea_memory_utilization'] = actual_bandwidth_gbps / peak_bandwidth_gbps
            else:
                metrics['oea_memory_utilization'] = None
            
            # 提取前5个瓶颈算子
            for i in range(5):
                prefix = f'oea_bottleneck_{i+1}'
                if i < len(bottleneck_ranking):
                    bottleneck = bottleneck_ranking[i]
                    metrics[f'{prefix}_operator'] = bottleneck.get('operator_type')
                    metrics[f'{prefix}_score'] = bottleneck.get('bottleneck_score')
                    metrics[f'{prefix}_efficiency'] = bottleneck.get('efficiency_degree')
                    metrics[f'{prefix}_time_proportion'] = bottleneck.get('kernel_time_proportion')
                else:
                    # 填充空值
                    metrics[f'{prefix}_operator'] = None
                    metrics[f'{prefix}_score'] = None
                    metrics[f'{prefix}_efficiency'] = None
                    metrics[f'{prefix}_time_proportion'] = None
        else:
            # OEA数据缺失，填充空值
            metrics['oea_overall_efficiency'] = None
            metrics['oea_total_compute_time_us'] = None
            metrics['oea_total_flops'] = None
            metrics['oea_memory_utilization'] = None
            
            for i in range(5):
                prefix = f'oea_bottleneck_{i+1}'
                metrics[f'{prefix}_operator'] = None
                metrics[f'{prefix}_score'] = None
                metrics[f'{prefix}_efficiency'] = None
                metrics[f'{prefix}_time_proportion'] = None
        
        # 硬件规格
        gpu_type = metrics.get('gpu_type', '')
        gpu_spec_key = self.map_gpu_type_to_spec_key(gpu_type)
        if gpu_spec_key in self.hardware_specs:
            hw_spec = self.hardware_specs[gpu_spec_key]
            metrics['hw_mem_bandwidth_gbps'] = hw_spec['mem_bandwidth_gbps']
            metrics['hw_fp16_tflops'] = hw_spec['fp16_tflops']
            metrics['hw_memory_size_gb'] = hw_spec['memory_size_gb']
        else:
            metrics['hw_mem_bandwidth_gbps'] = None
            metrics['hw_fp16_tflops'] = None
            metrics['hw_memory_size_gb'] = None
        
        return metrics
    
    def export_to_csv(self, integrated_data: List[Dict], output_file: str = "sglang_integrated_data.csv"):
        """导出为CSV格式，保持与vLLM框架一致的列顺序"""
        print(f"\n📤 导出数据到 {output_file}...")
        
        # 提取关键指标
        metrics_list = []
        for idx, case_data in enumerate(integrated_data):
            metrics = self.extract_key_metrics(case_data)
            metrics['case_id'] = idx + 1  # 填充case_id
            metrics_list.append(metrics)
        
        # 创建DataFrame
        df = pd.DataFrame(metrics_list)
        
        # 确保列顺序与vLLM框架一致
        column_order = [
            'case_id', 'pod_name', 'model_name', 'gpu_type', 'gpu_num',
            'group_name', 'group_id',
            'sea_qps', 'sea_fpr', 'sea_token_size',
            'mea_iips', 'mea_total_iterations', 'mea_avg_iteration_duration_us',
            'mea_std_iteration_duration_us', 'mea_mie',
            'gpu_utilization',
            'oea_overall_efficiency', 'oea_total_compute_time_us', 'oea_total_flops',
            'oea_memory_utilization',
            'oea_bottleneck_1_operator', 'oea_bottleneck_1_score',
            'oea_bottleneck_1_efficiency', 'oea_bottleneck_1_time_proportion',
            'oea_bottleneck_2_operator', 'oea_bottleneck_2_score',
            'oea_bottleneck_2_efficiency', 'oea_bottleneck_2_time_proportion',
            'oea_bottleneck_3_operator', 'oea_bottleneck_3_score',
            'oea_bottleneck_3_efficiency', 'oea_bottleneck_3_time_proportion',
            'oea_bottleneck_4_operator', 'oea_bottleneck_4_score',
            'oea_bottleneck_4_efficiency', 'oea_bottleneck_4_time_proportion',
            'oea_bottleneck_5_operator', 'oea_bottleneck_5_score',
            'oea_bottleneck_5_efficiency', 'oea_bottleneck_5_time_proportion',
            'hw_mem_bandwidth_gbps', 'hw_fp16_tflops', 'hw_memory_size_gb'
        ]
        
        # 重新排序列
        df = df[column_order]
        
        # 导出CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ 数据导出完成: {output_file}")
        print(f"   共 {len(df)} 行，{len(df.columns)} 列")
        
        return df
    
    def export_to_json(self, integrated_data: List[Dict], output_file: str = "sglang_integrated_data.json"):
        """导出为JSON格式"""
        print(f"\n📤 导出完整数据到 {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(integrated_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 完整数据导出完成: {output_file}")
        
        return integrated_data
    
    def print_summary(self, df: pd.DataFrame):
        """打印数据摘要"""
        print("\n" + "=" * 60)
        print("📈 数据概览")
        print("=" * 60)
        print(f"总案例数: {len(df)}")
        print(f"数据列数: {len(df.columns)}")
        
        # 按GPU类型统计
        if 'gpu_type' in df.columns:
            print("\n📊 GPU类型分布:")
            gpu_stats = df['gpu_type'].value_counts()
            for gpu, count in gpu_stats.items():
                print(f"   {gpu}: {count} 个案例")
        
        # 按模型统计
        if 'model_name' in df.columns:
            print("\n📊 模型分布:")
            model_stats = df['model_name'].value_counts()
            for model, count in model_stats.items():
                print(f"   {model}: {count} 个案例")
        
        # 关键指标统计
        key_metrics = {
            'sea_fpr': 'SEA FPR',
            'mea_iips': 'MEA IIPS',
            'mea_mie': 'MEA MIE',
            'oea_overall_efficiency': 'OEA整体效率',
            'gpu_utilization': 'GPU利用率'
        }
        
        print("\n📊 关键指标统计:")
        for metric, name in key_metrics.items():
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    print(f"   {name}:")
                    print(f"      均值={values.mean():.4f}, 标准差={values.std():.4f}")
                    print(f"      范围=[{values.min():.4f}, {values.max():.4f}]")

def main():
    """主函数"""
    print("🎯 SGLang框架 LLM-Prof 数据整合工具")
    print("=" * 60)
    
    # 创建数据整合器
    integrator = SGLangDataIntegrator()
    
    # 整合所有数据
    integrated_data = integrator.integrate_all_data()
    
    if not integrated_data:
        print("\n❌ 没有数据可以导出")
        return
    
    # 导出数据
    print("\n" + "=" * 60)
    print("📊 导出数据")
    print("=" * 60)
    
    df = integrator.export_to_csv(integrated_data)
    integrator.export_to_json(integrated_data)
    
    # 显示数据摘要
    integrator.print_summary(df)
    
    print("\n🎉 数据整合完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()