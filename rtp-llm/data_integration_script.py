#!/usr/bin/env python3
"""
LLM-Prof 数据整合脚本
整合22个案例的SEA、MEA、OEA三层数据，用于横向对比分析
"""

import json
import os
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

class LLMProfDataIntegrator:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.cases_data = []
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
            "AMD_MI308X": {"mem_bandwidth": 4000 * (1024**3), "FP16": 115e12, "INT8": 230e12, "memsize": 192 * (1024**3)}
        }
        
        # 转换为TFLOPs和GB/s单位
        for gpu_name, specs in hardware_specs_raw.items():
            self.hardware_specs[gpu_name] = {
                "mem_bandwidth_gbps": specs["mem_bandwidth"] / (1024**3),
                "fp16_tflops": specs["FP16"] / 1e12,
                "int8_tops": specs["INT8"] / 1e12,
                "memory_size_gb": specs["memsize"] / (1024**3)
            }
    
    def parse_sea_data(self) -> Dict[str, Dict]:
        """解析SEA层数据"""
        sea_data = {}
        
        # 从cases_after_sea.txt解析数据
        cases_file = self.base_dir / "cases_after_sea.txt"
        if not cases_file.exists():
            print(f"Warning: {cases_file} not found")
            return sea_data
            
        with open(cases_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析每个分组的数据
        groups = {
            "高效": 1,
            "大模型低效": 2, 
            "利用率失衡": 3,
            "硬件差异": 4
        }
        
        # 中文组名到英文组名的映射
        group_name_mapping = {
            "高效": "Low FPR with High QPS or Util",
            "大模型低效": "Large Parameter size with High FPR",
            "利用率失衡": "High FPR with Low Util",
            "硬件差异": "Same Model with Hardware Diff"
        }
        
        for group_name, group_id in groups.items():
            # 查找分组数据
            pattern = rf"分组{group_id}.*?:(.*?)(?=分组|$)"
            match = re.search(pattern, content, re.DOTALL)
            if not match:
                continue
                
            group_content = match.group(1)
            lines = [line.strip() for line in group_content.split('\n') if line.strip()]
            
            for line in lines:
                if line.startswith('序号') or not line:
                    continue
                    
                # 解析每行数据
                parts = line.split()
                if len(parts) >= 10:
                    case_id = parts[0]
                    pod_name = parts[1]
                    model_name = parts[2]
                    gpu_type = parts[3]
                    gpu_num = int(parts[4])
                    qps = float(parts[5])
                    gpu_util = float(parts[6])
                    f_peak = float(parts[7])
                    fpr = float(parts[8])
                    token_size = int(parts[9])
                    
                    # 使用英文组名
                    english_group_name = group_name_mapping.get(group_name, group_name)
                    
                    sea_data[case_id] = {
                        "case_id": case_id,
                        "pod_name": pod_name,
                        "model_name": model_name,
                        "gpu_type": gpu_type,
                        "gpu_num": gpu_num,
                        "qps": qps,
                        "gpu_utilization": gpu_util,
                        "f_peak": f_peak,
                        "fpr": fpr,
                        "token_size": token_size,
                        "group_name": english_group_name,
                        "group_id": group_id
                    }
        
        return sea_data
    
    def find_case_directories(self) -> Dict[str, Path]:
        """查找所有案例目录"""
        case_dirs = {}
        
        # 搜索traces_after_sea_section_part*目录
        for part_dir in self.base_dir.glob("traces_after_sea_section_part*"):
            if part_dir.is_dir():
                for case_dir in part_dir.iterdir():
                    if case_dir.is_dir():
                        case_name = case_dir.name
                        case_dirs[case_name] = case_dir
        
        return case_dirs
    
    def load_mea_data(self, case_dir: Path) -> Optional[Dict]:
        """加载MEA数据"""
        mea_file = case_dir / "stage4_mea_analysis_results.json"
        if not mea_file.exists():
            return None
            
        try:
            with open(mea_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading MEA data from {mea_file}: {e}")
            return None
    
    def load_oea_data(self, case_dir: Path) -> Optional[Dict]:
        """加载OEA数据"""
        oea_file = case_dir / "oea_summary.json"
        if not oea_file.exists():
            return None
            
        try:
            with open(oea_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading OEA data from {oea_file}: {e}")
            return None
    
    def load_config_data(self, case_dir: Path) -> Optional[Dict]:
        """加载配置数据"""
        config_file = case_dir / "prefill_metrics_with_config.txt"
        if not config_file.exists():
            return None
            
        try:
            config_data = {}
            with open(config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        config_data[key.strip()] = value.strip()
            return config_data
        except Exception as e:
            print(f"Error loading config data from {config_file}: {e}")
            return None
    
    def extract_key_metrics(self, case_data: Dict) -> Dict:
        """提取关键指标"""
        metrics = {}
        
        # SEA层指标
        sea_data = case_data.get('sea_data', {})
        metrics.update({
            'case_id': sea_data.get('case_id'),
            'pod_name': sea_data.get('pod_name'),
            'model_name': sea_data.get('model_name'),
            'gpu_type': sea_data.get('gpu_type'),
            'gpu_num': sea_data.get('gpu_num'),
            'group_name': sea_data.get('group_name'),
            'group_id': sea_data.get('group_id'),
            'sea_qps': sea_data.get('qps'),
            'sea_fpr': sea_data.get('fpr'),
            'sea_token_size': sea_data.get('token_size'),
        })
        
        # MEA层指标
        mea_data = case_data.get('mea_data', {})
        if mea_data:
            iips_analysis = mea_data.get('mea_analysis', {}).get('iips_analysis', {})
            mie_analysis = mea_data.get('mea_analysis', {}).get('mie_analysis', {})
            
            metrics.update({
                'mea_iips': iips_analysis.get('iips'),
                'mea_total_iterations': iips_analysis.get('total_iterations'),
                'mea_avg_iteration_duration_us': iips_analysis.get('average_iteration_duration_us'),
                'mea_std_iteration_duration_us': iips_analysis.get('std_iteration_duration_us'),
                'mea_mie': mie_analysis.get('mie'),
            })
        
        # 统一GPU利用率指标 - 优先使用SEA层数据，因为它是业务层面的真实利用率
        gpu_utilization = sea_data.get('gpu_utilization')
        if gpu_utilization is None and mea_data:
            # 如果SEA层没有，则使用MEA层的数据作为备选
            mie_analysis = mea_data.get('mea_analysis', {}).get('mie_analysis', {})
            gpu_utilization = mie_analysis.get('u_gpu')
        
        metrics['gpu_utilization'] = gpu_utilization
        
        # OEA层指标
        oea_data = case_data.get('oea_data', {})
        if oea_data:
            overall_metrics = oea_data.get('overall_metrics', {})
            bottleneck_ranking = oea_data.get('bottleneck_ranking', [])
            
            metrics.update({
                'oea_overall_efficiency': overall_metrics.get('overall_efficiency'),
                'oea_total_compute_time_us': overall_metrics.get('total_compute_time_us'),
                'oea_total_flops': overall_metrics.get('total_flops'),
                'oea_memory_utilization': overall_metrics.get('overall_memory_utilization'),
            })
            
            # 提取前5个瓶颈算子
            for i, bottleneck in enumerate(bottleneck_ranking[:5]):
                prefix = f'oea_bottleneck_{i+1}'
                metrics.update({
                    f'{prefix}_operator': bottleneck.get('operator_type'),
                    f'{prefix}_score': bottleneck.get('bottleneck_score'),
                    f'{prefix}_efficiency': bottleneck.get('efficiency_degree'),
                    f'{prefix}_time_proportion': bottleneck.get('kernel_time_proportion'),
                })
        
        # 硬件规格
        gpu_type = metrics.get('gpu_type', '')
        gpu_spec_key = self.map_gpu_type_to_spec_key(gpu_type)
        if gpu_spec_key in self.hardware_specs:
            hw_spec = self.hardware_specs[gpu_spec_key]
            metrics.update({
                'hw_mem_bandwidth_gbps': hw_spec['mem_bandwidth_gbps'],
                'hw_fp16_tflops': hw_spec['fp16_tflops'],
                'hw_memory_size_gb': hw_spec['memory_size_gb'],
            })
        
        return metrics
    
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
        print("🚀 开始数据整合...")
        
        # 1. 加载SEA数据
        print("📊 加载SEA层数据...")
        sea_data = self.parse_sea_data()
        print(f"   找到 {len(sea_data)} 个案例的SEA数据")
        
        # 2. 查找案例目录
        print("📁 查找案例目录...")
        case_dirs = self.find_case_directories()
        print(f"   找到 {len(case_dirs)} 个案例目录")
        
        # 3. 整合每个案例的数据
        integrated_data = []
        
        for case_name, case_dir in case_dirs.items():
            print(f"🔄 处理案例: {case_name}")
            
            case_data = {
                'case_name': case_name,
                'case_dir': str(case_dir)
            }
            
            # 匹配SEA数据
            matched_sea = None
            for case_id, sea_info in sea_data.items():
                if sea_info['pod_name'] in case_name:
                    matched_sea = sea_info
                    break
            
            if matched_sea:
                case_data['sea_data'] = matched_sea
                print(f"   ✅ SEA数据匹配成功 (案例ID: {matched_sea['case_id']})")
            else:
                print(f"   ❌ SEA数据匹配失败")
                continue
            
            # 加载MEA数据
            mea_data = self.load_mea_data(case_dir)
            if mea_data:
                case_data['mea_data'] = mea_data
                print(f"   ✅ MEA数据加载成功")
            else:
                print(f"   ❌ MEA数据加载失败")
            
            # 加载OEA数据
            oea_data = self.load_oea_data(case_dir)
            if oea_data:
                case_data['oea_data'] = oea_data
                print(f"   ✅ OEA数据加载成功")
            else:
                print(f"   ❌ OEA数据加载失败")
            
            # 加载配置数据
            config_data = self.load_config_data(case_dir)
            if config_data:
                case_data['config_data'] = config_data
                print(f"   ✅ 配置数据加载成功")
            
            integrated_data.append(case_data)
        
        print(f"✅ 数据整合完成，共处理 {len(integrated_data)} 个案例")
        return integrated_data
    
    def export_to_csv(self, integrated_data: List[Dict], output_file: str = "llm_prof_integrated_data.csv"):
        """导出为CSV格式"""
        print(f"📤 导出数据到 {output_file}...")
        
        # 提取关键指标
        metrics_list = []
        for case_data in integrated_data:
            metrics = self.extract_key_metrics(case_data)
            metrics_list.append(metrics)
        
        # 创建DataFrame并导出
        df = pd.DataFrame(metrics_list)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ 数据导出完成: {output_file}")
        print(f"   共 {len(df)} 行，{len(df.columns)} 列")
        
        return df
    
    def export_to_json(self, integrated_data: List[Dict], output_file: str = "llm_prof_integrated_data.json"):
        """导出为JSON格式"""
        print(f"📤 导出完整数据到 {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(integrated_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 完整数据导出完成: {output_file}")
        
        return integrated_data

def main():
    """主函数"""
    print("🎯 LLM-Prof 数据整合工具")
    print("=" * 50)
    
    # 创建数据整合器
    integrator = LLMProfDataIntegrator()
    
    # 整合所有数据
    integrated_data = integrator.integrate_all_data()
    
    # 导出数据
    print("\n📊 导出数据...")
    df = integrator.export_to_csv(integrated_data)
    integrator.export_to_json(integrated_data)
    
    # 显示数据概览
    print("\n📈 数据概览:")
    print(f"   总案例数: {len(df)}")
    print(f"   数据列数: {len(df.columns)}")
    
    # 按分组统计
    if 'group_name' in df.columns:
        group_stats = df['group_name'].value_counts()
        print("\n📊 分组统计:")
        for group, count in group_stats.items():
            print(f"   {group}: {count} 个案例")
    
    # 显示关键指标统计
    key_metrics = ['sea_fpr', 'mea_iips', 'mea_mie', 'oea_overall_efficiency']
    print("\n📊 关键指标统计:")
    for metric in key_metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                print(f"   {metric}: 均值={values.mean():.4f}, 标准差={values.std():.4f}, 范围=[{values.min():.4f}, {values.max():.4f}]")
    
    print("\n🎉 数据整合完成！")

if __name__ == "__main__":
    main()