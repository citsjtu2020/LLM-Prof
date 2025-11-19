#!/usr/bin/env python3
"""
MEA Analysis for vLLM Cases
专门针对vLLM框架的MEA分析脚本，跳过iteration验证，直接计算IIPS和MIE

已知条件：
- 可以直接从trace文件中提取时间信息计算IIPS和MIE
- 文件结构：{base_dir}/{GPU_type}/{pod_name}/trace文件
"""

import pandas as pd
import json
import gzip
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import argparse
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class vLLMMEAAnalyzer:
    """vLLM框架专用的MEA分析器"""
    
    def __init__(self):
        # GPU硬件规格配置 - 基于论文中的硬件规格
        self.gpu_specs = {
            'A100': {'F_peak': 312.0, 'memory_bandwidth': 1935.0},  # TFLOPs, GB/s
            'A800': {'F_peak': 312.0, 'memory_bandwidth': 1935.0},
            'H20': {'F_peak': 296.0, 'memory_bandwidth': 4800.0},
            'H800': {'F_peak': 989.0, 'memory_bandwidth': 3350.0}, 
            'L20': {'F_peak': 59.7, 'memory_bandwidth': 1229.0},
        }
        
    def load_trace_file(self, trace_file_path: str) -> Dict[str, Any]:
        """
        加载trace.json.gz文件
        
        Args:
            trace_file_path: trace文件路径
            
        Returns:
            解析后的trace数据
        """
        try:
            if trace_file_path.endswith('.gz'):
                with gzip.open(trace_file_path, 'rt', encoding='utf-8') as f:
                    trace_data = json.load(f)
            else:
                with open(trace_file_path, 'r', encoding='utf-8') as f:
                    trace_data = json.load(f)
            
            logger.info(f"Successfully loaded trace file: {trace_file_path}")
            return trace_data
            
        except Exception as e:
            logger.error(f"Failed to load trace file {trace_file_path}: {e}")
            return {}
    
    def extract_kernel_events(self, trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从trace数据中提取GPU kernel事件
        
        Args:
            trace_data: trace数据
            
        Returns:
            GPU kernel事件列表
        """
        events = trace_data.get('traceEvents', [])
        kernel_events = []
        
        for event in events:
            # 筛选GPU kernel事件 (ph='X' 表示完整事件，cat='kernel' 表示GPU kernel)
            if (event.get('ph') == 'X' and 
                event.get('cat') == 'kernel' and
                'ts' in event and 'dur' in event):
                
                kernel_events.append({
                    'name': event.get('name', ''),
                    'ts': event.get('ts', 0),  # 开始时间戳 (微秒)
                    'dur': event.get('dur', 0),  # 持续时间 (微秒)
                    'end_ts': event.get('ts', 0) + event.get('dur', 0)
                })
        
        # 按时间戳排序
        kernel_events.sort(key=lambda x: x['ts'])
        logger.info(f"Extracted {len(kernel_events)} kernel events")
        
        return kernel_events
    
    def calculate_iips_from_trace(self, kernel_events: List[Dict[str, Any]], 
                                 num_iterations: int = 10) -> Dict[str, Any]:
        """
        基于kernel事件计算IIPS
        
        对于vLLM，我们知道有固定的10个iteration，可以直接计算端到端时间
        
        Args:
            kernel_events: GPU kernel事件列表
            num_iterations: iteration数量（默认10）
            
        Returns:
            IIPS计算结果
        """
        if not kernel_events:
            logger.warning("No kernel events found for IIPS calculation")
            return {
                'iips': 0.0,
                'total_iterations': 0,
                'total_duration_us': 0.0,
                'total_duration_s': 0.0,
                'error': 'No kernel events'
            }
        
        # 计算端到端时间：从第一个kernel开始到最后一个kernel结束
        first_start_ts = kernel_events[0]['ts']
        last_end_ts = max(event['end_ts'] for event in kernel_events)
        
        total_duration_us = last_end_ts - first_start_ts
        total_duration_s = total_duration_us / 1_000_000.0
        
        if total_duration_s <= 0:
            logger.warning("Invalid total duration for IIPS calculation")
            return {
                'iips': 0.0,
                'total_iterations': num_iterations,
                'total_duration_us': total_duration_us,
                'total_duration_s': total_duration_s,
                'error': 'Invalid duration'
            }
        
        # 计算IIPS
        iips = num_iterations / total_duration_s
        
        result = {
            'iips': float(iips),
            'total_iterations': num_iterations,
            'total_duration_us': float(total_duration_us),
            'total_duration_s': float(total_duration_s),
            'first_start_ts': float(first_start_ts),
            'last_end_ts': float(last_end_ts),
            'num_kernel_events': len(kernel_events)
        }
        
        logger.info(f"IIPS calculated: {iips:.2f} iterations/second")
        logger.info(f"End-to-end duration: {total_duration_s:.3f} seconds")
        
        return result
    
    def calculate_mie(self, iips_result: Dict[str, Any], gpu_type: str, 
                     gpu_util: float, n_gpu: int = 1) -> Dict[str, Any]:
        """
        计算MIE (Model Inference Efficiency)
        
        基于论文公式：MIE = (F_peak × u_GPU × N_GPU) / IIPS
        
        Args:
            iips_result: IIPS计算结果
            gpu_type: GPU类型
            gpu_util: GPU利用率 (0-100)
            n_gpu: GPU数量
            
        Returns:
            MIE计算结果
        """
        iips = iips_result.get('iips', 0.0)
        
        if iips <= 0:
            logger.warning("IIPS is zero or negative, cannot calculate MIE")
            return {
                'mie': float('inf'),
                'error': 'Invalid IIPS value'
            }
        
        # 获取GPU规格
        if gpu_type not in self.gpu_specs:
            logger.warning(f"Unknown GPU type: {gpu_type}, using default values")
            f_peak = 100.0  # 默认值
            memory_bandwidth = 1000.0
        else:
            f_peak = self.gpu_specs[gpu_type]['F_peak']
            memory_bandwidth = self.gpu_specs[gpu_type]['memory_bandwidth']
        
        # 转换GPU利用率：从百分比转换为小数
        u_gpu = gpu_util / 100.0 if gpu_util > 1 else gpu_util
        
        # 计算MIE
        mie = (f_peak * u_gpu * n_gpu) / iips
        
        result = {
            'mie': float(mie),
            'f_peak': float(f_peak),
            'u_gpu': float(u_gpu),
            'n_gpu': int(n_gpu),
            'iips': float(iips),
            'gpu_type': gpu_type,
            'effective_compute_power_tflops': float(f_peak * u_gpu * n_gpu),
            'memory_bandwidth_gb_s': float(memory_bandwidth)
        }
        
        logger.info(f"MIE calculated: {mie:.6f} TFLOPs per iteration")
        logger.info(f"GPU: {gpu_type}, F_peak: {f_peak} TFLOPs, u_GPU: {u_gpu:.3f}")
        
        return result
    
    def find_trace_file_by_structure(self, base_dir: str, gpu_type: str, pod_name: str) -> Optional[str]:
        """
        根据文件结构自动查找trace文件
        
        文件结构：{base_dir}/{GPU_type}/{pod_name}/trace文件
        
        Args:
            base_dir: 基础目录
            gpu_type: GPU类型
            pod_name: pod名称
            
        Returns:
            trace文件路径，如果找不到返回None
        """
        # 构建预期的目录路径
        expected_dir = os.path.join(base_dir, gpu_type, pod_name)
        
        logger.info(f"Looking for trace file in: {expected_dir}")
        
        if not os.path.exists(expected_dir):
            logger.warning(f"Directory does not exist: {expected_dir}")
            return None
        
        # 在目录中查找trace文件
        trace_patterns = [
            "*.trace.json.gz",
            "*.trace.json",
            "**/mlflow-*.trace.json.gz",  # 支持子目录中的mlflow格式
            "**/mlflow-*.trace.json"
        ]
        
        for pattern in trace_patterns:
            search_path = os.path.join(expected_dir, pattern)
            matches = glob.glob(search_path, recursive=True)
            
            if matches:
                trace_file = matches[0]  # 取第一个匹配的文件
                logger.info(f"Found trace file: {trace_file}")
                return trace_file
        
        # 如果没有找到，列出目录内容以便调试
        try:
            files = os.listdir(expected_dir)
            logger.warning(f"No trace file found in {expected_dir}. Directory contents: {files}")
        except Exception as e:
            logger.error(f"Cannot list directory {expected_dir}: {e}")
        
        return None
    
    def find_trace_file_fallback(self, base_dir: str, gpu_type: str, pod_name: str) -> Optional[str]:
        """
        备用查找方法：在整个base_dir中递归搜索
        
        Args:
            base_dir: 基础目录
            gpu_type: GPU类型
            pod_name: pod名称
            
        Returns:
            trace文件路径，如果找不到返回None
        """
        logger.info(f"Fallback search for {pod_name} in {base_dir}")
        
        # 递归搜索包含pod_name的trace文件
        search_patterns = [
            f"**/{pod_name}*.trace.json.gz",
            f"**/{pod_name}*.trace.json",
            f"**/*{pod_name}*.trace.json.gz",
            f"**/*{pod_name}*.trace.json"
        ]
        
        for pattern in search_patterns:
            search_path = os.path.join(base_dir, pattern)
            matches = glob.glob(search_path, recursive=True)
            
            if matches:
                # 优先选择包含GPU类型的路径
                for match in matches:
                    if gpu_type in match:
                        logger.info(f"Found trace file (fallback): {match}")
                        return match
                
                # 如果没有包含GPU类型的，返回第一个匹配
                logger.info(f"Found trace file (fallback): {matches[0]}")
                return matches[0]
        
        return None
    
    def analyze_single_case(self, case_row: pd.Series, trace_base_dir: str) -> Dict[str, Any]:
        """
        分析单个案例
        
        Args:
            case_row: 案例数据行
            trace_base_dir: trace文件基础目录
            
        Returns:
            分析结果
        """
        pod_name = case_row['pod_name']
        gpu_type = case_row['GPU_type']
        gpu_util = case_row['GPU_util']
        
        logger.info(f"Analyzing case: {pod_name} on {gpu_type}")
        
        # 首先尝试按结构查找
        trace_file = self.find_trace_file_by_structure(trace_base_dir, gpu_type, pod_name)
        
        # 如果没找到，尝试备用方法
        if not trace_file:
            trace_file = self.find_trace_file_fallback(trace_base_dir, gpu_type, pod_name)
        
        if not trace_file:
            return {
                'pod_name': pod_name,
                'iips': None,
                'mie': None,
                'error': f'Trace file not found for {gpu_type}/{pod_name}'
            }
        
        # 加载trace数据
        trace_data = self.load_trace_file(trace_file)
        if not trace_data:
            return {
                'pod_name': pod_name,
                'iips': None,
                'mie': None,
                'error': 'Failed to load trace data'
            }
        
        # 提取kernel事件
        kernel_events = self.extract_kernel_events(trace_data)
        if not kernel_events:
            return {
                'pod_name': pod_name,
                'iips': None,
                'mie': None,
                'error': 'No kernel events found'
            }
        
        # 计算IIPS（固定10个iteration）
        iips_result = self.calculate_iips_from_trace(kernel_events, num_iterations=10)
        
        # 计算MIE
        mie_result = self.calculate_mie(iips_result, gpu_type, gpu_util, n_gpu=1)
        
        return {
            'pod_name': pod_name,
            'iips': iips_result.get('iips'),
            'mie': mie_result.get('mie'),
            'total_duration_s': iips_result.get('total_duration_s'),
            'num_kernel_events': iips_result.get('num_kernel_events'),
            'effective_compute_power_tflops': mie_result.get('effective_compute_power_tflops'),
            'trace_file': trace_file
        }
    
    def process_cases(self, csv_file: str, trace_base_dir: str, output_file: str):
        """
        处理所有案例并更新CSV文件
        
        Args:
            csv_file: 输入CSV文件路径
            trace_base_dir: trace文件基础目录
            output_file: 输出CSV文件路径
        """
        # 读取CSV文件
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} cases from {csv_file}")
            
            # 验证必要的列是否存在
            required_columns = ['pod_name', 'GPU_type', 'GPU_util']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns in CSV: {missing_columns}")
                return
                
            logger.info(f"CSV columns: {list(df.columns)}")
            
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            return
        
        # 显示数据概览
        gpu_types = df['GPU_type'].unique()
        logger.info(f"GPU types in dataset: {list(gpu_types)}")
        for gpu_type in gpu_types:
            count = len(df[df['GPU_type'] == gpu_type])
            logger.info(f"  {gpu_type}: {count} cases")
        
        # 显示案例预览
        logger.info(f"Sample cases:")
        for idx in range(min(3, len(df))):
            row = df.iloc[idx]
            logger.info(f"  Case {idx+1}: {row['pod_name']} ({row['GPU_type']}, GPU_util={row['GPU_util']}%)")
        
        # 初始化新列 - 只添加IIPS和MIE两列
        df['IIPS'] = None
        df['MIE'] = None
        
        # 处理每个案例
        success_count = 0
        for idx, row in df.iterrows():
            try:
                result = self.analyze_single_case(row, trace_base_dir)
                
                # 更新DataFrame - 只更新IIPS和MIE
                df.at[idx, 'IIPS'] = result.get('iips')
                df.at[idx, 'MIE'] = result.get('mie')
                
                if result.get('error'):
                    logger.error(f"❌ Failed to analyze {result['pod_name']}: {result['error']}")
                else:
                    success_count += 1
                    iips_val = result.get('iips', 0)
                    mie_val = result.get('mie', 0)
                    logger.info(f"✅ Successfully analyzed {result['pod_name']}: "
                              f"IIPS={iips_val:.2f}, MIE={mie_val:.6f}")
                
            except Exception as e:
                logger.error(f"❌ Error processing case {row['pod_name']}: {e}")
        
        # 保存结果
        try:
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            
            # 打印统计信息
            total_count = len(df)
            logger.info(f"\n📊 Analysis Summary:")
            logger.info(f"Total cases: {total_count}")
            logger.info(f"Successful: {success_count}")
            logger.info(f"Failed: {total_count - success_count}")
            logger.info(f"Success rate: {success_count/total_count*100:.1f}%")
            
            if success_count > 0:
                valid_df = df[df['IIPS'].notna() & df['MIE'].notna()]
                logger.info(f"\n📈 Performance Metrics:")
                logger.info(f"IIPS range: {valid_df['IIPS'].min():.2f} - {valid_df['IIPS'].max():.2f}")
                logger.info(f"MIE range: {valid_df['MIE'].min():.6f} - {valid_df['MIE'].max():.6f}")
                
                # 按GPU类型统计
                logger.info(f"\n🔧 By GPU Type:")
                for gpu_type in gpu_types:
                    gpu_df = valid_df[valid_df['GPU_type'] == gpu_type]
                    if len(gpu_df) > 0:
                        avg_iips = gpu_df['IIPS'].mean()
                        avg_mie = gpu_df['MIE'].mean()
                        logger.info(f"  {gpu_type}: {len(gpu_df)} cases, "
                                  f"avg IIPS={avg_iips:.2f}, avg MIE={avg_mie:.6f}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    parser = argparse.ArgumentParser(description='vLLM MEA Analysis Tool with Auto Path Detection')
    parser.add_argument('--csv', '-c', default='cases_after_sea.csv',
                       help='Input CSV file with cases (default: cases_after_sea.csv)')
    parser.add_argument('--trace-dir', '-t', default='.',
                       help='Base directory to search for trace files (default: current directory)')
    parser.add_argument('--output', '-o', default='cases_after_sea_with_mea.csv',
                       help='Output CSV file (default: cases_after_sea_with_mea.csv)')
    
    args = parser.parse_args()
    
    # 验证输入文件
    if not os.path.exists(args.csv):
        logger.error(f"Input CSV file not found: {args.csv}")
        sys.exit(1)
    
    if not os.path.exists(args.trace_dir):
        logger.error(f"Trace directory not found: {args.trace_dir}")
        sys.exit(1)
    
    # 显示目录结构信息
    logger.info(f"Base trace directory: {args.trace_dir}")
    try:
        subdirs = [d for d in os.listdir(args.trace_dir) if os.path.isdir(os.path.join(args.trace_dir, d))]
        logger.info(f"Found subdirectories: {subdirs}")
    except Exception as e:
        logger.warning(f"Cannot list trace directory: {e}")
    
    # 创建分析器并处理案例
    analyzer = vLLMMEAAnalyzer()
    analyzer.process_cases(args.csv, args.trace_dir, args.output)

if __name__ == "__main__":
    main()