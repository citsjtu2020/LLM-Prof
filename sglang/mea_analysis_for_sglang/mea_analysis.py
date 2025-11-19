#!/usr/bin/env python3
"""
MEA (Model Efficiency Analyzer) for SGLang Framework
基于 LLM-Prof 论文实现的简化版 MEA 分析器，专门用于 SGLang 框架
改进版本：自动根据 GPU_type 和 pod_name 定位 trace 文件
"""

import pandas as pd
import json
import gzip
import os
import sys
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import argparse
from pathlib import Path
import re
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SGLangMEAAnalyzer:
    """SGLang 框架的简化 MEA 分析器 - 改进版"""
    
    def __init__(self):
        # GPU 峰值算力配置 (TFLOPs, FP16)
        self.gpu_peak_flops = {
            'A100': 312.0,
            'A800': 312.0, 
            'H800': 989.0,
            'H20': 148.0,
            'L20': 59.7,
            'H100': 989.0,
            'V100': 125.0,
        }
        
        # 固定迭代数 (SGLang: 1 prefill + 9 decode)
        self.fixed_iterations = 10
        
        # 缓存已扫描的目录结构
        self.trace_file_cache = {}
        
    def get_gpu_peak_flops(self, gpu_type: str) -> float:
        """获取 GPU 峰值算力"""
        gpu_clean = gpu_type.strip().upper()
        
        # 尝试精确匹配
        for key, value in self.gpu_peak_flops.items():
            if key.upper() == gpu_clean:
                return value
        
        # 尝试部分匹配
        for key, value in self.gpu_peak_flops.items():
            if key.upper() in gpu_clean or gpu_clean in key.upper():
                return value
        
        # 默认值
        logger.warning(f"Unknown GPU type: {gpu_type}, using default H20 peak flops")
        return 148.0
    
    def scan_trace_files(self, base_dir: str = ".") -> Dict[str, List[str]]:
        """
        扫描并缓存所有 trace 文件的位置
        
        Args:
            base_dir: 基础目录
            
        Returns:
            按 GPU 类型分组的 trace 文件字典
        """
        if self.trace_file_cache:
            return self.trace_file_cache
        
        logger.info(f"Scanning trace files in {base_dir}...")
        
        # 使用 glob 模式匹配所有 .trace.json.gz 文件
        pattern = os.path.join(base_dir, "**", "*.trace.json.gz")
        trace_files = glob.glob(pattern, recursive=True)
        
        logger.info(f"Found {len(trace_files)} trace files")
        
        # 按 GPU 类型分组
        gpu_trace_map = {}
        
        for trace_file in trace_files:
            # 从路径中提取 GPU 类型
            path_parts = Path(trace_file).parts
            
            # 查找可能的 GPU 类型目录
            gpu_type = None
            for part in path_parts:
                if part.upper() in ['A100', 'A800', 'H800', 'H20', 'L20', 'H100', 'V100']:
                    gpu_type = part.upper()
                    break
            
            if gpu_type:
                if gpu_type not in gpu_trace_map:
                    gpu_trace_map[gpu_type] = []
                gpu_trace_map[gpu_type].append(trace_file)
            else:
                # 如果无法从路径确定 GPU 类型，放入通用列表
                if 'UNKNOWN' not in gpu_trace_map:
                    gpu_trace_map['UNKNOWN'] = []
                gpu_trace_map['UNKNOWN'].append(trace_file)
        
        self.trace_file_cache = gpu_trace_map
        
        # 打印扫描结果
        for gpu_type, files in gpu_trace_map.items():
            logger.info(f"  {gpu_type}: {len(files)} files")
        
        return gpu_trace_map
    
    def find_trace_file_optimized(self, pod_name: str, gpu_type: str, base_dir: str = ".") -> Optional[str]:
        """
        优化的 trace 文件查找方法
        
        Args:
            pod_name: pod 名称
            gpu_type: GPU 类型
            base_dir: 基础目录
            
        Returns:
            trace 文件路径，如果未找到返回 None
        """
        # 确保已扫描文件
        gpu_trace_map = self.scan_trace_files(base_dir)
        
        # 标准化 GPU 类型
        gpu_type_normalized = gpu_type.upper()
        
        # 构建候选文件名模式
        expected_filename = f"{pod_name}.trace.json.gz"
        
        # 优先在对应 GPU 类型目录中查找
        search_lists = []
        if gpu_type_normalized in gpu_trace_map:
            search_lists.append((gpu_type_normalized, gpu_trace_map[gpu_type_normalized]))
        
        # 如果在对应 GPU 类型中找不到，搜索所有文件
        for gt, files in gpu_trace_map.items():
            if gt != gpu_type_normalized:
                search_lists.append((gt, files))
        
        # 在每个搜索列表中查找
        for search_gpu_type, file_list in search_lists:
            # 精确匹配
            for trace_file in file_list:
                if os.path.basename(trace_file) == expected_filename:
                    logger.info(f"Found exact match: {trace_file} (expected GPU: {gpu_type}, found in: {search_gpu_type})")
                    return trace_file
            
            # 部分匹配（包含 pod_name）
            for trace_file in file_list:
                if pod_name in os.path.basename(trace_file):
                    logger.info(f"Found partial match: {trace_file} (expected GPU: {gpu_type}, found in: {search_gpu_type})")
                    return trace_file
        
        logger.warning(f"Trace file not found for pod: {pod_name}, GPU: {gpu_type}")
        return None
    
    def extract_trace_duration(self, trace_file: str) -> Optional[float]:
        """
        从 trace.json.gz 文件中提取总执行时间
        
        Args:
            trace_file: trace 文件路径
            
        Returns:
            总执行时间 (秒)，如果失败返回 None
        """
        try:
            logger.debug(f"Processing trace file: {trace_file}")
            
            # 检查文件是否存在
            if not os.path.exists(trace_file):
                logger.warning(f"Trace file not found: {trace_file}")
                return None
            
            # 读取压缩的 JSON 文件
            with gzip.open(trace_file, 'rt', encoding='utf-8') as f:
                trace_data = json.load(f)
            
            # 提取 traceEvents
            trace_events = trace_data.get('traceEvents', [])
            if not trace_events:
                logger.warning(f"No trace events found in {trace_file}")
                return None
            
            # 找到第一个和最后一个事件的时间戳
            timestamps = []
            for event in trace_events:
                ts = event.get('ts')
                dur = event.get('dur', 0)
                if ts is not None:
                    timestamps.append(ts)
                    if dur > 0:
                        timestamps.append(ts + dur)
            
            if len(timestamps) < 2:
                logger.warning(f"Insufficient timestamps in {trace_file}")
                return None
            
            # 计算总时间跨度 (微秒转秒)
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            duration_us = max_ts - min_ts
            duration_s = duration_us / 1_000_000.0
            
            logger.debug(f"Extracted duration: {duration_s:.6f} seconds ({duration_us:.1f} μs)")
            return duration_s
            
        except Exception as e:
            logger.error(f"Error processing trace file {trace_file}: {e}")
            return None
    
    def calculate_iips(self, duration_s: float) -> float:
        """
        计算 IIPS (Inference Iterations Per Second)
        
        Args:
            duration_s: 总执行时间 (秒)
            
        Returns:
            IIPS 值
        """
        if duration_s <= 0:
            return 0.0
        
        iips = self.fixed_iterations / duration_s
        return iips
    
    def calculate_mie(self, iips: float, gpu_type: str, gpu_util: float) -> float:
        """
        计算 MIE (Model Inference Efficiency)
        
        根据论文公式：MIE = (F_peak × u_GPU × N_GPU) / IIPS
        
        Args:
            iips: 每秒推理迭代数
            gpu_type: GPU 型号
            gpu_util: GPU 利用率 (百分比)
            
        Returns:
            MIE 值 (TFLOPs per iteration)
        """
        if iips <= 0:
            return float('inf')
        
        f_peak = self.get_gpu_peak_flops(gpu_type)  # TFLOPs
        u_gpu = gpu_util / 100.0  # 转换为 [0,1]
        n_gpu = 1  # SGLang 单卡推理
        
        mie = (f_peak * u_gpu * n_gpu) / iips
        return mie
    
    def analyze_case(self, row: pd.Series, base_dir: str = ".") -> Dict[str, Any]:
        """
        分析单个案例
        
        Args:
            row: CSV 行数据
            base_dir: 基础目录
            
        Returns:
            分析结果字典
        """
        pod_name = row['pod_name']
        gpu_type = row['GPU_type']
        gpu_util = row['GPU_util']
        
        logger.info(f"Analyzing case: {pod_name} on {gpu_type}")
        
        # 使用优化的文件查找方法
        trace_file = self.find_trace_file_optimized(pod_name, gpu_type, base_dir)
        if trace_file is None:
            return {
                'iips': 0.0,
                'mie': float('inf'),
                'duration_s': 0.0,
                'error': 'Trace file not found'
            }
        
        # 提取执行时间
        duration_s = self.extract_trace_duration(trace_file)
        if duration_s is None:
            return {
                'iips': 0.0,
                'mie': float('inf'),
                'duration_s': 0.0,
                'error': 'Failed to extract duration'
            }
        
        # 计算 IIPS
        iips = self.calculate_iips(duration_s)
        
        # 计算 MIE
        mie = self.calculate_mie(iips, gpu_type, gpu_util)
        
        result = {
            'iips': round(iips, 6),
            'mie': round(mie, 6) if mie != float('inf') else float('inf'),
            'duration_s': round(duration_s, 6),
            'trace_file': trace_file,
            'f_peak': self.get_gpu_peak_flops(gpu_type),
            'u_gpu': gpu_util / 100.0,
            'n_gpu': 1,
            'iterations': self.fixed_iterations
        }
        
        logger.info(f"Results for {pod_name}: IIPS={iips:.6f}, MIE={mie:.6f}")
        return result
    
    def analyze_all_cases(self, input_csv: str, output_csv: str, base_dir: str = ".") -> None:
        """
        分析所有筛选后的案例
        
        Args:
            input_csv: 输入 CSV 文件 (cases_after_sea.csv)
            output_csv: 输出 CSV 文件
            base_dir: 基础目录
        """
        logger.info(f"Starting MEA analysis for all cases")
        logger.info(f"Input: {input_csv}")
        logger.info(f"Output: {output_csv}")
        logger.info(f"Base directory: {base_dir}")
        
        # 预先扫描所有 trace 文件
        self.scan_trace_files(base_dir)
        
        # 读取输入 CSV
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} cases from {input_csv}")
        
        # 显示案例概览
        logger.info("Cases overview:")
        gpu_counts = df['GPU_type'].value_counts()
        for gpu_type, count in gpu_counts.items():
            logger.info(f"  {gpu_type}: {count} cases")
        
        # 只添加必要的两列：IIPS 和 MIE
        df['IIPS'] = 0.0
        df['MIE'] = float('inf')
        
        # 分析每个案例
        successful_cases = 0
        failed_cases = 0
        
        for idx, row in df.iterrows():
            try:
                result = self.analyze_case(row, base_dir)
                
                # 更新结果 - 只更新 IIPS 和 MIE
                df.at[idx, 'IIPS'] = result['iips']
                df.at[idx, 'MIE'] = result['mie']
                
                if result.get('error'):
                    failed_cases += 1
                    logger.warning(f"Failed to analyze {row['pod_name']}: {result['error']}")
                else:
                    successful_cases += 1
                    
            except Exception as e:
                logger.error(f"Error analyzing case {row['pod_name']}: {e}")
                df.at[idx, 'IIPS'] = 0.0
                df.at[idx, 'MIE'] = float('inf')
                failed_cases += 1
        
        # 保存结果
        df.to_csv(output_csv, index=False)
        logger.info(f"Results saved to {output_csv}")
        
        # 生成统计报告
        self._generate_analysis_report(df, successful_cases, failed_cases)
    
    def _generate_analysis_report(self, df: pd.DataFrame, successful_cases: int, failed_cases: int) -> None:
        """生成分析报告"""
        logger.info("\n" + "="*60)
        logger.info("MEA Analysis Report")
        logger.info("="*60)
        
        logger.info(f"Total cases: {len(df)}")
        logger.info(f"Successful analyses: {successful_cases}")
        logger.info(f"Failed analyses: {failed_cases}")
        logger.info(f"Success rate: {successful_cases/len(df)*100:.1f}%")
        
        # 统计有效结果
        valid_results = df[df['IIPS'] > 0]
        if len(valid_results) > 0:
            logger.info(f"\nIIPS Statistics:")
            logger.info(f"  Mean: {valid_results['IIPS'].mean():.6f}")
            logger.info(f"  Std:  {valid_results['IIPS'].std():.6f}")
            logger.info(f"  Min:  {valid_results['IIPS'].min():.6f}")
            logger.info(f"  Max:  {valid_results['IIPS'].max():.6f}")
            
            finite_mie = valid_results[valid_results['MIE'] != float('inf')]
            if len(finite_mie) > 0:
                logger.info(f"\nMIE Statistics:")
                logger.info(f"  Mean: {finite_mie['MIE'].mean():.6f}")
                logger.info(f"  Std:  {finite_mie['MIE'].std():.6f}")
                logger.info(f"  Min:  {finite_mie['MIE'].min():.6f}")
                logger.info(f"  Max:  {finite_mie['MIE'].max():.6f}")
        
        # 按 GPU 类型统计
        logger.info(f"\nResults by GPU Type:")
        for gpu_type in df['GPU_type'].unique():
            gpu_data = valid_results[valid_results['GPU_type'] == gpu_type]
            if len(gpu_data) > 0:
                avg_iips = gpu_data['IIPS'].mean()
                avg_mie = gpu_data[gpu_data['MIE'] != float('inf')]['MIE'].mean()
                logger.info(f"  {gpu_type}: {len(gpu_data)}/{len(df[df['GPU_type'] == gpu_type])} successful, "
                           f"avg IIPS={avg_iips:.6f}, avg MIE={avg_mie:.6f}")
            else:
                total_gpu_cases = len(df[df['GPU_type'] == gpu_type])
                logger.info(f"  {gpu_type}: 0/{total_gpu_cases} successful")
        
        # 失败案例分析
        if failed_cases > 0:
            logger.info(f"\nFailed Cases:")
            failed_results = df[(df['IIPS'] == 0) | (df['MIE'] == float('inf'))]
            logger.info(f"  Total failed cases: {len(failed_results)}")
            for idx, row in failed_results.iterrows():
                logger.info(f"    {row['pod_name']} ({row['GPU_type']})")
        
        logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(description='MEA Analysis for SGLang Framework - Improved Version')
    parser.add_argument('--input', '-i', default='cases_after_sea.csv',
                        help='Input CSV file (default: cases_after_sea.csv)')
    parser.add_argument('--output', '-o', default='cases_after_sea_with_mea.csv',
                        help='Output CSV file (default: cases_after_sea_with_mea.csv)')
    parser.add_argument('--base-dir', '-d', default='.',
                        help='Base directory containing GPU folders (default: current directory)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # 检查基础目录
    if not os.path.exists(args.base_dir):
        logger.error(f"Base directory not found: {args.base_dir}")
        sys.exit(1)
    
    # 创建分析器
    analyzer = SGLangMEAAnalyzer()
    
    try:
        # 执行分析
        analyzer.analyze_all_cases(args.input, args.output, args.base_dir)
        logger.info(f"✅ MEA analysis completed successfully!")
        logger.info(f"📁 Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"❌ MEA analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()