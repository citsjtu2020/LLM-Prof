#!/usr/bin/env python3
"""
MEA (Model Efficiency Analyzer) - 模型效率分析器
基于论文框架实现 IIPS 和 MIE 计算，用于评估模型推理效率

主要功能：
1. 计算 IIPS (Inference Iterations Per Second) - 每秒推理迭代数
2. 计算 MIE (Model Inference Efficiency) - 模型推理效率
3. 提供模型级别的性能分析和瓶颈识别
"""

import json
import sys
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import argparse
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    return obj

class MEAAnalyzer:
    """
    Model Efficiency Analyzer (MEA) - 模型效率分析器
    
    基于论文 Section 3.3 的 MEA 设计实现：
    1. 在线模型时间线追踪
    2. 基于内核级模式建模的推理迭代识别  
    3. 考虑推理吞吐量和资源使用的模型效率估计
    """
    
    def __init__(self, gpu_config_file: Optional[str] = None):
        """初始化 MEA 分析器"""
        self.gpu_config = self.load_gpu_config(gpu_config_file)
        
        # 确保必要的GPU配置字段存在
        if 'u_GPU' not in self.gpu_config or self.gpu_config['u_GPU'] is None:
            self.gpu_config['u_GPU'] = 0.8  # 默认GPU利用率
            logger.warning("u_GPU not found in config, using default value: 0.8")
        
        logger.info(f"MEA Analyzer initialized with GPU config: {self.gpu_config}")
    
    def load_gpu_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """加载 GPU 配置"""
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"Loaded GPU config from {config_file}")
                return config
            except Exception as e:
                logger.warning(f"Error loading GPU config: {e}")
        
        # 默认配置 - 基于H20 GPU
        default_config = {
            'F_peak': 148.0,  # H20 GPU peak TFLOPs (FP16)
            'N_GPU': 1,
            'u_GPU': 0.58,  
            'memory_bandwidth': 4800.0,  # GB/s
            'gpu_model': 'H20',
            'execute_token_size': None
        }
        
        logger.info("Using default H20 GPU configuration")
        return default_config
    
    def load_stage3_results(self, stage3_results_file: str) -> Dict[str, Any]:
        """加载 Stage 3 统计验证结果"""
        logger.info(f"Loading Stage 3 results from {stage3_results_file}")
        
        with open(stage3_results_file, 'r', encoding='utf-8') as f:
            stage3_data = json.load(f)
        
        if stage3_data.get('stage') != 3:
            raise ValueError(f"Invalid Stage 3 results file: expected stage=3, got {stage3_data.get('stage')}")
        
        logger.info(f"Loaded {len(stage3_data.get('iterations', []))} iterations from Stage 3")
        return stage3_data
    
    def load_external_config(self, config_file: str) -> Dict[str, Any]:
        """
        加载外部配置文档，支持prefill_metrics_with_config.txt格式
        该文件包含模型架构信息、硬件环境信息和服务运行时指标
        
        Args:
            config_file: 外部配置文件路径 (prefill_metrics_with_config.txt)
            
        Returns:
            包含GPU配置信息的字典
        """
        logger.info(f"Loading external GPU config from {config_file}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析配置文件内容
            external_config = self._parse_prefill_metrics_config(content)
            
            # 更新GPU配置 - 从硬件环境信息中提取
            hardware_info = external_config.get('hardware_info', {})
            if 'f_peak' in hardware_info:
                self.gpu_config['F_peak'] = hardware_info['f_peak']
                logger.info(f"Successfully loaded F_peak from hardware_info: {hardware_info['f_peak']}")
            if 'n_gpu' in hardware_info:
                self.gpu_config['N_GPU'] = hardware_info['n_gpu']
                logger.info(f"Successfully loaded N_GPU from hardware_info: {hardware_info['n_gpu']}")
            if 'memory_bandwidth' in hardware_info:
                self.gpu_config['memory_bandwidth'] = hardware_info['memory_bandwidth']
                logger.info(f"Successfully loaded memory_bandwidth from hardware_info: {hardware_info['memory_bandwidth']}")
            if 'gpu_type' in hardware_info:
                self.gpu_config['gpu_model'] = hardware_info['gpu_type']
                logger.info(f"Successfully loaded gpu_model from hardware_info: {hardware_info['gpu_type']}")
            
            # 处理NVIDIA GPU规格 - 从详细规格中提取信息
            for gpu_model, specs in hardware_info.items():
                if isinstance(specs, dict) and 'FP16' in specs:
                    extracted_f_peak = specs['FP16'] / 1e12  # 转换为TFLOPs
                    extracted_bandwidth = specs['mem_bandwidth'] / (1024**3)  # 转换为GB/s
                    extracted_gpu_type = gpu_model  # 保持完整的GPU型号名称
                    
                    # 更新GPU配置信息
                    self.gpu_config['F_peak'] = extracted_f_peak
                    self.gpu_config['memory_bandwidth'] = extracted_bandwidth
                    self.gpu_config['gpu_model'] = extracted_gpu_type
                    logger.info(f"Extracted F_peak from {gpu_model} specs: {extracted_f_peak}")
                    logger.info(f"Extracted memory_bandwidth from {gpu_model} specs: {extracted_bandwidth}")
                    logger.info(f"Extracted gpu_model from {gpu_model} specs: {extracted_gpu_type}")
            
            # 处理 GPU规格 - 从详细规格中提取信息
            for gpu_model, specs in hardware_info.items():
                if isinstance(specs, dict) and 'FP16' in specs:
                    self.gpu_config['F_peak'] = specs['FP16'] / 1e12  # 转换为TFLOPs
                    self.gpu_config['memory_bandwidth'] = specs['mem_bandwidth'] / (1024**3)  # 转换为GB/s
                    self.gpu_config['gpu_model'] = gpu_model  # 保持完整的GPU型号名称，不要截取
            
            # 更新GPU配置 - 从服务运行时指标中提取
            runtime_metrics = external_config.get('runtime_metrics', {})
            if 'u_gpu' in runtime_metrics:
                self.gpu_config['u_GPU'] = runtime_metrics['u_gpu']  # 正确映射u_gpu -> u_GPU
                logger.info(f"Successfully loaded GPU utilization: {runtime_metrics['u_gpu']}")
            if 'n_gpu' in runtime_metrics:
                # 优先使用runtime_metrics中的n_gpu，因为它是实际运行时的GPU数量
                self.gpu_config['N_GPU'] = runtime_metrics['n_gpu']
                logger.info(f"Successfully loaded GPU count: {runtime_metrics['n_gpu']}")
            if 'execute_token_size' in runtime_metrics:
                # 计算平均token size
                token_data = runtime_metrics['execute_token_size']
                if isinstance(token_data, dict) and 'values' in token_data:
                    avg_token_size = sum(token_data['values']) / len(token_data['values'])
                    self.gpu_config['execute_token_size'] = avg_token_size
                elif isinstance(token_data, (int, float)):
                    self.gpu_config['execute_token_size'] = token_data
            
            # 添加模型信息
            model_info = external_config.get('model_info', {})
            if model_info:
                self.gpu_config['model_name'] = model_info.get('model_name', 'Unknown')
                self.gpu_config['app_name'] = model_info.get('app_name', 'Unknown')
                self.gpu_config['inference_engine'] = runtime_metrics.get('inference_engine', 'Unknown')
            
            logger.info(f"Updated GPU config from prefill_metrics_with_config.txt: {self.gpu_config}")
            return external_config
            
        except FileNotFoundError:
            logger.warning(f"External config file {config_file} not found, using default values")
            return {}
        except Exception as e:
            logger.error(f"Error loading external config: {e}")
            return {}
    
    def _parse_prefill_metrics_config(self, content: str) -> Dict[str, Any]:
        """
        解析prefill_metrics_with_config.txt文件内容
        支持混合格式：键值对 + JSON块
        """
        config = {
            'model_info': {},
            'hardware_info': {},
            'runtime_metrics': {}
        }
        
        lines = content.split('\n')
        current_section = None
        in_json_block = False
        json_buffer = ""
        
        for line in lines:
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            
            # 检测章节标题
            if '模型架构信息' in line or 'Model Architecture' in line or '1.' in line:
                current_section = 'model_info'
                continue
            elif '硬件环境信息' in line or 'Hardware Environment' in line or '2.' in line:
                current_section = 'hardware_info'
                continue
            elif '服务运行时指标' in line or 'Runtime Metrics' in line or '3.' in line:
                current_section = 'runtime_metrics'
                continue
            
            # 解析键值对格式 (key = value)
            if '=' in line and not in_json_block:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if current_section == 'model_info':
                    config['model_info'][key] = value
                elif current_section == 'runtime_metrics':
                    # 特殊处理数值类型
                    if key == 'u_gpu':
                        try:
                            config['runtime_metrics'][key] = float(value)
                        except ValueError:
                            config['runtime_metrics'][key] = value
                    elif key == 'n_gpu':
                        try:
                            config['runtime_metrics'][key] = int(value)
                        except ValueError:
                            config['runtime_metrics'][key] = value
                    else:
                        config['runtime_metrics'][key] = value
                continue
            
            # 检测JSON块开始
            if line.startswith('```json'):
                in_json_block = True
                json_buffer = ""
                continue
            elif line.startswith('```') and in_json_block:
                in_json_block = False
                try:
                    # 预处理JSON内容，处理特殊的时间序列数据格式
                    processed_json = self._preprocess_json_content(json_buffer, current_section)
                    json_data = json.loads(processed_json)
                    
                    if current_section == 'hardware_info':
                        # 解析硬件信息
                        if 'gpu_type' in json_data:
                            config['hardware_info']['gpu_type'] = json_data['gpu_type']
                        if 'f_peak' in json_data:
                            config['hardware_info']['f_peak'] = json_data['f_peak']
                        if 'n_gpu' in json_data:
                            config['hardware_info']['n_gpu'] = json_data['n_gpu']
                        if 'memory_bandwidth' in json_data:
                            config['hardware_info']['memory_bandwidth'] = json_data['memory_bandwidth']
                        # 处理NVIDIA GPU规格
                        for gpu_model, specs in json_data.items():
                            if isinstance(specs, dict) and 'FP16' in specs:
                                config['hardware_info']['f_peak'] = specs['FP16'] / 1e12  # 转换为TFLOPs
                                config['hardware_info']['memory_bandwidth'] = specs['mem_bandwidth'] / (1024**3)  # 转换为GB/s
                                config['hardware_info']['gpu_type'] = gpu_model.split('_')[1] if '_' in gpu_model else gpu_model
                    elif current_section == 'runtime_metrics':
                        # 解析运行时指标
                        if 'u_gpu' in json_data:
                            config['runtime_metrics']['u_gpu'] = json_data['u_gpu']
                            logger.info(f"Successfully parsed u_gpu from JSON: {json_data['u_gpu']}")
                        if 'n_gpu' in json_data:
                            config['runtime_metrics']['n_gpu'] = json_data['n_gpu']
                            logger.info(f"Successfully parsed n_gpu from JSON: {json_data['n_gpu']}")
                        if 'qps' in json_data:
                            config['runtime_metrics']['qps'] = json_data['qps']
                        if 'execute_token_size' in json_data:
                            config['runtime_metrics']['execute_token_size'] = json_data['execute_token_size']
                        if 'inference_engine' in json_data:
                            config['runtime_metrics']['inference_engine'] = json_data['inference_engine']
                        if 'task_function' in json_data:
                            config['runtime_metrics']['task_function'] = json_data['task_function']
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON block in section {current_section}: {e}")
                    # 如果JSON解析失败，尝试手动解析关键字段
                    if current_section == 'runtime_metrics':
                        self._manual_extract_runtime_metrics(json_buffer, config)
                    elif current_section == 'hardware_info':
                        self._manual_extract_hardware_info(json_buffer, config)
                json_buffer = ""
            elif in_json_block:
                json_buffer += line + "\n"
        
        return config
    
    def _preprocess_json_content(self, json_content: str, section: str) -> str:
        """预处理JSON内容，处理特殊格式的时间序列数据"""
        if section != 'runtime_metrics':
            return json_content
        
        # 处理时间序列数据格式，将其转换为标准JSON
        lines = json_content.split('\n')
        processed_lines = []
        in_time_series = False
        time_series_key = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检测时间序列数据开始
            if ('"qps":' in line or '"execute_token_size":' in line) and line.endswith(':'):
                time_series_key = line.split(':')[0].strip().strip('"')
                in_time_series = True
                processed_lines.append(f'"{time_series_key}": "time_series_data",')
                continue
            elif in_time_series:
                # 跳过时间序列数据行，直到遇到下一个字段或结束
                if line.startswith('"') and ':' in line and not line.startswith('2025-'):
                    in_time_series = False
                    processed_lines.append(line)
                elif line == '}' or line == '},':
                    in_time_series = False
                    processed_lines.append(line)
                # 跳过时间序列数据行
                continue
            else:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _manual_extract_runtime_metrics(self, json_buffer: str, config: Dict[str, Any]):
        """手动提取运行时指标，当JSON解析失败时使用"""
        import re
        
        # 提取u_gpu
        u_gpu_match = re.search(r'"u_gpu":\s*([0-9.]+)', json_buffer)
        if u_gpu_match:
            try:
                config['runtime_metrics']['u_gpu'] = float(u_gpu_match.group(1))
                logger.info(f"Manually extracted u_gpu: {config['runtime_metrics']['u_gpu']}")
            except ValueError:
                pass
        
        # 提取n_gpu
        n_gpu_match = re.search(r'"n_gpu":\s*([0-9]+)', json_buffer)
        if n_gpu_match:
            try:
                config['runtime_metrics']['n_gpu'] = int(n_gpu_match.group(1))
                logger.info(f"Manually extracted n_gpu: {config['runtime_metrics']['n_gpu']}")
            except ValueError:
                pass
        
        # 提取inference_engine
        engine_match = re.search(r'"inference_engine":\s*"([^"]+)"', json_buffer)
        if engine_match:
            config['runtime_metrics']['inference_engine'] = engine_match.group(1)
            logger.info(f"Manually extracted inference_engine: {config['runtime_metrics']['inference_engine']}")
        
        # 提取task_function
        task_match = re.search(r'"task_function":\s*"([^"]+)"', json_buffer)
        if task_match:
            config['runtime_metrics']['task_function'] = task_match.group(1)
            logger.info(f"Manually extracted task_function: {config['runtime_metrics']['task_function']}")
    
    def _manual_extract_hardware_info(self, json_buffer: str, config: Dict[str, Any]):
        """手动提取硬件信息，当JSON解析失败时使用"""
        import re
        
        # 提取gpu_type
        gpu_type_match = re.search(r'"gpu_type":\s*"([^"]+)"', json_buffer)
        if gpu_type_match:
            config['hardware_info']['gpu_type'] = gpu_type_match.group(1)
            logger.info(f"Manually extracted gpu_type: {config['hardware_info']['gpu_type']}")
        
        # 提取f_peak
        f_peak_match = re.search(r'"f_peak":\s*([0-9.]+)', json_buffer)
        if f_peak_match:
            try:
                config['hardware_info']['f_peak'] = float(f_peak_match.group(1))
                logger.info(f"Manually extracted f_peak: {config['hardware_info']['f_peak']}")
            except ValueError:
                pass
        
        # 提取n_gpu
        n_gpu_match = re.search(r'"n_gpu":\s*([0-9]+)', json_buffer)
        if n_gpu_match:
            try:
                config['hardware_info']['n_gpu'] = int(n_gpu_match.group(1))
                logger.info(f"Manually extracted n_gpu from hardware_info: {config['hardware_info']['n_gpu']}")
            except ValueError:
                pass
        
        # 提取memory_bandwidth
        mem_bw_match = re.search(r'"memory_bandwidth":\s*([0-9.]+)', json_buffer)
        if mem_bw_match:
            try:
                config['hardware_info']['memory_bandwidth'] = float(mem_bw_match.group(1))
                logger.info(f"Manually extracted memory_bandwidth: {config['hardware_info']['memory_bandwidth']}")
            except ValueError:
                pass
    
    def extract_valid_iterations(self, stage3_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取最终验证通过的有效迭代"""
        iterations = stage3_data.get('iterations', [])
        valid_iterations = [
            iteration for iteration in iterations 
            if iteration.get('final_validated', False)
        ]
        
        logger.info(f"Extracted {len(valid_iterations)} valid iterations out of {len(iterations)} total")
        return valid_iterations
    
    def calculate_iips(self, valid_iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算 IIPS (Inference Iterations Per Second) - 基于端到端时间
        
        从第一个iteration的开始时间到最后一个iteration的结束时间
        
        Args:
            valid_iterations: 验证通过的迭代列表
            
        Returns:
            包含 IIPS 计算结果的字典
        """
        if not valid_iterations:
            logger.warning("No valid iterations found for IIPS calculation")
            return {
                'iips': 0.0,
                'total_iterations': 0,
                'total_duration_us': 0.0,
                'total_duration_s': 0.0,
                'average_iteration_duration_us': 0.0,
                'error': 'No valid iterations'
            }
        
        # 获取第一个iteration的开始时间和最后一个iteration的结束时间
        first_start_ts = None
        last_end_ts = None
        iteration_durations = []
        
        for iteration in valid_iterations:
            start_ts = iteration.get('start_ts')
            end_ts = iteration.get('end_ts')
            duration_us = iteration.get('duration_us', 0.0)
            
            if start_ts is not None and end_ts is not None:
                if first_start_ts is None or start_ts < first_start_ts:
                    first_start_ts = start_ts
                if last_end_ts is None or end_ts > last_end_ts:
                    last_end_ts = end_ts
                    
            if duration_us > 0:
                iteration_durations.append(duration_us)
        
        if first_start_ts is None or last_end_ts is None:
            logger.warning("Could not find valid start/end timestamps")
            return {
                'iips': 0.0,
                'total_iterations': len(valid_iterations),
                'total_duration_us': 0.0,
                'total_duration_s': 0.0,
                'average_iteration_duration_us': 0.0,
                'error': 'Invalid timestamp data'
            }
        
        # 计算端到端总时间
        total_duration_us = last_end_ts - first_start_ts
        if total_duration_us <= 0:
            logger.warning("Total duration is zero or negative")
            return {
                'iips': 0.0,
                'total_iterations': len(valid_iterations),
                'total_duration_us': total_duration_us,
                'total_duration_s': 0.0,
                'average_iteration_duration_us': 0.0,
                'error': 'Invalid duration data'
            }
        
        # 转换为秒
        total_duration_s = total_duration_us / 1_000_000.0
        
        # 计算 IIPS
        num_iterations = len(valid_iterations)
        iips = num_iterations / total_duration_s
        
        # 计算统计信息
        avg_duration_us = np.mean(iteration_durations) if iteration_durations else 0.0
        std_duration_us = np.std(iteration_durations) if len(iteration_durations) > 1 else 0.0
        min_duration_us = np.min(iteration_durations) if iteration_durations else 0.0
        max_duration_us = np.max(iteration_durations) if iteration_durations else 0.0
        
        result = {
            'iips': float(iips),
            'total_iterations': num_iterations,
            'total_duration_us': float(total_duration_us),
            'total_duration_s': float(total_duration_s),
            'average_iteration_duration_us': float(avg_duration_us),
            'std_iteration_duration_us': float(std_duration_us),
            'min_iteration_duration_us': float(min_duration_us),
            'max_iteration_duration_us': float(max_duration_us),
            'first_start_ts': float(first_start_ts),
            'last_end_ts': float(last_end_ts),
            'throughput_tokens_per_second': float(iips)  # 每秒生成的 token 数
        }
        
        logger.info(f"IIPS calculated: {iips:.2f} iterations/second")
        logger.info(f"End-to-end duration: {total_duration_s:.3f} seconds ({total_duration_us:.1f} μs)")
        logger.info(f"Average iteration duration: {avg_duration_us:.2f} μs")
        
        return result
    
    def validate_gpu_config(self) -> bool:
        """
        验证GPU配置是否完整
        
        Returns:
            配置是否有效
        """
        required_fields = ['u_GPU', 'F_peak', 'N_GPU']
        missing_fields = []
        
        for field in required_fields:
            if self.gpu_config.get(field) is None:
                missing_fields.append(field)
        
        if missing_fields:
            logger.error(f"Missing required GPU config fields: {missing_fields}")
            logger.error("Please provide external config file with gpu_utilization, gpu_count, and peak_flops")
            return False
        
        return True
    
    def calculate_mie(self, iips_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算 MIE (Model Inference Efficiency)
        
        根据论文公式：MIE = (F_peak × u_GPU × N_GPU) / IIPS
        
        Args:
            iips_result: IIPS 计算结果
            
        Returns:
            包含 MIE 计算结果的字典
        """
        # 验证GPU配置
        if not self.validate_gpu_config():
            return {
                'mie': float('inf'),
                'error': 'Invalid or incomplete GPU configuration'
            }
        
        iips = iips_result.get('iips', 0.0)
        
        if iips <= 0:
            logger.warning("IIPS is zero or negative, cannot calculate MIE")
            return {
                'mie': float('inf'),
                'f_peak': self.gpu_config['F_peak'],
                'u_gpu': self.gpu_config['u_GPU'],
                'n_gpu': self.gpu_config['N_GPU'],
                'iips': iips,
                'error': 'Invalid IIPS value'
            }
        
        # 计算 MIE - 使用外部配置提供的真实GPU利用率
        f_peak = self.gpu_config['F_peak']  # TFLOPs
        u_gpu = self.gpu_config['u_GPU']    # 从外部配置获取的真实利用率
        n_gpu = self.gpu_config['N_GPU']    # 从外部配置获取的GPU数量
        
        mie = (f_peak * u_gpu * n_gpu) / iips
        
        # 计算相关指标
        effective_compute_power = f_peak * u_gpu * n_gpu  # 有效计算能力 (TFLOPs)
        compute_per_token = mie  # 每个 token 消耗的计算资源 (TFLOPs)
        
        result = {
            'mie': float(mie),
            'f_peak': float(f_peak),
            'u_gpu': float(u_gpu),
            'n_gpu': int(n_gpu),
            'iips': float(iips),
            'effective_compute_power_tflops': float(effective_compute_power),
            'compute_per_token_tflops': float(compute_per_token),
            'gpu_model': self.gpu_config.get('gpu_model', 'Unknown'),
            'execute_token_size': self.gpu_config.get('execute_token_size'),
            'efficiency_interpretation': self._interpret_mie(mie),
            'config_source': 'external_config'  # 标明配置来源
        }
        
        logger.info(f"MIE calculated: {mie:.6f} TFLOPs per iteration")
        logger.info(f"Using external GPU utilization: {u_gpu:.3f}")
        logger.info(f"Effective compute power: {effective_compute_power:.2f} TFLOPs")
        
        return result
    
    def _interpret_mie(self, mie: float) -> str:
        """解释 MIE 值的含义"""
        if mie < 0.001:
            return "Excellent efficiency - very low compute cost per token"
        elif mie < 0.01:
            return "Good efficiency - reasonable compute cost per token"
        elif mie < 0.1:
            return "Moderate efficiency - moderate compute cost per token"
        elif mie < 1.0:
            return "Poor efficiency - high compute cost per token"
        else:
            return "Very poor efficiency - extremely high compute cost per token"
    
    def analyze_iteration_patterns(self, valid_iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析迭代执行模式"""
        if not valid_iterations:
            return {'error': 'No valid iterations to analyze'}
        
        # 分析操作符分布变化
        operator_distributions = []
        for iteration in valid_iterations:
            op_dist = iteration.get('operator_distribution', {})
            if op_dist:
                operator_distributions.append(op_dist)
        
        if not operator_distributions:
            return {'error': 'No operator distribution data found'}
        
        # 计算操作符分布的稳定性
        all_operators = set()
        for dist in operator_distributions:
            all_operators.update(dist.keys())
        
        operator_stability = {}
        for op in all_operators:
            values = [dist.get(op, 0.0) for dist in operator_distributions]
            operator_stability[op] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'cv': float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0.0
            }
        
        # 分析执行时间变化
        durations = [it.get('duration_us', 0) for it in valid_iterations if it.get('duration_us', 0) > 0]
        duration_stats = {
            'mean_us': float(np.mean(durations)) if durations else 0.0,
            'std_us': float(np.std(durations)) if durations else 0.0,
            'cv': float(np.std(durations) / np.mean(durations)) if durations and np.mean(durations) > 0 else 0.0,
            'min_us': float(np.min(durations)) if durations else 0.0,
            'max_us': float(np.max(durations)) if durations else 0.0
        }
        
        return {
            'operator_stability': operator_stability,
            'duration_statistics': duration_stats,
            'execution_consistency': 'High' if duration_stats['cv'] < 0.1 else 'Medium' if duration_stats['cv'] < 0.3 else 'Low'
        }
    
    def analyze_model_efficiency(self, stage3_results_file: str, 
                                external_config_file: Optional[str] = None) -> Dict[str, Any]:
        """执行完整的 MEA 分析"""
        logger.info("Starting MEA (Model Efficiency Analyzer) analysis")
        
        try:
            # 加载外部配置
            external_config = {}
            if external_config_file:
                external_config = self.load_external_config(external_config_file)
            
            # 加载 Stage 3 结果
            stage3_data = self.load_stage3_results(stage3_results_file)
            
            # 提取有效迭代
            valid_iterations = self.extract_valid_iterations(stage3_data)
            
            # 计算IIPS（基于端到端时间）
            iips_result = self.calculate_iips(valid_iterations)
            
            # 计算 MIE
            mie_result = self.calculate_mie(iips_result)
            
            # 分析迭代模式
            pattern_analysis = self.analyze_iteration_patterns(valid_iterations)
            
            # 生成性能洞察
            insights = self._generate_performance_insights(iips_result, mie_result)
            
            # 构建完整结果
            result = {
                'mea_analysis': {
                    'iips_analysis': iips_result,
                    'mie_analysis': mie_result,
                    'pattern_analysis': pattern_analysis,
                    'performance_insights': insights
                },
                'stage3_summary': {
                    'total_iterations': stage3_data.get('metadata', {}).get('total_iterations', 0),
                    'final_valid_iterations': stage3_data.get('metadata', {}).get('final_valid_iterations', 0),
                    'validation_rate': stage3_data.get('metadata', {}).get('validation_rate', 0.0),
                    'global_operator_distribution': stage3_data.get('metadata', {}).get('global_operator_distribution', {})
                },
                'gpu_configuration': self.gpu_config,
                'external_configuration': external_config,
                'analysis_metadata': {
                    'framework_version': 'MEA v1.0',
                    'analysis_timestamp': self._get_timestamp(),
                    'input_file': stage3_results_file,
                    'external_config_file': external_config_file,
                    'methodology': 'Based on LLM-Prof MEA framework with end-to-end IIPS calculation'
                }
            }
            
            logger.info("MEA analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"MEA analysis failed: {e}")
            return {
                'error': str(e),
                'stage': 'MEA Analysis',
                'timestamp': self._get_timestamp()
            }
    
    def _generate_performance_insights(self, iips_result: Dict[str, Any], mie_result: Dict[str, Any]) -> List[str]:
        """生成性能洞察"""
        insights = []
        
        # IIPS分析
        iips = iips_result.get('iips', 0)
        avg_duration = iips_result.get('average_iteration_duration_us', 0) / 1000  # 转换为ms
        
        # MIE分析
        mie = mie_result.get('mie', float('inf'))
        
        # GPU利用率分析 - 添加安全检查
        u_gpu = self.gpu_config.get('u_GPU', 0)
        if u_gpu is None:
            u_gpu = 0
        
        # IIPS分析
        if iips > 0:
            insights.append(f"📊 IIPS: {iips:.1f} iterations/second")
        
        # MIE分析
        if mie < float('inf'):
            if mie > 0.1:
                insights.append(f"⚠️  High MIE ({mie:.4f}) indicates poor model efficiency - each token requires significant compute resources")
            elif mie > 0.01:
                insights.append(f"⚠️  Moderate MIE ({mie:.4f}) suggests room for optimization")
            else:
                insights.append(f"✅ Good MIE ({mie:.4f}) indicates efficient compute utilization")
        
        # 执行时间分析
        if avg_duration > 0:
            if avg_duration < 5:
                insights.append(f"✅ Fast iteration execution ({avg_duration:.1f}ms average)")
            elif avg_duration < 20:
                insights.append(f"⚠️  Moderate iteration latency ({avg_duration:.1f}ms average)")
            else:
                insights.append(f"⚠️  High iteration latency ({avg_duration:.1f}ms average)")
        
        # GPU利用率分析 - 添加数值检查
        if isinstance(u_gpu, (int, float)) and u_gpu > 0:
            if u_gpu < 0.3:
                insights.append(f"⚠️  Low GPU utilization ({u_gpu:.1%}) suggests underutilized compute resources")
            elif u_gpu < 0.7:
                insights.append(f"⚠️  Moderate GPU utilization ({u_gpu:.1%}) - potential for improvement")
            else:
                insights.append(f"✅ High GPU utilization ({u_gpu:.1%}) indicates good resource usage")
        
        # 吞吐量分析
        if iips > 0:
            if iips > 100:
                insights.append(f"✅ High throughput ({iips:.1f} iterations/sec) indicates good performance")
            elif iips > 50:
                insights.append(f"⚠️  Moderate throughput ({iips:.1f} iterations/sec)")
            else:
                insights.append(f"⚠️  Low throughput ({iips:.1f} iterations/sec) may indicate bottlenecks")
        
        # 一致性分析
        std_duration = iips_result.get('std_iteration_duration_us', 0) / 1000
        if std_duration > 0 and avg_duration > 0:
            cv = std_duration / avg_duration
            if cv > 0.5:
                insights.append("⚠️  High execution variability may indicate dynamic batching or workload variations")
            elif cv > 0.2:
                insights.append("⚠️  Moderate execution variability detected")
            else:
                insights.append("✅ Consistent execution timing")
        
        return insights
    
    def calculate_wall_clock_iips(self, valid_iterations: List[Dict[str, Any]], stage1_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算基于端到端时间的 IIPS (Wall-clock IIPS)
        
        从第一个有效iteration的第一个HtoD到最后一个有效iteration的最后一个DtoH的总时间
        
        Args:
            valid_iterations: 验证通过的迭代列表
            stage1_data: Stage1结果数据，包含完整的iteration信息
            
        Returns:
            包含端到端 IIPS 计算结果的字典
        """
        if not valid_iterations:
            logger.warning("No valid iterations found for wall-clock IIPS calculation")
            return {
                'wall_clock_iips': 0.0,
                'wall_clock_duration_us': 0.0,
                'wall_clock_duration_s': 0.0,
                'first_htod_ts': None,
                'last_dtoh_ts': None,
                'error': 'No valid iterations'
            }
        
        # 获取所有stage1 iterations
        all_iterations = stage1_data.get('iterations', [])
        if not all_iterations:
            logger.warning("No iterations found in stage1 data")
            return {
                'wall_clock_iips': 0.0,
                'wall_clock_duration_us': 0.0,
                'wall_clock_duration_s': 0.0,
                'error': 'No stage1 iterations'
            }
        
        # 获取有效iteration的ID列表
        valid_iteration_ids = set()
        for iteration in valid_iterations:
            # 从stage3结果中获取iteration_id，如果没有则使用索引
            iteration_id = iteration.get('iteration_id')
            if iteration_id is None:
                # 如果没有explicit ID，尝试从其他字段推断
                for key in ['id', 'index', 'segment_id']:
                    if key in iteration:
                        iteration_id = iteration[key]
                        break
            if iteration_id is not None:
                valid_iteration_ids.add(iteration_id)
        
        # 如果无法获取ID，使用前N个iterations（N=有效iteration数量）
        if not valid_iteration_ids:
            logger.info("No explicit iteration IDs found, using first N iterations")
            valid_iteration_ids = set(range(1, len(valid_iterations) + 1))
        
        # 过滤出有效的iterations
        valid_stage1_iterations = []
        for iteration in all_iterations:
            iteration_id = iteration.get('iteration_id')
            if iteration_id in valid_iteration_ids:
                valid_stage1_iterations.append(iteration)
        
        if not valid_stage1_iterations:
            logger.warning("No matching iterations found between stage1 and stage3")
            return {
                'wall_clock_iips': 0.0,
                'wall_clock_duration_us': 0.0,
                'wall_clock_duration_s': 0.0,
                'error': 'No matching iterations'
            }
        
        # 按iteration_id排序
        valid_stage1_iterations.sort(key=lambda x: x.get('iteration_id', 0))
        
        # 找到第一个iteration的第一个HtoD事件
        first_htod_ts = None
        first_iteration = valid_stage1_iterations[0]
        for event in first_iteration.get('events', []):
            if event.get('name') == 'Memcpy HtoD (PINNED -> DEVICE)':
                first_htod_ts = event.get('ts')
                break
        
        # 找到最后一个iteration的最后一个DtoH事件
        last_dtoh_ts = None
        last_iteration = valid_stage1_iterations[-1]
        events = last_iteration.get('events', [])
        # 从后往前查找DtoH事件
        for event in reversed(events):
            if event.get('name') == 'Memcpy DtoH (DEVICE -> PINNED)':
                # DtoH事件的结束时间 = ts + dur
                event_ts = event.get('ts', 0)
                event_dur = event.get('dur', 0)
                last_dtoh_ts = event_ts + event_dur
                break
        
        if first_htod_ts is None or last_dtoh_ts is None:
            logger.warning(f"Could not find HtoD/DtoH events: first_htod_ts={first_htod_ts}, last_dtoh_ts={last_dtoh_ts}")
            return {
                'wall_clock_iips': 0.0,
                'wall_clock_duration_us': 0.0,
                'wall_clock_duration_s': 0.0,
                'first_htod_ts': first_htod_ts,
                'last_dtoh_ts': last_dtoh_ts,
                'error': 'Missing HtoD/DtoH events'
            }
        
        # 计算端到端时间
        wall_clock_duration_us = last_dtoh_ts - first_htod_ts
        wall_clock_duration_s = wall_clock_duration_us / 1_000_000.0
        
        # 计算wall-clock IIPS
        num_iterations = len(valid_iterations)
        wall_clock_iips = num_iterations / wall_clock_duration_s if wall_clock_duration_s > 0 else 0.0
        
        result = {
            'wall_clock_iips': float(wall_clock_iips),
            'wall_clock_duration_us': float(wall_clock_duration_us),
            'wall_clock_duration_s': float(wall_clock_duration_s),
            'first_htod_ts': float(first_htod_ts),
            'last_dtoh_ts': float(last_dtoh_ts),
            'total_iterations': num_iterations,
            'first_iteration_id': valid_stage1_iterations[0].get('iteration_id'),
            'last_iteration_id': valid_stage1_iterations[-1].get('iteration_id')
        }
        
        logger.info(f"Wall-clock IIPS calculated: {wall_clock_iips:.2f} iterations/second")
        logger.info(f"Wall-clock duration: {wall_clock_duration_s:.3f} seconds ({wall_clock_duration_us:.1f} μs)")
        logger.info(f"First HtoD at: {first_htod_ts:.3f} μs, Last DtoH at: {last_dtoh_ts:.3f} μs")
        
        return result
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """保存 MEA 分析结果"""
        # 转换 numpy 类型
        results_converted = convert_numpy_types(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)
        
        logger.info(f"MEA analysis results saved to {output_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """打印 MEA 分析摘要"""
        if 'error' in results:
            print(f"\n❌ MEA Analysis Error: {results['error']}")
            return
        
        mea = results.get('mea_analysis', {})
        iips_analysis = mea.get('iips_analysis', {})
        mie = mea.get('mie_analysis', {})
        insights = mea.get('performance_insights', [])
        
        print("\n" + "="*80)
        print("🚀 MEA (Model Efficiency Analyzer) Analysis Results")
        print("="*80)
        
        # IIPS 结果
        print(f"\n📊 IIPS (Inference Iterations Per Second) Analysis:")
        print(f"   Throughput: {iips_analysis.get('iips', 0):.2f} iterations/second")
        print(f"   Total valid iterations: {iips_analysis.get('total_iterations', 0)}")
        print(f"   End-to-end execution time: {iips_analysis.get('total_duration_s', 0):.3f} seconds")
        print(f"   Average iteration duration: {iips_analysis.get('average_iteration_duration_us', 0)/1000:.2f} ms")
        
        # 时间戳信息
        if iips_analysis.get('first_start_ts') and iips_analysis.get('last_end_ts'):
            print(f"   First iteration start: {iips_analysis.get('first_start_ts', 0):.3f} μs")
            print(f"   Last iteration end: {iips_analysis.get('last_end_ts', 0):.3f} μs")
        
        # MIE 结果
        print(f"\n⚡ MIE (Model Inference Efficiency) Analysis:")
        print(f"   MIE: {mie.get('mie', 0):.6f} TFLOPs per iteration")
        print(f"   GPU utilization: {mie.get('u_gpu', 0):.1%}")
        print(f"   Effective compute power: {mie.get('effective_compute_power_tflops', 0):.2f} TFLOPs")
        print(f"   Compute per token: {mie.get('compute_per_token_tflops', 0):.6f} TFLOPs")
        print(f"   Efficiency level: {mie.get('efficiency_interpretation', 'Unknown')}")
        
        # GPU 配置
        gpu_config = results.get('gpu_configuration', {})
        print(f"\n🖥️  GPU Configuration:")
        print(f"   Model: {gpu_config.get('gpu_model', 'Unknown')}")
        print(f"   Count: {gpu_config.get('N_GPU', 'Unknown')}")
        print(f"   Peak Performance: {gpu_config.get('F_peak', 'Unknown')} TFLOPs")
        print(f"   Utilization: {gpu_config.get('u_GPU', 'Unknown'):.1%}")
        if gpu_config.get('execute_token_size'):
            print(f"   Execute Token Size: {gpu_config.get('execute_token_size')}")
        
        # 性能洞察
        print(f"\n💡 Performance Insights:")
        for insight in insights:
            print(f"   {insight}")
        
        print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='MEA (Model Efficiency Analyzer)')
    parser.add_argument('stage3_results', type=str, help='Path to Stage 3 results JSON file')
    parser.add_argument('--external-config', type=str, help='Path to external configuration file')
    parser.add_argument('--output', type=str, help='Output file path (optional)')
    args = parser.parse_args()
    
    analyzer = MEAAnalyzer()
    results = analyzer.analyze_model_efficiency(args.stage3_results, args.external_config)
    analyzer.print_summary(results)
    
    # 自动生成输出文件路径
    if args.output:
        output_file = args.output
    else:
        # 基于输入文件路径生成输出路径
        input_dir = os.path.dirname(args.stage3_results)
        output_file = os.path.join(input_dir, 'stage4_mea_analysis_results.json')
    
    analyzer.save_results(results, output_file)

if __name__ == "__main__":
    main()