#!/usr/bin/env python3
"""
基于 DtoH 事件的 iteration 检测脚本 - vLLM版本
逻辑：
- 第1个iteration：从第一个kernel到第一个memcpy DtoH
- 第2-N个iteration：从上一个DtoH之后的第一个kernel到下一个DtoH
- vLLM固定模式：11个iterations（2个prefill + 9个decode）
"""

import json
import gzip
import sys
import logging
from typing import List, Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IterationCandidate:
    """表示一个候选的推理 iteration 段"""
    def __init__(self, start_ts: float, end_ts: float, events: List[Dict]):
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.duration = end_ts - start_ts
        self.events = events
        self.iteration_id = None
        self.phase = None  # 'prefill' or 'decode'
    
    def __repr__(self):
        return f"Iteration(id={self.iteration_id}, phase={self.phase}, start={self.start_ts:.1f}, duration={self.duration:.1f}μs, events={len(self.events)})"

class DtoHBasedIterationDetector:
    """基于 DtoH 事件的 Iteration 检测器 - vLLM版本"""
    
    def __init__(self):
        pass
    
    def load_trace_file(self, trace_path: str) -> List[Dict]:
        """加载 trace 文件（支持 .gz 压缩）"""
        logger.info(f"加载 trace 文件: {trace_path}")
        
        try:
            if trace_path.endswith('.gz'):
                with gzip.open(trace_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(trace_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Chrome Trace Format 通常有 'traceEvents' 字段
            if 'traceEvents' in data:
                events = data['traceEvents']
            elif isinstance(data, list):
                events = data
            else:
                logger.error("无法识别的 trace 格式")
                return []
            
            logger.info(f"成功加载 {len(events)} 个 trace events")
            return events
            
        except Exception as e:
            logger.error(f"加载 trace 文件失败: {e}")
            raise
    
    def extract_dtoh_events(self, trace_events: List[Dict[str, Any]]) -> List[float]:
        """提取所有 DtoH 事件的时间戳"""
        dtoh_events = []
        
        for event in trace_events:
            if event.get("cat") == "gpu_memcpy":
                name = event.get("name", "")
                ts = event.get("ts", 0)
                
                # DtoH transfers (logits, output tokens)
                if "DtoH" in name or "Device to Host" in name:
                    dtoh_events.append(ts)
        
        dtoh_events.sort()
        logger.info(f"找到 {len(dtoh_events)} 个 DtoH 事件")
        return dtoh_events
    
    def find_first_kernel_after(self, trace_events: List[Dict[str, Any]], after_ts: float) -> float:
        """找到指定时间戳之后的第一个kernel的时间戳"""
        for event in trace_events:
            if (event.get('ph') == 'X' and  # Complete events
                event.get('cat', '').lower() in ['kernel', 'gpu', 'cuda'] and
                'ts' in event):
                event_ts = event.get('ts', 0)
                if event_ts > after_ts:
                    return event_ts
        return None
    
    def find_first_kernel(self, trace_events: List[Dict[str, Any]]) -> float:
        """找到第一个kernel的时间戳"""
        for event in trace_events:
            if (event.get('ph') == 'X' and  # Complete events
                event.get('cat', '').lower() in ['kernel', 'gpu', 'cuda'] and
                'ts' in event):
                return event.get('ts', 0)
        return None
    
    def detect_iterations_vllm(self, trace_events: List[Dict[str, Any]], 
                               dtoh_events: List[float]) -> List[IterationCandidate]:
        """
        基于 DtoH 事件检测 iterations（vLLM模式）：
        - 第1个iteration：从第一个kernel到第一个DtoH
        - 第2-N个iteration：从上一个DtoH之后的第一个kernel到下一个DtoH
        - 最小持续时间阈值：1000μs
        """
        candidates = []
        
        if len(dtoh_events) < 1:
            logger.warning("DtoH 事件不足，无法检测 iterations")
            return candidates
        
        logger.info(f"基于 {len(dtoh_events)} 个 DtoH 事件检测 vLLM iterations")
        
        # 最小 iteration 持续时间阈值 (1000 微秒)
        MIN_ITERATION_DURATION = 1000.0
        
        # 第1个iteration：从第一个kernel到第一个DtoH
        first_kernel_ts = self.find_first_kernel(trace_events)
        if first_kernel_ts is None:
            logger.error("找不到第一个kernel")
            return candidates
        
        first_dtoh_ts = dtoh_events[0]
        duration = first_dtoh_ts - first_kernel_ts
        
        if duration >= MIN_ITERATION_DURATION:
            # 提取第1个iteration的events
            segment_events = [
                e for e in trace_events
                if first_kernel_ts <= e.get('ts', 0) <= first_dtoh_ts
            ]
            
            if len(segment_events) > 0:
                candidate = IterationCandidate(first_kernel_ts, first_dtoh_ts, segment_events)
                candidate.iteration_id = 1
                candidates.append(candidate)
                logger.debug(f"创建 iteration 1: {first_kernel_ts:.1f} - {first_dtoh_ts:.1f} μs (持续时间: {duration:.1f} μs)")
        
        # 第2-N个iteration：从上一个DtoH之后的第一个kernel到下一个DtoH
        for i in range(len(dtoh_events) - 1):
            prev_dtoh_ts = dtoh_events[i]
            next_dtoh_ts = dtoh_events[i + 1]
            
            # 找到prev_dtoh之后的第一个kernel
            start_kernel_ts = self.find_first_kernel_after(trace_events, prev_dtoh_ts)
            if start_kernel_ts is None:
                logger.debug(f"在 DtoH {i+1} 之后找不到kernel，跳过")
                continue
            
            # 检查持续时间
            duration = next_dtoh_ts - start_kernel_ts
            if duration < MIN_ITERATION_DURATION:
                logger.debug(f"跳过 iteration {i+2}: 持续时间 {duration:.1f}μs < {MIN_ITERATION_DURATION}μs 阈值")
                continue
            
            # 提取这个iteration的events
            segment_events = [
                e for e in trace_events
                if start_kernel_ts <= e.get('ts', 0) <= next_dtoh_ts
            ]
            
            if len(segment_events) > 0:
                candidate = IterationCandidate(start_kernel_ts, next_dtoh_ts, segment_events)
                candidate.iteration_id = len(candidates) + 1
                candidates.append(candidate)
                logger.debug(f"创建 iteration {candidate.iteration_id}: {start_kernel_ts:.1f} - {next_dtoh_ts:.1f} μs (持续时间: {duration:.1f} μs)")
        
        logger.info(f"检测到 {len(candidates)} 个 iterations")
        return candidates
    
    def assign_phases_vllm(self, candidates: List[IterationCandidate]) -> List[IterationCandidate]:
        """
        为 vLLM 动态分配阶段：
        - 如果有11个iterations：前2个为Prefill，后9个为Decode
        - 如果有10个iterations：第1个为Prefill，后9个为Decode
        """
        total_iterations = len(candidates)
        
        if total_iterations == 11:
            # 2 prefill + 9 decode
            prefill_count = 2
            logger.info("检测到11个iterations，分配模式：2 prefill + 9 decode")
        elif total_iterations == 10:
            # 1 prefill + 9 decode
            prefill_count = 1
            logger.info("检测到10个iterations，分配模式：1 prefill + 9 decode")
        else:
            # 默认：如果iterations数量异常，前20%为prefill
            prefill_count = max(1, int(total_iterations * 0.2))
            logger.warning(f"检测到{total_iterations}个iterations（非标准数量），使用默认分配：前{prefill_count}个为prefill")
        
        for candidate in candidates:
            if candidate.iteration_id <= prefill_count:
                candidate.phase = 'prefill'
            else:
                candidate.phase = 'decode'
        
        return candidates
    
    def analyze_kernels_by_phase(self, candidates: List[IterationCandidate]) -> Dict:
        """分析各阶段的 kernel 分布"""
        
        prefill_kernels = []
        decode_kernels = []
        total_kernels = 0
        
        for candidate in candidates:
            gpu_events = []
            for event in candidate.events:
                if (event.get('ph') == 'X' and  # Complete events
                    event.get('cat', '').lower() in ['kernel', 'gpu', 'cuda'] and
                    'name' in event and 'ts' in event and 'dur' in event):
                    gpu_events.append(event)
            
            total_kernels += len(gpu_events)
            
            if candidate.phase == 'prefill':
                prefill_kernels.extend(gpu_events)
            elif candidate.phase == 'decode':
                decode_kernels.extend(gpu_events)
        
        return {
            'total_iterations': len(candidates),
            'prefill_iterations': len([c for c in candidates if c.phase == 'prefill']),
            'decode_iterations': len([c for c in candidates if c.phase == 'decode']),
            'total_kernels': total_kernels,
            'prefill_kernels': len(prefill_kernels),
            'decode_kernels': len(decode_kernels),
            'prefill_kernels_list': prefill_kernels,
            'decode_kernels_list': decode_kernels
        }
    
    def analyze_trace(self, trace_path: str) -> Dict:
        """完整分析 trace 文件的 iteration 划分"""
        
        logger.info(f"开始基于 DtoH 的 iteration 分析 (vLLM模式): {trace_path}")
        
        # 1. 加载 trace 数据
        events = self.load_trace_file(trace_path)
        
        # 2. 提取 DtoH 事件
        dtoh_events = self.extract_dtoh_events(events)
        
        if len(dtoh_events) == 0:
            logger.warning("没有找到 DtoH 事件，无法检测 iterations")
            return {}
        
        # 3. 基于 DtoH 检测 iterations (vLLM模式)
        candidates = self.detect_iterations_vllm(events, dtoh_events)
        
        # 4. 为 vLLM 分配阶段
        candidates = self.assign_phases_vllm(candidates)
        
        # 5. 分析 kernel 分布
        kernel_analysis = self.analyze_kernels_by_phase(candidates)
        
        # 6. 生成详细报告
        logger.info(f"\n=== DtoH-based Iteration 分析结果 (vLLM) ===")
        logger.info(f"总 iterations: {kernel_analysis['total_iterations']}")
        logger.info(f"Prefill iterations: {kernel_analysis['prefill_iterations']}")
        logger.info(f"Decode iterations: {kernel_analysis['decode_iterations']}")
        logger.info(f"总 kernels: {kernel_analysis['total_kernels']}")
        logger.info(f"Prefill kernels: {kernel_analysis['prefill_kernels']}")
        logger.info(f"Decode kernels: {kernel_analysis['decode_kernels']}")
        
        if candidates:
            logger.info(f"\n=== Iteration 详情 ===")
            for candidate in candidates:
                gpu_events = [e for e in candidate.events 
                             if (e.get('ph') == 'X' and 
                                 e.get('cat', '').lower() in ['kernel', 'gpu', 'cuda'])]
                logger.info(f"{candidate} GPU kernels: {len(gpu_events)}")
        
        return {
            'iterations': candidates,
            'kernel_analysis': kernel_analysis,
            'trace_file': trace_path
        }

def main():
    if len(sys.argv) < 2:
        print("Usage: python iteration_detector_dtoh_based.py <trace_file>")
        print("Example: python3 iteration_detector_dtoh_based.py cases_after_sea/A100/Qwen2.5-32B_batch8_input128_output10/mlflow-*.trace.json")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    
    try:
        detector = DtoHBasedIterationDetector()
        result = detector.analyze_trace(trace_file)
        
        if result:
            # 从trace文件路径中提取pod_name（使用目录名）
            trace_dir = os.path.dirname(trace_file)
            if not trace_dir:
                trace_dir = '.'
            
            # 使用目录名作为pod_name
            pod_name = os.path.basename(trace_dir)
            
            # 生成输出文件路径：trace文件所在文件夹/dtoh_iteration_analysis_pod_name.json
            output_file = os.path.join(trace_dir, f"dtoh_iteration_analysis_{pod_name}.json")
            
            # 准备可序列化的结果
            serializable_result = {
                'trace_file': result['trace_file'],
                'kernel_analysis': result['kernel_analysis'],
                'iterations': []
            }
            
            # 转换 iterations 为可序列化格式
            for iteration in result['iterations']:
                serializable_result['iterations'].append({
                    'iteration_id': iteration.iteration_id,
                    'phase': iteration.phase,
                    'start_ts': iteration.start_ts,
                    'end_ts': iteration.end_ts,
                    'duration_us': iteration.duration,
                    'event_count': len(iteration.events)
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"分析结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()