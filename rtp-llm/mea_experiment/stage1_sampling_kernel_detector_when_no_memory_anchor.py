#!/usr/bin/env python3
"""
MEA Stage 1: Sampling Kernel-based Iteration Detection and Segmentation
Modified version that uses sampling kernels to segment iterations instead of memory anchors.
Based on the original MEA Stage 1 approach but adapted for kernel-based segmentation.
"""

import json
import sys
import logging
import os
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IterationCandidate:
    """Represents a candidate inference iteration segment"""
    def __init__(self, start_ts: float, end_ts: float, events: List[Dict]):
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.duration = end_ts - start_ts
        self.events = events
        self.iteration_id = None
    
    def __repr__(self):
        return f"IterationCandidate(id={self.iteration_id}, start={self.start_ts:.3f}, duration={self.duration:.3f}μs)"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'iteration_id': self.iteration_id,
            'start_ts': self.start_ts,
            'end_ts': self.end_ts,
            'duration_us': self.duration,
            'event_count': len(self.events),
            'events': self.events
        }

class SamplingKernelDetector:
    """
    Stage 1: Sampling Kernel-based Iteration Detection and Segmentation
    Uses sampling kernels as iteration boundaries instead of memory anchors
    """
    
    def __init__(self):
        pass
    
    def extract_sampling_kernels(self, trace_events: List[Dict[str, Any]]) -> List[float]:
        """
        Extract sampling kernel events as iteration boundaries
        Look for kernels with 'sampling' in their name
        """
        sampling_kernels = []
        
        for event in trace_events:
            if event.get("cat") == "kernel":
                name = event.get("name", "").lower()
                ts = event.get("ts", 0)
                
                # Look for kernels with 'sampling' in the name
                if "sampling" in name:
                    sampling_kernels.append(ts)
                    logger.debug(f"Found sampling kernel at {ts:.3f}: {event.get('name', '')}")
        
        sampling_kernels.sort()
        logger.info(f"Found {len(sampling_kernels)} sampling kernel events")
        return sampling_kernels
    
    def extract_candidate_segments(self, trace_events: List[Dict[str, Any]], 
                                  sampling_kernels: List[float]) -> List[IterationCandidate]:
        """
        Extract candidate iteration segments based on sampling kernels
        Each segment starts from one sampling kernel and ends at the next
        """
        candidates = []
        
        if len(sampling_kernels) < 2:
            logger.warning("Need at least 2 sampling kernels for iteration detection")
            return candidates
        
        logger.info(f"Processing {len(sampling_kernels)} sampling kernel events")
        
        # Minimum iteration duration threshold (1000 microseconds)
        MIN_ITERATION_DURATION = 1000.0
        
        # Create segments between consecutive sampling kernels
        for i in range(len(sampling_kernels) - 1):
            start_ts = sampling_kernels[i]
            end_ts = sampling_kernels[i + 1]
            duration = end_ts - start_ts
            
            # Skip segments that are too short
            if duration < MIN_ITERATION_DURATION:
                logger.debug(f"Skipping short segment: {start_ts:.1f} - {end_ts:.1f} μs (duration: {duration:.1f}μs < {MIN_ITERATION_DURATION}μs threshold)")
                continue
            
            # Extract events within this iteration
            segment_events = []
            for event in trace_events:
                event_ts = event.get("ts", 0)
                if start_ts <= event_ts < end_ts:
                    segment_events.append(event)
            
            # Create candidate if segment has events
            if len(segment_events) > 0:
                candidate = IterationCandidate(start_ts, end_ts, segment_events)
                candidate.iteration_id = len(candidates) + 1
                candidates.append(candidate)
                logger.debug(f"Created iteration {candidate.iteration_id}: {start_ts:.1f} - {end_ts:.1f} μs (duration: {duration:.1f} μs, events: {len(segment_events)})")
            else:
                logger.debug(f"Filtered out empty segment: {start_ts:.1f} - {end_ts:.1f} μs")
        
        logger.info(f"Extracted {len(candidates)} candidate segments based on sampling kernels")
        
        # Log iteration statistics for debugging
        if candidates:
            durations = [c.duration for c in candidates]
            logger.info(f"Iteration duration stats: min={min(durations):.1f}μs, max={max(durations):.1f}μs, avg={sum(durations)/len(durations):.1f}μs")
        
        return candidates
    
    def detect_iterations(self, trace_file: str) -> List[IterationCandidate]:
        """
        Main method for Stage 1: Sampling kernel-based iteration detection and segmentation
        """
        logger.info(f"Starting Stage 1: Sampling kernel-based iteration detection for {trace_file}")
        
        # Load trace data
        with open(trace_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        trace_events = data.get("traceEvents", [])
        logger.info(f"Loaded {len(trace_events)} trace events")
        
        # Extract sampling kernels
        sampling_kernels = self.extract_sampling_kernels(trace_events)
        
        if len(sampling_kernels) < 2:
            logger.warning("Insufficient sampling kernels found. Cannot detect iterations.")
            return []
        
        # Extract candidate segments
        candidates = self.extract_candidate_segments(trace_events, sampling_kernels)
        
        logger.info(f"Stage 1 completed: {len(candidates)} candidate iterations detected")
        return candidates
    
    def save_results(self, candidates: List[IterationCandidate], output_file: str):
        """Save Stage 1 results to file"""
        results = {
            'stage': 1,
            'description': 'Sampling kernel-based iteration detection and segmentation',
            'metadata': {
                'total_candidates': len(candidates),
                'stage_1_completed': True,
                'segmentation_method': 'sampling_kernel'
            },
            'iterations': [candidate.to_dict() for candidate in candidates]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Stage 1 results saved to {output_file}")

def get_case_directory_from_trace_path(trace_path: str) -> str:
    """
    从trace文件路径中提取case目录路径
    例如: traces_after_sea_section_part1/bc-online-question-recognize-20250819-na61-h20.inference-part0-deaa5dd1-a-fc4a/20251009113352-53761c978044b87a2892/ecos-trace-209515-1759980853551740372-7.json
    返回: traces_after_sea_section_part1/bc-online-question-recognize-20250819-na61-h20.inference-part0-deaa5dd1-a-fc4a
    """
    # 规范化路径
    trace_path = os.path.normpath(trace_path)
    
    # 分割路径
    path_parts = trace_path.split(os.sep)
    
    # 查找包含 traces_after_sea_section_part 的部分
    case_dir_parts = []
    found_traces_dir = False
    
    for i, part in enumerate(path_parts):
        if part.startswith('traces_after_sea_section_part'):
            found_traces_dir = True
            case_dir_parts.append(part)
        elif found_traces_dir and not part.startswith('2025') and '.' not in part:
            # 这是case目录名（不是时间戳目录，也不是文件）
            case_dir_parts.append(part)
            break
        elif found_traces_dir:
            case_dir_parts.append(part)
    
    if len(case_dir_parts) >= 2:
        return os.path.join(*case_dir_parts)
    else:
        # 如果无法识别case目录，返回trace文件所在目录
        return os.path.dirname(trace_path)

def main():
    if len(sys.argv) < 2:
        print("Usage: python stage1_sampling_kernel_detector.py <input_trace_file> [--output <output_file>]")
        print("Example: python stage1_sampling_kernel_detector.py trace.json")
        print("         python stage1_sampling_kernel_detector.py trace.json --output stage1_results.json")
        print("")
        print("If no output file is specified, results will be saved to the case directory as 'stage1_results.json'")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # 解析命令行参数
    output_file = None
    if len(sys.argv) > 2:
        if '--output' in sys.argv:
            try:
                output_idx = sys.argv.index('--output')
                if output_idx + 1 < len(sys.argv):
                    output_file = sys.argv[output_idx + 1]
            except (ValueError, IndexError):
                pass
        else:
            # 兼容旧的参数格式
            output_file = sys.argv[2]
    
    # 如果没有指定输出文件，自动生成到case目录下
    if output_file is None:
        case_dir = get_case_directory_from_trace_path(input_file)
        output_file = os.path.join(case_dir, "stage1_results.json")
        logger.info(f"No output file specified. Results will be saved to: {output_file}")
    
    try:
        detector = SamplingKernelDetector()
        candidates = detector.detect_iterations(input_file)
        detector.save_results(candidates, output_file)
        
        # Print summary
        print(f"\nStage 1: Sampling Kernel-based Detection Summary:")
        print(f"Input trace file: {input_file}")
        print(f"Candidate iterations detected: {len(candidates)}")
        if candidates:
            durations = [c.duration for c in candidates]
            print(f"Duration range: {min(durations):.1f} - {max(durations):.1f} μs")
            print(f"Average duration: {sum(durations)/len(durations):.1f} μs")
        print(f"Results saved to: {output_file}")
        print(f"\nReady for Stage 2: Operator fingerprint validation")
        
    except Exception as e:
        logger.error(f"Error during Stage 1 processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()