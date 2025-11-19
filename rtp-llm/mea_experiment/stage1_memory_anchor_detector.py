#!/usr/bin/env python3
"""
MEA Stage 1: Memory Anchor Detection and Iteration Segmentation
Based on the paper's first stage approach:
1. Memory anchor detection (memcpyHtoD/DtoH events)
2. Non-overlapping iteration segmentation
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

class MemoryAnchorDetector:
    """
    Stage 1: Memory Anchor Detection and Iteration Segmentation
    Implements non-overlapping HtoD-DtoH pairing logic
    """
    
    def __init__(self):
        pass
    
    def extract_memory_anchors(self, trace_events: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
        """
        Extract memory transfer anchors (HtoD and DtoH events)
        Based on paper Algorithm 1, lines 2-8
        """
        start_anchors = []  # memcpyHtoD events
        end_anchors = []    # memcpyDtoH events
        
        for event in trace_events:
            if event.get("cat") == "gpu_memcpy":
                name = event.get("name", "")
                ts = event.get("ts", 0)
                
                # Start anchors: HtoD transfers (token indices, position embeddings)
                if "HtoD" in name or "Host to Device" in name:
                    start_anchors.append(ts)
                
                # End anchors: DtoH transfers (logits, output tokens)
                elif "DtoH" in name or "Device to Host" in name:
                    end_anchors.append(ts)
        
        start_anchors.sort()
        end_anchors.sort()
        
        logger.info(f"Found {len(start_anchors)} start anchors and {len(end_anchors)} end anchors")
        return start_anchors, end_anchors
    
    def extract_candidate_segments(self, trace_events: List[Dict[str, Any]], 
                                  start_anchors: List[float], 
                                  end_anchors: List[float]) -> List[IterationCandidate]:
        """
        Extract candidate iteration segments based on memory anchors
        Non-overlapping pairing: each DtoH can only be used once
        Modified to ensure minimum iteration duration of 1000μs
        """
        candidates = []
        
        if not start_anchors or not end_anchors:
            logger.warning("Insufficient memory anchors for iteration detection")
            return candidates
        
        logger.info(f"Processing {len(start_anchors)} HtoD and {len(end_anchors)} DtoH events")
        
        # Sort anchors by timestamp
        htod_sorted = sorted(start_anchors)
        dtoh_sorted = sorted(end_anchors)
        
        # Track which DtoH events have been used
        used_dtoh = set()
        current_htod_idx = 0
        
        # Minimum iteration duration threshold (1000 microseconds)
        MIN_ITERATION_DURATION = 1000.0
        
        while current_htod_idx < len(htod_sorted):
            start_ts = htod_sorted[current_htod_idx]
            
            # Find the first unused DtoH after this HtoD that results in sufficient duration
            end_ts = None
            used_dtoh_ts = None
            
            for dtoh_ts in dtoh_sorted:
                if dtoh_ts > start_ts and dtoh_ts not in used_dtoh:
                    # Check if this DtoH would result in sufficient iteration duration
                    potential_duration = dtoh_ts - start_ts
                    if potential_duration >= MIN_ITERATION_DURATION:
                        end_ts = dtoh_ts
                        used_dtoh_ts = dtoh_ts
                        break
                    else:
                        logger.debug(f"Skipping DtoH at {dtoh_ts:.1f} - duration {potential_duration:.1f}μs < {MIN_ITERATION_DURATION}μs threshold")
            
            # If no valid DtoH found with sufficient duration, remaining HtoD events are fragments
            if end_ts is None:
                logger.info(f"No more DtoH with sufficient duration available after HtoD at {start_ts:.1f}. Remaining {len(htod_sorted) - current_htod_idx} HtoD events are fragments.")
                break
            
            # Mark this DtoH as used
            used_dtoh.add(used_dtoh_ts)
            
            # Extract events within this iteration
            segment_events = []
            for event in trace_events:
                event_ts = event.get("ts", 0)
                if start_ts <= event_ts < end_ts:
                    segment_events.append(event)
            
            # Create candidate if segment has events
            duration = end_ts - start_ts
            if len(segment_events) > 0:
                candidate = IterationCandidate(start_ts, end_ts, segment_events)
                candidate.iteration_id = len(candidates) + 1
                candidates.append(candidate)
                logger.debug(f"Created iteration {candidate.iteration_id}: {start_ts:.1f} - {end_ts:.1f} μs (duration: {duration:.1f} μs)")
            else:
                logger.debug(f"Filtered out empty segment: {start_ts:.1f} - {end_ts:.1f} μs")
            
            # Find next HtoD that starts after current iteration ends
            next_htod_idx = current_htod_idx + 1
            while next_htod_idx < len(htod_sorted) and htod_sorted[next_htod_idx] < end_ts:
                logger.debug(f"Skipping internal HtoD at {htod_sorted[next_htod_idx]:.1f} (within current iteration)")
                next_htod_idx += 1
            
            current_htod_idx = next_htod_idx
        
        logger.info(f"Extracted {len(candidates)} non-overlapping candidate segments")
        logger.info(f"Used {len(used_dtoh)} out of {len(dtoh_sorted)} DtoH events")
        
        # Log iteration statistics for debugging
        if candidates:
            durations = [c.duration for c in candidates]
            logger.info(f"Iteration duration stats: min={min(durations):.1f}μs, max={max(durations):.1f}μs, avg={sum(durations)/len(durations):.1f}μs")
        
        return candidates
    
    def detect_iterations(self, trace_file: str) -> List[IterationCandidate]:
        """
        Main method for Stage 1: Memory anchor detection and iteration segmentation
        """
        logger.info(f"Starting Stage 1: Memory anchor detection for {trace_file}")
        
        # Load trace data
        with open(trace_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        trace_events = data.get("traceEvents", [])
        logger.info(f"Loaded {len(trace_events)} trace events")
        
        # Extract memory anchors
        start_anchors, end_anchors = self.extract_memory_anchors(trace_events)
        
        if len(start_anchors) == 0 and len(end_anchors) == 0:
            logger.warning("No memory transfer events found. Cannot detect iterations.")
            return []
        
        # Extract candidate segments
        candidates = self.extract_candidate_segments(trace_events, start_anchors, end_anchors)
        
        logger.info(f"Stage 1 completed: {len(candidates)} candidate iterations detected")
        return candidates
    
    def save_results(self, candidates: List[IterationCandidate], output_file: str):
        """Save Stage 1 results to file"""
        results = {
            'stage': 1,
            'description': 'Memory anchor detection and iteration segmentation',
            'metadata': {
                'total_candidates': len(candidates),
                'stage_1_completed': True
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
        print("Usage: python stage1_memory_anchor_detector.py <input_trace_file> [--output <output_file>]")
        print("Example: python stage1_memory_anchor_detector.py trace.json")
        print("         python stage1_memory_anchor_detector.py trace.json --output stage1_results.json")
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
        detector = MemoryAnchorDetector()
        candidates = detector.detect_iterations(input_file)
        detector.save_results(candidates, output_file)
        
        # Print summary
        print(f"\nStage 1: Memory Anchor Detection Summary:")
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