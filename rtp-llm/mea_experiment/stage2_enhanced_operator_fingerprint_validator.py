#!/usr/bin/env python3
"""
MEA Stage 2: Enhanced Operator Fingerprint Validation
Based on the paper's methodology with three key optimizations:
1. Dynamic thresholds: Adaptive threshold determination based on global distribution
2. Structural awareness: Clear distinction of head-middle-tail operator patterns  
3. Sequence patterns: Analysis of operator execution order and transitions
"""

import json
import sys
import csv
import logging
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
import numpy as np

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

class EnhancedOperatorFingerprintValidator:
    """
    Enhanced Stage 2: Operator Fingerprint Validation with dynamic thresholds,
    structural awareness, and sequence pattern recognition
    """
    
    def __init__(self):
        self.kernel_to_operator_map = self._build_kernel_operator_mapping()
        
    def _build_kernel_operator_mapping(self) -> Dict[str, str]:
        """Build mapping from GPU kernel names to high-level operators"""
        mapping = {
            # Linear operators
            'rowparallellinear': 'linear',
            'mergedcolumnparallellinear': 'linear', 
            'qkvparallellinear': 'linear',
            'logitsprocessor': 'linear',
            'linear': 'linear',
            'gemm': 'linear',
            'sgemm': 'linear',
            'hgemm': 'linear',
            'cutlass': 'linear',
            'cublasgemm': 'linear',
            'matmul': 'linear',
            'nvjet': 'linear',
            'gemvt': 'linear',
            'gemvn': 'linear',
            
            # LayerNorm operators
            'rmsnorm': 'layernorm',
            'rms_norm': 'layernorm',
            'fused_add_rms_norm': 'layernorm',
            'layer_norm': 'layernorm',
            'layernorm': 'layernorm',
            'norm': 'layernorm',
            
            # Vocab Parallel Embedding
            'vocabparallelembedding': 'vocab_parallel_embedding',
            'vocab_parallel_embedding': 'vocab_parallel_embedding',
            'embedding': 'vocab_parallel_embedding',
            'embed': 'vocab_parallel_embedding',
            
            # Activation functions
            'siluandmul': 'activation',
            'silu_and_mul': 'activation',
            'gelu': 'activation',
            'relu': 'activation',
            'silu': 'activation',
            'swish': 'activation',
            'sigmoid': 'activation',
            
            # Attention operators
            'masked_multihead_attention_kernel': 'attention',  # 专门处理FasterTransformer的attention kernel
            'multihead_attention_kernel': 'attention',         # 通用的multihead attention kernel
            'flash_attn': 'attention',
            'flashinfer_attn': 'attention',
            'flash_attention': 'attention',
            'flashinfer_attention': 'attention',
            'fmha': 'attention',
            'attention': 'attention',
            'scaled_dot_product': 'attention',
            'attn': 'attention',
            
            # Sampler
            'sampler': 'sampler',
            'sampling': 'sampler',
            'sample': 'sampler',
            'top_k': 'sampler',
            'top_p': 'sampler',
            'multinomial': 'sampler',
            
            # Memory operations
            'memcpy': 'memory_op',
            'transpose': 'transpose',
            
            # Communication operations
            'nccl': 'communication',
            'allreduce': 'communication',
            'allgather': 'communication',
        }
        return mapping
    
    def _map_kernel_to_operator(self, kernel_name: str) -> str:
        """Map a kernel name to its corresponding operator type"""
        kernel_lower = kernel_name.lower()
        for pattern, operator in self.kernel_to_operator_map.items():
            if pattern in kernel_lower:
                return operator
        return 'other'
    
    def compute_dynamic_thresholds(self, global_distribution: Dict[str, float]) -> Dict[str, float]:
        """
        Optimization 1: Dynamic thresholds based on global operator distribution
        Following the paper's adaptive threshold methodology
        """
        if not global_distribution:
            return {
                'linear_min': 0.15,
                'layernorm_min': 0.05,
                'attention_min': 0.03
            }
        
        # Extract key operator ratios
        linear_ratio = global_distribution.get('linear', 0.0)
        layernorm_ratio = global_distribution.get('layernorm', 0.0)
        attention_ratio = global_distribution.get('attention', 0.0)
        
        # Adaptive thresholds: use 50% of observed global ratio as minimum
        # but ensure reasonable lower bounds for robustness
        thresholds = {
            'linear_min': max(0.10, linear_ratio * 0.5),
            'layernorm_min': max(0.03, layernorm_ratio * 0.5),
            'attention_min': max(0.02, attention_ratio * 0.5),
            'communication_min': max(0.01, global_distribution.get('communication', 0.0) * 0.3),
            'activation_min': max(0.01, global_distribution.get('activation', 0.0) * 0.3)
        }
        
        logger.info(f"Dynamic thresholds computed: {thresholds}")
        return thresholds
    
    def analyze_sequence_patterns(self, operator_sequence: List[str]) -> Dict[str, Any]:
        """
        Optimization 3: Adaptive sequence pattern analysis with data-driven pattern discovery
        Based on the paper's sequence pattern recognition methodology
        """
        if len(operator_sequence) < 3:
            return {
                'has_valid_sequence': False,
                'reason': 'Sequence too short for pattern analysis'
            }
        
        # 1. Transition Pattern Analysis
        transitions = []
        transition_counts = Counter()
        
        for i in range(len(operator_sequence) - 1):
            transition = f"{operator_sequence[i]}->{operator_sequence[i+1]}"
            transitions.append(transition)
            transition_counts[transition] += 1
        
        # 2. Data-driven Canonical Pattern Discovery
        # Instead of hardcoded patterns, discover frequent transitions dynamically
        total_transitions = len(transitions)
        frequent_patterns = []
        
        # Extract patterns that appear frequently (top 20% of transitions)
        if total_transitions > 0:
            # Get top frequent transitions as canonical patterns
            sorted_transitions = transition_counts.most_common()
            threshold = max(2, int(total_transitions * 0.05))  # At least 5% frequency
            
            for transition, count in sorted_transitions:
                if count >= threshold:
                    frequent_patterns.append(transition)
                if len(frequent_patterns) >= 10:  # Limit to top 10 patterns
                    break
        
        # 3. Fallback to domain knowledge patterns if no frequent patterns found
        if not frequent_patterns:
            # Use domain knowledge as fallback, but make it more comprehensive
            domain_patterns = [
                # Standard Transformer patterns
                'layernorm->linear', 'linear->attention', 'attention->linear',
                'linear->layernorm', 'linear->activation', 'activation->linear',
                # Additional patterns for different architectures
                'rmsnorm->linear', 'linear->rmsnorm', 'rmsnorm->attention',
                'attention->rmsnorm', 'embedding->layernorm', 'layernorm->embedding',
                # MoE patterns
                'linear->router', 'router->expert', 'expert->linear',
                # Communication patterns
                'communication->linear', 'linear->communication',
                # Memory patterns
                'memory_op->linear', 'linear->memory_op'
            ]
            frequent_patterns = domain_patterns
        
        # Count pattern matches using discovered frequent patterns
        pattern_matches = 0
        for pattern in frequent_patterns:
            if pattern in transition_counts:
                pattern_matches += transition_counts[pattern]
        
        # 4. Adaptive Layer Repetition Detection
        layer_patterns = self._detect_adaptive_layer_repetition(operator_sequence)
        
        # 5. Compute Sequence Entropy
        sequence_entropy = self._compute_sequence_entropy(operator_sequence)
        
        # 6. Adaptive Validation Criteria
        pattern_ratio = pattern_matches / total_transitions if total_transitions > 0 else 0
        
        # Dynamic threshold based on sequence characteristics
        min_pattern_ratio = self._compute_adaptive_pattern_threshold(
            operator_sequence, frequent_patterns, layer_patterns
        )
        
        # Enhanced validation criteria with adaptive thresholds
        has_valid_sequence = (
            pattern_ratio >= min_pattern_ratio and
            layer_patterns['estimated_layers'] >= 1 and  # At least 1 layer repetition
            sequence_entropy < 4.0  # Slightly more permissive entropy threshold
        )
        
        return {
            'has_valid_sequence': has_valid_sequence,
            'transitions': dict(transition_counts.most_common(10)),
            'discovered_patterns': frequent_patterns[:10],  # Top discovered patterns
            'canonical_pattern_matches': pattern_matches,
            'pattern_ratio': pattern_ratio,
            'min_pattern_ratio': min_pattern_ratio,
            'layer_patterns': layer_patterns,
            'sequence_entropy': sequence_entropy,
            'total_transitions': total_transitions,
            'adaptation_info': {
                'used_data_driven_patterns': len(frequent_patterns) > 0,
                'pattern_discovery_threshold': threshold if total_transitions > 0 else 0,
                'adaptive_validation': True
            }
        }
    
    def _detect_adaptive_layer_repetition(self, sequence: List[str]) -> Dict[str, Any]:
        """Adaptive layer repetition detection with dynamic pattern length range"""
        if len(sequence) < 4:
            return {'estimated_layers': 0, 'repetition_score': 0.0, 'best_pattern': None}
        
        # Dynamic pattern length range based on sequence characteristics
        seq_len = len(sequence)
        unique_ops = len(set(sequence))
        
        # Adaptive range calculation
        min_pattern_len = max(2, min(3, unique_ops // 2))
        max_pattern_len = min(seq_len // 3, max(6, unique_ops), 12)
        
        # Ensure reasonable bounds
        max_pattern_len = min(max_pattern_len, 12)  # Cap at 20 for performance
        
        best_pattern = None
        max_repetitions = 0
        best_score = 0.0
        
        for pattern_length in range(min_pattern_len, max_pattern_len + 1):
            for start in range(len(sequence) - pattern_length * 2):
                pattern = sequence[start:start + pattern_length]
                repetitions = 1
                coverage = pattern_length
                
                # Count consecutive repetitions
                pos = start + pattern_length
                while pos + pattern_length <= len(sequence):
                    if sequence[pos:pos + pattern_length] == pattern:
                        repetitions += 1
                        coverage += pattern_length
                        pos += pattern_length
                    else:
                        # Allow some flexibility - check for partial matches
                        partial_match = 0
                        remaining = min(pattern_length, len(sequence) - pos)
                        for i in range(remaining):
                            if pos + i < len(sequence) and sequence[pos + i] == pattern[i]:
                                partial_match += 1
                        
                        if partial_match >= pattern_length * 0.7:  # 70% match threshold
                            repetitions += partial_match / pattern_length
                            coverage += partial_match
                        pos += 1
                
                # Score based on repetitions and coverage
                coverage_ratio = coverage / len(sequence)
                score = repetitions * coverage_ratio
                
                if score > best_score:
                    max_repetitions = repetitions
                    best_pattern = pattern
                    best_score = score
        
        repetition_score = min(1.0, best_score)
        
        return {
            'estimated_layers': int(max_repetitions),
            'best_pattern': best_pattern,
            'repetition_score': repetition_score,
            'adaptive_range': (min_pattern_len, max_pattern_len),
            'pattern_length': len(best_pattern) if best_pattern else 0
        }
    
    def _compute_adaptive_pattern_threshold(self, sequence: List[str], 
                                          discovered_patterns: List[str],
                                          layer_info: Dict[str, Any]) -> float:
        """Compute adaptive pattern threshold based on sequence characteristics"""
        
        # Base threshold
        base_threshold = 0.15
        
        # Adjust based on sequence diversity
        unique_ops = len(set(sequence))
        total_ops = len(sequence)
        diversity_ratio = unique_ops / total_ops if total_ops > 0 else 1.0
        
        # More diverse sequences need lower thresholds
        diversity_adjustment = max(0.05, 0.25 - diversity_ratio * 0.2)
        
        # Adjust based on layer repetition quality
        repetition_score = layer_info.get('repetition_score', 0.0)
        repetition_adjustment = max(0.0, 0.1 - repetition_score * 0.05)
        
        # Adjust based on number of discovered patterns
        pattern_count = len(discovered_patterns)
        pattern_adjustment = max(0.0, 0.05 - pattern_count * 0.005)
        
        adaptive_threshold = base_threshold + diversity_adjustment + repetition_adjustment + pattern_adjustment
        
        # Ensure reasonable bounds
        return max(0.10, min(0.35, adaptive_threshold))

    def _compute_sequence_entropy(self, sequence: List[str]) -> float:
        """Compute Shannon entropy of operator sequence to measure regularity"""
        if not sequence:
            return 0.0
        
        counts = Counter(sequence)
        total = len(sequence)
        
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy

    def analyze_structural_segments(self, operator_sequence: List[str], 
                                   global_distribution: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimization 2: Structural awareness with clear head-middle-tail segmentation
        Based on the paper's structural awareness methodology
        """
        if not operator_sequence:
            return {
                'has_valid_structure': False,
                'reason': 'Empty operator sequence'
            }
        
        seq_len = len(operator_sequence)
        if seq_len < 10:
            return {
                'has_valid_structure': False,
                'reason': f'Sequence too short ({seq_len} operators)'
            }
        
        # Dynamic segmentation based on sequence length
        head_ratio = max(0.05, min(0.15, 10 / seq_len))
        head_end = max(2, int(seq_len * head_ratio))
        
        tail_ratio = max(0.03, min(0.10, 5 / seq_len))
        tail_start = seq_len - max(2, int(seq_len * tail_ratio))
        
        # Extract segments
        head_section = operator_sequence[:head_end]
        middle_section = operator_sequence[head_end:tail_start]
        tail_section = operator_sequence[tail_start:]
        
        # Define expected operators for each phase
        head_operators = {'vocab_parallel_embedding', 'communication', 'linear', 'layernorm'}
        tail_operators = {'sampler', 'memory_op', 'linear', 'communication'}
        middle_operators = {'linear', 'layernorm', 'attention', 'activation'}
        
        # Analyze each segment
        head_analysis = self._analyze_segment(head_section, head_operators, "head")
        middle_analysis = self._analyze_segment(middle_section, middle_operators, "middle")
        tail_analysis = self._analyze_segment(tail_section, tail_operators, "tail")
        
        # Dynamic validation with adaptive thresholds
        thresholds = self.compute_dynamic_thresholds(global_distribution)
        
        # Compute operator ratios
        op_counts = Counter(operator_sequence)
        total_ops = len(operator_sequence)
        
        operator_ratios = {
            'linear': op_counts.get('linear', 0) / total_ops,
            'layernorm': op_counts.get('layernorm', 0) / total_ops,
            'attention': op_counts.get('attention', 0) / total_ops,
            'communication': op_counts.get('communication', 0) / total_ops,
            'activation': op_counts.get('activation', 0) / total_ops
        }
        
        # Dynamic validation based on computed thresholds
        ratio_validation = (
            operator_ratios['linear'] >= thresholds['linear_min'] and
            operator_ratios['layernorm'] >= thresholds['layernorm_min'] and
            operator_ratios['attention'] >= thresholds['attention_min']
        )
        
        # Structural validation
        structural_validation = (
            head_analysis['coverage_score'] >= 0.3 and
            middle_analysis['coverage_score'] >= 0.6 and
            tail_analysis['coverage_score'] >= 0.2
        )
        
        is_valid_structure = ratio_validation and structural_validation
        
        return {
            'has_valid_structure': is_valid_structure,
            'head_section': head_analysis,
            'middle_section': middle_analysis,
            'tail_section': tail_analysis,
            'operator_ratios': operator_ratios,
            'dynamic_thresholds': thresholds,
            'ratio_validation': ratio_validation,
            'structural_validation': structural_validation,
            'segmentation': {
                'head_range': (0, head_end),
                'middle_range': (head_end, tail_start),
                'tail_range': (tail_start, seq_len)
            }
        }
    
    def _analyze_segment(self, segment: List[str], expected_ops: Set[str], 
                        segment_name: str) -> Dict[str, Any]:
        """Analyze a specific segment for expected operator patterns"""
        if not segment:
            return {
                'operators': [],
                'unique_ops': [],
                'coverage_score': 0.0,
                'dominant_ops': [],
                'segment_name': segment_name
            }
        
        segment_ops = set(segment)
        op_counts = Counter(segment)
        
        # Coverage score: how well this segment matches expected operators
        overlap = segment_ops & expected_ops
        coverage_score = len(overlap) / len(expected_ops) if expected_ops else 0.0
        
        # Dominant operators (top 3 by frequency)
        dominant_ops = [op for op, _ in op_counts.most_common(3)]
        
        return {
            'operators': segment,
            'unique_ops': list(segment_ops),
            'coverage_score': coverage_score,
            'expected_overlap': list(overlap),
            'dominant_ops': dominant_ops,
            'segment_name': segment_name,
            'length': len(segment)
        }

    def analyze_enhanced_operator_fingerprint(self, operator_sequence: List[str], 
                                            global_distribution: Dict[str, float]) -> Dict[str, Any]:
        """
        Enhanced operator fingerprint analysis integrating all three optimizations:
        1. Dynamic thresholds
        2. Structural awareness  
        3. Sequence pattern recognition
        
        Modified to reduce false negatives while maintaining accuracy
        """
        if not operator_sequence:
            return {
                'has_valid_fingerprint': False,
                'reason': 'Empty operator sequence'
            }
        
        seq_len = len(operator_sequence)
        if seq_len < 10:
            return {
                'has_valid_fingerprint': False,
                'reason': f'Sequence too short ({seq_len} operators)'
            }
        
        # 1. Structural Analysis with Dynamic Segmentation
        structural_analysis = self.analyze_structural_segments(operator_sequence, global_distribution)
        
        # 2. Sequence Pattern Analysis
        sequence_analysis = self.analyze_sequence_patterns(operator_sequence)
        
        # 3. Dynamic Thresholds
        thresholds = self.compute_dynamic_thresholds(global_distribution)
        
        # 4. Enhanced Validation Logic - More tolerant for edge cases
        # Primary validation: Strong indicators of valid transformer iteration
        strong_indicators = (
            sequence_analysis['pattern_ratio'] >= 0.8 and  # Very high pattern ratio
            sequence_analysis['layer_patterns']['estimated_layers'] >= 45  # High layer count
        )
        
        # Standard validation with slightly relaxed criteria
        standard_validation = (
            structural_analysis['has_valid_structure'] and
            sequence_analysis['has_valid_sequence'] and
            sequence_analysis['pattern_ratio'] >= 0.25
        )
        
        # Fallback validation for borderline cases
        fallback_validation = (
            sequence_analysis['has_valid_sequence'] and
            sequence_analysis['pattern_ratio'] >= 0.7 and  # High pattern ratio compensates
            structural_analysis['ratio_validation']  # At least operator ratios are good
        )
        
        # Accept if any validation path succeeds
        is_valid_fingerprint = strong_indicators or standard_validation or fallback_validation
        
        # 5. Enhanced Confidence Score - boost for high-performing iterations
        confidence_components = [
            structural_analysis['ratio_validation'],
            structural_analysis['structural_validation'],
            sequence_analysis['has_valid_sequence'],
            sequence_analysis['pattern_ratio'] >= 0.25
        ]
        confidence_score = sum(confidence_components) / len(confidence_components)
        
        # Boost confidence for iterations with strong patterns
        if sequence_analysis['pattern_ratio'] >= 0.8:
            confidence_score = min(1.0, confidence_score + 0.15)
        elif sequence_analysis['pattern_ratio'] >= 0.7:
            confidence_score = min(1.0, confidence_score + 0.10)
        
        return {
            'has_valid_fingerprint': is_valid_fingerprint,
            'confidence_score': confidence_score,
            'structural_analysis': structural_analysis,
            'sequence_analysis': sequence_analysis,
            'dynamic_thresholds': thresholds,
            'sequence_length': seq_len,
            'unique_operator_types': len(set(operator_sequence)),
            'validation_summary': {
                'structural_valid': structural_analysis['has_valid_structure'],
                'sequence_valid': sequence_analysis['has_valid_sequence'],
                'pattern_ratio': sequence_analysis['pattern_ratio'],
                'estimated_layers': sequence_analysis['layer_patterns']['estimated_layers']
            }
        }
    
    def extract_kernels_with_frequency(self, trace_file: str) -> Dict[str, int]:
        """Extract distinct kernels with their frequencies from trace data"""
        logger.info(f"Extracting kernels with frequencies from {trace_file}")
        
        with open(trace_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both direct trace events and stage1 results format
        if 'iterations' in data:
            # This is stage1 results format
            kernel_counts = Counter()
            iterations = data.get('iterations', [])
            
            for iteration in iterations:
                events = iteration.get('events', [])
                for event in events:
                    # Look for kernel events with correct category
                    if event.get("cat") == "kernel" and event.get("ph") == "X":
                        kernel_name = event.get("name", "")
                        if kernel_name and not kernel_name.startswith(("process_", "thread_")):
                            kernel_counts[kernel_name] += 1
        else:
            # This is direct trace events format
            trace_events = data.get("traceEvents", [])
            kernel_counts = Counter()
            
            for event in trace_events:
                if event.get("cat") == "kernel" and event.get("ph") == "X":
                    kernel_name = event.get("name", "")
                    if kernel_name and not kernel_name.startswith(("process_", "thread_")):
                        kernel_counts[kernel_name] += 1
        
        logger.info(f"Found {len(kernel_counts)} distinct kernels with frequencies")
        return dict(kernel_counts)
    
    def save_kernels_with_frequency_csv(self, kernel_counts: Dict[str, int], output_file: str):
        """Save kernels with frequencies and operator mapping to CSV"""
        logger.info(f"Saving kernels with frequencies to {output_file}")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['kernel_name', 'operator_type', 'frequency'])
            
            sorted_kernels = sorted(kernel_counts.items(), key=lambda x: (-x[1], x[0]))
            
            for kernel_name, frequency in sorted_kernels:
                operator_type = self._map_kernel_to_operator(kernel_name)
                writer.writerow([kernel_name, operator_type, frequency])
        
        logger.info(f"Kernels with frequencies saved to {output_file}")
    
    def validate_stage1_candidates(self, stage1_results_file: str) -> Dict[str, Any]:
        """
        Enhanced Stage 1 validation with all three optimizations:
        1. Dynamic thresholds 2. Structural awareness 3. Sequence patterns
        """
        logger.info(f"Starting Enhanced Stage 2: Operator fingerprint validation with three optimizations")
        
        # Load Stage 1 results
        with open(stage1_results_file, 'r', encoding='utf-8') as f:
            stage1_data = json.load(f)
        
        iterations = stage1_data.get('iterations', [])
        logger.info(f"Validating {len(iterations)} candidate iterations from Stage 1")
        
        # First pass: collect global statistics
        global_operator_counts = Counter()
        
        for iteration in iterations:
            events = iteration.get('events', [])
            
            for event in events:
                if event.get("cat") in ["gpu_op", "kernel"] and event.get("ph") == "X":
                    kernel_name = event.get("name", "")
                    if kernel_name:
                        operator_type = self._map_kernel_to_operator(kernel_name)
                        global_operator_counts[operator_type] += 1
        
        # Calculate global operator distribution
        total_global_ops = sum(global_operator_counts.values())
        global_operator_distribution = {}
        if total_global_ops > 0:
            global_operator_distribution = {
                op: count / total_global_ops 
                for op, count in global_operator_counts.items()
            }
        
        # Second pass: enhanced validation for each iteration
        validated_iterations = []
        
        for iteration in iterations:
            iteration_id = iteration.get('iteration_id')
            events = iteration.get('events', [])
            
            # Extract operators from this iteration
            operator_sequence = []
            operator_counts = Counter()
            
            for event in events:
                if event.get("cat") in ["gpu_op", "kernel"] and event.get("ph") == "X":
                    kernel_name = event.get("name", "")
                    if kernel_name:
                        operator_type = self._map_kernel_to_operator(kernel_name)
                        operator_sequence.append(operator_type)
                        operator_counts[operator_type] += 1
            
            # Calculate operator distribution for this iteration
            total_ops = sum(operator_counts.values())
            operator_distribution = {}
            if total_ops > 0:
                operator_distribution = {
                    op: count / total_ops 
                    for op, count in operator_counts.items()
                }
            
            # Enhanced fingerprint analysis with all three optimizations
            fingerprint_analysis = self.analyze_enhanced_operator_fingerprint(
                operator_sequence, global_operator_distribution
            )
            
            # Add enhanced validation results
            validated_iteration = iteration.copy()
            validated_iteration.update({
                'operator_sequence': operator_sequence,
                'operator_counts': dict(operator_counts),
                'operator_distribution': operator_distribution,
                'total_operators': total_ops,
                'unique_operators': len(operator_counts),
                'fingerprint_analysis': fingerprint_analysis,
                'fingerprint_validated': fingerprint_analysis['has_valid_fingerprint'],
                'confidence_score': fingerprint_analysis['confidence_score'],
                'stage2_enhanced': True
            })
            
            validated_iterations.append(validated_iteration)
            
            # Enhanced logging
            status = "✓" if validated_iteration['fingerprint_validated'] else "✗"
            confidence = fingerprint_analysis['confidence_score']
            layers = fingerprint_analysis['validation_summary']['estimated_layers']
            pattern_ratio = fingerprint_analysis['validation_summary']['pattern_ratio']
            
            logger.info(f"{status} Iteration {iteration_id}: "
                       f"Valid={validated_iteration['fingerprint_validated']}, "
                       f"Confidence={confidence:.2f}, "
                       f"Layers={layers}, "
                       f"PatternRatio={pattern_ratio:.2f}")
        
        # Build execution fingerprint
        execution_fingerprint = [op for op, _ in global_operator_counts.most_common(10)]
        
        # Count validated iterations
        valid_iterations = [it for it in validated_iterations 
                          if it.get('fingerprint_validated', False)]
        
        # Compute statistics
        avg_confidence = float(np.mean([it['confidence_score'] for it in validated_iterations]))
        avg_pattern_ratio = float(np.mean([it['fingerprint_analysis']['validation_summary']['pattern_ratio'] 
                                   for it in validated_iterations]))
        avg_layers = float(np.mean([it['fingerprint_analysis']['validation_summary']['estimated_layers'] 
                            for it in validated_iterations]))
        
        results = {
            'stage': 2,
            'description': 'Enhanced operator fingerprint validation with dynamic thresholds, structural awareness, and sequence patterns',
            'metadata': {
                'total_candidates': len(validated_iterations),
                'valid_fingerprint_iterations': len(valid_iterations),
                'average_confidence_score': avg_confidence,
                'average_pattern_ratio': avg_pattern_ratio,
                'average_estimated_layers': avg_layers,
                'global_operator_counts': dict(global_operator_counts),
                'global_operator_distribution': global_operator_distribution,
                'execution_fingerprint': execution_fingerprint,
                'stage1_completed': True,
                'stage2_enhanced_completed': True,
                'optimizations_enabled': {
                    'dynamic_thresholds': True,
                    'structural_awareness': True,
                    'sequence_pattern_recognition': True
                }
            },
            'iterations': validated_iterations
        }
        
        logger.info(f"Enhanced Stage 2 completed: "
                   f"Validated {len(valid_iterations)}/{len(validated_iterations)} iterations "
                   f"with avg confidence {avg_confidence:.3f}, "
                   f"avg pattern ratio {avg_pattern_ratio:.3f}, "
                   f"avg layers {avg_layers:.1f}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save Enhanced Stage 2 results to file with numpy type conversion"""
        # Convert numpy types to native Python types
        results_converted = convert_numpy_types(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2)
        
        logger.info(f"Enhanced Stage 2 results saved to {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python stage2_enhanced_operator_fingerprint_validator.py <trace_file> [stage1_results_file] [output_file]")
        print("Example: python stage2_enhanced_operator_fingerprint_validator.py trace.json stage1_results.json stage2_enhanced_results.json")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    
    # Fix: Use the same input file for both kernel extraction and validation
    # The trace_file parameter is actually the stage1_results.json file
    stage1_results_file = trace_file  # Use the same file
    
    # Generate output file path in the same directory as input
    import os
    input_dir = os.path.dirname(trace_file)
    output_file = os.path.join(input_dir, "stage2_results.json")
    
    # Override with command line arguments if provided
    if len(sys.argv) > 2:
        stage1_results_file = sys.argv[2]
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    try:
        validator = EnhancedOperatorFingerprintValidator()
        
        # Step 1: Extract distinct kernels and save to CSV
        kernel_counts = validator.extract_kernels_with_frequency(trace_file)
        csv_output = trace_file.replace('.json', '_distinct_kernels_enhanced.csv')
        validator.save_kernels_with_frequency_csv(kernel_counts, csv_output)
        
        # Step 2: Enhanced validation with all three optimizations
        results = validator.validate_stage1_candidates(stage1_results_file)
        validator.save_results(results, output_file)
        
        # Print enhanced summary
        metadata = results['metadata']
        optimizations = metadata['optimizations_enabled']
        
        print(f"\n🚀 Enhanced Stage 2: Operator Fingerprint Validation Summary:")
        print(f"📊 Distinct kernels found: {len(kernel_counts)}")
        print(f"📁 Kernels CSV saved to: {csv_output}")
        print(f"🎯 Total candidate iterations: {metadata['total_candidates']}")
        print(f"✅ Valid fingerprint iterations: {metadata['valid_fingerprint_iterations']}")
        print(f"📈 Average confidence score: {metadata['average_confidence_score']:.3f}")
        print(f"🔄 Average pattern ratio: {metadata['average_pattern_ratio']:.3f}")
        print(f"🏗️  Average estimated layers: {metadata['average_estimated_layers']:.1f}")
        print(f"🔧 Global operator types: {len(metadata['global_operator_counts'])}")
        print(f"🔍 Execution fingerprint: {metadata['execution_fingerprint']}")
        print(f"\n🎨 Enhanced optimizations enabled:")
        print(f"  ⚡ Dynamic thresholds: {optimizations['dynamic_thresholds']}")
        print(f"  🏗️  Structural awareness: {optimizations['structural_awareness']}")
        print(f"  🔄 Sequence pattern recognition: {optimizations['sequence_pattern_recognition']}")
        print(f"💾 Results saved to: {output_file}")
        print(f"\n🎯 Ready for Stage 3: Statistical validation")
        
    except Exception as e:
        logger.error(f"Error during Enhanced Stage 2 processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()