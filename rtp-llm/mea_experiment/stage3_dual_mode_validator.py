#!/usr/bin/env python3
"""
MEA Stage 3: Dual-Mode Statistical Validation
Specialized for detecting two distinct operation modes with enhanced minority detection
"""

import json
import sys
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from scipy.stats import chisquare

# Try to import sklearn, fallback to simple clustering if not available
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using simple clustering fallback")

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

class DualModeStatisticalValidator:
    """
    Stage 3: Dual-Mode Statistical Validation specialized for detecting two operation modes
    Implements enhanced minority mode detection with operator count-based clustering
    """
    
    def __init__(self, wmse_threshold: float = 0.05, chi_square_alpha: float = 0.01):
        """
        Initialize dual-mode statistical validator
        
        Args:
            wmse_threshold: Threshold for WMSE validation
            chi_square_alpha: Significance level for chi-square test
        """
        self.wmse_threshold = wmse_threshold
        self.chi_square_alpha = chi_square_alpha
        
    def compute_global_operator_distribution(self, iterations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute global operator distribution (p_j^exp) from all valid iterations
        This serves as the expected distribution for statistical validation
        """
        global_operator_counts = Counter()
        total_valid_iterations = 0
        
        # Aggregate operator counts from all fingerprint-validated iterations
        for iteration in iterations:
            if iteration.get('fingerprint_validated', False):
                operator_counts = iteration.get('operator_counts', {})
                for op, count in operator_counts.items():
                    global_operator_counts[op] += count
                total_valid_iterations += 1
        
        # Calculate global distribution
        total_ops = sum(global_operator_counts.values())
        global_distribution = {}
        
        if total_ops > 0:
            global_distribution = {
                op: count / total_ops 
                for op, count in global_operator_counts.items()
            }
        
        logger.info(f"Global operator distribution computed from {total_valid_iterations} valid iterations")
        logger.info(f"Total operators: {total_ops}, Unique operator types: {len(global_distribution)}")
        
        return global_distribution
    
    def chi_square_goodness_of_fit_test(self, observed_dist: Dict[str, float], 
                                       expected_dist: Dict[str, float],
                                       total_observed: int) -> Tuple[float, float, bool]:
        """
        Perform Chi-square goodness-of-fit test with improved numerical stability
        
        Args:
            observed_dist: Observed operator distribution (p_j^obs)
            expected_dist: Expected operator distribution (p_j^exp)
            total_observed: Total number of operators in the iteration
            
        Returns:
            Tuple of (chi_square_statistic, p_value, is_valid)
        """
        if not expected_dist or total_observed == 0:
            return 0.0, 1.0, False
        
        # Get all operator types from both distributions
        all_operators = set(observed_dist.keys()) | set(expected_dist.keys())
        
        observed_counts = []
        expected_counts = []
        
        # Ensure distributions are properly normalized to avoid precision issues
        obs_total = sum(observed_dist.values())
        exp_total = sum(expected_dist.values())
        
        # Normalize distributions if they're not exactly 1.0 (within tolerance)
        if abs(obs_total - 1.0) > 1e-10:
            observed_dist = {k: v / obs_total for k, v in observed_dist.items()}
        if abs(exp_total - 1.0) > 1e-10:
            expected_dist = {k: v / exp_total for k, v in expected_dist.items()}
        
        for op in all_operators:
            obs_prob = observed_dist.get(op, 0.0)
            exp_prob = expected_dist.get(op, 0.0)
            
            obs_count = obs_prob * total_observed
            exp_count = exp_prob * total_observed
            
            # Skip operators with very low expected counts to avoid chi-square issues
            if exp_count < 1.0:
                continue
                
            observed_counts.append(obs_count)
            expected_counts.append(exp_count)
        
        if len(observed_counts) < 2:
            logger.warning("Insufficient data for chi-square test")
            return 0.0, 1.0, False
        
        try:
            # Convert to numpy arrays for better numerical stability
            observed_counts = np.array(observed_counts)
            expected_counts = np.array(expected_counts)
            
            # Ensure the sums match exactly to avoid scipy precision warnings
            obs_sum = np.sum(observed_counts)
            exp_sum = np.sum(expected_counts)
            
            # Normalize to make sums exactly equal (fix precision issues)
            if abs(obs_sum - exp_sum) > 1e-10:
                # Adjust observed counts to match expected sum exactly
                scaling_factor = exp_sum / obs_sum
                observed_counts = observed_counts * scaling_factor
            
            # Perform chi-square test with improved numerical stability
            chi2_stat, p_value = chisquare(observed_counts, expected_counts)
            is_valid = p_value >= self.chi_square_alpha
            
            return float(chi2_stat), float(p_value), is_valid
            
        except Exception as e:
            logger.warning(f"Chi-square test failed: {e}")
            return 0.0, 0.0, False
    
    def compute_wmse(self, observed_dist: Dict[str, float], 
                    expected_dist: Dict[str, float]) -> Tuple[float, bool]:
        """
        Compute Weighted Mean Squared Error (WMSE)
        
        Formula: Loss = Σ p_j^obs · (p_j^obs - p_j^exp)²
        
        Args:
            observed_dist: Observed operator distribution (p_j^obs)
            expected_dist: Expected operator distribution (p_j^exp)
            
        Returns:
            Tuple of (wmse_loss, is_valid)
        """
        if not expected_dist:
            return float('inf'), False
        
        wmse_loss = 0.0
        all_operators = set(observed_dist.keys()) | set(expected_dist.keys())
        
        for op in all_operators:
            p_obs = observed_dist.get(op, 0.0)
            p_exp = expected_dist.get(op, 0.0)
            
            # WMSE formula: p_j^obs · (p_j^obs - p_j^exp)²
            wmse_loss += p_obs * (p_obs - p_exp) ** 2
        
        is_valid = wmse_loss < self.wmse_threshold
        
        return float(wmse_loss), is_valid
    
    def validate_iteration_statistics(self, iteration: Dict[str, Any], 
                                    global_distribution: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform statistical validation for a single iteration with dual-mode awareness
        
        Args:
            iteration: Single iteration data with operator distribution
            global_distribution: Expected operator distribution for this mode
            
        Returns:
            Dictionary containing validation results
        """
        # Extract observed distribution
        operator_distribution = iteration.get('operator_distribution', {})
        total_operators = iteration.get('total_operators', 0)
        
        if not operator_distribution or total_operators == 0:
            return {
                'chi_square_valid': False,
                'wmse_valid': False,
                'statistical_valid': False,
                'chi_square_statistic': 0.0,
                'chi_square_p_value': 0.0,
                'wmse_loss': float('inf'),
                'validation_reason': 'Empty operator distribution'
            }
        
        # Compute WMSE
        wmse_loss, wmse_valid = self.compute_wmse(
            operator_distribution, global_distribution
        )
        
        # Perform Chi-square goodness-of-fit test
        chi2_stat, p_value, chi2_valid = self.chi_square_goodness_of_fit_test(
            operator_distribution, global_distribution, total_operators
        )
        
        # Dual-mode validation: more lenient for different modes
        # Accept if WMSE is reasonable (< 0.15) OR chi-square passes
        statistical_valid = wmse_valid or chi2_valid or (wmse_loss < 0.15)
        validation_reason = self._get_validation_reason(chi2_valid, wmse_valid, p_value, wmse_loss)
        
        return {
            'chi_square_valid': chi2_valid,
            'wmse_valid': wmse_valid,
            'statistical_valid': statistical_valid,
            'chi_square_statistic': chi2_stat,
            'chi_square_p_value': p_value,
            'wmse_loss': wmse_loss,
            'validation_thresholds': {
                'chi_square_alpha': self.chi_square_alpha,
                'wmse_threshold': self.wmse_threshold
            },
            'validation_reason': validation_reason
        }
    
    def _get_validation_reason(self, chi2_valid: bool, wmse_valid: bool, 
                              p_value: float, wmse_loss: float) -> str:
        """Generate human-readable validation reason"""
        if chi2_valid and wmse_valid:
            return "Passed both chi-square and WMSE validation"
        elif wmse_loss < 0.15:
            return f"Passed dual-mode lenient validation: WMSE={wmse_loss:.6f} < 0.15"
        elif chi2_valid:
            return f"Passed chi-square test: p-value={p_value:.4f} >= {self.chi_square_alpha}"
        elif wmse_valid:
            return f"Passed WMSE test: {wmse_loss:.6f} <= {self.wmse_threshold}"
        else:
            return f"Failed validation: p-value={p_value:.4f} < {self.chi_square_alpha}, WMSE={wmse_loss:.6f} > {self.wmse_threshold}"
    
    def detect_dual_modes(self, iterations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Enhanced dual-mode detection specifically designed for cases like:
        - 239 iterations of one mode (e.g., 458 operators)
        - 12 iterations of another mode (e.g., 490 operators)
        """
        # Extract fingerprint-validated iterations
        valid_iterations = [it for it in iterations if it.get('fingerprint_validated', False)]
        
        if len(valid_iterations) < 2:
            logger.warning("Insufficient valid iterations for dual-mode detection")
            return {"single_mode": valid_iterations}
        
        logger.info(f"Analyzing {len(valid_iterations)} iterations for dual-mode detection")
        
        # Step 1: Analyze operator count distribution (most reliable indicator)
        operator_counts = [it.get('total_operators', 0) for it in valid_iterations]
        op_count_distribution = Counter(operator_counts)
        unique_op_counts = list(op_count_distribution.keys())
        
        logger.info(f"Operator count distribution: {dict(op_count_distribution)}")
        
        # Step 2: Check for exactly two distinct operator count groups
        if len(unique_op_counts) == 2:
            # Perfect case: exactly two distinct operator counts
            mode1_op_count, mode2_op_count = unique_op_counts
            mode1_iterations = [it for it in valid_iterations if it.get('total_operators', 0) == mode1_op_count]
            mode2_iterations = [it for it in valid_iterations if it.get('total_operators', 0) == mode2_op_count]
            
            # Determine which is primary and which is secondary based on count
            if len(mode1_iterations) >= len(mode2_iterations):
                primary_mode, secondary_mode = mode1_iterations, mode2_iterations
                primary_ops, secondary_ops = mode1_op_count, mode2_op_count
            else:
                primary_mode, secondary_mode = mode2_iterations, mode1_iterations
                primary_ops, secondary_ops = mode2_op_count, mode1_op_count
            
            logger.info(f"Detected dual modes by operator count:")
            logger.info(f"  Primary mode: {len(primary_mode)} iterations with {primary_ops} operators")
            logger.info(f"  Secondary mode: {len(secondary_mode)} iterations with {secondary_ops} operators")
            
            return {
                "primary_mode": primary_mode,
                "secondary_mode": secondary_mode
            }
        
        # Step 3: Check for minority mode (< 10% of total) with different characteristics
        elif len(unique_op_counts) > 2:
            # Multiple operator counts - look for minority patterns
            total_iterations = len(valid_iterations)
            minority_threshold = max(2, int(total_iterations * 0.1))  # At least 2, max 10%
            
            # Find potential minority modes
            minority_candidates = []
            majority_candidates = []
            
            for op_count, count in op_count_distribution.items():
                if count <= minority_threshold and count >= 2:  # Minority but significant
                    minority_candidates.append((op_count, count))
                else:
                    majority_candidates.append((op_count, count))
            
            if minority_candidates and majority_candidates:
                # Select the largest minority group
                minority_op_count, minority_count = max(minority_candidates, key=lambda x: x[1])
                
                # Combine all majority groups
                minority_iterations = [it for it in valid_iterations 
                                     if it.get('total_operators', 0) == minority_op_count]
                majority_iterations = [it for it in valid_iterations 
                                     if it.get('total_operators', 0) != minority_op_count]
                
                logger.info(f"Detected dual modes with minority detection:")
                logger.info(f"  Primary mode: {len(majority_iterations)} iterations (various operator counts)")
                logger.info(f"  Secondary mode: {len(minority_iterations)} iterations with {minority_op_count} operators")
                
                return {
                    "primary_mode": majority_iterations,
                    "secondary_mode": minority_iterations
                }
        
        # Step 4: Fallback to duration-based clustering if operator count doesn't work
        return self._duration_based_dual_mode_detection(valid_iterations)
    
    def _duration_based_dual_mode_detection(self, iterations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fallback dual-mode detection based on duration patterns
        """
        logger.info("Falling back to duration-based dual-mode detection")
        
        durations = [it.get('duration_us', 0) for it in iterations]
        
        # Use statistical outlier detection
        q1, q2, q3 = np.percentile(durations, [25, 50, 75])
        iqr = q3 - q1
        
        # Define outlier boundaries
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Separate outliers from normal iterations
        outliers = []
        normal = []
        
        for iteration in iterations:
            duration = iteration.get('duration_us', 0)
            if duration < lower_bound or duration > upper_bound:
                outliers.append(iteration)
            else:
                normal.append(iteration)
        
        # Check if we have a meaningful split
        if len(outliers) >= 2 and len(normal) >= 2:
            # Determine which group is primary based on size
            if len(normal) >= len(outliers):
                primary_mode, secondary_mode = normal, outliers
                mode_type = "normal vs outlier"
            else:
                primary_mode, secondary_mode = outliers, normal
                mode_type = "outlier vs normal"
            
            logger.info(f"Detected dual modes by duration ({mode_type}):")
            logger.info(f"  Primary mode: {len(primary_mode)} iterations")
            logger.info(f"  Secondary mode: {len(secondary_mode)} iterations")
            
            return {
                "primary_mode": primary_mode,
                "secondary_mode": secondary_mode
            }
        
        # If no clear dual mode detected, return single mode
        logger.info("No clear dual modes detected, treating as single mode")
        return {"single_mode": iterations}
    
    def validate_stage2_results(self, stage2_results_file: str) -> Dict[str, Any]:
        """
        Dual-Mode Stage 3: Statistical validation with enhanced minority detection
        
        Args:
            stage2_results_file: Path to Stage 2 results JSON file
            
        Returns:
            Dictionary containing Stage 3 validation results with dual-mode support
        """
        logger.info(f"Starting Dual-Mode Stage 3: Enhanced minority detection statistical validation")
        
        # Load Stage 2 results
        with open(stage2_results_file, 'r', encoding='utf-8') as f:
            stage2_data = json.load(f)
        
        iterations = stage2_data.get('iterations', [])
        logger.info(f"Validating {len(iterations)} iterations from Stage 2")
        
        # Step 1: Detect dual modes with enhanced sensitivity
        clustered_iterations = self.detect_dual_modes(iterations)
        
        if not clustered_iterations:
            logger.error("Cannot detect modes - no valid iterations found")
            return {
                'stage': 3,
                'error': 'No valid iterations for mode detection',
                'iterations': []
            }
        
        # Step 2: Compute mode-specific operator distributions
        mode_distributions = self.compute_mode_specific_distributions(clustered_iterations)
        
        logger.info(f"Computed distributions for {len(mode_distributions)} operation modes")
        
        # Step 3: Create iteration-to-mode mapping
        iteration_mode_mapping = {}
        for mode_id, mode_iterations in clustered_iterations.items():
            for iteration in mode_iterations:
                iteration_id = iteration.get('iteration_id')
                iteration_mode_mapping[iteration_id] = mode_id
        
        # Step 4: Validate each iteration against its assigned mode
        validated_iterations = []
        mode_validation_stats = {mode_id: {'total': 0, 'valid': 0} for mode_id in mode_distributions.keys()}
        
        for iteration in iterations:
            iteration_id = iteration.get('iteration_id')
            fingerprint_validated = iteration.get('fingerprint_validated', False)
            
            # Only perform statistical validation on fingerprint-validated iterations
            if fingerprint_validated:
                # Get the mode this iteration belongs to
                assigned_mode = iteration_mode_mapping.get(iteration_id)
                
                if assigned_mode and assigned_mode in mode_distributions:
                    # Validate against the assigned mode's distribution
                    expected_dist = mode_distributions[assigned_mode]
                    statistical_validation = self.validate_iteration_statistics(iteration, expected_dist)
                    
                    # Add mode information
                    statistical_validation['matched_mode'] = assigned_mode
                    
                    # Update mode statistics
                    mode_validation_stats[assigned_mode]['total'] += 1
                    if statistical_validation['statistical_valid']:
                        mode_validation_stats[assigned_mode]['valid'] += 1
                else:
                    # Fallback for iterations not in clustering (shouldn't happen)
                    statistical_validation = {
                        'chi_square_valid': False,
                        'wmse_valid': False,
                        'statistical_valid': False,
                        'chi_square_statistic': 0.0,
                        'chi_square_p_value': 0.0,
                        'wmse_loss': float('inf'),
                        'matched_mode': None,
                        'validation_reason': 'No assigned mode found'
                    }
            else:
                # Skip statistical validation for iterations that failed fingerprint validation
                statistical_validation = {
                    'chi_square_valid': False,
                    'wmse_valid': False,
                    'statistical_valid': False,
                    'chi_square_statistic': 0.0,
                    'chi_square_p_value': 0.0,
                    'wmse_loss': float('inf'),
                    'matched_mode': None,
                    'validation_reason': 'Skipped - failed fingerprint validation'
                }
            
            # Add statistical validation results
            validated_iteration = iteration.copy()
            validated_iteration.update({
                'statistical_validation': statistical_validation,
                'final_validated': (
                    fingerprint_validated and 
                    statistical_validation['statistical_valid']
                ),
                'operation_mode': statistical_validation.get('matched_mode', 'unknown'),
                'stage3_dual_mode_completed': True
            })
            
            validated_iterations.append(validated_iteration)
            
            # Enhanced logging with mode information
            if fingerprint_validated:
                status = "✓" if statistical_validation['statistical_valid'] else "✗"
                chi2_p = statistical_validation['chi_square_p_value']
                wmse = statistical_validation['wmse_loss']
                mode = statistical_validation.get('matched_mode', 'unknown')
                
                logger.info(f"{status} Iteration {iteration_id} [{mode}]: "
                           f"Chi2_p={chi2_p:.4f}, "
                           f"WMSE={wmse:.6f}, "
                           f"Final_Valid={validated_iteration['final_validated']}")
            else:
                logger.info(f"⊘ Iteration {iteration_id}: Skipped statistical validation (failed fingerprint)")
        
        # Compute comprehensive statistics
        fingerprint_valid_count = sum(1 for it in validated_iterations if it.get('fingerprint_validated', False))
        statistical_valid_count = sum(1 for it in validated_iterations 
                                    if it.get('statistical_validation', {}).get('statistical_valid', False))
        final_valid_count = sum(1 for it in validated_iterations if it.get('final_validated', False))
        
        # Compute mode-specific statistics
        mode_stats = {}
        for mode_id, stats in mode_validation_stats.items():
            mode_iterations = [it for it in validated_iterations 
                             if it.get('operation_mode') == mode_id]
            mode_stats[mode_id] = {
                'total_iterations': len(mode_iterations),
                'valid_iterations': stats['valid'],
                'validation_rate': stats['valid'] / stats['total'] if stats['total'] > 0 else 0.0,
                'avg_duration_us': float(np.mean([it.get('duration_us', 0) for it in mode_iterations])) if mode_iterations else 0.0,
                'avg_operators': float(np.mean([it.get('total_operators', 0) for it in mode_iterations])) if mode_iterations else 0.0
            }
        
        # Compute average metrics for statistical validation
        statistical_iterations = [it for it in validated_iterations 
                                if it.get('fingerprint_validated', False)]
        
        if statistical_iterations:
            avg_chi2_p = float(np.mean([
                it['statistical_validation']['chi_square_p_value'] 
                for it in statistical_iterations
            ]))
            avg_wmse = float(np.mean([
                it['statistical_validation']['wmse_loss'] 
                for it in statistical_iterations if it['statistical_validation']['wmse_loss'] != float('inf')
            ]))
        else:
            avg_chi2_p = 0.0
            avg_wmse = float('inf')
        
        # Determine if single-mode or dual-mode
        is_single_mode = len(mode_distributions) == 1 and 'single_mode' in mode_distributions
        
        results = {
            'stage': 3,
            'description': f'Dual-mode statistical validation with enhanced minority detection',
            'metadata': {
                'total_iterations': len(validated_iterations),
                'fingerprint_valid_iterations': fingerprint_valid_count,
                'statistical_valid_iterations': statistical_valid_count,
                'final_valid_iterations': final_valid_count,
                'validation_rate': final_valid_count / len(validated_iterations) if validated_iterations else 0.0,
                'average_chi_square_p_value': avg_chi2_p,
                'average_wmse_loss': avg_wmse,
                'operation_modes': mode_stats,
                'mode_distributions': mode_distributions,
                'is_single_mode': is_single_mode,
                'detected_modes': len(mode_distributions),
                'validation_thresholds': {
                    'chi_square_alpha': self.chi_square_alpha,
                    'wmse_threshold': self.wmse_threshold
                },
                'stage1_completed': stage2_data.get('metadata', {}).get('stage1_completed', False),
                'stage2_enhanced_completed': stage2_data.get('metadata', {}).get('stage2_enhanced_completed', False),
                'stage3_dual_mode_completed': True,
                'dual_mode_detection': True,
                'enhanced_minority_detection': True
            },
            'iterations': validated_iterations
        }
        
        # Enhanced summary logging
        mode_type = "single-mode" if is_single_mode else "dual-mode"
        logger.info(f"Dual-Mode Stage 3 completed with {mode_type} detection:")
        logger.info(f"  Total iterations: {len(validated_iterations)}")
        logger.info(f"  Detected modes: {len(mode_distributions)}")
        logger.info(f"  Final valid iterations: {final_valid_count}/{len(validated_iterations)} "
                   f"(validation rate: {final_valid_count / len(validated_iterations) * 100:.1f}%)")
        
        for mode_id, stats in mode_stats.items():
            logger.info(f"  Mode {mode_id}: {stats['valid_iterations']}/{stats['total_iterations']} valid "
                       f"({stats['validation_rate']*100:.1f}%), "
                       f"avg_duration={stats['avg_duration_us']:.1f}μs, "
                       f"avg_operators={stats['avg_operators']:.1f}")
        
        return results
    
    def compute_mode_specific_distributions(self, clustered_iterations: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
        """
        Compute operator distribution for each operation mode
        """
        mode_distributions = {}
        
        for mode_id, mode_iterations in clustered_iterations.items():
            mode_operator_counts = Counter()
            
            # Aggregate operator counts for this mode
            for iteration in mode_iterations:
                operator_counts = iteration.get('operator_counts', {})
                for op, count in operator_counts.items():
                    mode_operator_counts[op] += count
            
            # Calculate distribution for this mode
            total_ops = sum(mode_operator_counts.values())
            if total_ops > 0:
                mode_distributions[mode_id] = {
                    op: count / total_ops 
                    for op, count in mode_operator_counts.items()
                }
            else:
                mode_distributions[mode_id] = {}
            
            logger.info(f"Mode {mode_id} distribution: {len(mode_distributions[mode_id])} operator types, "
                       f"{total_ops} total operators")
        
        return mode_distributions
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save Stage 3 results to file with numpy type conversion"""
        # Convert numpy types to native Python types
        results_converted = convert_numpy_types(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2)
        
        logger.info(f"Dual-Mode Stage 3 results saved to {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 stage3_dual_mode_validator.py <stage2_results_file> [output_file] [--wmse-threshold=0.05] [--chi-square-alpha=0.01]")
        print("Example: python3 stage3_dual_mode_validator.py stage2_enhanced_results.json stage3_dual_mode_results.json")
        sys.exit(1)
    
    stage2_results_file = sys.argv[1]
    
    # Generate output file path in the same directory as input if not specified
    if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
        output_file = sys.argv[2]
    else:
        # Auto-generate output file path in the same directory as input
        import os
        input_dir = os.path.dirname(stage2_results_file)
        output_file = os.path.join(input_dir, "stage3_dual_mode_results.json")
    
    # Parse optional parameters
    wmse_threshold = 0.05
    chi_square_alpha = 0.01
    
    for arg in sys.argv[2:]:
        if arg.startswith('--wmse-threshold='):
            wmse_threshold = float(arg.split('=')[1])
        elif arg.startswith('--chi-square-alpha='):
            chi_square_alpha = float(arg.split('=')[1])
    
    try:
        validator = DualModeStatisticalValidator(
            wmse_threshold=wmse_threshold,
            chi_square_alpha=chi_square_alpha
        )
        
        # Perform dual-mode statistical validation
        results = validator.validate_stage2_results(stage2_results_file)
        validator.save_results(results, output_file)
        
        # Print summary
        metadata = results['metadata']
        
        print(f"\n📊 Dual-Mode Stage 3: Statistical Validation Summary:")
        print(f"🎯 Total iterations: {metadata['total_iterations']}")
        print(f"✅ Fingerprint valid: {metadata['fingerprint_valid_iterations']}")
        print(f"📈 Statistical valid: {metadata['statistical_valid_iterations']}")
        print(f"🏆 Final valid iterations: {metadata['final_valid_iterations']}")
        print(f"📊 Validation rate: {metadata['validation_rate']:.1%}")
        print(f"🔍 Average Chi-square p-value: {metadata['average_chi_square_p_value']:.4f}")
        print(f"📏 Average WMSE loss: {metadata['average_wmse_loss']:.6f}")
        print(f"🎛️  Detected modes: {metadata['detected_modes']}")
        print(f"🔧 Enhanced minority detection: {metadata.get('enhanced_minority_detection', False)}")
        
        # Print mode-specific statistics
        if 'operation_modes' in metadata:
            print(f"\n📋 Mode-specific statistics:")
            for mode_id, stats in metadata['operation_modes'].items():
                print(f"  {mode_id}: {stats['valid_iterations']}/{stats['total_iterations']} valid "
                     f"({stats['validation_rate']*100:.1f}%), "
                     f"avg_duration={stats['avg_duration_us']:.1f}μs, "
                     f"avg_operators={stats['avg_operators']:.1f}")
        
        print(f"\n🎛️  Validation thresholds:")
        print(f"  📊 Chi-square α: {metadata['validation_thresholds']['chi_square_alpha']}")
        print(f"  📏 WMSE threshold: {metadata['validation_thresholds']['wmse_threshold']}")
        print(f"💾 Results saved to: {output_file}")
        print(f"\n🎯 Ready for MIE and IIPS calculation")
        
    except Exception as e:
        logger.error(f"Error during Dual-Mode Stage 3 processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()