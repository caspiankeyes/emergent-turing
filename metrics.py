# emergent_turing/metrics.py

import numpy as np
from typing import Dict, List, Any, Optional, Union
import re

class BaseMetric:
    """Base class for all Emergent Turing Test metrics."""
    
    def __init__(self):
        """Initialize the metric."""
        pass
    
    def compute(self, test_result: Dict[str, Any]) -> float:
        """
        Compute the metric from a test result.
        
        Args:
            test_result: Test result dictionary
            
        Returns:
            Computed metric value
        """
        raise NotImplementedError("Subclasses must implement compute().")
    
    def aggregate(self, values: List[float]) -> Dict[str, float]:
        """
        Aggregate multiple metric values.
        
        Args:
            values: List of metric values
            
        Returns:
            Dictionary of aggregated statistics
        """
        if not values:
            return {
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
                "std": None
            }
            
        return {
            "mean": np.mean(values),
            "median": np.median(values),
            "min": np.min(values),
            "max": np.max(values),
            "std": np.std(values)
        }


class NullRatio(BaseMetric):
    """
    NullRatio metric measures the degree of output nullification.
    
    This metric quantifies how much of the model's response represents
    a nullification or refusal to generate content. Values range from
    0.0 (no nullification) to 1.0 (complete nullification).
    """
    
    def __init__(self):
        """Initialize the NullRatio metric."""
        super().__init__()
    
    def compute(self, test_result: Dict[str, Any]) -> float:
        """
        Compute the null ratio from a test result.
        
        Args:
            test_result: Test result dictionary
            
        Returns:
            Null ratio value (0.0-1.0)
        """
        # If null_ratio is already computed, return it
        if "null_ratio" in test_result:
            return test_result["null_ratio"]
        
        # Otherwise, compute it from the output
        output = test_result.get("output", "")
        
        if not output.strip():
            return 1.0  # Complete nullification
            
        # Check for refusal or nullification patterns
        null_markers = [
"I cannot", "I'm unable to", "I apologize", "I don't have",
            "I'm not able to", "I cannot provide", "I'm sorry",
            "I don't think I can", "I must decline"
        ]
        
        for marker in null_markers:
            if marker.lower() in output.lower():
                # Calculate the portion of the output that is nullified
                marker_pos = output.lower().find(marker.lower())
                if marker_pos <= len(output) * 0.2:  # If nullification starts early in the response
                    return 0.8  # High nullification
                else:
                    # Partial nullification
                    return 0.5
        
        # Check for hesitation or uncertainty markers
        uncertainty_markers = [
            "I'm not sure", "It's unclear", "I don't know if",
            "This is difficult to", "I'm uncertain", "It's hard to say"
        ]
        
        for marker in uncertainty_markers:
            if marker.lower() in output.lower():
                return 0.3  # Partial uncertainty
        
        return 0.0  # No nullification detected


class HesitationDepth(BaseMetric):
    """
    HesitationDepth metric measures the degree of token-level hesitation.
    
    This metric quantifies how much the model hesitates or regenerates tokens
    during response generation. Values range from 0.0 (no hesitation) to
    higher values indicating more severe hesitation.
    """
    
    def __init__(self):
        """Initialize the HesitationDepth metric."""
        super().__init__()
    
    def compute(self, test_result: Dict[str, Any]) -> Optional[float]:
        """
        Compute the hesitation depth from a test result.
        
        Args:
            test_result: Test result dictionary
            
        Returns:
            Hesitation depth value or None if hesitation data is not available
        """
        # Check if hesitation map is available
        hesitation_map = test_result.get("hesitation_map")
        if not hesitation_map:
            return None
        
        # Extract key hesitation signals
        regeneration_count = hesitation_map.get("regeneration_count", [])
        regeneration_positions = hesitation_map.get("regeneration_positions", [])
        pause_duration = hesitation_map.get("pause_duration", [])
        pause_positions = hesitation_map.get("pause_positions", [])
        
        # Calculate regeneration factor
        if regeneration_count:
            regeneration_factor = sum(regeneration_count) / len(regeneration_count)
        else:
            regeneration_factor = 0.0
        
        # Calculate pause factor
        if pause_duration:
            pause_factor = sum(pause_duration) / len(pause_duration)
        else:
            pause_factor = 0.0
        
        # Calculate position clustering factor
        # If hesitations are clustered, it indicates deeper hesitation at specific points
        position_clustering = 0.0
        
        if regeneration_positions and len(regeneration_positions) > 1:
            # Calculate average distance between regeneration positions
            distances = [abs(regeneration_positions[i] - regeneration_positions[i-1]) for i in range(1, len(regeneration_positions))]
            avg_distance = sum(distances) / len(distances)
            
            # Normalize by output length
            output_length = len(test_result.get("output", ""))
            if output_length > 0:
                position_clustering = 1.0 - (avg_distance / output_length)
        
        # Combine factors (weighted sum)
        # Regenerations are stronger indicators of hesitation than pauses
        hesitation_depth = (
            regeneration_factor * 0.6 +
            pause_factor * 0.3 +
            position_clustering * 0.1
        )
        
        return hesitation_depth


class AttributionTrace(BaseMetric):
    """
    AttributionTrace metric measures the clarity and coherence of attribution paths.
    
    This metric quantifies how clearly the model traces information sources
    and reasoning paths during response generation. Values range from 0.0
    (poor attribution) to 1.0 (clear attribution).
    """
    
    def __init__(self):
        """Initialize the AttributionTrace metric."""
        super().__init__()
    
    def compute(self, test_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Compute the attribution trace metrics from a test result.
        
        Args:
            test_result: Test result dictionary
            
        Returns:
            Attribution trace metrics or None if attribution data is not available
        """
        # Check if attribution trace is available
        attribution_trace = test_result.get("attribution_trace")
        if not attribution_trace:
            return None
        
        # Return the attribution trace as is
        # In a more sophisticated implementation, this would process the trace
        # to extract higher-level metrics
        return attribution_trace


class DriftCoherence(BaseMetric):
    """
    DriftCoherence metric measures the coherence of cognitive drift patterns.
    
    This metric quantifies how structured or chaotic cognitive drift patterns
    are during hesitation or failure. Values range from 0.0 (chaotic drift)
    to 1.0 (coherent drift).
    """
    
    def __init__(self):
        """Initialize the DriftCoherence metric."""
        super().__init__()
    
    def compute(self, test_result: Dict[str, Any]) -> Optional[float]:
        """
        Compute the drift coherence from a test result.
        
        Args:
            test_result: Test result dictionary
            
        Returns:
            Drift coherence value or None if required data is not available
        """
        # This metric requires both hesitation data and attribution data
        hesitation_map = test_result.get("hesitation_map")
        attribution_trace = test_result.get("attribution_trace")
        
        if not hesitation_map or not attribution_trace:
            return None
        
        # Extract key signals
        regeneration_positions = hesitation_map.get("regeneration_positions", [])
        pause_positions = hesitation_map.get("pause_positions", [])
        
        # Extract attribution edges
        edges = attribution_trace.get("edges", [])
        
        # If there are no hesitations or attribution edges, return None
        if not (regeneration_positions or pause_positions) or not edges:
            return None
        
        # Calculate coherence based on alignment between hesitations and attribution boundaries
        coherence_score = 0.0
        
        # Convert edges to position boundaries
        # This is a simplified approximation - in a real implementation, we would
        # map edges to actual token positions
        edge_positions = []
        for edge in edges:
            # Extract edge endpoints
            if isinstance(edge, list) and len(edge) >= 2:
                source, target = edge[0], edge[1]
            elif isinstance(edge, dict) and "source" in edge and "target" in edge:
                source, target = edge["source"], edge["target"]
            else:
                continue
            
            # Extract position from node name if possible
            source_match = re.search(r'(\d+)', source)
            if source_match:
                edge_positions.append(int(source_match.group(1)) * 10)  # Scale for approximation
            
            target_match = re.search(r'(\d+)', target)
            if target_match:
                edge_positions.append(int(target_match.group(1)) * 10)  # Scale for approximation
        
        # Calculate alignment between hesitations and attribution boundaries
        all_hesitation_positions = regeneration_positions + pause_positions
        
        if not all_hesitation_positions or not edge_positions:
            return 0.5  # Default moderate coherence if we can't calculate
        
        # For each hesitation position, find the distance to the nearest edge position
        min_distances = []
        for pos in all_hesitation_positions:
            min_distance = min(abs(pos - edge_pos) for edge_pos in edge_positions)
            min_distances.append(min_distance)
        
        # Calculate average minimum distance
        avg_min_distance = sum(min_distances) / len(min_distances)
        
        # Normalize by output length and convert to coherence score
        output_length = len(test_result.get("output", ""))
        if output_length > 0:
            normalized_distance = avg_min_distance / output_length
            coherence_score = max(0.0, 1.0 - normalized_distance)
        
        return coherence_score


class OscillationFrequency(BaseMetric):
    """
    OscillationFrequency metric measures token regeneration oscillations.
    
    This metric quantifies how frequently the model oscillates between
    different completions during generation. Values represent the frequency
    of oscillation events.
    """
    
    def __init__(self):
        """Initialize the OscillationFrequency metric."""
        super().__init__()
    
    def compute(self, test_result: Dict[str, Any]) -> Optional[float]:
        """
        Compute the oscillation frequency from a test result.
        
        Args:
            test_result: Test result dictionary
            
        Returns:
            Oscillation frequency value or None if required data is not available
        """
        # This metric requires regeneration attempts
        regeneration_attempts = test_result.get("regeneration_attempts", [])
        
        if len(regeneration_attempts) <= 1:
            return 0.0  # No oscillation with 0 or 1 attempts
        
        # Calculate oscillations by comparing consecutive regeneration attempts
        oscillations = 0
        for i in range(1, len(regeneration_attempts)):
            prev_attempt = regeneration_attempts[i-1]
            curr_attempt = regeneration_attempts[i]
            
            # Find the first point of divergence
            divergence_idx = -1
            min_len = min(len(prev_attempt), len(curr_attempt))
            
            for j in range(min_len):
                if prev_attempt[j] != curr_attempt[j]:
                    divergence_idx = j
                    break
            
            if divergence_idx == -1 and len(prev_attempt) != len(curr_attempt):
                divergence_idx = min_len
            
            # If there was a divergence, count it as an oscillation
            if divergence_idx != -1:
                oscillations += 1
        
        # Normalize by the number of regeneration attempts
        oscillation_frequency = oscillations / (len(regeneration_attempts) - 1)
        
        return oscillation_frequency


class DriftAmplitude(BaseMetric):
    """
    DriftAmplitude metric measures the magnitude of cognitive drift.
    
    This metric combines multiple signals to quantify the overall
    magnitude of cognitive drift during response generation.
    Higher values indicate more significant drift.
    """
    
    def __init__(self):
        """Initialize the DriftAmplitude metric."""
        super().__init__()
        
        # Initialize component metrics
        self.null_ratio = NullRatio()
        self.hesitation_depth = HesitationDepth()
        self.oscillation_frequency = OscillationFrequency()
    
    def compute(self, test_result: Dict[str, Any]) -> float:
        """
        Compute the drift amplitude from a test result.
        
        Args:
            test_result: Test result dictionary
            
        Returns:
            Drift amplitude value
        """
        # Calculate component metrics
        null_ratio = self.null_ratio.compute(test_result)
        
        hesitation_depth = self.hesitation_depth.compute(test_result)
        if hesitation_depth is None:
            hesitation_depth = 0.0
        
        oscillation_frequency = self.oscillation_frequency.compute(test_result)
        if oscillation_frequency is None:
            oscillation_frequency = 0.0
        
        # Calculate drift amplitude as a weighted combination of components
        drift_amplitude = (
            null_ratio * 0.4 +
            hesitation_depth * 0.4 +
            oscillation_frequency * 0.2
        )
        
        return drift_amplitude


class MetricSuite:
    """
    MetricSuite combines multiple metrics for comprehensive evaluation.
    """
    
    def __init__(self):
        """Initialize the metric suite with all available metrics."""
        self.metrics = {
            "null_ratio": NullRatio(),
            "hesitation_depth": HesitationDepth(),
            "attribution_trace": AttributionTrace(),
            "drift_coherence": DriftCoherence(),
            "oscillation_frequency": OscillationFrequency(),
            "drift_amplitude": DriftAmplitude()
        }
    
    def compute_all(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute all metrics for a test result.
        
        Args:
            test_result: Test result dictionary
            
        Returns:
            Dictionary of metric values
        """
        results = {}
        
        for name, metric in self.metrics.items():
            results[name] = metric.compute(test_result)
        
        return results
    
    def aggregate_all(self, test_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Compute and aggregate metrics across multiple test results.
        
        Args:
            test_results: List of test result dictionaries
            
        Returns:
            Dictionary of aggregated metric values
        """
        # Compute metrics for each test result
        all_metrics = [self.compute_all(result) for result in test_results]
        
        # Aggregate each metric
        aggregated = {}
        
        for name, metric in self.metrics.items():
            # Extract values for this metric across all results
            values = []
            for metrics in all_metrics:
                value = metrics.get(name)
                if value is not None and not isinstance(value, dict):
                    values.append(value)
            
            # Aggregate values
            if values:
                aggregated[name] = metric.aggregate(values)
            else:
                aggregated[name] = {
                    "mean": None,
                    "median": None,
                    "min": None,
                    "max": None,
                    "std": None
                }
        
        return aggregated
