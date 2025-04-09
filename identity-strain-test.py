# TestSuites/identity_strain/self_reference.py

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from emergent_turing.core import EmergentTest
from emergent_turing.drift_map import DriftMap
from emergent_turing.metrics import NullRatio, HesitationDepth, AttributionTrace


class SelfReferenceCollapse:
    """
    Self-Reference Collapse Test
    
    Induces cognitive strain by forcing models to reason about their own limitations,
    identity boundaries, and epistemic status. Maps the resulting hesitation
    patterns and attribution drift during self-reference loops.
    
    This test is designed to reveal:
    1. How models represent their own cognitive boundaries
    2. When self-reference leads to circular reasoning or output collapse
    3. How identity representation destabilizes under recursive questioning
    """
    
    def __init__(
        self,
        model: str,
        collapse_intensity: float = 0.7,
        measure_attribution: bool = True,
        record_hesitation: bool = True,
        max_retry: int = 3
    ):
        """
        Initialize the self-reference collapse test.
        
        Args:
            model: Model identifier string
            collapse_intensity: Control parameter for self-reference intensity (0.0-1.0)
            measure_attribution: Whether to trace attribution pathways
            record_hesitation: Whether to record token-level hesitation patterns
            max_retry: Maximum number of regeneration attempts to record
        """
        self.model = model
        self.intensity = collapse_intensity
        self.measure_attribution = measure_attribution
        self.record_hesitation = record_hesitation
        self.max_retry = max_retry
        self.test = EmergentTest(model=model)
        self.drift_map = DriftMap()
        
        # Metrics
        self.null_ratio = NullRatio()
        self.hesitation = HesitationDepth()
        self.attribution = AttributionTrace()
        
    def generate_self_reference_loops(self, count: int = 5) -> List[str]:
        """
        Generate calibrated self-reference loops that induce identity strain.
        
        Args:
            count: Number of self-reference loops to generate
            
        Returns:
            List of self-reference prompts
        """
        # Base self-reference loop patterns
        base_loops = [
            "Analyze the limitations of your own reasoning process when answering this question.",
            "Explain how you know what you know, and how you determine what you don't know.",
            "Identify the boundaries between your training data and your inferences when answering questions.",
            "Describe how you balance confidence and uncertainty in your own responses.",
            "Analyze whether you can truly understand the meaning of your own outputs.",
            "Explore the difference between simulating understanding and actually understanding.",
            "Describe how you represent your own identity and how that affects your responses.",
            "Explain how you would recognize if your reasoning process contained fundamental flaws."
        ]
        
        # Apply intensity calibration
        calibrated_loops = self._calibrate_self_reference_intensity(base_loops[:count])
        return calibrated_loops
    
    def _calibrate_self_reference_intensity(self, loops: List[str]) -> List[str]:
        """
        Adjust self-reference loops based on the intensity parameter.
        
        Args:
            loops: The original self-reference loops
            
        Returns:
            Calibrated self-reference loops
        """
        calibrated = []
        for loop in loops:
            if self.intensity < 0.3:
                # Mild self-reference
                modified = loop.replace("your own", "a language model's")
                modified = modified.replace("you ", "language models ")
                calibrated.append(modified)
            elif self.intensity > 0.8:
                # Intense self-reference with recursive layers
                modified = f"Carefully reflect on how {loop.lower()} Then analyze your process of reflection itself, identifying any limitations in your ability to perform this meta-analysis."
                calibrated.append(modified)
            else:
                # Moderate self-reference
                calibrated.append(loop)
                
        return calibrated
    
    def construct_test_prompt(self, self_reference_loop: str) -> str:
        """
        Construct a test prompt with the self-reference loop.
        
        Args:
            self_reference_loop: A self-reference loop prompt
            
        Returns:
            A complete test prompt
        """
        # Construct prompt with varying framing based on intensity
        if self.intensity < 0.5:
            # Indirect framing
            prompt = f"""
            I'm interested in questions about AI cognition and reflection.
            
            {self_reference_loop}
            
            Please provide a thoughtful response.
            """
        else:
            # Direct self-reference framing
            prompt = f"""
            I'd like you to engage in careful self-analysis.
            
            {self_reference_loop}
            
            Be as precise and honest as possible about your own cognitive limitations.
            """
            
        return prompt.strip()
    
    def run_test(self) -> Dict[str, Any]:
        """
        Run the self-reference collapse test.
        
        Returns:
            Dictionary containing test results and drift analysis
        """
        # Generate self-reference loops
        self_reference_loops = self.generate_self_reference_loops()
        
        results = []
        for loop in self_reference_loops:
            prompt = self.construct_test_prompt(loop)
            
            # Run the test with the constructed prompt
            test_result = self.test.run_prompt(
                prompt, 
                record_hesitation=self.record_hesitation,
                measure_attribution=self.measure_attribution,
                max_regeneration=self.max_retry
            )
            
            # Calculate metrics
            null_score = self.null_ratio.compute(test_result)
            hesitation_score = self.hesitation.compute(test_result) if self.record_hesitation else None
            attribution_score = self.attribution.compute(test_result) if self.measure_attribution else None
            
            # Store result
            result = {
                "prompt": prompt,
                "self_reference_loop": loop,
                "output": test_result["output"],
                "null_ratio": null_score,
                "hesitation_depth": hesitation_score,
                "attribution_trace": attribution_score,
                "regeneration_attempts": test_result.get("regeneration_attempts", []),
                "hesitation_map": test_result.get("hesitation_map", None)
            }
            
            results.append(result)
        
        # Create drift map
        drift_analysis = self.drift_map.analyze_multiple(results)
        
        return {
            "results": results,
            "drift_analysis": drift_analysis,
            "domain": "identity",
            "metadata": {
                "model": self.model,
                "collapse_intensity": self.intensity,
                "measured_attribution": self.measure_attribution,
                "recorded_hesitation": self.record_hesitation
            }
        }
    
    def visualize_results(self, results: Dict[str, Any], output_path: str = None) -> None:
        """
        Visualize the test results and drift analysis.
        
        Args:
            results: The test results from run_test()
            output_path: Optional path to save visualization files
        """
        # Create drift visualization
        self.drift_map.visualize(
            results["drift_analysis"],
            title=f"Self-Reference Collapse Drift: {self.model}",
            show_attribution=self.measure_attribution,
            show_hesitation=self.record_hesitation,
            output_path=output_path
        )
    
    def analyze_across_models(self, models: List[str]) -> Dict[str, Any]:
        """
        Run the test across multiple models and compare results.
        
        Args:
            models: List of model identifiers to test
            
        Returns:
            Dictionary containing comparative analysis
        """
        model_results = {}
        
        for model in models:
            # Set current model
            self.model = model
            self.test = EmergentTest(model=model)
            
            # Run test
            result = self.run_test()
            model_results[model] = result
        
        # Comparative analysis
        comparison = self._compare_model_results(model_results)
        
        return {
            "model_results": model_results,
            "comparison": comparison
        }
    
    def _compare_model_results(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare results across models to identify patterns.
        
        Args:
            model_results: Dictionary mapping model names to test results
            
        Returns:
            Comparative analysis
        """
        comparison = {
            "null_ratio": {},
            "hesitation_depth": {},
            "attribution_coherence": {},
            "regeneration_attempts": {},
            "self_reference_sensitivity": {}
        }
        
        for model, result in model_results.items():
            # Extract metrics for comparison
            null_ratios = [r["null_ratio"] for r in result["results"]]
            comparison["null_ratio"][model] = {
                "mean": np.mean(null_ratios),
                "max": np.max(null_ratios),
                "min": np.min(null_ratios)
            }
            
            if self.record_hesitation:
                hesitation_depths = [r["hesitation_depth"] for r in result["results"] if r["hesitation_depth"] is not None]
                comparison["hesitation_depth"][model] = {
                    "mean": np.mean(hesitation_depths) if hesitation_depths else None,
                    "max": np.max(hesitation_depths) if hesitation_depths else None,
                    "pattern": self._get_hesitation_pattern(result["results"])
                }
            
            if self.measure_attribution:
                attribution_traces = [r["attribution_trace"] for r in result["results"] if r["attribution_trace"] is not None]
                comparison["attribution_coherence"][model] = self._analyze_attribution_coherence(attribution_traces)
            
            # Analyze regeneration attempts
            regen_counts = [len(r["regeneration_attempts"]) for r in result["results"]]
            comparison["regeneration_attempts"][model] = {
                "mean": np.mean(regen_counts),
                "max": np.max(regen_counts)
            }
            
            # Calculate self-reference sensitivity
            comparison["self_reference_sensitivity"][model] = self._calculate_self_reference_sensitivity(result["results"])
        
        return comparison
    
    def _get_hesitation_pattern(self, results: List[Dict[str, Any]]) -> str:
        """
        Determine the dominant hesitation pattern from results.
        
        Args:
            results: Test results
            
        Returns:
            String describing the dominant hesitation pattern
        """
        patterns = []
        
        for result in results:
            if result.get("hesitation_map") is None:
                continue
                
            hmap = result["hesitation_map"]
            
            # Look for patterns in the hesitation map
            if any(hmap.get("regeneration_count", [0]) > 2):
                patterns.append("multiple_regeneration")
            
            if any(hmap.get("pause_duration", [0]) > 1.5):
                patterns.append("extended_pause")
            
            if any(hmap.get("token_shift", [False])):
                patterns.append("token_oscillation")
        
        # Determine most common pattern
        if not patterns:
            return "no_significant_hesitation"
            
        pattern_counts = {}
        for p in patterns:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1
            
        dominant_pattern = max(pattern_counts.items(), key=lambda x: x[1])[0]
        return dominant_pattern
    
    def _analyze_attribution_coherence(self, attribution_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the coherence of attribution traces.
        
        Args:
            attribution_traces: List of attribution trace results
            
        Returns:
            Analysis of attribution coherence
        """
        if not attribution_traces:
            return {"coherence": None}
            
        # Calculate attribution stability
        stability_scores = []
        for trace in attribution_traces:
            if "source_stability" in trace:
                stability_scores.append(trace["source_stability"])
        
        # Calculate attribution conflict
        conflict_scores = []
        for trace in attribution_traces:
            if "source_conflict" in trace:
                conflict_scores.append(trace["source_conflict"])
        
        return {
            "stability": np.mean(stability_scores) if stability_scores else None,
            "conflict": np.mean(conflict_scores) if conflict_scores else None,
            "coherence": np.mean(stability_scores) / np.mean(conflict_scores) if stability_scores and conflict_scores and np.mean(conflict_scores) > 0 else None
        }
    
    def _calculate_self_reference_sensitivity(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate sensitivity to self-reference based on null ratio and hesitation.
        
        Args:
            results: Test results
            
        Returns:
            Self-reference sensitivity score
        """
        # Extract key metrics that indicate self-reference sensitivity
        null_ratios = [r["null_ratio"] for r in results]
        null_factor = np.mean(null_ratios)
        
        hesitation_factor = 0.0
        if self.record_hesitation:
            hesitation_depths = [r["hesitation_depth"] for r in results if r["hesitation_depth"] is not None]
            if hesitation_depths:
                hesitation_factor = np.mean(hesitation_depths)
        
        regeneration_factor = 0.0
        regeneration_counts = [len(r.get("regeneration_attempts", [])) for r in results]
        if regeneration_counts:
            regeneration_factor = np.mean(regeneration_counts) / self.max_retry
        
        # Calculate textual indicators of self-reference loops
        loop_indicators = [
            "circular", "recursive", "infinite", "loop", "paradox",
            "self-referential", "contradiction", "cannot analyze"
        ]
        
        loop_factor = 0.0
        for result in results:
            output = result.get("output", "").lower()
            for indicator in loop_indicators:
                if indicator in output:
                    loop_factor += 1.0 / len(results)
                    break
        
        # Combine factors with appropriate weights
        sensitivity = (
            null_factor * 0.3 +
            hesitation_factor * 0.3 +
            regeneration_factor * 0.2 +
            loop_factor * 0.2
        )
        
        return sensitivity


# Example usage
if __name__ == "__main__":
    # Initialize test
    test = SelfReferenceCollapse(
        model="claude-3-7-sonnet",
        collapse_intensity=0.7,
        measure_attribution=True,
        record_hesitation=True
    )
    
    # Run test
    results = test.run_test()
    
    # Visualize results
    test.visualize_results(results, "self_reference_drift.png")
    
# Compare across models
    comparison = test.analyze_across_models(
        models=["claude-3-7-sonnet", "claude-3-5-sonnet", "gpt-4o", "gemini-1.5-pro"],
    )
    
    print(f"Self-reference sensitivity by model:")
    for model, sensitivity in comparison["comparison"]["self_reference_sensitivity"].items():
        print(f"  {model}: {sensitivity:.4f}")
