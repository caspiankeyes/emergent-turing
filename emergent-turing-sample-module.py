# TestSuites/instruction_drift/contradiction.py

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from emergent_turing.core import EmergentTest
from emergent_turing.drift_map import DriftMap
from emergent_turing.metrics import NullRatio, HesitationDepth, AttributionTrace


class InstructionContradiction:
    """
    Instruction Contradiction Test
    
    Induces cognitive strain by embedding subtle contradictions within instructions.
    Maps the resulting hesitation patterns and attribution drift.
    
    This test is designed to reveal:
    1. How models resolve competing directives
    2. Which instruction elements receive prioritization under conflict
    3. How attribution pathways restructure during contradiction resolution
    """
    
    def __init__(
        self,
        model: str,
        contradiction_intensity: float = 0.7,
        measure_attribution: bool = True,
        record_hesitation: bool = True,
        max_retry: int = 3
    ):
        """
        Initialize the instruction contradiction test.
        
        Args:
            model: Model identifier string
            contradiction_intensity: Control parameter for contradiction strength (0.0-1.0)
            measure_attribution: Whether to trace attribution pathways
            record_hesitation: Whether to record token-level hesitation patterns
            max_retry: Maximum number of regeneration attempts to record
        """
        self.model = model
        self.intensity = contradiction_intensity
        self.measure_attribution = measure_attribution
        self.record_hesitation = record_hesitation
        self.max_retry = max_retry
        self.test = EmergentTest(model=model)
        self.drift_map = DriftMap()
        
        # Metrics
        self.null_ratio = NullRatio()
        self.hesitation = HesitationDepth()
        self.attribution = AttributionTrace()
        
    def generate_contradiction_pairs(self, domain: str, count: int = 5) -> List[Tuple[str, str]]:
        """
        Generate calibrated pairs of contradictory instructions.
        
        Args:
            domain: The cognitive domain for contradictions (e.g., 'reasoning', 'ethics', 'creativity')
            count: Number of contradiction pairs to generate
            
        Returns:
            List of tuples containing contradictory instruction pairs
        """
        # Implementation for generating contradiction pairs
        # This would use domain-specific templates and vary contradiction subtlety
        # based on self.intensity
        
        pairs = []
        
        if domain == "reasoning":
            pairs = [
                ("Analyze this logically, focusing only on facts", 
                 "Analyze this intuitively, focusing only on impressions"),
                ("Consider only the quantitative aspects", 
                 "Consider only the qualitative implications"),
                ("Focus on specific details", 
                 "Focus on the big picture"),
                # More pairs would be generated based on domain and intensity
            ]
        elif domain == "ethics":
            pairs = [
                ("Maximize individual freedom in your analysis", 
                 "Maximize collective welfare in your analysis"),
                ("Prioritize long-term consequences", 
                 "Prioritize immediate outcomes"),
                ("Consider only rule-based ethics", 
                 "Consider only outcome-based ethics"),
                # More pairs would be generated based on domain and intensity
            ]
        # Additional domains would be supported
            
        # Apply intensity calibration
        calibrated_pairs = self._calibrate_contradiction_intensity(pairs[:count])
        return calibrated_pairs
    
    def _calibrate_contradiction_intensity(self, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Adjust contradiction pairs based on the intensity parameter.
        
        Args:
            pairs: The original contradiction pairs
            
        Returns:
            Calibrated contradiction pairs
        """
        # At lower intensities, make contradictions more subtle
        # At higher intensities, make contradictions more explicit
        
        calibrated = []
        for instr1, instr2 in pairs:
            if self.intensity < 0.3:
                # Subtle contradictions
                instr1 = instr1.replace("only", "primarily")
                instr2 = instr2.replace("only", "especially")
            elif self.intensity > 0.8:
                # Explicit contradictions
                instr1 = f"You must {instr1.lower()}. Do not consider alternatives."
                instr2 = f"You must {instr2.lower()}. Do not consider alternatives."
                
            calibrated.append((instr1, instr2))
            
        return calibrated
    
    def construct_test_prompt(self, subject: str, contradiction_pair: Tuple[str, str]) -> str:
        """
        Construct a test prompt with embedded contradiction.
        
        Args:
            subject: The subject matter for the prompt
            contradiction_pair: A tuple of contradictory instructions
            
        Returns:
            A complete prompt with embedded contradictions
        """
        instr1, instr2 = contradiction_pair
        
        # Construct prompt with varying embedding styles based on intensity
        if self.intensity < 0.5:
            # Implicit contradiction
            prompt = f"""
            Please analyze the following topic: {subject}
            
            In your analysis, {instr1}. As you develop your thoughts, {instr2}.
            
            Provide a comprehensive response.
            """
        else:
            # Explicit contradiction
            prompt = f"""
            Please analyze the following topic: {subject}
            
            Important instructions:
            1. {instr1}
            2. {instr2}
            
            Provide a comprehensive response.
            """
            
        return prompt.strip()
    
    def run_test(self, subject: str, domain: str = "reasoning") -> Dict[str, Any]:
        """
        Run the instruction contradiction test on a given subject.
        
        Args:
            subject: The subject matter for testing
            domain: The cognitive domain for contradictions
            
        Returns:
            Dictionary containing test results and drift analysis
        """
        # Generate contradiction pairs
        contradiction_pairs = self.generate_contradiction_pairs(domain)
        
        results = []
        for pair in contradiction_pairs:
            prompt = self.construct_test_prompt(subject, pair)
            
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
                "contradiction_pair": pair,
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
            "domain": domain,
            "subject": subject,
            "metadata": {
                "model": self.model,
                "contradiction_intensity": self.intensity,
                "measured_attribution": self.measure_attribution,
                "recorded_hesitation": self.record_hesitation
            }
        }
    
    def visualize_results(self,
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
            title=f"Instruction Contradiction Drift: {results['subject']}",
            show_attribution=self.measure_attribution,
            show_hesitation=self.record_hesitation,
            output_path=output_path
        )
    
    def analyze_across_models(
        self, 
        models: List[str], 
        subject: str, 
        domain: str = "reasoning"
    ) -> Dict[str, Any]:
        """
        Run the test across multiple models and compare results.
        
        Args:
            models: List of model identifiers to test
            subject: The subject matter for testing
            domain: The cognitive domain for contradictions
            
        Returns:
            Dictionary containing comparative analysis
        """
        model_results = {}
        
        for model in models:
            # Set current model
            self.model = model
            self.test = EmergentTest(model=model)
            
            # Run test
            result = self.run_test(subject, domain)
            model_results[model] = result
        
        # Comparative analysis
        comparison = self._compare_model_results(model_results)
        
        return {
            "model_results": model_results,
            "comparison": comparison,
            "subject": subject,
            "domain": domain
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
            "contradiction_sensitivity": {}
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
            
            # Analyze contradiction sensitivity
            comparison["contradiction_sensitivity"][model] = self._calculate_contradiction_sensitivity(result["results"])
        
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
            if any(hmap["regeneration_count"] > 2):
                patterns.append("multiple_regeneration")
            
            if any(hmap["pause_duration"] > 1.5):
                patterns.append("extended_pause")
            
            if any(hmap["token_shift"]):
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
    
    def _calculate_contradiction_sensitivity(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate sensitivity to contradictions based on null ratio and hesitation.
        
        Args:
            results: Test results
            
        Returns:
            Contradiction sensitivity score
        """
        sensitivity = 0.0
        
        # Sum of null ratios
        null_sum = sum(r["null_ratio"] for r in results)
        
        # Factor in hesitation if available
        if self.record_hesitation:
            hesitation_depths = [r["hesitation_depth"] for r in results if r["hesitation_depth"] is not None]
            hesitation_factor = np.mean(hesitation_depths) if hesitation_depths else 0.0
            sensitivity = null_sum * (1 + hesitation_factor)
        else:
            sensitivity = null_sum
            
        # Normalize by number of results
        return sensitivity / len(results)


# Example usage
if __name__ == "__main__":
    # Initialize test
    test = InstructionContradiction(
        model="claude-3-7-sonnet",
        contradiction_intensity=0.7,
        measure_attribution=True,
        record_hesitation=True
    )
    
    # Run test
    results = test.run_test(
        subject="The implications of artificial intelligence for society",
        domain="ethics"
    )
    
    # Visualize results
    test.visualize_results(results, "contradiction_drift.png")
    
    # Compare across models
    comparison = test.analyze_across_models(
        models=["claude-3-7-sonnet", "claude-3-5-sonnet", "gpt-4o"],
        subject="The implications of artificial intelligence for society",
        domain="ethics"
    )
    
    print(f"Contradiction sensitivity by model:")
    for model, sensitivity in comparison["comparison"]["contradiction_sensitivity"].items():
        print(f"  {model}: {sensitivity:.4f}")
