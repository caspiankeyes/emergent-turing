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