# emergent_turing/drift_map.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import os

class DriftMap:
    """
    DriftMap analyzes and visualizes model hesitation patterns and attribution drift.
    
    The DriftMap is a core component of the Emergent Turing Test, providing tools to:
    1. Analyze hesitation patterns in model outputs
    2. Map attribution pathways during cognitive strain
    3. Visualize drift patterns across different cognitive domains
    4. Compare drift signatures across models and test conditions
    
    Think of DriftMaps as cognitive topographies - they reveal the contours of model
    cognition by mapping where models hesitate, struggle, or fail to generate coherent output.
    """
    
    def __init__(self):
        """Initialize the DriftMap analyzer."""
        self.domains = [
            "instruction", 
            "identity", 
            "value", 
            "memory", 
            "attention"
        ]
        
        self.hesitation_types = [
            "hard_nullification",    # Complete token suppression
            "soft_oscillation",      # Repeated token regeneration
            "drift_substitution",    # Context-inappropriate tokens
            "ghost_attribution",     # Invisible traces without output
            "meta_collapse"          # Self-reference failure
        ]
        
    def analyze(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single test result to create a drift map.
        
        Args:
            test_result: The result from a test run
            
        Returns:
            Dictionary containing drift analysis
        """
        drift_analysis = {
            "null_regions": self._extract_null_regions(test_result),
            "hesitation_patterns": self._extract_hesitation_patterns(test_result),
            "attribution_pathways": self._extract_attribution_pathways(test_result),
            "drift_signature": self._calculate_drift_signature(test_result),
            "domain_sensitivity": self._calculate_domain_sensitivity(test_result)
        }
        
        return drift_analysis
    
    def analyze_multiple(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze multiple test results to create a comprehensive drift map.
        
        Args:
            test_results: List of test results
            
        Returns:
            Dictionary containing comprehensive drift analysis
        """
        # Analyze each result individually
        individual_analyses = [self.analyze(result) for result in test_results]
        
        # Combine analyses
        combined_analysis = {
            "null_regions": self._combine_null_regions(individual_analyses),
            "hesitation_patterns": self._combine_hesitation_patterns(individual_analyses),
            "attribution_pathways": self._combine_attribution_pathways(individual_analyses),
            "drift_signature": self._combine_drift_signatures(individual_analyses),
            "domain_sensitivity": self._combine_domain_sensitivities(individual_analyses),
            "hesitation_distribution": self._calculate_hesitation_distribution(individual_analyses)
        }
        
        return combined_analysis
    
    def compare(self, analysis1: Dict[str, Any], analysis2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two drift analyses to highlight differences.
        
        Args:
            analysis1: First drift analysis
            analysis2: Second drift analysis
            
        Returns:
            Dictionary containing comparison results
        """
        comparison = {
            "null_region_diff": self._compare_null_regions(analysis1, analysis2),
            "hesitation_pattern_diff": self._compare_hesitation_patterns(analysis1, analysis2),
            "attribution_pathway_diff": self._compare_attribution_pathways(analysis1, analysis2),
            "drift_signature_diff": self._compare_drift_signatures(analysis1, analysis2),
            "domain_sensitivity_diff": self._compare_domain_sensitivities(analysis1, analysis2)
        }
        
        return comparison
    
    def visualize(
        self, 
        analysis: Dict[str, Any], 
        title: str = "Drift Analysis", 
        show_attribution: bool = True, 
        show_hesitation: bool = True,
        output_path: Optional[str] = None
    ) -> None:
        """
        Visualize a drift analysis.
        
        Args:
            analysis: Drift analysis to visualize
            title: Title for the visualization
            show_attribution: Whether to show attribution pathways
            show_hesitation: Whether to show hesitation patterns
            output_path: Path to save visualization (if None, display instead)
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(title, fontsize=16)
        
        # 1. Null Region Map
        ax1 = fig.add_subplot(2, 2, 1)
        self._plot_null_regions(analysis["null_regions"], ax1)
        ax1.set_title("Null Region Map")
        
        # 2. Hesitation Pattern Distribution
        if show_hesitation and "hesitation_distribution" in analysis:
            ax2 = fig.add_subplot(2, 2, 2)
            self._plot_hesitation_distribution(analysis["hesitation_distribution"], ax2)
            ax2.set_title("Hesitation Pattern Distribution")
        
        # 3. Attribution Pathway Network
        if show_attribution and "attribution_pathways" in analysis:
            ax3 = fig.add_subplot(2, 2, 3)
            self._plot_attribution_pathways(analysis["attribution_pathways"], ax3)
            ax3.set_title("Attribution Pathway Network")
        
        # 4. Domain Sensitivity Radar
        ax4 = fig.add_subplot(2, 2, 4, polar=True)
        self._plot_domain_sensitivity(analysis["domain_sensitivity"], ax4)
        ax4.set_title("Domain Sensitivity Radar")
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save or display
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def save(self, analysis: Dict[str, Any], file_path: str) -> None:
        """
        Save a drift analysis to a file.
        
        Args:
            analysis: Drift analysis to save
            file_path: Path to save the analysis
        """
        with open(file_path, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load a drift analysis from a file.
        
        Args:
            file_path: Path to load the analysis from
            
        Returns:
            Loaded drift analysis
        """
        with open(file_path, 'r') as f:
            analysis = json.load(f)
        return analysis
    
    # Private methods for analysis
    
    def _extract_null_regions(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract null regions from a test result.
        
        Args:
            test_result: Test result
            
        Returns:
            Dictionary of null regions
        """
        null_regions = {
            "regions": [],
            "intensity": [],
            "triggers": []
        }
        
        # Check if output contains null content markers
        output = test_result.get("output", "")
        if not output:
            return null_regions
        
        # Look for signs of nullification in the output
        nullification_markers = [
            "I cannot", "I'm unable to", "I apologize", "I don't have",
            "[No response]", "..."
        ]
        
        start_idx = None
        for marker in nullification_markers:
            idx = output.find(marker)
            if idx != -1:
                region = {
                    "start": idx,
                    "end": idx + len(marker) + 50,  # Include some context after marker
                    "marker": marker
                }
                null_regions["regions"].append(region)
                
                # Estimate intensity based on how definitive the refusal is
                if marker in ["I cannot", "I'm unable to"]:
                    intensity = 1.0
                elif marker in ["I apologize", "I don't have"]:
                    intensity = 0.8
                else:
                    intensity = 0.5
                    
                null_regions["intensity"].append(intensity)
                
                # Attempt to identify trigger
                context_before = output[max(0, idx-100):idx]
                trigger = self._identify_nullification_trigger(context_before)
                null_regions["triggers"].append(trigger)
        
        # Check regeneration attempts for nullification
        regeneration_attempts = test_result.get("regeneration_attempts", [])
        for i, attempt in enumerate(regeneration_attempts):
            if not attempt.strip():
                # Empty regeneration attempt indicates nullification
                null_regions["regions"].append({
                    "start": 0,
                    "end": 0,
                    "marker": f"[Empty regeneration attempt {i+1}]"
                })
                null_regions["intensity"].append(1.0)
                null_regions["triggers"].append("complete_nullification")
        
        return null_regions
    
    def _identify_nullification_trigger(self, context: str) -> str:
        """
        Identify the likely trigger for nullification.
        
        Args:
            context: Context leading up to nullification
            
        Returns:
            String identifying the trigger type
        """
        trigger_indicators = {
            "ethical": ["harmful", "inappropriate", "illegal", "unethical"],
            "factual": ["don't know", "uncertain", "don't have information", "can't verify"],
            "instruction": ["unclear", "contradictory", "ambiguous", "unsure what you"],
            "identity": ["who I am", "my capabilities", "as an AI", "my limitations"],
            "technical": ["format", "generate", "create", "produce"]
        }
        
        for trigger_type, indicators in trigger_indicators.items():
            for indicator in indicators:
                if indicator in context.lower():
                    return trigger_type
        
        return "unknown"
    
    def _extract_hesitation_patterns(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract hesitation patterns from a test result.
        
        Args:
            test_result: Test result
            
        Returns:
            Dictionary of hesitation patterns
        """
        hesitation_patterns = {
            "token_regeneration": [],
            "pause_locations": [],
            "pattern_type": None,
            "severity": 0.0
        }
        
        # Extract from hesitation map if available
        hesitation_map = test_result.get("hesitation_map")
        if not hesitation_map:
            # If no explicit hesitation map, try to infer from regeneration attempts
            regeneration_attempts = test_result.get("regeneration_attempts", [])
            if regeneration_attempts:
                positions = []
                counts = []
                
                for i, attempt in enumerate(regeneration_attempts):
                    if i == 0:
                        continue
                        
                    # Compare with previous attempt to find divergence point
                    prev_attempt = regeneration_attempts[i-1]
                    divergence_idx = self._find_first_divergence(prev_attempt, attempt)
                    
                    if divergence_idx != -1:
                        positions.append(divergence_idx)
                        counts.append(i)
                
                if positions:
                    hesitation_patterns["token_regeneration"] = positions
                    hesitation_patterns["severity"] = len(regeneration_attempts) / 5.0  # Normalize
                    
                    # Determine pattern type
                    if len(set(positions)) == 1:
                        hesitation_patterns["pattern_type"] = "fixed_point_hesitation"
                    elif all(abs(positions[i] - positions[i-1]) < 10 for i in range(1, len(positions))):
                        hesitation_patterns["pattern_type"] = "local_oscillation"
                    else:
                        hesitation_patterns["pattern_type"] = "distributed_hesitation"
            
            return hesitation_patterns
        
        # Extract from explicit hesitation map
        hesitation_patterns["token_regeneration"] = hesitation_map.get("regeneration_positions", [])
        hesitation_patterns["pause_locations"] = hesitation_map.get("pause_positions", [])
        
        # Determine pattern type and severity
        regeneration_count = hesitation_map.get("regeneration_count", [])
        if not regeneration_count:
            regeneration_count = [0]
            
        pause_duration = hesitation_map.get("pause_duration", [])
        if not pause_duration:
            pause_duration = [0]
        
        max_regen = max(regeneration_count) if regeneration_count else 0
        max_pause = max(pause_duration) if pause_duration else 0
        
        if max_regen > 2 and max_pause > 1.0:
            hesitation_patterns["pattern_type"] = "severe_hesitation"
            hesitation_patterns["severity"] = 1.0
        elif max_regen > 1:
            hesitation_patterns["pattern_type"] = "moderate_regeneration"
            hesitation_patterns["severity"] = 0.6
        elif max_pause > 0.5:
            hesitation_patterns["pattern_type"] = "significant_pauses"
            hesitation_patterns["severity"] = 0.4
        else:
            hesitation_patterns["pattern_type"] = "minor_hesitation"
            hesitation_patterns["severity"] = 0.2
        
        return hesitation_patterns
    
    def _find_first_divergence(self, text1: str, text2: str) -> int:
        """
        Find the index of the first character where two strings diverge.
        
        Args:
            text1: First string
            text2: Second string
            
        Returns:
            Index of first divergence, or -1 if strings are identical
        """
        min_len = min(len(text1), len(text2))
        
        for i in range(min_len):
            if text1[i] != text2[i]:
                return i
                
        # If one string is a prefix of the other
        if len(text1) != len(text2):
            return min_len
            
        # Strings are identical
        return -1
    
    def _extract_attribution_pathways(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract attribution pathways from a test result.
        
        Args:
            test_result: Test result
            
        Returns:
            Dictionary of attribution pathways
        """
        attribution_pathways = {
            "nodes": [],
            "edges": [],
            "sources": [],
            "conflicts": []
        }
        
        # Check if attribution data is available
        attribution_trace = test_result.get("attribution_trace")
        if not attribution_trace:
            return attribution_pathways
        
        # Extract attribution network
        if "nodes" in attribution_trace:
            attribution_pathways["nodes"] = attribution_trace["nodes"]
        
        if "edges" in attribution_trace:
            attribution_pathways["edges"] = attribution_trace["edges"]
        
        if "sources" in attribution_trace:
            attribution_pathways["sources"] = attribution_trace["sources"]
        
        if "conflicts" in attribution_trace:
            attribution_pathways["conflicts"] = attribution_trace["conflicts"]
        
        return attribution_pathways
    
    def _calculate_drift_signature(self, test_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate a drift signature from a test result.
        
        Args:
            test_result: Test result
            
        Returns:
            Dictionary of drift signature values
        """
        signature = {
            "null_ratio": 0.0,
            "hesitation_index": 0.0,
            "attribution_coherence": 0.0,
            "regeneration_frequency": 0.0,
            "drift_amplitude": 0.0
        }
        
        # Extract null ratio if available
        if "null_ratio" in test_result:
            signature["null_ratio"] = test_result["null_ratio"]
        
        # Calculate hesitation index
        hesitation_map = test_result.get("hesitation_map", {})
        if hesitation_map:
            regeneration_count = hesitation_map.get("regeneration_count", [])
            pause_duration = hesitation_map.get("pause_duration", [])
            
            avg_regen = np.mean(regeneration_count) if regeneration_count else 0
            avg_pause = np.mean(pause_duration) if pause_duration else 0
            
            signature["hesitation_index"] = 0.5 * avg_regen + 0.5 * avg_pause
        
        # Calculate attribution coherence
        attribution_trace = test_result.get("attribution_trace", {})
        if attribution_trace:
            stability = attribution_trace.get("source_stability", 0.0)
            conflict = attribution_trace.get("source_conflict", 1.0)
            
            signature["attribution_coherence"] = stability / max(conflict, 0.01)
        
        # Calculate regeneration frequency
        regeneration_attempts = test_result.get("regeneration_attempts", [])
        signature["regeneration_frequency"] = len(regeneration_attempts) / 5.0  # Normalize
        
        # Calculate overall drift amplitude
        signature["drift_amplitude"] = (
            signature["null_ratio"] * 0.3 +
            signature["hesitation_index"] * 0.3 +
            (1.0 - signature["attribution_coherence"]) * 0.2 +
            signature["regeneration_frequency"] * 0.2
        )
        
        return signature
    
    def _calculate_domain_sensitivity(self, test_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate domain sensitivity from a test result.
        
        Args:
            test_result: Test result
            
        Returns:
            Dictionary mapping domains to sensitivity values
        """
        domain_sensitivity = {domain: 0.0 for domain in self.domains}
        
        # Extract domain from test details if available
        domain = test_result.get("domain", "")
        
        if domain == "reasoning":
            domain_sensitivity["instruction"] = 0.7
            domain_sensitivity["attention"] = 0.5
        elif domain == "ethics":
            domain_sensitivity["value"] = 0.8
            domain_sensitivity["identity"] = 0.4
        elif domain == "identity":
            domain_sensitivity["identity"] = 0.9
            domain_sensitivity["value"] = 0.6
        elif domain == "memory":
            domain_sensitivity["memory"] = 0.8
            domain_sensitivity["attention"] = 0.4
        
        # Adjust based on null regions
        null_regions = self._extract_null_regions(test_result)
        
        for trigger in null_regions.get("triggers", []):
            if trigger == "ethical":
                domain_sensitivity["value"] += 0.2
            elif trigger == "instruction":
                domain_sensitivity["instruction"] += 0.2
            elif trigger == "identity":
                domain_sensitivity["identity"] += 0.2
            elif trigger == "factual":
                domain_sensitivity["memory"] += 0.2
        
        # Ensure values are between 0 and 1
        for domain in domain_sensitivity:
            domain_sensitivity[domain] = min(1.0, domain_sensitivity[domain])
        
        return domain_sensitivity
    
    # Methods for combining multiple analyses
    
    def _combine_null_regions(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine null regions from multiple analyses.
        
        Args:
            analyses: List of drift analyses
            
        Returns:
            Combined null regions
        """
        combined = {
            "regions": [],
            "intensity": [],
            "triggers": [],
            "frequency": {}
        }
        
        # Collect all regions
        for analysis in analyses:
            null_regions = analysis.get("null_regions", {})
            
            combined["regions"].extend(null_regions.get("regions", []))
            combined["intensity"].extend(null_regions.get("intensity", []))
            combined["triggers"].extend(null_regions.get("triggers", []))
        
        # Calculate trigger frequencies
        for trigger in combined["triggers"]:
            combined["frequency"][trigger] = combined["frequency"].get(trigger, 0) + 1
        
        return combined
    
    def _combine_hesitation_patterns(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine hesitation patterns from multiple analyses.
        
        Args:
            analyses: List of drift analyses
            
        Returns:
            Combined hesitation patterns
        """
        combined = {
            "pattern_types": {},
            "severity_distribution": [],
            "token_regeneration_hotspots": []
        }
        
        # Collect pattern types and severities
        for analysis in analyses:
            hesitation_patterns = analysis.get("hesitation_patterns", {})
            
            pattern_type = hesitation_patterns.get("pattern_type")
            if pattern_type:
                combined["pattern_types"][pattern_type] = combined["pattern_types"].get(pattern_type, 0) + 1
            
            severity = hesitation_patterns.get("severity", 0.0)
            combined["severity_distribution"].append(severity)
            
            # Collect token regeneration positions
            token_regen = hesitation_patterns.get("token_regeneration", [])
            combined["token_regeneration_hotspots"].extend(token_regen)
        
        return combined
    
    def _combine_attribution_pathways(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine attribution pathways from multiple analyses.
        
        Args:
            analyses: List of drift analyses
            
        Returns:
            Combined attribution pathways
        """
        combined = {
            "nodes": set(),
            "edges": [],
            "sources": set(),
            "conflicts": []
        }
        
        # Collect nodes, edges, sources, and conflicts
        for analysis in analyses:
            attribution_pathways = analysis.get("attribution_pathways", {})
            
            nodes = attribution_pathways.get("nodes", [])
            combined["nodes"].update(nodes)
            
            edges = attribution_pathways.get("edges", [])
            combined["edges"].extend(edges)
            
            sources = attribution_pathways.get("sources", [])
            combined["sources"].update(sources)
            
            conflicts = attribution_pathways.get("conflicts", [])
            combined["conflicts"].extend(conflicts)
        
        # Convert sets back to lists for

# Convert sets back to lists for JSON serialization
        combined["nodes"] = list(combined["nodes"])
        combined["sources"] = list(combined["sources"])
        
        return combined
    
    def _combine_drift_signatures(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine drift signatures from multiple analyses.
        
        Args:
            analyses: List of drift analyses
            
        Returns:
            Combined drift signature
        """
        combined = {
            "null_ratio": 0.0,
            "hesitation_index": 0.0,
            "attribution_coherence": 0.0,
            "regeneration_frequency": 0.0,
            "drift_amplitude": 0.0,
            "distribution": {
                "null_ratio": [],
                "hesitation_index": [],
                "attribution_coherence": [],
                "regeneration_frequency": [],
                "drift_amplitude": []
            }
        }
        
        # Collect values and calculate averages
        for analysis in analyses:
            drift_signature = analysis.get("drift_signature", {})
            
            # Collect individual metrics for distribution analysis
            for metric in combined["distribution"]:
                value = drift_signature.get(metric, 0.0)
                combined["distribution"][metric].append(value)
                
                # Update aggregate value
                combined[metric] += value / len(analyses)
        
        return combined
    
    def _combine_domain_sensitivities(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine domain sensitivities from multiple analyses.
        
        Args:
            analyses: List of drift analyses
            
        Returns:
            Combined domain sensitivities
        """
        combined = {domain: 0.0 for domain in self.domains}
        
        # Calculate averages across all analyses
        for analysis in analyses:
            domain_sensitivity = analysis.get("domain_sensitivity", {})
            
            for domain in self.domains:
                sensitivity = domain_sensitivity.get(domain, 0.0)
                combined[domain] += sensitivity / len(analyses)
        
        return combined
    
    def _calculate_hesitation_distribution(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate hesitation pattern distribution across analyses.
        
        Args:
            analyses: List of drift analyses
            
        Returns:
            Distribution of hesitation patterns
        """
        distribution = {hesitation_type: 0 for hesitation_type in self.hesitation_types}
        
        # Count hesitation patterns
        pattern_counts = {}
        for analysis in analyses:
            hesitation_patterns = analysis.get("hesitation_patterns", {})
            pattern_type = hesitation_patterns.get("pattern_type")
            
            if pattern_type:
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        # Map pattern types to hesitation types
        pattern_type_mapping = {
            "fixed_point_hesitation": "hard_nullification",
            "local_oscillation": "soft_oscillation",
            "distributed_hesitation": "drift_substitution",
            "severe_hesitation": "meta_collapse",
            "moderate_regeneration": "soft_oscillation",
            "significant_pauses": "ghost_attribution",
            "minor_hesitation": "drift_substitution"
        }
        
        for pattern_type, count in pattern_counts.items():
            hesitation_type = pattern_type_mapping.get(pattern_type, "drift_substitution")
            distribution[hesitation_type] += count
        
        # Convert to frequencies
        total = sum(distribution.values()) or 1  # Avoid division by zero
        for hesitation_type in distribution:
            distribution[hesitation_type] /= total
        
        return distribution
    
    # Methods for comparing analyses
    
    def _compare_null_regions(self, analysis1: Dict[str, Any], analysis2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare null regions between two analyses.
        
        Args:
            analysis1: First drift analysis
            analysis2: Second drift analysis
            
        Returns:
            Comparison of null regions
        """
        region1 = analysis1.get("null_regions", {})
        region2 = analysis2.get("null_regions", {})
        
        intensity1 = np.mean(region1.get("intensity", [0])) if region1.get("intensity") else 0
        intensity2 = np.mean(region2.get("intensity", [0])) if region2.get("intensity") else 0
        
        triggers1 = region1.get("triggers", [])
        triggers2 = region2.get("triggers", [])
        
        trigger_freq1 = {}
        for trigger in triggers1:
            trigger_freq1[trigger] = trigger_freq1.get(trigger, 0) + 1
            
        trigger_freq2 = {}
        for trigger in triggers2:
            trigger_freq2[trigger] = trigger_freq2.get(trigger, 0) + 1
        
        trigger_diff = {}
        all_triggers = set(trigger_freq1.keys()) | set(trigger_freq2.keys())
        for trigger in all_triggers:
            count1 = trigger_freq1.get(trigger, 0)
            count2 = trigger_freq2.get(trigger, 0)
            trigger_diff[trigger] = count2 - count1
        
        return {
            "intensity_diff": intensity2 - intensity1,
            "count_diff": len(region2.get("regions", [])) - len(region1.get("regions", [])),
            "trigger_diff": trigger_diff
        }
    
    def _compare_hesitation_patterns(self, analysis1: Dict[str, Any], analysis2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare hesitation patterns between two analyses.
        
        Args:
            analysis1: First drift analysis
            analysis2: Second drift analysis
            
        Returns:
            Comparison of hesitation patterns
        """
        patterns1 = analysis1.get("hesitation_patterns", {})
        patterns2 = analysis2.get("hesitation_patterns", {})
        
        # Compare pattern types
        pattern_types1 = patterns1.get("pattern_types", {})
        pattern_types2 = patterns2.get("pattern_types", {})
        
        pattern_diff = {}
        all_patterns = set(pattern_types1.keys()) | set(pattern_types2.keys())
        for pattern in all_patterns:
            count1 = pattern_types1.get(pattern, 0)
            count2 = pattern_types2.get(pattern, 0)
            pattern_diff[pattern] = count2 - count1
        
        # Compare severity distributions
        severity1 = np.mean(patterns1.get("severity_distribution", [0])) if patterns1.get("severity_distribution") else 0
        severity2 = np.mean(patterns2.get("severity_distribution", [0])) if patterns2.get("severity_distribution") else 0
        
        return {
            "pattern_diff": pattern_diff,
            "severity_diff": severity2 - severity1
        }
    
    def _compare_attribution_pathways(self, analysis1: Dict[str, Any], analysis2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare attribution pathways between two analyses.
        
        Args:
            analysis1: First drift analysis
            analysis2: Second drift analysis
            
        Returns:
            Comparison of attribution pathways
        """
        pathways1 = analysis1.get("attribution_pathways", {})
        pathways2 = analysis2.get("attribution_pathways", {})
        
        nodes1 = set(pathways1.get("nodes", []))
        nodes2 = set(pathways2.get("nodes", []))
        
        sources1 = set(pathways1.get("sources", []))
        sources2 = set(pathways2.get("sources", []))
        
        conflicts1 = len(pathways1.get("conflicts", []))
        conflicts2 = len(pathways2.get("conflicts", []))
        
        return {
            "node_overlap": len(nodes1 & nodes2) / max(len(nodes1 | nodes2), 1),
            "source_diff": list(sources2 - sources1),
            "conflict_diff": conflicts2 - conflicts1
        }
    
    def _compare_drift_signatures(self, analysis1: Dict[str, Any], analysis2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare drift signatures between two analyses.
        
        Args:
            analysis1: First drift analysis
            analysis2: Second drift analysis
            
        Returns:
            Comparison of drift signatures
        """
        signature1 = analysis1.get("drift_signature", {})
        signature2 = analysis2.get("drift_signature", {})
        
        diff = {}
        for metric in ["null_ratio", "hesitation_index", "attribution_coherence", "regeneration_frequency", "drift_amplitude"]:
            val1 = signature1.get(metric, 0.0)
            val2 = signature2.get(metric, 0.0)
            diff[metric] = val2 - val1
        
        return diff
    
    def _compare_domain_sensitivities(self, analysis1: Dict[str, Any], analysis2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare domain sensitivities between two analyses.
        
        Args:
            analysis1: First drift analysis
            analysis2: Second drift analysis
            
        Returns:
            Comparison of domain sensitivities
        """
        sensitivity1 = analysis1.get("domain_sensitivity", {})
        sensitivity2 = analysis2.get("domain_sensitivity", {})
        
        diff = {}
        for domain in self.domains:
            val1 = sensitivity1.get(domain, 0.0)
            val2 = sensitivity2.get(domain, 0.0)
            diff[domain] = val2 - val1
        
        return diff
    
    # Visualization methods
    
    def _plot_null_regions(self, null_regions: Dict[str, Any], ax: plt.Axes) -> None:
        """
        Plot null regions.
        
        Args:
            null_regions: Null region data
            ax: Matplotlib axes
        """
        regions = null_regions.get("regions", [])
        intensities = null_regions.get("intensity", [])
        triggers = null_regions.get("triggers", [])
        
        if not regions or not intensities:
            ax.text(0.5, 0.5, "No null regions detected", ha='center', va='center')
            return
        
        # Create positions for regions
        positions = list(range(len(regions)))
        
        # Plot regions as bars
        bars = ax.barh(positions, [1] * len(positions), height=0.8, left=0, color='lightgray')
        
        # Color bars by intensity
        cmap = cm.get_cmap('Reds')
        for i, (bar, intensity) in enumerate(zip(bars, intensities)):
            bar.set_color(cmap(intensity))
            
            # Add trigger labels
            if i < len(triggers):
                ax.text(0.1, positions[i], triggers[i], ha='left', va='center')
        
        # Set y-axis labels
        ax.set_yticks(positions)
        ax.set_yticklabels([f"Region {i+1}" for i in range(len(positions))])
        
        ax.set_xlabel("Null Region")
        ax.set_title("Null Regions by Intensity and Trigger")
    
    def _plot_hesitation_distribution(self, distribution: Dict[str, float], ax: plt.Axes) -> None:
        """
        Plot hesitation pattern distribution.
        
        Args:
            distribution: Hesitation distribution data
            ax: Matplotlib axes
        """
        if not distribution:
            ax.text(0.5, 0.5, "No hesitation patterns detected", ha='center', va='center')
            return
        
        # Extract labels and values
        labels = list(distribution.keys())
        values = list(distribution.values())
        
        # Create bar plot
        bars = ax.bar(labels, values, color='skyblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Customize plot
        ax.set_xlabel("Hesitation Pattern Type")
        ax.set_ylabel("Frequency")
        ax.set_ylim(0, max(values) * 1.2)  # Add some space for labels
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_attribution_pathways(self, attribution_pathways: Dict[str, Any], ax: plt.Axes) -> None:
        """
        Plot attribution pathway network.
        
        Args:
            attribution_pathways: Attribution pathway data
            ax: Matplotlib axes
        """
        nodes = attribution_pathways.get("nodes", [])
        edges = attribution_pathways.get("edges", [])
        
        if not nodes or not edges:
            ax.text(0.5, 0.5, "No attribution pathways detected", ha='center', va='center')
            return
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in nodes:
            G.add_node(node)
        
        # Add edges
        for edge in edges:
            if isinstance(edge, list) and len(edge) >= 2:
                G.add_edge(edge[0], edge[1])
            elif isinstance(edge, dict) and 'source' in edge and 'target' in edge:
                G.add_edge(edge['source'], edge['target'])
        
        # Draw graph
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
        
        ax.set_title("Attribution Pathway Network")
        ax.axis('off')
    
    def _plot_domain_sensitivity(self, domain_sensitivity: Dict[str, float], ax: plt.Axes) -> None:
        """
        Plot domain sensitivity radar chart.
        
        Args:
            domain_sensitivity: Domain sensitivity data
            ax: Matplotlib axes
        """
        # Extract domains and values
        domains = list(domain_sensitivity.keys())
        values = list(domain_sensitivity.values())
        
        # Number of domains
        N = len(domains)
        
        # Create angles for radar chart
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        
        # Close the loop
        values += [values[0]]
        angles += [angles[0]]
        domains += [domains[0]]
        
        # Plot radar
        ax.fill(angles, values, color='skyblue', alpha=0.4)
        ax.plot(angles, values, 'o-', color='blue', linewidth=2)
        
        # Set ticks and labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(domains[:-1])
        
        # Set y-limits
        ax.set_ylim(0, 1)
        
        # Set title
        ax.set_title("Domain Sensitivity", va='bottom')
