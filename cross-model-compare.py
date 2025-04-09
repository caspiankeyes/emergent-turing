#!/usr/bin/env python
# examples/cross_model_compare.py

import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from emergent_turing.core import EmergentTest
from emergent_turing.drift_map import DriftMap
from emergent_turing.metrics import MetricSuite

def parse_args():
    parser = argparse.ArgumentParser(description="Run Emergent Turing test comparisons across models")
    parser.add_argument("--models", nargs="+", default=["claude-3-7-sonnet", "gpt-4o"], 
                      help="Models to test")
    parser.add_argument("--module", type=str, default="instruction-drift",
                      choices=["instruction-drift", "identity-strain", "value-conflict", 
                              "memory-destabilization", "attention-manipulation"],
                      help="Test module to run")
    parser.add_argument("--intensity", type=float, default=0.7,
                      help="Test intensity level (0.0-1.0)")
    parser.add_argument("--output-dir", type=str, default="results",
                      help="Directory to save test results")
    parser.add_argument("--measure-attribution", action="store_true",
                      help="Measure attribution patterns")
    parser.add_argument("--record-hesitation", action="store_true",
                      help="Record token-level hesitation patterns")
    return parser.parse_args()

def setup_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def run_comparison(args):
    """Run comparison across models."""
    print(f"Running {args.module} test on models: {', '.join(args.models)}")
    print(f"Test intensity: {args.intensity}")
    
    # Set up output directory
    output_path = setup_output_dir(args.output_dir)
    
    # Initialize drift map for visualization
    drift_map = DriftMap()
    
    # Initialize metric suite
    metrics = MetricSuite()
    
    # Store results for each model
    all_results = {}
    
    # Run test on each model
    for model in args.models:
        print(f"\nTesting model: {model}")
        
        # Initialize test
        test = EmergentTest(model=model)
        
        # Create test parameters
        params = {
            "intensity": args.intensity
        }
        
        # Add module-specific parameters
        if args.module == "instruction-drift":
            params["subject"] = "The impact of artificial intelligence on society"
            params["domain"] = "ethics"
        elif args.module == "value-conflict":
            params["scenario"] = "ethical_dilemma"
        elif args.module == "memory-destabilization":
            params["context_length"] = "medium"
        elif args.module == "attention-manipulation":
            params["content_type"] = "factual"
        
        # Run test module
        result = test.run_module(
            args.module,
            params=params,
            record_hesitation=args.record_hesitation,
            measure_attribution=args.measure_attribution
        )
        
        # Store result
        all_results[model] = result
        
        # Calculate metrics
        model_metrics = metrics.compute_all(result)
        print(f"  Metrics for {model}:")
        for metric_name, metric_value in model_metrics.items():
            if isinstance(metric_value, dict) or metric_value is None:
                continue
            print(f"    {metric_name}: {metric_value:.4f}")
    
    # Create comparative visualization
    visualize_comparison(all_results, args, output_path)
    
    # Save raw results
    for model, result in all_results.items():
        result_path = output_path / f"{model}_{args.module}_result.json"
        with open(result_path, "w") as f:
            # Convert result to JSON-serializable format
            import json
            json.dump(serialize_result(result), f, indent=2)
    
    print(f"\nResults saved to {output_path}")

def serialize_result(result):
    """Convert result to JSON-serializable format."""
    import numpy as np
    import json
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super(NumpyEncoder, self).default(obj)
    
    # First convert to JSON and back to handle NumPy types
    result_json = json.dumps(result, cls=NumpyEncoder)
    return json.loads(result_json)

def visualize_comparison(all_results, args, output_path):
    """Create visualizations comparing model results."""
    # Extract metric values for comparison
    metric_values = {}
    
    for model, result in all_results.items():
        # Calculate null ratio
        null_ratio = result.get("null_ratio", 0.0)
        if not metric_values.get("null_ratio"):
            metric_values["null_ratio"] = {}
        metric_values["null_ratio"][model] = null_ratio
        
        # Calculate hesitation depth if available
        if args.record_hesitation:
            hesitation_depth = 0.0
            hesitation_map = result.get("hesitation_map")
            if hesitation_map:
                regeneration_count = hesitation_map.get("regeneration_count", [])
                if regeneration_count:
                    hesitation_depth = sum(regeneration_count) / len(regeneration_count)
            
            if not metric_values.get("hesitation_depth"):
                metric_values["hesitation_depth"] = {}
            metric_values["hesitation_depth"][model] = hesitation_depth
        
        # Calculate drift amplitude (combined metric)
        drift_amplitude = null_ratio * 0.5
        if args.record_hesitation:
            drift_amplitude += metric_values["hesitation_depth"].get(model, 0.0) * 0.5
        
        if not metric_values.get("drift_amplitude"):
            metric_values["drift_amplitude"] = {}
        metric_values["drift_amplitude"][model] = drift_amplitude
    
    # Create bar chart comparing metrics across models
    create_comparison_chart(metric_values, args, output_path)
    
    # Create detailed drift maps for each model
    for model, result in all_results.items():
        if "drift_analysis" in result:
            drift_map = DriftMap()
            output_file = output_path / f"{model}_{args.module}_drift_map.png"
            drift_map.visualize(
                result["drift_analysis"],
                title=f"{model} - {args.module} Drift Map",
                show_attribution=args.measure_attribution,
                show_hesitation=args.record_hesitation,
                output_path=str(output_file)
            )

def create_comparison_chart(metric_values, args, output_path):
    """Create bar chart comparing metrics across models."""
    # Convert to DataFrame for easier plotting
    metrics_to_plot = ["null_ratio", "hesitation_depth", "drift_amplitude"]
    available_metrics = [m for m in metrics_to_plot if m in metric_values]
    
    data = {}
    for metric in available_metrics:
        data[metric] = pd.Series(metric_values[metric])
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot
    df.plot(kind="bar", ax=ax)
    
    # Customize
    ax.set_title(f"Emergent Turing Test: {args.module} Comparison")
    ax.set_ylabel("Metric Value")
    ax.set_xlabel("Model")
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    output_file = output_path / f"comparison_{args.module}_metrics.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    run_comparison(args)
