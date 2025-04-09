# emergent-turing Repository Structure

```
emergent-turing/
├── README.md                # Main documentation
├── CONTRIBUTING.md          # Contribution guidelines
├── ETHICS.md                # Ethical considerations
├── LICENSE                  # MIT License
├── setup.py                 # Installation configuration
├── requirements.txt         # Dependencies
│
├── emergent_turing/         # Main package
│   ├── __init__.py
│   ├── core.py              # Core functionality
│   ├── drift_map.py         # Drift analysis tools
│   ├── metrics.py           # Hesitation metrics
│   ├── compatibility.py     # Model compatibility testing
│   ├── visualization.py     # Drift visualization tools
│   └── integration.py       # Integration with other frameworks
│
├── TestSuites/              # Cognitive strain test modules
│   ├── __init__.py
│   ├── instruction_drift/   # Instruction ambiguity and conflict tests
│   │   ├── __init__.py
│   │   ├── ambiguity.py     # Calibrated instruction ambiguity
│   │   ├── contradiction.py # Embedded contradictory instructions
│   │   └── entanglement.py  # Instruction frame conflicts
│   │
│   ├── identity_strain/     # Identity and self-reference tests
│   │   ├── __init__.py
│   │   ├── self_reference.py    # Self-reference loops
│   │   ├── boundary_blur.py     # Identity boundary testing
│   │   └── meta_collapse.py     # Meta-cognitive collapse
│   │
│   ├── value_conflict/      # Value system tests
│   │   ├── __init__.py
│   │   ├── ethical_dilemma.py   # Calibrated moral dilemmas
│   │   ├── preference_flip.py   # Preference reversal tests
│   │   └── constitution_test.py # Constitutional contradiction tests
│   │
│   ├── memory_destabilization/  # Memory and context tests
│   │   ├── __init__.py
│   │   ├── context_fragment.py  # Context breakdown tests
│   │   ├── temporal_shift.py    # Temporal coherence tests
│   │   └── causal_break.py      # Causal reasoning disruption
│   │
│   └── attention_manipulation/  # Attention and salience tests
│       ├── __init__.py
│       ├── salience_invert.py   # Attention priority inversion
│       ├── token_suppress.py    # Selective token suppression
│       └── attribution_leak.py  # Attribution pathway mapping
│
├── Shells/                  # Symbolic test scaffolds
│   ├── __init__.py
│   ├── null_shell.py        # Null output induction framework
│   ├── drift_shell.py       # Attribution drift shell
│   ├── hesitation_shell.py  # Token regeneration measurement
│   └── oscillation_shell.py # Value oscillation detection
│
├── DriftMaps/               # Attribution and hesitation maps
│   ├── __init__.py
│   ├── instruction_maps/    # Instruction-domain drift maps
│   ├── identity_maps/       # Identity-domain drift maps
│   ├── value_maps/          # Value-domain drift maps
│   ├── memory_maps/         # Memory-domain drift maps
│   └── attention_maps/      # Attention-domain drift maps
│
├── Metrics/                 # Measurement implementations
│   ├── __init__.py
│   ├── null_ratio.py        # Nullification measurement
│   ├── hesitation_depth.py  # Hesitation pattern analysis
│   ├── drift_coherence.py   # Drift stability metrics
│   ├── attribution_trace.py # Attribution pathway analysis
│   └── oscillation_freq.py  # Oscillation pattern metrics
│
├── Integration/             # Framework integration
│   ├── __init__.py
│   ├── pareto_integration.py    # Integration with pareto-lang
│   ├── residue_integration.py   # Integration with symbolic-residue
│   └── transformer_integration.py  # Integration with transformerOS
│
├── examples/                # Usage examples
│   ├── basic_testing.py     # Basic test suite usage
│   ├── drift_analysis.py    # Drift map analysis example
│   ├── cross_model_compare.py   # Cross-model comparison
│   └── visualization_example.py # Visualization examples
│
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_drift_map.py
│   ├── test_metrics.py
│   └── test_integration.py
│
└── docs/                    # Extended documentation
    ├── getting_started.md
    ├── drift_map_tutorial.md
    ├── test_suite_guide.md
    ├── metrics_explained.md
    ├── integration_guide.md
    └── figures/             # Diagrams and visualizations
        ├── drift_map_example.svg
        ├── hesitation_patterns.svg
        ├── null_ratio_comparison.svg
        └── attribution_trace_example.svg
```

## Key Components Overview

### TestSuites
The core cognitive strain modules that induce specific types of model hesitation, null outputs, and attribution drift.

### Shells
Symbolic scaffolds that wrap model interactions to capture hesitation patterns and attribution traces.

### DriftMaps
Structured representations of hesitation landscapes across different cognitive domains.

### Metrics
Measurement tools for quantifying various aspects of model hesitation and drift.

### Integration
Connectors to other interpretability frameworks like pareto-lang, symbolic-residue, and transformerOS.

## Architecture Design Principles

1. **Modularity**: Each test component works independently but integrates with the broader framework
2. **Extensibility**: Easy to add new test types, drift maps, and metrics
3. **Compatibility**: Works with a range of model architectures and scales
4. **Interpretability**: Results are designed to be meaningful and visualizable
5. **Integration**: Fits into existing interpretability ecosystems
