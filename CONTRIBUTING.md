# Contributing to the Emergent Turing Test

We welcome contributions from the interpretability research community. The Emergent Turing Test is an evolving framework designed to map the cognitive boundaries of language models through hesitation patterns and attribution drift.

## Core Design Principles

When contributing to this project, please keep these foundational principles in mind:

1. **Interpretability Through Hesitation**: The framework prioritizes interpreting model behavior through where it hesitates, not just where it succeeds.

2. **Open-Ended Diagnostics**: Tests are designed to map behavior, not pass/fail models. They reveal interpretive landscapes, not singular verdicts.

3. **Signal in Silence**: Null outputs and refusals contain rich interpretive information about model boundaries.

4. **Integration-First Architecture**: Components should seamlessly integrate with existing interpretability tools and frameworks.

5. **Evidence-Based Expansion**: New test modules should be based on observable hesitation patterns in real model behavior.

## Contribution Areas

We particularly welcome contributions in these areas:

### Test Modules

- **New Cognitive Strain Patterns**: Novel ways to induce and measure specific types of model hesitation
- **Domain-Specific Collapse Tests**: Tests targeting specialized knowledge domains or reasoning types
- **Cross-Model Calibration**: Methods to ensure test comparability across different model architectures

### Drift Metrics

- **Novel Hesitation Metrics**: New ways to quantify model hesitation patterns
- **Attribution Analysis**: Improved methods for tracing information flow during hesitation
- **Visualization Tools**: Better ways to map and visualize drift patterns

### Integration Extensions

- **Framework Connectors**: Tools to integrate with other interpretability frameworks
- **Model Adapters**: Support for additional model architectures
- **Dataset Collections**: Curated test cases that reveal interesting drift patterns

## Contribution Process

1. **Discuss First**: For significant contributions, open an issue to discuss your idea before implementing

2. **Follow Standards**: Follow the existing code style and documentation patterns

3. **Test Thoroughly**: Include unit tests for any new functionality

4. **Explain Intent**: Document not just what your code does, but why it matters for interpretability

5. **Submit PR**: Create a pull request with a clear description of the contribution

## Development Setup

```bash
# Clone the repository
git clone https://github.com/caspiankeyes/emergent-turing.git
cd emergent-turing

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Code Style

We follow standard Python style guidelines:

- Use meaningful variable and function names
- Document functions with docstrings
- Keep functions focused on a single responsibility
- Write tests for new functionality
- Use type hints where appropriate

## Ethical Considerations

The Emergent Turing Test is designed to improve model interpretability, which has important ethical implications:

- **Dual Use**: Be mindful that techniques for inducing model hesitation could potentially be misused
- **Privacy**: Ensure test suites don't unnecessarily expose user data or private model information
- **Representation**: Consider how test design might impact different stakeholders and communities
- **Transparency**: Document limitations and potential biases in test methods

We are committed to developing this framework in a way that advances beneficial uses of AI while mitigating potential harms.

## Questions?

If you have questions about contributing, please open an issue or reach out to the project maintainers. We're excited to collaborate with the interpretability research community on this evolving framework.

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.
