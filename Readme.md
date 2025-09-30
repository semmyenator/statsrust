# statsrust - Lean Statistical Analysis Library for Rust

![Apache 2.0 | MIT License](https://img.shields.io/badge/License-Apache%202.0%20%7C%20MIT-blue.svg)
![CC BY 4.0 License](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)

## Intent & Purpose
statsrust is built to address a specific need: a minimal, high-performance statistical analysis library for Rust developers that prioritizes **numerical stability**, **mathematical correctness**, and **zero-cost abstractions**. It is intentionally designed to serve use cases where control, performance, and minimal dependency overhead are critical—avoiding the bloat of general-purpose statistical libraries.


## Core Specifications
### Functional Scope
statsrust provides rigorously implemented core statistical functionality, including:
- Descriptive statistics (measures of central tendency, position, and variability)
- Non-parametric methods (Kernel Density Estimation with multiple kernel functions)
- Parametric distributions (focused on normal distribution operations)


### Design Constraints & Organizing Principles
To achieve its purpose, the library adheres to these intentional design decisions:
- **Direct enum usage** over string-to-enum conversion (reduces runtime overhead)
- **Function pointers** (`fn(f64) -> f64`) instead of trait objects (`Box<dyn Fn>`) (minimizes indirection)
- **Pure `Vec<f64>` implementation** without `ndarray` dependency (simplifies integration)
- **Hand-rolled core algorithms** (avoids unnecessary dependencies while ensuring numerical stability)
- **GitHub-only distribution** (not published to crates.io) (targets specialized use cases)


## Usage
This lean implementation is designed for direct integration via GitHub:

```toml
[dependencies]
statsrust = { git = "https://github.com/semmyenator/statsrust" }
```


## Documentation
Comprehensive specifications and guidance are available in:
- [API Reference](Api.md) (technical interfaces and contracts)
- [Mathematical Foundations](Mathlogic.md) (theoretical underpinnings and correctness guarantees)
- [Usage Guide](Instructions.md) (practical implementation patterns)
- [Contribution Guidelines](Contributing.md) (how to extend the library while preserving its intent)


## Scope Clarification
This implementation is intentionally minimalistic:
- Optimized for specific use cases rather than general-purpose statistical analysis
- Prioritizes control and performance over broad-case robustness
- Serves as a specialized tool, not a replacement for comprehensive statistical libraries


## Contributing
We welcome contributions that align with the library's core intent. Please review our [Contribution Guidelines](Contributing.md) for details on how to propose enhancements, report issues, or submit pull requests while maintaining the library's design constraints.


## License
This project uses a dual-licensing model:
- Software code (Rust source) is licensed under both [Apache License 2.0](LICENSE-APACHE) and [MIT License](LICENSE-MIT)
- Documentation and non-code content are licensed under [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE-CC)

This document is licensed under CC BY 4.0.  
Original author: statsrust Authors
