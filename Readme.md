# statsrust - Lean Statistical Analysis Library for Rust

![Apache 2.0 | MIT License](https://img.shields.io/badge/License-Apache%202.0%20%7C%20MIT-blue.svg)
![CC BY 4.0 License](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)

statsrust is a minimal, high-performance statistical analysis library for Rust developers, designed with emphasis on numerical stability, mathematical correctness, and zero-cost abstractions. This lean version focuses on core statistical functionality without heavy dependencies.

## Key Design Decisions

- **Direct enum usage** instead of string-to-enum conversion
- **Function pointers** (`fn(f64) -> f64`) instead of trait objects (`Box<dyn Fn>`)
- **Pure Vec<f64> implementation** without ndarray dependency
- **Hand-rolled core algorithms** instead of statrs dependency
- **GitHub-only distribution** (not published to crates.io) for specific use cases

## Features

- **Descriptive Statistics**: Measures of central tendency, position, and variability
- **Kernel Density Estimation**: Multiple kernel functions with efficient implementation
- **Normal Distribution Model**: Complete algebraic operations on normal distributions
- **Numerical Stability**: Carefully designed algorithms to prevent overflow, underflow, and cancellation errors
- **Comprehensive Error Handling**: Detailed error messages for edge cases
- **Zero-cost Abstractions**: Function pointers and direct enum usage for optimal performance

## Usage

This lean version is designed for direct integration via GitHub rather than crates.io:

```toml
[dependencies]
statsrust = { git = "https://github.com/semmyenator/statsrust" }
```

## Quick Start

```rust
use statsrust::*;

fn main() -> Result<(), StatError> {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    // Basic descriptive statistics
    let mean = mean(&data)?;
    let median = median(&data)?;
    let variance = variance(&data, None)?;
    
    println!("Mean: {:.2}, Median: {:.2}, Variance: {:.2}", mean, median, variance);
    
    // Kernel Density Estimation (using enum directly)
    let kde = kde(&data, 0.5, Kernel::Normal, false)?;
    println!("Density at 3.0: {:.4}", kde(3.0));
    
    // Normal distribution operations
    let dist = NormalDist::from_samples(&data)?;
    println!("Distribution: N(μ={:.2}, σ={:.2})", dist.mean(), dist.stdev());
    
    Ok(())
}
```

## Key Highlights

### Zero-Cost Abstractions
- Direct enum usage eliminates runtime string parsing
- Function pointers (`fn(f64) -> f64`) replace trait objects for zero allocation overhead
- No "magic strings" - kernels are specified via `Kernel::Normal` enum variants

### Numerical Stability
- Geometric mean uses logarithmic transformation to prevent overflow
- Variance calculation uses centered data to avoid catastrophic cancellation
- Inverse CDF approximations maintain precision while balancing performance

### Kernel Density Estimation
- Efficient implementation with multiple kernel functions:
  - Gaussian, Epanechnikov, Triangular, Quartic, Triweight, and more
- Optimized for bounded kernels using binary search
- Direct function pointer implementation for minimal overhead

### Normal Distribution Model
- Complete algebraic operations:
  ```rust
  let dist1 = NormalDist::new(0.0, 1.0)?;
  let dist2 = NormalDist::new(1.0, 2.0)?;
  
  // Distribution operations
  let sum_dist = dist1 + dist2;  // N(1.0, √5)
  let scaled_dist = dist1 * 2.0; // N(0.0, 2.0)
  let overlap = dist1.overlap(&dist2); // Calculate distribution overlap
  ```

## Documentation
For comprehensive documentation:
- [API Reference](Api.md)
- [Mathematical Foundations](Mathlogic.md)
- [Usage Guide](Instructions.md)
- [Contribution Guidelines](Contributing.md)

## Note on Library Scope

[leaner version of statsrust](https://github.com/semmyenator/statsrust/blob/main/src/statsrust.rs)

This leaner version of statsrust is intentionally minimalistic:
- It's optimized for specific use cases rather than being a general-purpose statistical library
- It prioritizes control and performance over broad-case robustness
- It's hosted on GitHub only (not published to crates.io)
- It's designed as a specialized tool, not a replacement for comprehensive statistical libraries

## Contributing
We welcome contributions! Please see our [Contribution Guidelines](Contributing.md) for details on how to report bugs, suggest enhancements, or submit pull requests.

## License
This project is dual-licensed under:
- [Apache License 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

Documentation content is licensed under:
- [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE-CC)

This document is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
Original author: statsrust Authors

