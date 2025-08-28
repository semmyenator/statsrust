# statsrust - Statistical Analysis Library for Rust

![Apache 2.0 | MIT License](https://img.shields.io/badge/License-Apache%202.0%20%7C%20MIT-blue.svg)
![CC BY 4.0 License](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)

**statsrust** is a high-performance statistical analysis library for Rust developers, designed with emphasis on numerical stability, mathematical correctness, and user-friendly API.

## Features

- **Descriptive Statistics**: Comprehensive implementation of measures of central tendency, position, and variability
- **Kernel Density Estimation**: Multiple kernel functions with efficient implementation
- **Normal Distribution Model**: Complete algebraic operations on normal distributions
- **Numerical Stability**: Carefully designed algorithms to prevent overflow, underflow, and cancellation errors
- **Comprehensive Error Handling**: Detailed error messages for edge cases
- **Generic Input Support**: Works with various numeric types

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
statsrust = "0.1.0"  # Replace with the latest version
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
    
    // Kernel Density Estimation
    let kde = kde(&data, 0.5, "normal", false)?;
    println!("Density at 3.0: {:.4}", kde(3.0));
    
    // Normal distribution operations
    let dist = NormalDist::from_samples(&data)?;
    println!("Distribution: N(μ={:.2}, σ={:.2})", dist.mean(), dist.stdev());
    
    Ok(())
}
```

## Key Highlights

### Numerical Stability
Our algorithms are designed to avoid common numerical issues:
- Geometric mean uses logarithmic transformation to prevent overflow
- Variance calculation uses centered data to avoid catastrophic cancellation
- Inverse CDF approximations maintain precision while balancing performance

### Kernel Density Estimation
Efficient implementation with multiple kernel functions:
- Gaussian, Epanechnikov, Triangular, Quartic, Triweight, and more
- Optimized for bounded kernels using binary search
- Supports both PDF and CDF estimation
- Random sample generation from estimated distributions

### Normal Distribution Model
Complete algebraic operations:
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

## Contributing

We welcome contributions! Please see our [Contribution Guidelines](Contributing.md) for details on how to report bugs, suggest enhancements, or submit pull requests.

## License

This project is dual-licensed under:
- [Apache License 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

Documentation content is licensed under:
- [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE-CC)

---

*This document is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).*  
*Original author: statsrust Authors*