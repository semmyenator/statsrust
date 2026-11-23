# StatsRust

**StatsRust** is a lightweight, zero-dependency statistical library for Rust, inspired by Python's standard `statistics` module.

Built entirely on top of the Rust standard library (`std`), it requires no third-party crates (such as `rand`, `statrs`, `num_traits`, or `thiserror`). This makes it ideal for embedded systems, WebAssembly, rapid prototyping, or any project aiming to keep its dependency tree minimal and compile times fast.

## ✨ Key Features

* **Zero External Dependencies**: Relies strictly on Rust `std`, ensuring fast compilation and zero dependency conflicts.
* **Comprehensive Descriptive Statistics**: Covers measures of central tendency (mean, median, mode), spread (variance, standard deviation, quantiles), and bivariate relationships (correlation, covariance, linear regression).
* **Kernel Density Estimation (KDE)**: Includes 9 built-in kernels supporting Probability Density Functions (PDF), Cumulative Distribution Functions (CDF), and random sampling (`kde_random`).
* **Normal Distribution (`NormalDist`)**: Supports PDF, CDF, Inverse CDF, Z-scores, distribution overlap calculation, Box-Muller random sampling, and arithmetic operator overloading (`+`, `-`, `*`, `/`) for linear combinations of normal random variables.
* **Generic Numeric Support**: Powered by a built-in `ToPrimitive` trait, allowing seamless input of slices (`&[T]`) containing `i32`, `u64`, `f32`, `f64`, and other primitive numeric types.

## 📦 Installation & Integration

Add `statsrust` to your `Cargo.toml` (if using locally or via path/git):

```toml
[dependencies]
statsrust = { path = "." }
```

Alternatively, since it is a self-contained single-file library, you can copy `src/lib.rs` (or `statsrust.rs`) directly into your project and import it as a module:

```rust
mod statsrust;
use statsrust::*;
```

## 🚀 Quick Start

### 1. Central Tendency & Spread

```rust
use statsrust::{
    fmean, geometric_mean, harmonic_mean, mean, median, mode, multimode, quantiles, stdev, variance,
};

fn main() -> Result<(), statsrust::StatsrustError> {
    let data = vec![10.0, 20.0, 20.0, 30.0, 40.0, 50.0];
    let weights = vec![1.0, 1.0, 2.0, 1.0, 1.0, 1.0];

    // Averages
    println!("Arithmetic Mean: {}", mean(&data)?);
    println!("Weighted Mean: {}", fmean(&data, Some(&weights))?);
    println!("Geometric Mean: {}", geometric_mean(&data)?);
    println!("Harmonic Mean: {}", harmonic_mean(&data, None)?);

    // Median & Mode
    println!("Median: {}", median(&data)?);
    println!("Mode: {}", mode(&[1, 2, 2, 3])?);
    println!("Multimode: {:?}", multimode(&[1, 1, 2, 2, 3]));

    // Variance & Standard Deviation (Sample)
    println!("Sample Stdev: {}", stdev(&data, None)?);
    println!("Sample Variance: {}", variance(&data, None)?);

    // Quartiles
    println!("Quartiles (inclusive): {:?}", quantiles(&data, 4, "inclusive")?);

    Ok(())
}
```

### 2. Correlation & Linear Regression

```rust
use statsrust::{correlation, covariance, linear_regression};

fn main() -> Result<(), statsrust::StatsrustError> {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];

    // Pearson ("linear") and Spearman ("ranked") correlation
    let r = correlation(&x, &y, "linear")?;
    let rho = correlation(&x, &y, "ranked")?;
    let cov = covariance(&x, &y)?;

    // Ordinary Least Squares (OLS) linear regression: (slope, intercept)
    let (slope, intercept) = linear_regression(&x, &y, false)?;

    println!("Correlation r = {:.4}, rho = {:.4}", r, rho);
    println!("Covariance = {:.4}", cov);
    println!("Regression line: y = {:.4}x + {:.4}", slope, intercept);

    Ok(())
}
```

### 3. Normal Distribution (`NormalDist`)

```rust
use statsrust::NormalDist;

fn main() -> Result<(), statsrust::StatsrustError> {
    // Create a normal distribution N(mu = 100.0, sigma = 15.0)
    let iq = NormalDist::new(100.0, 15.0)?;

    println!("PDF at 115: {:.5}", iq.pdf(115.0));
    println!("CDF at 130: {:.5}", iq.cdf(130.0));
    println!("95th percentile: {:.2}", iq.inv_cdf(0.95)?);
    println!("Z-score (130): {:.2}", iq.zscore(130.0)?);

    // Linear combinations of independent normal distributions
    let d1 = NormalDist::new(10.0, 3.0)?;
    let d2 = NormalDist::new(20.0, 4.0)?;
    let sum_dist = d1 + d2; // N(30.0, 5.0)
    println!("Sum distribution: mean = {}, stdev = {}", sum_dist.mean(), sum_dist.stdev());

    // Overlapping coefficient between two distributions
    println!("Overlap: {:.4}", d1.overlap(&d2));

    // Generate random samples (with optional seed for reproducibility)
    let samples = iq.samples(1000, Some(42));
    let fitted = NormalDist::from_samples(&samples)?;
    println!("Fitted: mean = {:.2}, stdev = {:.2}", fitted.mean(), fitted.stdev());

    Ok(())
}
```

### 4. Kernel Density Estimation (KDE)

```rust
use statsrust::{kde, kde_random, Kernel};

fn main() -> Result<(), statsrust::StatsrustError> {
    let data = vec![1.2, 1.5, 1.8, 2.0, 2.1, 3.5, 3.8, 4.0];
    let bandwidth = 0.5;

    // Estimate Probability Density Function (PDF)
    let pdf_fn = kde(&data, bandwidth, Kernel::Parabolic, false)?;
    println!("Density estimate at x = 2.0: {:.4}", pdf_fn(2.0));

    // Estimate Cumulative Distribution Function (CDF)
    let cdf_fn = kde(&data, bandwidth, Kernel::Normal, true)?;
    println!("Estimated P(X <= 2.5): {:.4}", cdf_fn(2.5));

    // Draw random samples from the KDE
    let mut sampler = kde_random(&data, bandwidth, Kernel::Triangular, Some(123))?;
    println!("Random KDE sample: {:.4}", sampler());

    Ok(())
}
```

## 📚 API Overview

### Averages & Measures of Central Location
* `mean(data)` — Arithmetic mean
* `fmean(data, weights)` — Floating-point weighted arithmetic mean
* `geometric_mean(data)` — Geometric mean
* `harmonic_mean(data, weights)` — Harmonic mean (with optional weights)
* `median(data)` — Median (middle value)
* `median_low(data)` — Low median
* `median_high(data)` — High median
* `median_grouped(data, interval)` — Interpolated median of grouped continuous data
* `mode(data)` — Single mode (most common value)
* `multimode(data)` — List of most frequently occurring values

### Spread & Bivariate Relationships
* `pstdev(data, mu)` / `pvariance(data, mu)` — Population standard deviation and variance
* `stdev(data, xbar)` / `variance(data, xbar)` — Sample standard deviation and variance
* `quantiles(data, n, method)` — Divide data into $n$ continuous intervals (`"inclusive"` or `"exclusive"`)
* `covariance(x, y)` — Sample covariance
* `correlation(x, y, method)` — Pearson's (`"linear"`) or Spearman's (`"ranked"`) correlation coefficient
* `linear_regression(x, y, proportional)` — Slope and intercept for simple linear regression

### Supported Kernels (`Kernel`)
Construct via the `Kernel` enum or `Kernel::from_name(&str)`:
* `Kernel::Normal` (`"normal"`, `"gauss"`)
* `Kernel::Logistic` (`"logistic"`)
* `Kernel::Sigmoid` (`"sigmoid"`)
* `Kernel::Rectangular` (`"rectangular"`, `"uniform"`)
* `Kernel::Triangular` (`"triangular"`)
* `Kernel::Parabolic` (`"parabolic"`, `"epanechnikov"`)
* `Kernel::Quartic` (`"quartic"`, `"biweight"`)
* `Kernel::Triweight` (`"triweight"`)
* `Kernel::Cosine` (`"cosine"`)

## 📄 License

Licensed under either of:

* **Apache License, Version 2.0** ([LICENSE-APACHE](http://www.apache.org/licenses/LICENSE-2.0))
* **MIT License** ([LICENSE-MIT](http://opensource.org/licenses/MIT))

at your option.
