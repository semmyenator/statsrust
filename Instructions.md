# Rust statsrust Library - Comprehensive Usage Guide

![Apache 2.0 | MIT License](https://img.shields.io/badge/License-Apache%202.0%20%7C%20MIT-blue.svg)
![CC BY 4.0 License](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)

## Table of Contents

- [Introduction](#introduction)
- [Installation and Setup](#installation-and-setup)
- [Basic Error Handling](#basic-error-handling)
- [Core Statistical Functions](#core-statistical-functions)
  - [Measures of Central Tendency](#measures-of-central-tendency)
  - [Measures of Position](#measures-of-position)
  - [Measures of Variability](#measures-of-variability)
  - [Correlation and Regression](#correlation-and-regression)
  - [Quantile Calculations](#quantile-calculations)
- [Kernel Density Estimation (KDE)](#kernel-density-estimation-kde)
  - [Density Estimation](#density-estimation)
  - [Random Sample Generation](#random-sample-generation)
- [Normal Distribution Model](#normal-distribution-model)
  - [Creation and Properties](#creation-and-properties)
  - [Distribution Operations](#distribution-operations)
- [Advanced Usage Patterns](#advanced-usage-patterns)
- [Numerical Stability Best Practices](#numerical-stability-best-practices)
- [Troubleshooting Common Issues](#troubleshooting-common-issues)
- [Integration with Other Rust Libraries](#integration-with-other-rust-libraries)

## Introduction

The Rust statsrust Library is a comprehensive statistical analysis toolkit designed for Rust developers. It provides robust implementations of descriptive statistics, probability distributions, and non-parametric methods with emphasis on numerical stability, mathematical precision, and user-friendly error handling.

This guide provides detailed instructions for effectively using the library in your Rust projects, covering everything from basic setup to advanced statistical techniques. Unlike the [API documentation](Api.md) which focuses on technical specifications or the [Mathematical Foundations](Mathlogic.md) which details theoretical underpinnings, this guide emphasizes practical usage with real-world examples.

## Installation and Setup

### Adding to Your Project

Add the following to your `Cargo.toml`:

```toml
[dependencies]
statsrust = "0.1.0"  # Replace with the latest version
```

### Importing the Library

In your Rust code:

```rust
use statsrust::*;
// Or import specific modules:
use statsrust::{mean, median, NormalDist, kde};
```

### Required Dependencies

The library depends on these external crates, which will be automatically resolved when you build your project:

```toml
thiserror = "1.0"
num-traits = "0.2"
ndarray = "0.15"
statrs = "0.18"
rand = "0.8"
```

## Basic Error Handling

The library uses a comprehensive error enumeration system to handle various edge cases. All functions return `Result<T, StatError>`, so proper error handling is essential.

### Error Handling Pattern

```rust
// Using match pattern
match mean(&data) {
    Ok(m) => println!("Mean: {}", m),
    Err(e) => eprintln!("Error calculating mean: {}", e),
}

// Using the `?` operator in functions that return Result
fn process_data(data: &[f64]) -> Result<(), StatError> {
    let m = mean(data)?;
    let med = median(data)?;
    println!("Mean: {:.4}, Median: {:.4}", m, med);
    Ok(())
}
```

### Common Error Types and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `NoDataPoints` | Empty dataset | Check data length before operations |
| `InsufficientSampleData` | Not enough data points | Ensure dataset meets minimum requirements |
| `NegativeValueNotAllowed` | Negative values where positive required | Filter or transform data |
| `InvalidBandwidth` | Bandwidth ≤ 0 | Use positive bandwidth values |
| `ConstantInput` | All values identical | Check data variability |

## Core Statistical Functions

### Measures of Central Tendency

#### Arithmetic Mean

Calculates the average value of a dataset.

```rust
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let m = mean(&data)?;  // Returns 3.0
```

#### Weighted Mean

Calculates the mean with specified weights.

```rust
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let weights = vec![0.1, 0.2, 0.3, 0.2, 0.2];
let wm = fmean(&data, Some(&weights))?;  // Returns weighted average
```

#### Geometric Mean

Useful for averaging ratios or exponential growth rates.

```rust
let data = vec![1.0, 2.0, 4.0, 8.0];
let gm = geometric_mean(&data)?;  // Returns approximately 2.828 (2√2)
```

#### Harmonic Mean

Commonly used for averaging rates.

```rust
let speeds = vec![30.0, 60.0];  // km/h
let avg_speed = harmonic_mean(&speeds, None)?;  // Returns 40.0
```

### Measures of Position

#### Median

Returns the middle value of a sorted dataset.

```rust
let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
let m = median(&data)?;  // Returns 5.0
// For even-sized datasets, returns average of two middle values
let even_data = vec![1.0, 3.0, 5.0, 7.0];
let m_even = median(&even_data)?;  // Returns 4.0
```

#### Median Variants

Different methods for handling even-sized datasets:

```rust
let data = vec![1.0, 3.0, 5.0, 7.0];
let median_low = median_low(&data)?;  // Returns 3.0 (smaller middle value)
let median_high = median_high(&data)?;  // Returns 5.0 (larger middle value)
```

#### Grouped Median

For pre-grouped data with specified intervals:

```rust
let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0];
let grouped_median = median_grouped(&data, 1.0)?;  // Interval width = 1.0
```

### Measures of Variability

#### Variance and Standard Deviation

Two versions are provided: population (dividing by N) and sample (dividing by n-1).

```rust
// Population variance (dividing by N)
let pop_var = pvariance(&data, None)?;  // Uses calculated mean

// Sample variance (unbiased estimator, dividing by n-1)
let samp_var = variance(&data, None)?;

// Standard deviations
let pop_std = pstdev(&data, None)?;
let samp_std = stdev(&data, None)?;
```

### Correlation and Regression

#### Pearson Correlation

Measures linear relationship between two datasets:

```rust
let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
let corr = correlation(&x, &y, "linear")?;  // Returns Pearson correlation
```

#### Spearman Rank Correlation

Measures monotonic relationship:

```rust
let rank_corr = correlation(&x, &y, "ranked")?;  // Returns Spearman correlation
```

#### Covariance

Measures how two variables change together:

```rust
let cov = covariance(&x, &y)?;
```

#### Linear Regression

Fits a line to data points:

```rust
let (slope, intercept) = linear_regression(&x, &y, false)?;
// Returns slope and intercept of regression line

// For proportional regression (through origin):
let (slope, _) = linear_regression(&x, &y, true)?;
```

### Quantile Calculations

Calculates division points that split data into equal-sized intervals:

```rust
// Calculate quartiles (3 points dividing into 4 equal parts)
let quartiles = quantiles(&data, 4, "inclusive")?;

// Calculate deciles
let deciles = quantiles(&data, 10, "exclusive")?;
```

**Method Selection Guide:**
- `"inclusive"`: Use when data represents the entire population
- `"exclusive"`: Use when data is a sample from a larger population

## Kernel Density Estimation (KDE)

### Density Estimation

#### Creating a KDE Function

```rust
// Create a PDF estimation function
let kde_func = kde(&data, 1.0, "normal", false)?;  

// Evaluate at specific point
let density_at_3 = kde_func(3.0);

// Create a CDF estimation function
let cdf_func = kde(&data, 1.0, "normal", true)?;
let cdf_at_3 = cdf_func(3.0);
```

#### Supported Kernel Functions

| Kernel | Names | Support | Best For |
|--------|-------|---------|----------|
| Normal (Gaussian) | "normal", "gauss" | (-∞, ∞) | General purpose, smooth |
| Parabolic (Epanechnikov) | "parabolic", "epanechnikov" | [-1, 1] | Optimal MSE, efficient |
| Triangular | "triangular" | [-1, 1] | Simple, linear decay |
| Quartic (Biweight) | "quartic", "biweight" | [-1, 1] | Smoother than Epanechnikov |
| Rectangular | "rectangular", "uniform" | [-1, 1] | Simplest kernel |
| Triweight | "triweight" | [-1, 1] | Higher-order smoothness |

#### Bandwidth Selection

Choosing an appropriate bandwidth is crucial for good KDE results:

```rust
// Rule of thumb for bandwidth (Silverman's rule)
let n = data.len() as f64;
let std_dev = stdev(data, None).unwrap_or(1.0);
let iqr = if data.len() > 3 {
    let q = quantiles(data, 4, "inclusive").unwrap();
    q[2] - q[0]
} else {
    std_dev * 1.349 // Approximation when IQR can't be calculated
};
let bandwidth = 0.9 * n.powf(-0.2) * f64::min(std_dev, iqr / 1.349);
```

### Random Sample Generation

Generate random samples from the estimated distribution:

```rust
let mut kde_rand = kde_random(&data, 1.0, "normal", Some(42))?;
let sample = kde_rand();  // Generate a single random sample
let samples: Vec<f64> = (0..1000).map(|_| kde_rand()).collect();  // Generate 1000 samples
```

**Practical Tip:** Using a seed (`Some(42)`) ensures reproducible results for testing or simulations.

## Normal Distribution Model

### Creation and Properties

#### Creating a Normal Distribution

```rust
// Create with specific parameters
let dist = NormalDist::new(0.0, 1.0)?;  // Standard normal distribution

// Estimate from sample data
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let dist = NormalDist::from_samples(&data)?;
```

#### Distribution Properties

```rust
// Basic properties
let mean = dist.mean();        // Returns μ
let median = dist.median();    // Returns μ (same as mean)
let mode = dist.mode();        // Returns μ (same as mean)
let stdev = dist.stdev();      // Returns σ
let variance = dist.variance(); // Returns σ²

// Distribution functions
let pdf_val = dist.pdf(0.0)?;      // Probability density at 0.0
let cdf_val = dist.cdf(0.0)?;      // Cumulative probability up to 0.0
let inv_cdf_val = dist.inv_cdf(0.5)?; // Inverse CDF (median)
let z_score = dist.zscore(1.0)?;   // Z-score calculation

// Quantiles
let quartiles = dist.quantiles(4); // Returns 3 quartile points
```

#### Random Sampling

```rust
// Generate 1000 random samples
let samples = dist.samples(1000, Some(42)); 

// For reproducible results, use the same seed
let same_samples = dist.samples(1000, Some(42));
```

#### Overlap Calculation

Calculate the area where two normal distributions overlap:

```rust
let dist1 = NormalDist::new(0.0, 1.0)?;
let dist2 = NormalDist::new(1.0, 1.0)?;
let overlap = dist1.overlap(&dist2);  // Returns value between 0.0 and 1.0
```

### Distribution Operations

#### Scalar Operations

```rust
let dist = NormalDist::new(0.0, 1.0)?;

// Addition: N(μ+c, σ)
let dist_plus = dist + 2.0;  

// Subtraction: N(μ-c, σ)
let dist_minus = dist - 2.0;  

// Multiplication: N(μ×c, σ×|c|)
let dist_times = dist * 2.0;  

// Division: N(μ/c, σ/|c|)
let dist_div = dist / 2.0;    
```

#### Distribution Operations (Independent Distributions)

```rust
let dist1 = NormalDist::new(0.0, 1.0)?;
let dist2 = NormalDist::new(1.0, 2.0)?;

// Addition: N(μ₁+μ₂, √(σ₁²+σ₂²))
let sum_dist = dist1 + dist2;  

// Subtraction: N(μ₁-μ₂, √(σ₁²+σ₂²))
let diff_dist = dist1 - dist2; 
```

**Important:** These operations assume the distributions are independent.

## Advanced Usage Patterns

### Complete Statistical Analysis Example

```rust
use statsrust::*;

fn analyze_data(data: &[f64]) -> Result<(), StatError> {
    // Basic descriptive statistics
    let m = mean(data)?;
    let med = median(data)?;
    let mode_val = mode(data)?;
    let var = variance(data, None)?;
    let std_dev = stdev(data, None)?;
    
    println!("Mean: {:.4}, Median: {:.4}, Mode: {:.4}", m, med, mode_val);
    println!("Variance: {:.4}, Std Dev: {:.4}", var, std_dev);
    
    // Quantile analysis
    let quartiles = quantiles(data, 4, "inclusive")?;
    println!("Quartiles: [{:.4}, {:.4}, {:.4}]", 
             quartiles[0], quartiles[1], quartiles[2]);
    
    // Correlation analysis (if we have another dataset)
    let data2 = vec![2.1, 3.8, 5.2, 3.9, 5.1];
    let corr = correlation(data, &data2, "linear")?;
    let (slope, intercept) = linear_regression(data, &data2, false)?;
    
    println!("Correlation: {:.4}", corr);
    println!("Regression: y = {:.4}x + {:.4}", slope, intercept);
    
    // KDE analysis
    let kde_func = kde(data, 0.5, "epanechnikov", false)?;
    println!("Density at mean: {:.4}", kde_func(m));
    
    // Normal distribution fit
    let normal_dist = NormalDist::from_samples(data)?;
    println!("Normal fit: N(μ={:.4}, σ={:.4})", 
             normal_dist.mean(), normal_dist.stdev());
    
    Ok(())
}
```

### KDE Visualization Example

```rust
use statsrust::*;
use std::fs::File;
use std::io::Write;

fn generate_kde_plot(data: &[f64], filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Create KDE function
    let kde = kde(data, 0.5, "gaussian", false)?;
    
    // Determine plot range
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min) - 2.0;
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 2.0;
    
    // Generate points for plotting
    let num_points = 100;
    let step = (max - min) / num_points as f64;
    let mut points = Vec::with_capacity(num_points); 
    
    for i in 0..num_points {
        let x = min + i as f64 * step;
        let y = kde(x);
        points.push((x, y));
    }
    
    // Write to CSV
    let mut file = File::create(filename)?;
    writeln!(file, "x,y")?;
    for (x, y) in points {
        writeln!(file, "{},{}", x, y)?;
    }
    
    println!("KDE plot data written to {}", filename);
    Ok(())
}
```

### Hypothesis Testing with Normal Distributions

```rust
use statsrust::*;

fn two_sample_z_test(
    sample1: &[f64], 
    sample2: &[f64],
    null_hypothesis_mean_diff: f64
) -> Result<(f64, f64), StatError> {
    // Estimate distributions from samples
    let dist1 = NormalDist::from_samples(sample1)?;
    let dist2 = NormalDist::from_samples(sample2)?;
    
    // Calculate difference distribution
    let diff_dist = dist1 - dist2;
    
    // Calculate observed difference
    let observed_diff = dist1.mean() - dist2.mean();
    
    // Calculate p-value for two-tailed test
    let z_score = (observed_diff - null_hypothesis_mean_diff) / diff_dist.stdev();
    let p_value = 2.0 * (1.0 - diff_dist.cdf(z_score.abs())?);
    
    Ok((z_score, p_value))
}
```

## Numerical Stability Best Practices

The statsrust library implements several techniques to ensure numerical stability, but following these best practices will help you get the most accurate results:

### Handling Large Datasets

For large datasets, consider using streaming algorithms when possible:

```rust
// Instead of storing all data points
let mut sum = 0.0;
let mut count = 0;
for value in large_dataset {
    sum += value;
    count += 1;
}
let mean = sum / count as f64;
```

### Geometric Mean with Large Values

The library automatically handles overflow/underflow for geometric mean, but it's important to understand the limitation:

```rust
// This will work even with extremely large or small values
let data = vec![1e-100, 1e100, 1e50, 1e-50];
let gm = geometric_mean(&data)?;  // Handles without overflow
```

### Variance Calculation for Large Values

When working with large values, the library's two-pass algorithm ensures numerical stability:

```rust
// This is numerically stable even when values are large but close together
let data = vec![1_000_000.0, 1_000_000.1, 1_000_000.2, 1_000_000.3];
let var = variance(&data, None)?;
```

### Avoiding Division by Zero

When calculating harmonic means or other operations that involve division:

```rust
// Always check for zero values
let data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
let positive_data: Vec<f64> = data.into_iter()
    .filter(|&x| x > 0.0)
    .collect();

if !positive_data.is_empty() {
    let hm = harmonic_mean(&positive_data, None)?;
}
```

## Troubleshooting Common Issues

### 1. "NoDataPoints" Error

**Cause:** Trying to calculate statistics on an empty dataset  
**Solution:** Always check that your data has at least one element

```rust
if !data.is_empty() {
    let m = mean(&data)?;
}
```

### 2. "InsufficientSampleData" Error

**Cause:** Sample variance requires at least two data points  
**Solution:** Ensure your dataset has enough points for the operation

```rust
if data.len() >= 2 {
    let var = variance(&data, None)?;
}
```

### 3. Negative Values in Geometric/Harmonic Mean

**Cause:** These means require positive values  
**Solution:** Filter or transform your data

```rust
let positive_data: Vec<f64> = data.iter()
    .filter(|&&x| x > 0.0)
    .cloned()
    .collect();
    
if !positive_data.is_empty() {
    let gm = geometric_mean(&positive_data)?;
}
```

### 4. Bandwidth Too Small in KDE

**Cause:** Very small bandwidth leads to overfitting and numerical issues  
**Solution:** Use a reasonable bandwidth value

```rust
// Using Silverman's rule for bandwidth selection
let n = data.len() as f64;
let std_dev = stdev(data, None).unwrap_or(1.0);
let iqr = if data.len() > 3 {
    let q = quantiles(data, 4, "inclusive").unwrap();
    q[2] - q[0]
} else {
    std_dev * 1.349
};
let bandwidth = 0.9 * n.powf(-0.2) * f64::min(std_dev, iqr / 1.349);
```

### 5. Correlation Calculation Fails

**Cause:** Constant input or insufficient data points  
**Solution:** Check data variability and size

```rust
if data_x.len() >= 2 && data_y.len() == data_x.len() {
    if data_x.iter().any(|&x| x != data_x[0]) && 
       data_y.iter().any(|&y| y != data_y[0]) {
        let corr = correlation(&data_x, &data_y, "linear")?;
    }
}
```

### 6. Inaccurate KDE Results

**Cause:** Inappropriate kernel choice or bandwidth  
**Solution:** Try different kernels and bandwidth values

```rust
// Compare results with different kernels
let kde_normal = kde(&data, 0.5, "normal", false)?;
let kde_epanechnikov = kde(&data, 0.5, "epanechnikov", false)?;

println!("Density at 3.0 (normal): {:.4}", kde_normal(3.0));
println!("Density at 3.0 (epanechnikov): {:.4}", kde_epanechnikov(3.0));
```

## Integration with Other Rust Libraries

### With `ndarray` for Matrix Operations

```rust
use ndarray::{Array1, Array2};
use statsrust::*;

fn analyze_matrix_columns(matrix: &Array2<f64>) -> Result<(), StatError> {
    for col in matrix.columns() {
        let column_data: Vec<f64> = col.to_vec();
        let mean_val = mean(&column_data)?;
        let std_dev = stdev(&column_data, None)?;
        println!("Column stats: mean={:.4}, std={:.4}", mean_val, std_dev);
    }
    Ok(())
}
```

### With `plotters` for Data Visualization

```rust
use plotters::prelude::*;
use statsrust::*;

fn plot_kde(data: &[f64], output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let kde = kde(data, 0.5, "gaussian", false)?;
    
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min) - 2.0;
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 2.0;
    
    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Kernel Density Estimation", ("sans-serif", 40))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_ranged(min..max, 0.0..1.0)?;
    
    chart.configure_mesh().draw()?;
    
    chart.draw_series(
        (0..100)
            .map(|i| {
                let x = min + (max - min) * (i as f64) / 99.0;
                (x, kde(x))
            })
            .map(|(x, y)| (x, y))
            .collect::<Vec<_>>(),
    )?
    .stroke_width(2)
    .stroke(&BLUE);
    
    root.present()?;
    Ok(())
}
```

### With `polars` for DataFrames

```rust
use polars::prelude::*;
use statsrust::*;

fn analyze_dataframe(df: &DataFrame) -> Result<(), StatError> {
    for col in df.get_columns() {
        if let Ok(series) = col.f64() {
            let values: Vec<f64> = series.into_no_null_iter().collect();
            if !values.is_empty() {
                let mean_val = mean(&values)?;
                let median_val = median(&values)?;
                println!("Column {}: mean={:.4}, median={:.4}", 
                         col.name(), mean_val, median_val);
            }
        }
    }
    Ok(())
}
```

## Conclusion

The statsrust library provides a comprehensive set of statistical tools for Rust developers. By following this guide, you should be able to effectively utilize all major features of the library, from basic descriptive statistics to advanced distribution modeling.

Remember these key principles for successful usage:
- Always handle potential errors
- Choose appropriate methods based on your data characteristics
- Consider numerical stability when working with extreme values
- Select proper bandwidth and kernel for KDE operations
- Understand the difference between population and sample statistics

For additional information, please refer to:
- [API Documentation](Api.md) for technical specifications
- [Mathematical Foundations](Mathlogic.md) for theoretical background
- [Contribution Guidelines](Contributing.md) if you'd like to improve the library

Happy analyzing!

This document is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).  
Original author: statsrust Authors