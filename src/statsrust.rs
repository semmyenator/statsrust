// Copyright (c) StatsRust Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
use thiserror::Error;
use num_traits::ToPrimitive;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::collections::HashMap;

/// Custom error type for Statsrust operations
#[derive(Error, Debug)]
pub enum StatsrustError {
    #[error("No data points provided")]
    NoDataPoints,
    #[error("At least two data points required")]
    NotEnoughDataPoints,
    #[error("Negative value not allowed")]
    NegativeValueNotAllowed,
    #[error("Weight sum must be non-zero")]
    ZeroWeightSum,
    #[error("Data and weights must have the same length")]
    MismatchedLengths,
    #[error("Bandwidth must be positive")]
    InvalidBandwidth,
    #[error("Input must contain numeric values")]
    NonNumericInput,
    #[error("p must be in the range 0.0 < p < 1.0")]
    InvalidProbability,
    #[error("Population variance requires at least one data point")]
    InsufficientPopulationData,
    #[error("Sample variance requires at least two data points")]
    InsufficientSampleData,
    #[error("Constant input - cannot compute correlation")]
    ConstantInput,
}

/// Enumeration of kernel functions for KDE
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kernel {
    Normal,
    Logistic,
    Sigmoid,
    Rectangular,
    Triangular,
    Parabolic,
    Quartic,
    Triweight,
    Cosine,
}

impl Kernel {
    /// Returns the kernel function
    pub fn kernel(&self) -> KernelFn {
        match self {
            Kernel::Normal => normal_kernel,
            Kernel::Logistic => logistic_kernel,
            Kernel::Sigmoid => sigmoid_kernel,
            Kernel::Rectangular => rectangular_kernel,
            Kernel::Triangular => triangular_kernel,
            Kernel::Parabolic => parabolic_kernel,
            Kernel::Quartic => quartic_kernel,
            Kernel::Triweight => triweight_kernel,
            Kernel::Cosine => cosine_kernel,
        }
    }
    
    /// Returns the CDF function
    pub fn cdf(&self) -> KernelFn {
        match self {
            Kernel::Normal => normal_cdf,
            Kernel::Logistic => logistic_cdf,
            Kernel::Sigmoid => sigmoid_cdf,
            Kernel::Rectangular => rectangular_cdf,
            Kernel::Triangular => triangular_cdf,
            Kernel::Parabolic => parabolic_cdf,
            Kernel::Quartic => quartic_cdf,
            Kernel::Triweight => triweight_cdf,
            Kernel::Cosine => cosine_cdf,
        }
    }
    
    /// Returns the inverse CDF function for random sampling
    pub fn inv_cdf(&self) -> KernelFn {
        match self {
            Kernel::Normal => normal_inv_cdf,
            Kernel::Logistic => logistic_inv_cdf,
            Kernel::Sigmoid => sigmoid_inv_cdf,
            Kernel::Rectangular => rectangular_inv_cdf,
            Kernel::Triangular => triangular_inv_cdf,
            Kernel::Parabolic => parabolic_inv_cdf,
            Kernel::Quartic => quartic_inv_cdf,
            Kernel::Triweight => triweight_inv_cdf,
            Kernel::Cosine => cosine_inv_cdf,
        }
    }
    
    /// Returns kernel support range (None for infinite support)
    pub fn support(&self) -> Option<f64> {
        match self {
            Kernel::Normal | Kernel::Logistic | Kernel::Sigmoid => None,
            _ => Some(1.0),
        }
    }
}

/// Type alias for kernel functions
pub type KernelFn = fn(f64) -> f64;

// ======================
// Core Kernel Functions
// ======================

// Normal kernel
fn normal_kernel(t: f64) -> f64 {
    (-(t * t) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

// Logistic kernel
fn logistic_kernel(t: f64) -> f64 {
    0.5 / (1.0 + t.cosh())
}

// Sigmoid kernel
fn sigmoid_kernel(t: f64) -> f64 {
    (1.0 / std::f64::consts::PI) / t.cosh()
}

// Rectangular kernel
fn rectangular_kernel(t: f64) -> f64 {
    if t.abs() <= 1.0 { 0.5 } else { 0.0 }
}

// Triangular kernel
fn triangular_kernel(t: f64) -> f64 {
    if t.abs() <= 1.0 { 1.0 - t.abs() } else { 0.0 }
}

// Parabolic (Epanechnikov) kernel
fn parabolic_kernel(t: f64) -> f64 {
    if t.abs() <= 1.0 { 0.75 * (1.0 - t * t) } else { 0.0 }
}

// Quartic (Biweight) kernel
fn quartic_kernel(t: f64) -> f64 {
    if t.abs() <= 1.0 { 15.0 / 16.0 * (1.0 - t * t).powi(2) } else { 0.0 }
}

// Triweight kernel
fn triweight_kernel(t: f64) -> f64 {
    if t.abs() <= 1.0 { 35.0 / 32.0 * (1.0 - t * t).powi(3) } else { 0.0 }
}

// Cosine kernel
fn cosine_kernel(t: f64) -> f64 {
    if t.abs() <= 1.0 { 
        (std::f64::consts::PI / 4.0) * ((std::f64::consts::PI / 2.0) * t).cos() 
    } else { 
        0.0 
    }
}

// ======================
// CDF Functions
// ======================

/// Error function approximation (Abramowitz and Stegun)
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    
    // Constants for approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
    sign * y
}

// Normal CDF
fn normal_cdf(t: f64) -> f64 {
    let z = t / std::f64::consts::SQRT_2;
    (1.0 + erf(z)) / 2.0
}

// Logistic CDF
fn logistic_cdf(t: f64) -> f64 {
    1.0 - 1.0 / (t.exp() + 1.0)
}

// Sigmoid CDF
fn sigmoid_cdf(t: f64) -> f64 {
    (2.0 / std::f64::consts::PI) * (t.exp()).atan()
}

// Rectangular CDF
fn rectangular_cdf(t: f64) -> f64 {
    if t < -1.0 {
        0.0
    } else if t > 1.0 {
        1.0
    } else {
        0.5 * t + 0.5
    }
}

// Triangular CDF
fn triangular_cdf(t: f64) -> f64 {
    if t < -1.0 {
        0.0
    } else if t > 1.0 {
        1.0
    } else if t < 0.0 {
        0.5 * t * t + t + 0.5
    } else {
        -0.5 * t * t + t + 0.5
    }
}

// Parabolic CDF
fn parabolic_cdf(t: f64) -> f64 {
    if t < -1.0 {
        0.0
    } else if t > 1.0 {
        1.0
    } else {
        -0.25 * t.powi(3) + 0.75 * t + 0.5
    }
}

// Quartic CDF
fn quartic_cdf(t: f64) -> f64 {
    if t < -1.0 {
        0.0
    } else if t > 1.0 {
        1.0
    } else {
        3.0/16.0 * t.powi(5) - 5.0/8.0 * t.powi(3) + 15.0/16.0 * t + 0.5
    }
}

// Triweight CDF
fn triweight_cdf(t: f64) -> f64 {
    if t < -1.0 {
        0.0
    } else if t > 1.0 {
        1.0
    } else {
        35.0/32.0 * (-1.0/7.0 * t.powi(7) + 3.0/5.0 * t.powi(5) - t.powi(3) + t) + 0.5
    }
}

// Cosine CDF
fn cosine_cdf(t: f64) -> f64 {
    if t < -1.0 {
        0.0
    } else if t > 1.0 {
        1.0
    } else {
        0.5 * ((std::f64::consts::PI / 2.0) * t).sin() + 0.5
    }
}

// ======================
// Inverse CDF Functions
// ======================

// Normal inverse CDF (using Box-Muller transform for samples, but this is for theoretical calculations)
fn normal_inv_cdf(p: f64) -> f64 {
    // For simplicity, we'll use a basic approximation
    // In a production library, a more accurate approximation would be used
    const A: [f64; 4] = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637];
    const B: [f64; 4] = [-8.4735109309, 23.08336743743, -21.06224101826, 3.13082909833];
    const C: [f64; 9] = [
        0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 0.0276438810333863, 
        0.0038405729373609, 0.0003951896511919, 0.0000321767881768, 0.0000002844213583, 
        0.0000003914339187
    ];
    
    if p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    
    let q = p - 0.5;
    if q.abs() <= 0.42 {
        let r = q * q;
        let num = (((A[3] * r + A[2]) * r + A[1]) * r + A[0]) * q;
        let den = (((B[3] * r + B[2]) * r + B[1]) * r + B[0]) * r + 1.0;
        return num / den;
    }
    
    let r = if p < 0.5 { p } else { 1.0 - p };
    let r = (-r.ln()).sqrt();
    let mut val = C[0];
    for i in 1..9 {
        val = val * r + C[i];
    }
    if p < 0.5 { -val } else { val }
}

// Logistic inverse CDF
fn logistic_inv_cdf(p: f64) -> f64 {
    p.ln() - (1.0 - p).ln()
}

// Sigmoid inverse CDF
fn sigmoid_inv_cdf(p: f64) -> f64 {
    (p * std::f64::consts::PI / 2.0).tan().ln()
}

// Rectangular inverse CDF
fn rectangular_inv_cdf(p: f64) -> f64 {
    2.0 * p - 1.0
}

// Triangular inverse CDF
fn triangular_inv_cdf(p: f64) -> f64 {
    if p < 0.5 {
        (2.0 * p).sqrt() - 1.0
    } else {
        1.0 - (2.0 - 2.0 * p).sqrt()
    }
}

// Parabolic inverse CDF
fn parabolic_inv_cdf(p: f64) -> f64 {
    2.0 * (((p * 2.0 - 1.0).acos() + std::f64::consts::PI) / 3.0).cos()
}

// Quartic inverse CDF
fn quartic_inv_cdf(p: f64) -> f64 {
    let mut x = if p <= 0.5 { 
        (2.0 * p).powf(0.4258865685331) - 1.0 
    } else { 
        1.0 - (2.0 * (1.0 - p)).powf(0.4258865685331) 
    };
    if p > 0.004 && p < 0.499 {
        let sign = if p <= 0.5 { 1.0 } else { -1.0 };
        x += 0.026818732 * (7.101753784 * p + 2.73230839482953).sin() * sign;
    }
    x
}

// Triweight inverse CDF
fn triweight_inv_cdf(p: f64) -> f64 {
    let sign = if p <= 0.5 { 1.0 } else { -1.0 };
    let p_adj = if p <= 0.5 { p } else { 1.0 - p };
    (2.0 * p_adj).powf(0.3400218741872791) - 1.0 * sign
}

// Cosine inverse CDF
fn cosine_inv_cdf(p: f64) -> f64 {
    2.0 * ((2.0 * p - 1.0).asin()) / std::f64::consts::PI
}

// ======================
// Core Statistics Functions
// ======================

/// Calculates arithmetic mean of data points
pub fn mean<T>(data: &[T]) -> Result<f64, StatsrustError> 
where 
    T: ToPrimitive + Copy
{
    if data.is_empty() {
        return Err(StatsrustError::NoDataPoints);
    }
    
    let sum: f64 = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .sum();
    
    Ok(sum / data.len() as f64)
}

/// Calculates weighted mean of data points
pub fn fmean<T>(data: &[T], weights: Option<&[T]>) -> Result<f64, StatsrustError> 
where 
    T: ToPrimitive + Copy
{
    if data.is_empty() {
        return Err(StatsrustError::NoDataPoints);
    }
    
    let weights = match weights {
        Some(w) => {
            if w.len() != data.len() {
                return Err(StatsrustError::MismatchedLengths);
            }
            w
        },
        None => {
            return mean(data);
        },
    };
    
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    
    for (&x, &w) in data.iter().zip(weights.iter()) {
        let x = x.to_f64().ok_or(StatsrustError::NonNumericInput)?;
        let w = w.to_f64().ok_or(StatsrustError::NonNumericInput)?;
        
        if w < 0.0 {
            return Err(StatsrustError::NegativeValueNotAllowed);
        }
        
        weighted_sum += x * w;
        weight_sum += w;
    }
    
    if weight_sum == 0.0 {
        return Err(StatsrustError::ZeroWeightSum);
    }
    
    Ok(weighted_sum / weight_sum)
}

/// Calculates geometric mean of data points
pub fn geometric_mean<T>(data: &[T]) -> Result<f64, StatsrustError> 
where 
    T: ToPrimitive + Copy
{
    if data.is_empty() {
        return Err(StatsrustError::NoDataPoints);
    }
    
    let mut log_sum = 0.0;
    let mut found_zero = false;
    
    for &x in data {
        let x = x.to_f64().ok_or(StatsrustError::NonNumericInput)?;
        
        if x < 0.0 {
            return Err(StatsrustError::NegativeValueNotAllowed);
        } else if x == 0.0 {
            found_zero = true;
        } else {
            log_sum += x.ln();
        }
    }
    
    if found_zero {
        return Ok(0.0);
    }
    
    Ok((log_sum / data.len() as f64).exp())
}

/// Calculates harmonic mean of data points
pub fn harmonic_mean<T>(data: &[T], weights: Option<&[T]>) -> Result<f64, StatsrustError> 
where 
    T: ToPrimitive + Copy
{
    if data.is_empty() {
        return Err(StatsrustError::NoDataPoints);
    }
    
    let weights = match weights {
        Some(w) => {
            if w.len() != data.len() {
                return Err(StatsrustError::MismatchedLengths);
            }
            w
        },
        None => {
            return mean(data).and_then(|m| Ok(1.0 / m));
        },
    };
    
    let mut sum_weights = 0.0;
    let mut sum_reciprocal = 0.0;
    
    for (&x, &w) in data.iter().zip(weights.iter()) {
        let x = x.to_f64().ok_or(StatsrustError::NonNumericInput)?;
        let w = w.to_f64().ok_or(StatsrustError::NonNumericInput)?;
        
        if x <= 0.0 || w < 0.0 {
            return Err(StatsrustError::NegativeValueNotAllowed);
        }
        
        sum_weights += w;
        sum_reciprocal += w / x;
    }
    
    if sum_weights == 0.0 {
        return Err(StatsrustError::ZeroWeightSum);
    }
    
    Ok(sum_weights / sum_reciprocal)
}

/// Calculates median of data points
pub fn median<T>(data: &[T]) -> Result<f64, StatsrustError> 
where 
    T: ToPrimitive + Copy + PartialOrd
{
    if data.is_empty() {
        return Err(StatsrustError::NoDataPoints);
    }
    
    let mut values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    
    if n % 2 == 1 {
        Ok(values[n / 2])
    } else {
        Ok((values[n / 2 - 1] + values[n / 2]) / 2.0)
    }
}

/// Calculates low median of data points
pub fn median_low<T>(data: &[T]) -> Result<f64, StatsrustError> 
where 
    T: ToPrimitive + Copy + PartialOrd
{
    if data.is_empty() {
        return Err(StatsrustError::NoDataPoints);
    }
    
    let mut values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    
    if n % 2 == 1 {
        Ok(values[n / 2])
    } else {
        Ok(values[n / 2 - 1])
    }
}

/// Calculates high median of data points
pub fn median_high<T>(data: &[T]) -> Result<f64, StatsrustError> 
where 
    T: ToPrimitive + Copy + PartialOrd
{
    if data.is_empty() {
        return Err(StatsrustError::NoDataPoints);
    }
    
    let mut values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    
    if n % 2 == 1 {
        Ok(values[n / 2])
    } else {
        Ok(values[n / 2])
    }
}

/// Calculates median for grouped data
pub fn median_grouped<T>(data: &[T], interval: f64) -> Result<f64, StatsrustError> 
where 
    T: ToPrimitive + Copy + PartialOrd
{
    if data.is_empty() {
        return Err(StatsrustError::NoDataPoints);
    }
    
    let mut values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    
    let x = values[n / 2];
    let i = values.partition_point(|&v| v < x);
    let j = values.partition_point(|&v| v <= x);
    
    let f = (j - i) as f64;
    let cf = i as f64;
    let l = x - interval / 2.0;
    
    Ok(l + interval * (n as f64 / 2.0 - cf) / f)
}

/// Calculates mode of data points
pub fn mode<T: Eq + std::hash::Hash + Copy>(data: &[T]) -> Result<T, StatsrustError> {
    if data.is_empty() {
        return Err(StatsrustError::NoDataPoints);
    }
    
    let mut counts = HashMap::new();
    for &item in data {
        *counts.entry(item).or_insert(0) += 1;
    }
    
    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(val, _)| val)
        .ok_or(StatsrustError::NoDataPoints)
}

/// Calculates multiple modes of data points
pub fn multimode<T: Eq + std::hash::Hash + Copy>(data: &[T]) -> Vec<T> {
    if data.is_empty() {
        return Vec::new();
    }
    
    let mut counts = HashMap::new();
    for &item in data {
        *counts.entry(item).or_insert(0) += 1;
    }
    
    let max_count = *counts.values().max().unwrap_or(&0);
    
    counts
        .into_iter()
        .filter(|&(_, count)| count == max_count)
        .map(|(val, _)| val)
        .collect()
}

/// Calculates population variance
pub fn pvariance<T>(data: &[T], mu: Option<f64>) -> Result<f64, StatsrustError> 
where 
    T: ToPrimitive + Copy
{
    if data.is_empty() {
        return Err(StatsrustError::InsufficientPopulationData);
    }
    
    let values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    
    let mu = match mu {
        Some(mu) => mu,
        None => mean(&values)?,
    };
    
    let variance = values.iter()
        .map(|&x| (x - mu).powi(2))
        .sum::<f64>() / values.len() as f64;
    
    Ok(variance)
}

/// Calculates population standard deviation
pub fn pstdev<T>(data: &[T], mu: Option<f64>) -> Result<f64, StatsrustError> 
where 
    T: ToPrimitive + Copy
{
    pvariance(data, mu).map(|var| var.sqrt())
}

/// Calculates sample variance
pub fn variance<T>(data: &[T], xbar: Option<f64>) -> Result<f64, StatsrustError> 
where 
    T: ToPrimitive + Copy
{
    if data.len() < 2 {
        return Err(StatsrustError::InsufficientSampleData);
    }
    
    let values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    
    let xbar = match xbar {
        Some(xbar) => xbar,
        None => mean(&values)?,
    };
    
    let variance = values.iter()
        .map(|&x| (x - xbar).powi(2))
        .sum::<f64>() / (values.len() - 1) as f64;
    
    Ok(variance)
}

/// Calculates sample standard deviation
pub fn stdev<T>(data: &[T], xbar: Option<f64>) -> Result<f64, StatsrustError> 
where 
    T: ToPrimitive + Copy
{
    variance(data, xbar).map(|var| var.sqrt())
}

/// Calculates quantiles of data points
pub fn quantiles<T>(data: &[T], n: usize, method: &str) -> Result<Vec<f64>, StatsrustError> 
where 
    T: ToPrimitive + Copy + PartialOrd
{
    if n < 1 {
        return Err(StatsrustError::NoDataPoints);
    }
    if data.is_empty() {
        return Err(StatsrustError::NoDataPoints);
    }
    
    let mut values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ld = values.len();
    
    if ld < 2 {
        if ld == 1 {
            return Ok(vec![values[0]; n - 1]);
        }
        return Err(StatsrustError::NoDataPoints);
    }
    
    match method {
        "inclusive" => {
            let m = ld - 1;
            let mut result = Vec::with_capacity(n - 1);
            for i in 1..n {
                let (j, delta) = ((i * m) / n, (i * m) % n);
                let interpolated = (values[j] * (n - delta) as f64 + values[j + 1] * delta as f64) / n as f64;
                result.push(interpolated);
            }
            Ok(result)
        },
        "exclusive" => {
            let m = ld + 1;
            let mut result = Vec::with_capacity(n - 1);
            for i in 1..n {
                let mut j = i * m / n;
                j = if j < 1 { 1 } else if j > ld - 1 { ld - 1 } else { j };
                let delta = i * m - j * n;
                let interpolated = (values[j - 1] * (n - delta) as f64 + values[j] * delta as f64) / n as f64;
                result.push(interpolated);
            }
            Ok(result)
        },
        _ => Err(StatsrustError::InvalidProbability),
    }
}

/// Calculates correlation coefficient
pub fn correlation<T>(x: &[T], y: &[T], method: &str) -> Result<f64, StatsrustError> 
where 
    T: ToPrimitive + Copy
{
    if x.len() != y.len() {
        return Err(StatsrustError::MismatchedLengths);
    }
    if x.len() < 2 {
        return Err(StatsrustError::NotEnoughDataPoints);
    }
    
    let x: Vec<f64> = x.iter().map(|&v| v.to_f64().unwrap()).collect();
    let y: Vec<f64> = y.iter().map(|&v| v.to_f64().unwrap()).collect();
    
    match method {
        "linear" => {
            let x_mean = mean(&x)?;
            let y_mean = mean(&y)?;
            
            let (sxx, syy, sxy) = x.iter().zip(y.iter()).fold((0.0, 0.0, 0.0), |(sxx, syy, sxy), (&xi, &yi)| {
                let xi_centered = xi - x_mean;
                let yi_centered = yi - y_mean;
                (sxx + xi_centered * xi_centered, syy + yi_centered * yi_centered, sxy + xi_centered * yi_centered)
            });
            
            if sxx == 0.0 || syy == 0.0 {
                return Err(StatsrustError::ConstantInput);
            }
            
            Ok(sxy / (sxx * syy).sqrt())
        },
        "ranked" => {
            let x_ranks = rank(&x)?;
            let y_ranks = rank(&y)?;
            correlation(&x_ranks, &y_ranks, "linear")
        },
        _ => Err(StatsrustError::InvalidProbability),
    }
}

/// Calculates covariance
pub fn covariance<T>(x: &[T], y: &[T]) -> Result<f64, StatsrustError> 
where 
    T: ToPrimitive + Copy
{
    if x.len() != y.len() {
        return Err(StatsrustError::MismatchedLengths);
    }
    if x.len() < 2 {
        return Err(StatsrustError::NotEnoughDataPoints);
    }
    
    let x: Vec<f64> = x.iter().map(|&v| v.to_f64().unwrap()).collect();
    let y: Vec<f64> = y.iter().map(|&v| v.to_f64().unwrap()).collect();
    
    let x_mean = mean(&x)?;
    let y_mean = mean(&y)?;
    
    let sxy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| {
        (xi - x_mean) * (yi - y_mean)
    }).sum();
    
    Ok(sxy / (x.len() - 1) as f64)
}

/// Calculates linear regression parameters
pub fn linear_regression<T>(x: &[T], y: &[T], proportional: bool) -> Result<(f64, f64), StatsrustError> 
where 
    T: ToPrimitive + Copy
{
    if x.len() != y.len() {
        return Err(StatsrustError::MismatchedLengths);
    }
    if x.len() < 2 {
        return Err(StatsrustError::NotEnoughDataPoints);
    }
    
    let x: Vec<f64> = x.iter().map(|&v| v.to_f64().unwrap()).collect();
    let y: Vec<f64> = y.iter().map(|&v| v.to_f64().unwrap()).collect();
    
    let x_mean = if proportional { 0.0 } else { mean(&x)? };
    let y_mean = if proportional { 0.0 } else { mean(&y)? };
    
    let (sxx, sxy) = x.iter().zip(y.iter()).fold((0.0, 0.0), |(sxx, sxy), (&xi, &yi)| {
        let xi_centered = if proportional { xi } else { xi - x_mean };
        let yi_centered = if proportional { yi } else { yi - y_mean };
        (sxx + xi_centered * xi_centered, sxy + xi_centered * yi_centered)
    });
    
    if sxx == 0.0 {
        return Err(StatsrustError::ConstantInput);
    }
    
    let slope = sxy / sxx;
    let intercept = if proportional { 0.0 } else { y_mean - slope * x_mean };
    
    Ok((slope, intercept))
}

/// Helper function for ranking data
fn rank(data: &[f64]) -> Result<Vec<f64>, StatsrustError> {
    let n = data.len();
    let mut indexed: Vec<(f64, usize)> = data.iter().copied().zip(0..n).collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].0 - indexed[i].0).abs() < 1e-10 {
            j += 1;
        }
        
        let rank = (i + j - 1) as f64 / 2.0 + 1.0;
        for k in i..j {
            ranks[indexed[k].1] = rank;
        }
        
        i = j;
    }
    
    Ok(ranks)
}

/// Kernel Density Estimation function
pub fn kde<T>(data: &[T], h: f64, kernel: Kernel, cumulative: bool) -> Result<Box<dyn Fn(f64) -> f64 + Send + Sync>, StatsrustError> 
where 
    T: ToPrimitive + Copy
{
    if data.is_empty() {
        return Err(StatsrustError::NoDataPoints);
    }
    if h <= 0.0 {
        return Err(StatsrustError::InvalidBandwidth);
    }
    
    let values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    
    let n = values.len() as f64;
    
    // Sort values for efficient range queries
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let kernel_fn = kernel.kernel();
    let cdf_fn = kernel.cdf();
    let support = kernel.support();
    
    if cumulative {
        match support {
            Some(s) => {
                let bandwidth = h * s;
                Ok(Box::new(move |x: f64| -> f64 {
                    // Find indices within the bandwidth
                    let i = sorted.partition_point(|&v| v < x - bandwidth);
                    let j = sorted.partition_point(|&v| v <= x + bandwidth);
                    
                    let mut sum = 0.0;
                    for k in i..j {
                        let t = (x - sorted[k]) / h;
                        sum += cdf_fn(t);
                    }
                    sum / n
                }))
            },
            None => {
                Ok(Box::new(move |x: f64| -> f64 {
                    values.iter()
                        .map(|&x_i| cdf_fn((x - x_i) / h))
                        .sum::<f64>() / n
                }))
            }
        }
    } else {
        match support {
            Some(s) => {
                let bandwidth = h * s;
                Ok(Box::new(move |x: f64| -> f64 {
                    // Find indices within the bandwidth
                    let i = sorted.partition_point(|&v| v < x - bandwidth);
                    let j = sorted.partition_point(|&v| v <= x + bandwidth);
                    
                    let mut sum = 0.0;
                    for k in i..j {
                        let t = (x - sorted[k]) / h;
                        sum += kernel_fn(t);
                    }
                    sum / (n * h)
                }))
            },
            None => {
                Ok(Box::new(move |x: f64| -> f64 {
                    values.iter()
                        .map(|&x_i| kernel_fn((x - x_i) / h))
                        .sum::<f64>() / (n * h)
                }))
            }
        }
    }
}

/// Generates random samples based on KDE
pub fn kde_random<T>(data: &[T], h: f64, kernel: Kernel, seed: Option<u64>) -> Result<Box<dyn FnMut() -> f64 + Send + Sync>, StatsrustError> 
where 
    T: ToPrimitive + Copy
{
    if data.is_empty() {
        return Err(StatsrustError::NoDataPoints);
    }
    if h <= 0.0 {
        return Err(StatsrustError::InvalidBandwidth);
    }
    
    let inv_cdf = kernel.inv_cdf();
    let values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    
    let mut rng = match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };
    
    let values_clone = values.clone();
    
    Ok(Box::new(move || -> f64 {
        let index = rng.gen_range(0..values_clone.len());
        let x_i = values_clone[index];
        x_i + h * inv_cdf(rng.gen())
    }))
}

/// Represents a normal distribution
#[derive(Debug, Clone, Copy)]
pub struct NormalDist {
    mu: f64,    // Mean
    sigma: f64, // Standard deviation
}

impl NormalDist {
    /// Creates a new normal distribution with given parameters
    pub fn new(mu: f64, sigma: f64) -> Result<Self, StatsrustError> {
        if sigma < 0.0 {
            return Err(StatsrustError::NegativeValueNotAllowed);
        }
        Ok(Self { mu, sigma })
    }
    
    /// Estimates normal distribution parameters from sample data
    pub fn from_samples<T>(data: &[T]) -> Result<Self, StatsrustError> 
    where 
        T: ToPrimitive + Copy
    {
        if data.len() < 2 {
            return Err(StatsrustError::NotEnoughDataPoints);
        }
        
        let values: Vec<f64> = data.iter()
            .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
            .collect::<Result<Vec<_>, _>>()?;
        
        let mean = mean(&values)?;
        let variance = variance(&values, Some(mean))?;
        
        Ok(Self {
            mu: mean,
            sigma: variance.sqrt(),
        })
    }
    
    /// Generates random samples from the distribution using Box-Muller transform
    pub fn samples(&self, n: usize, seed: Option<u64>) -> Vec<f64> {
        let mut rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        
        let mut samples = Vec::with_capacity(n);
        for _ in 0..(n + 1) / 2 {
            let u1: f64 = rng.gen_range(0.0..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            
            samples.push(self.mu + self.sigma * r * theta.cos());
            if samples.len() < n {
                samples.push(self.mu + self.sigma * r * theta.sin());
            }
        }
        
        samples
    }
    
    /// Probability density function
    pub fn pdf(&self, x: f64) -> f64 {
        if self.sigma == 0.0 {
            if (x - self.mu).abs() < 1e-10 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            let norm = 1.0 / (self.sigma * (2.0 * std::f64::consts::PI).sqrt());
            let exponent = -((x - self.mu) * (x - self.mu)) / (2.0 * self.sigma * self.sigma);
            norm * exponent.exp()
        }
    }
    
    /// Cumulative distribution function
    pub fn cdf(&self, x: f64) -> f64 {
        if self.sigma == 0.0 {
            if x < self.mu {
                0.0
            } else {
                1.0
            }
        } else {
            normal_cdf((x - self.mu) / self.sigma)
        }
    }
    
    /// Inverse cumulative distribution function (quantile function)
    pub fn inv_cdf(&self, p: f64) -> Result<f64, StatsrustError> {
        if p <= 0.0 || p >= 1.0 {
            return Err(StatsrustError::InvalidProbability);
        }
        if self.sigma == 0.0 {
            return Ok(self.mu);
        }
        Ok(self.mu + self.sigma * normal_inv_cdf(p))
    }
    
    /// Calculates equally spaced quantiles
    pub fn quantiles(&self, n: usize) -> Vec<f64> {
        (1..n).map(|i| self.inv_cdf(i as f64 / n as f64).unwrap_or(f64::NAN)).collect()
    }
    
    /// Calculates the overlapping area between two normal distributions
    pub fn overlap(&self, other: &Self) -> f64 {
        let (x, y) = if (other.sigma, other.mu) < (self.sigma, self.mu) {
            (other, self)
        } else {
            (self, other)
        };
        
        if x.sigma == 0.0 || y.sigma == 0.0 {
            return 0.0;
        }
        
        let x_var = x.sigma * x.sigma;
        let y_var = y.sigma * y.sigma;
        let dv = y_var - x_var;
        let dm = (y.mu - x.mu).abs();
        
        if dv.abs() < 1e-10 {
            return 1.0 - erf(dm / (2.0 * x.sigma * 2f64.sqrt()));
        }
        
        let a = x.mu * y_var - y.mu * x_var;
        let b = x.sigma * y.sigma * (dm * dm + dv * (y_var / x_var).ln()).sqrt();
        let x1 = (a + b) / dv;
        let x2 = (a - b) / dv;
        
        let y_cdf_x1 = y.cdf(x1);
        let x_cdf_x1 = x.cdf(x1);
        let y_cdf_x2 = y.cdf(x2);
        let x_cdf_x2 = x.cdf(x2);
        
        1.0 - (y_cdf_x1 - x_cdf_x1 + y_cdf_x2 - x_cdf_x2).abs()
    }
    
    /// Calculates Z-score
    pub fn zscore(&self, x: f64) -> Result<f64, StatsrustError> {
        if self.sigma == 0.0 {
            return Err(StatsrustError::NegativeValueNotAllowed);
        }
        Ok((x - self.mu) / self.sigma)
    }
    
    /// Returns the mean of the distribution
    pub fn mean(&self) -> f64 {
        self.mu
    }
    
    /// Returns the median of the distribution
    pub fn median(&self) -> f64 {
        self.mu
    }
    
    /// Returns the mode of the distribution
    pub fn mode(&self) -> f64 {
        self.mu
    }
    
    /// Returns the standard deviation of the distribution
    pub fn stdev(&self) -> f64 {
        self.sigma
    }
    
    /// Returns the variance of the distribution
    pub fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }
}

/// Adds a constant to a normal distribution
impl std::ops::Add<f64> for NormalDist {
    type Output = NormalDist;
    fn add(self, rhs: f64) -> Self::Output {
        NormalDist::new(self.mu + rhs, self.sigma).unwrap()
    }
}

/// Subtracts a constant from a normal distribution
impl std::ops::Sub<f64> for NormalDist {
    type Output = NormalDist;
    fn sub(self, rhs: f64) -> Self::Output {
        NormalDist::new(self.mu - rhs, self.sigma).unwrap()
    }
}

/// Multiplies a normal distribution by a constant
impl std::ops::Mul<f64> for NormalDist {
    type Output = NormalDist;
    fn mul(self, rhs: f64) -> Self::Output {
        NormalDist::new(self.mu * rhs, self.sigma * rhs.abs()).unwrap()
    }
}

/// Divides a normal distribution by a constant
impl std::ops::Div<f64> for NormalDist {
    type Output = NormalDist;
    fn div(self, rhs: f64) -> Self::Output {
        NormalDist::new(self.mu / rhs, self.sigma / rhs.abs()).unwrap()
    }
}

/// Adds two normal distributions
impl std::ops::Add<NormalDist> for NormalDist {
    type Output = NormalDist;
    fn add(self, rhs: NormalDist) -> Self::Output {
        NormalDist::new(
            self.mu + rhs.mu,
            (self.sigma.powi(2) + rhs.sigma.powi(2)).sqrt()
        ).unwrap()
    }
}

/// Subtracts two normal distributions
impl std::ops::Sub<NormalDist> for NormalDist {
    type Output = NormalDist;
    fn sub(self, rhs: NormalDist) -> Self::Output {
        NormalDist::new(
            self.mu - rhs.mu,
            (self.sigma.powi(2) + rhs.sigma.powi(2)).sqrt()
        ).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mean() {
        let data = vec![1, 2, 3, 4, 5];
        assert_eq!(mean(&data).unwrap(), 3.0);
        let data = vec![1.5, 2.5, 3.5];
        assert_eq!(mean(&data).unwrap(), 2.5);
        let data: Vec<i32> = vec![];
        assert!(mean(&data).is_err());
    }
    
    #[test]
    fn test_median() {
        let data = vec![1, 3, 5];
        assert_eq!(median(&data).unwrap(), 3.0);
        let data = vec![1, 3, 5, 7];
        assert_eq!(median(&data).unwrap(), 4.0);
        let data = vec![1, 3, 3, 5, 7];
        assert_eq!(median(&data).unwrap(), 3.0);
    }
    
    #[test]
    fn test_variance() {
        let data = vec![1, 2, 3, 4, 5];
        assert!((variance(&data, None).unwrap() - 2.5).abs() < 1e-10);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((variance(&data, None).unwrap() - 2.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_normal_dist() {
        let dist = NormalDist::new(0.0, 1.0).unwrap();
        assert_eq!(dist.mean(), 0.0);
        assert_eq!(dist.stdev(), 1.0);
        let samples = dist.samples(1000, Some(42));
        assert_eq!(samples.len(), 1000);
        let dist2 = NormalDist::from_samples(&samples).unwrap();
        // Since the Box-Muller transform is more accurate, tighter tolerances can be used
        assert!((dist2.mean() - 0.0).abs() < 0.1);
        assert!((dist2.stdev() - 1.0).abs() < 0.1);
    }
    
    #[test]
    fn test_kde() {
        let data = vec![1, 2, 3, 4, 5];
        let kde_func = kde(&data, 1.0, Kernel::Normal, false).unwrap();
        let value = kde_func(3.0);
        assert!(value > 0.0);
        let kde_cdf = kde(&data, 1.0, Kernel::Normal, true).unwrap();
        let cdf_value = kde_cdf(3.0);
        assert!(cdf_value >= 0.0 && cdf_value <= 1.0);
    }
    
    #[test]
    fn test_kde_random() {
        let data = vec![1, 2, 3, 4, 5];
        let mut kde_rand = kde_random(&data, 1.0, Kernel::Normal, Some(42)).unwrap();
        let sample1 = kde_rand();
        let sample2 = kde_rand();
        assert_ne!(sample1, sample2);
    }
    
    #[test]
    fn test_parabolic_kernel() {
        let kernel = Kernel::Parabolic;
        let inv_cdf = kernel.inv_cdf();
        assert!((inv_cdf(0.5) - 0.0).abs() < 1e-10);
        assert!((inv_cdf(0.0) + 1.0).abs() < 1e-10);
        assert!((inv_cdf(1.0) - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_box_muller() {
        let dist = NormalDist::new(0.0, 1.0).unwrap();
        let samples = dist.samples(10000, Some(42));
        
        // Check statistical properties under large samples
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;
        
        // Under large samples, it should be very close to the theoretical value
        assert!((mean - 0.0).abs() < 0.05);
        assert!((variance - 1.0).abs() < 0.05);
    }
}
