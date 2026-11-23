// Copyright (c) StatsRust Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Depends only on the Rust standard library (no thiserror / num_traits / ndarray / statrs / rand)

use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// ==========================================
// ToPrimitive trait (replaces num_traits)
// ==========================================

pub trait ToPrimitive {
    fn to_f64(self) -> Option<f64>;
}

macro_rules! impl_to_primitive {
    ($($t:ty),*) => {
        $(
            impl ToPrimitive for $t {
                #[inline]
                fn to_f64(self) -> Option<f64> {
                    Some(self as f64)
                }
            }
        )*
    };
}

impl_to_primitive!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);

// ==========================================
// SplitMix64 PRNG (replaces rand crate)
// ==========================================

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn from_entropy() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Self::new(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn gen_range_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u64() as usize) % max
    }
}

// ==========================================
// Error Handling (replaces thiserror)
// ==========================================

#[derive(Debug)]
pub enum StatsrustError {
    NoDataPoints,
    NotEnoughDataPoints,
    NegativeValueNotAllowed,
    ZeroWeightSum,
    MismatchedLengths,
    UnknownKernel(String),
    InvalidBandwidth,
    NonNumericInput,
    InvalidProbability,
    InsufficientPopulationData,
    InsufficientSampleData,
    ConstantInput,
}

impl fmt::Display for StatsrustError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDataPoints => write!(f, "No data points provided"),
            Self::NotEnoughDataPoints => write!(f, "At least two data points required"),
            Self::NegativeValueNotAllowed => write!(f, "Negative value not allowed"),
            Self::ZeroWeightSum => write!(f, "Weight sum must be non-zero"),
            Self::MismatchedLengths => write!(f, "Data and weights must have the same length"),
            Self::UnknownKernel(n) => write!(f, "Invalid kernel name: {}", n),
            Self::InvalidBandwidth => write!(f, "Bandwidth must be positive"),
            Self::NonNumericInput => write!(f, "Input must contain numeric values"),
            Self::InvalidProbability => write!(f, "p must be in the range 0.0 < p < 1.0"),
            Self::InsufficientPopulationData => write!(f, "Population variance requires at least one data point"),
            Self::InsufficientSampleData => write!(f, "Sample variance requires at least two data points"),
            Self::ConstantInput => write!(f, "Constant input - cannot compute correlation"),
        }
    }
}

impl std::error::Error for StatsrustError {}

// ==========================================
// Kernel Enum & Dispatch
// ==========================================

pub type KernelFn = fn(f64) -> f64;

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
    pub fn from_name(name: &str) -> Result<Self, StatsrustError> {
        match name.to_lowercase().trim() {
            "normal" | "gauss" => Ok(Kernel::Normal),
            "logistic" => Ok(Kernel::Logistic),
            "sigmoid" => Ok(Kernel::Sigmoid),
            "rectangular" | "uniform" => Ok(Kernel::Rectangular),
            "triangular" => Ok(Kernel::Triangular),
            "parabolic" | "epanechnikov" => Ok(Kernel::Parabolic),
            "quartic" | "biweight" => Ok(Kernel::Quartic),
            "triweight" => Ok(Kernel::Triweight),
            "cosine" => Ok(Kernel::Cosine),
            _ => Err(StatsrustError::UnknownKernel(name.to_string())),
        }
    }

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

    pub fn support(&self) -> Option<f64> {
        match self {
            Kernel::Normal | Kernel::Logistic | Kernel::Sigmoid => None,
            _ => Some(1.0),
        }
    }
}

// ==========================================
// Math Helpers
// ==========================================

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
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

// ==========================================
// Kernel PDF Implementations
// ==========================================

fn normal_kernel(t: f64) -> f64 {
    (-(t * t) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
}
fn logistic_kernel(t: f64) -> f64 {
    0.5 / (1.0 + t.cosh())
}
fn sigmoid_kernel(t: f64) -> f64 {
    (1.0 / std::f64::consts::PI) / t.cosh()
}
fn rectangular_kernel(t: f64) -> f64 {
    if t.abs() <= 1.0 { 0.5 } else { 0.0 }
}
fn triangular_kernel(t: f64) -> f64 {
    if t.abs() <= 1.0 { 1.0 - t.abs() } else { 0.0 }
}
fn parabolic_kernel(t: f64) -> f64 {
    if t.abs() <= 1.0 { 0.75 * (1.0 - t * t) } else { 0.0 }
}
fn quartic_kernel(t: f64) -> f64 {
    if t.abs() <= 1.0 { 15.0 / 16.0 * (1.0 - t * t).powi(2) } else { 0.0 }
}
fn triweight_kernel(t: f64) -> f64 {
    if t.abs() <= 1.0 { 35.0 / 32.0 * (1.0 - t * t).powi(3) } else { 0.0 }
}
fn cosine_kernel(t: f64) -> f64 {
    if t.abs() <= 1.0 {
        (std::f64::consts::PI / 4.0) * ((std::f64::consts::PI / 2.0) * t).cos()
    } else {
        0.0
    }
}

// ==========================================
// CDF Implementations
// ==========================================

fn normal_cdf(t: f64) -> f64 {
    let z = t / std::f64::consts::SQRT_2;
    (1.0 + erf(z)) / 2.0
}
fn logistic_cdf(t: f64) -> f64 {
    1.0 - 1.0 / (t.exp() + 1.0)
}
fn sigmoid_cdf(t: f64) -> f64 {
    (2.0 / std::f64::consts::PI) * (t.exp()).atan()
}
fn rectangular_cdf(t: f64) -> f64 {
    if t < -1.0 { 0.0 } else if t > 1.0 { 1.0 } else { 0.5 * t + 0.5 }
}
fn triangular_cdf(t: f64) -> f64 {
    if t < -1.0 { 0.0 } else if t > 1.0 { 1.0 } else if t < 0.0 { 0.5 * t * t + t + 0.5 } else { -0.5 * t * t + t + 0.5 }
}
fn parabolic_cdf(t: f64) -> f64 {
    if t < -1.0 { 0.0 } else if t > 1.0 { 1.0 } else { -0.25 * t.powi(3) + 0.75 * t + 0.5 }
}
fn quartic_cdf(t: f64) -> f64 {
    if t < -1.0 { 0.0 } else if t > 1.0 { 1.0 } else { 3.0 / 16.0 * t.powi(5) - 5.0 / 8.0 * t.powi(3) + 15.0 / 16.0 * t + 0.5 }
}
fn triweight_cdf(t: f64) -> f64 {
    if t < -1.0 { 0.0 } else if t > 1.0 { 1.0 } else { 35.0 / 32.0 * (-1.0 / 7.0 * t.powi(7) + 3.0 / 5.0 * t.powi(5) - t.powi(3) + t) + 0.5 }
}
fn cosine_cdf(t: f64) -> f64 {
    if t < -1.0 { 0.0 } else if t > 1.0 { 1.0 } else { 0.5 * ((std::f64::consts::PI / 2.0) * t).sin() + 0.5 }
}

// ==========================================
// Inverse CDF Implementations
// ==========================================

/// Hybrid numerical inverse CDF solver (bisection + Newton's method)
/// For symmetric compact kernels on [-1, 1]
fn numerical_inv_cdf(cdf_fn: KernelFn, pdf_fn: KernelFn, p: f64) -> f64 {
    if p <= 0.0 { return -1.0; }
    if p >= 1.0 { return 1.0; }
    if (p - 0.5).abs() < 1e-15 { return 0.0; }

    let negate = p < 0.5;
    let target = if negate { 1.0 - p } else { p };

    let mut lo: f64 = 0.0;
    let mut hi: f64 = 1.0;

    for _ in 0..10 {
        let mid = (lo + hi) * 0.5;
        if cdf_fn(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let mut x = (lo + hi) * 0.5;
    for _ in 0..50 {
        let fx = cdf_fn(x) - target;
        if fx.abs() < 1e-15 { break; }
        let dfx = pdf_fn(x);
        if dfx.abs() < 1e-30 { break; }
        let x_new = x - fx / dfx;
        if x_new < lo || x_new > hi {
            x = (lo + hi) * 0.5;
        } else {
            x = x_new;
        }
        if cdf_fn(x) < target { lo = x; } else { hi = x; }
    }

    if negate { -x } else { x }
}

fn normal_inv_cdf(p: f64) -> f64 {
    const A: [f64; 4] = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637];
    const B: [f64; 4] = [-8.4735109309, 23.08336743743, -21.06224101826, 3.13082909833];
    const C: [f64; 9] = [
        0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 0.0276438810333863,
        0.0038405729373609, 0.0003951896511919, 0.0000321767881768, 0.0000002844213583,
        0.0000003914339187
    ];
    if p <= 0.0 || p >= 1.0 { return f64::NAN; }
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
    for &c in C.iter().skip(1) {
        val = val * r + c;
    }
    if p < 0.5 { -val } else { val }
}

fn logistic_inv_cdf(p: f64) -> f64 {
    p.ln() - (1.0 - p).ln()
}
fn sigmoid_inv_cdf(p: f64) -> f64 {
    (p * std::f64::consts::PI / 2.0).tan().ln()
}
fn rectangular_inv_cdf(p: f64) -> f64 {
    2.0 * p - 1.0
}
fn triangular_inv_cdf(p: f64) -> f64 {
    if p < 0.5 { (2.0 * p).sqrt() - 1.0 } else { 1.0 - (2.0 - 2.0 * p).sqrt() }
}
fn parabolic_inv_cdf(p: f64) -> f64 {
    2.0 * (((p * 2.0 - 1.0).acos() + std::f64::consts::PI) / 3.0).cos()
}
fn quartic_inv_cdf(p: f64) -> f64 {
    numerical_inv_cdf(quartic_cdf, quartic_kernel, p)
}
fn triweight_inv_cdf(p: f64) -> f64 {
    numerical_inv_cdf(triweight_cdf, triweight_kernel, p)
}
fn cosine_inv_cdf(p: f64) -> f64 {
    2.0 * ((2.0 * p - 1.0).asin()) / std::f64::consts::PI
}

// ==========================================
// Core Statistics Functions
// ==========================================

pub fn mean<T>(data: &[T]) -> Result<f64, StatsrustError>
where T: ToPrimitive + Copy
{
    if data.is_empty() { return Err(StatsrustError::NoDataPoints); }
    let sum: f64 = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .sum();
    Ok(sum / data.len() as f64)
}

pub fn fmean<T>(data: &[T], weights: Option<&[T]>) -> Result<f64, StatsrustError>
where T: ToPrimitive + Copy
{
    if data.is_empty() { return Err(StatsrustError::NoDataPoints); }
    let weights = match weights {
        Some(w) => {
            if w.len() != data.len() { return Err(StatsrustError::MismatchedLengths); }
            w
        },
        None => return mean(data),
    };
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    for (&x, &w) in data.iter().zip(weights.iter()) {
        let x = x.to_f64().ok_or(StatsrustError::NonNumericInput)?;
        let w = w.to_f64().ok_or(StatsrustError::NonNumericInput)?;
        if w < 0.0 { return Err(StatsrustError::NegativeValueNotAllowed); }
        weighted_sum += x * w;
        weight_sum += w;
    }
    if weight_sum == 0.0 { return Err(StatsrustError::ZeroWeightSum); }
    Ok(weighted_sum / weight_sum)
}

pub fn geometric_mean<T>(data: &[T]) -> Result<f64, StatsrustError>
where T: ToPrimitive + Copy
{
    if data.is_empty() { return Err(StatsrustError::NoDataPoints); }
    let mut log_sum = 0.0;
    let mut found_zero = false;
    for &x in data {
        let x = x.to_f64().ok_or(StatsrustError::NonNumericInput)?;
        if x < 0.0 { return Err(StatsrustError::NegativeValueNotAllowed); }
        else if x == 0.0 { found_zero = true; }
        else { log_sum += x.ln(); }
    }
    if found_zero { return Ok(0.0); }
    Ok((log_sum / data.len() as f64).exp())
}

pub fn harmonic_mean<T>(data: &[T], weights: Option<&[T]>) -> Result<f64, StatsrustError>
where T: ToPrimitive + Copy
{
    if data.is_empty() { return Err(StatsrustError::NoDataPoints); }

    let weights = match weights {
        Some(w) => {
            if w.len() != data.len() { return Err(StatsrustError::MismatchedLengths); }
            w
        },
        None => {
            let mut sum_reciprocal = 0.0;
            for &x in data {
                let x = x.to_f64().ok_or(StatsrustError::NonNumericInput)?;
                if x <= 0.0 { return Err(StatsrustError::NegativeValueNotAllowed); }
                sum_reciprocal += 1.0 / x;
            }
            if sum_reciprocal == 0.0 { return Err(StatsrustError::ZeroWeightSum); }
            return Ok(data.len() as f64 / sum_reciprocal);
        },
    };

    let mut sum_weights = 0.0;
    let mut sum_reciprocal = 0.0;
    for (&x, &w) in data.iter().zip(weights.iter()) {
        let x = x.to_f64().ok_or(StatsrustError::NonNumericInput)?;
        let w = w.to_f64().ok_or(StatsrustError::NonNumericInput)?;
        if x <= 0.0 || w < 0.0 { return Err(StatsrustError::NegativeValueNotAllowed); }
        sum_weights += w;
        sum_reciprocal += w / x;
    }
    if sum_weights == 0.0 { return Err(StatsrustError::ZeroWeightSum); }
    if sum_reciprocal == 0.0 { return Err(StatsrustError::ZeroWeightSum); }
    Ok(sum_weights / sum_reciprocal)
}

pub fn median<T>(data: &[T]) -> Result<f64, StatsrustError>
where T: ToPrimitive + Copy + PartialOrd
{
    if data.is_empty() { return Err(StatsrustError::NoDataPoints); }
    let mut values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 1 { Ok(values[n / 2]) } else { Ok((values[n / 2 - 1] + values[n / 2]) / 2.0) }
}

pub fn median_low<T>(data: &[T]) -> Result<f64, StatsrustError>
where T: ToPrimitive + Copy + PartialOrd
{
    if data.is_empty() { return Err(StatsrustError::NoDataPoints); }
    let mut values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 1 { Ok(values[n / 2]) } else { Ok(values[n / 2 - 1]) }
}

pub fn median_high<T>(data: &[T]) -> Result<f64, StatsrustError>
where T: ToPrimitive + Copy + PartialOrd
{
    if data.is_empty() { return Err(StatsrustError::NoDataPoints); }
    let mut values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    Ok(values[n / 2])
}

pub fn median_grouped<T>(data: &[T], interval: f64) -> Result<f64, StatsrustError>
where T: ToPrimitive + Copy + PartialOrd
{
    if data.is_empty() { return Err(StatsrustError::NoDataPoints); }
    if interval <= 0.0 { return Err(StatsrustError::InvalidBandwidth); }
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

pub fn mode<T: Eq + std::hash::Hash + Copy>(data: &[T]) -> Result<T, StatsrustError> {
    if data.is_empty() { return Err(StatsrustError::NoDataPoints); }
    let mut counts = HashMap::new();
    for &item in data { *counts.entry(item).or_insert(0) += 1; }
    counts.into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(val, _)| val)
        .ok_or(StatsrustError::NoDataPoints)
}

pub fn multimode<T: Eq + std::hash::Hash + Copy>(data: &[T]) -> Vec<T> {
    if data.is_empty() { return Vec::new(); }
    let mut counts = HashMap::new();
    for &item in data { *counts.entry(item).or_insert(0) += 1; }
    let max_count = *counts.values().max().unwrap_or(&0);
    counts.into_iter()
        .filter(|&(_, count)| count == max_count)
        .map(|(val, _)| val)
        .collect()
}

pub fn pvariance<T>(data: &[T], mu: Option<f64>) -> Result<f64, StatsrustError>
where T: ToPrimitive + Copy
{
    if data.is_empty() { return Err(StatsrustError::InsufficientPopulationData); }
    let values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    let mu = match mu { Some(mu) => mu, None => mean(&values)? };
    let variance = values.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / values.len() as f64;
    Ok(variance)
}

pub fn pstdev<T>(data: &[T], mu: Option<f64>) -> Result<f64, StatsrustError>
where T: ToPrimitive + Copy
{
    pvariance(data, mu).map(|var| var.sqrt())
}

pub fn variance<T>(data: &[T], xbar: Option<f64>) -> Result<f64, StatsrustError>
where T: ToPrimitive + Copy
{
    if data.len() < 2 { return Err(StatsrustError::InsufficientSampleData); }
    let values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    let xbar = match xbar { Some(xbar) => xbar, None => mean(&values)? };
    let variance = values.iter().map(|&x| (x - xbar).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    Ok(variance)
}

pub fn stdev<T>(data: &[T], xbar: Option<f64>) -> Result<f64, StatsrustError>
where T: ToPrimitive + Copy
{
    variance(data, xbar).map(|var| var.sqrt())
}

pub fn quantiles<T>(data: &[T], n: usize, method: &str) -> Result<Vec<f64>, StatsrustError>
where T: ToPrimitive + Copy + PartialOrd
{
    if n < 1 || data.is_empty() { return Err(StatsrustError::NoDataPoints); }
    let mut values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ld = values.len();
    if ld < 2 {
        if ld == 1 { return Ok(vec![values[0]; n - 1]); }
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

pub fn correlation<T>(x: &[T], y: &[T], method: &str) -> Result<f64, StatsrustError>
where T: ToPrimitive + Copy
{
    if x.len() != y.len() { return Err(StatsrustError::MismatchedLengths); }
    if x.len() < 2 { return Err(StatsrustError::NotEnoughDataPoints); }
    let x: Vec<f64> = x.iter().map(|&v| v.to_f64().ok_or(StatsrustError::NonNumericInput)).collect::<Result<Vec<_>, _>>()?;
    let y: Vec<f64> = y.iter().map(|&v| v.to_f64().ok_or(StatsrustError::NonNumericInput)).collect::<Result<Vec<_>, _>>()?;
    match method {
        "linear" => {
            let x_mean = mean(&x)?;
            let y_mean = mean(&y)?;
            let (sxx, syy, sxy) = x.iter().zip(y.iter()).fold((0.0, 0.0, 0.0), |(sxx, syy, sxy), (&xi, &yi)| {
                let xc = xi - x_mean;
                let yc = yi - y_mean;
                (sxx + xc * xc, syy + yc * yc, sxy + xc * yc)
            });
            if sxx == 0.0 || syy == 0.0 { return Err(StatsrustError::ConstantInput); }
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

pub fn covariance<T>(x: &[T], y: &[T]) -> Result<f64, StatsrustError>
where T: ToPrimitive + Copy
{
    if x.len() != y.len() { return Err(StatsrustError::MismatchedLengths); }
    if x.len() < 2 { return Err(StatsrustError::NotEnoughDataPoints); }
    let x: Vec<f64> = x.iter().map(|&v| v.to_f64().ok_or(StatsrustError::NonNumericInput)).collect::<Result<Vec<_>, _>>()?;
    let y: Vec<f64> = y.iter().map(|&v| v.to_f64().ok_or(StatsrustError::NonNumericInput)).collect::<Result<Vec<_>, _>>()?;
    let x_mean = mean(&x)?;
    let y_mean = mean(&y)?;
    let sxy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean)).sum();
    Ok(sxy / (x.len() - 1) as f64)
}

pub fn linear_regression<T>(x: &[T], y: &[T], proportional: bool) -> Result<(f64, f64), StatsrustError>
where T: ToPrimitive + Copy
{
    if x.len() != y.len() { return Err(StatsrustError::MismatchedLengths); }
    if x.len() < 2 { return Err(StatsrustError::NotEnoughDataPoints); }
    let x: Vec<f64> = x.iter().map(|&v| v.to_f64().ok_or(StatsrustError::NonNumericInput)).collect::<Result<Vec<_>, _>>()?;
    let y: Vec<f64> = y.iter().map(|&v| v.to_f64().ok_or(StatsrustError::NonNumericInput)).collect::<Result<Vec<_>, _>>()?;
    let x_mean = if proportional { 0.0 } else { mean(&x)? };
    let y_mean = if proportional { 0.0 } else { mean(&y)? };
    let (sxx, sxy) = x.iter().zip(y.iter()).fold((0.0, 0.0), |(sxx, sxy), (&xi, &yi)| {
        let xc = if proportional { xi } else { xi - x_mean };
        let yc = if proportional { yi } else { yi - y_mean };
        (sxx + xc * xc, sxy + xc * yc)
    });
    if sxx == 0.0 { return Err(StatsrustError::ConstantInput); }
    let slope = sxy / sxx;
    let intercept = if proportional { 0.0 } else { y_mean - slope * x_mean };
    Ok((slope, intercept))
}

fn rank(data: &[f64]) -> Result<Vec<f64>, StatsrustError> {
    let n = data.len();
    let mut indexed: Vec<(f64, usize)> = data.iter().copied().zip(0..n).collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].0 - indexed[i].0).abs() < 1e-10 { j += 1; }
        let r = (i + j - 1) as f64 / 2.0 + 1.0;
        for rank_slot in ranks.iter_mut().take(j).skip(i) {
            *rank_slot = r;
        }
        i = j;
    }
    Ok(ranks)
}

// ==========================================
// KDE & Distributions
// ==========================================

pub fn kde<T>(data: &[T], h: f64, kernel: Kernel, cumulative: bool) -> Result<Box<dyn Fn(f64) -> f64 + Send + Sync>, StatsrustError>
where T: ToPrimitive + Copy
{
    if data.is_empty() { return Err(StatsrustError::NoDataPoints); }
    if h <= 0.0 { return Err(StatsrustError::InvalidBandwidth); }
    let values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    let n = values.len() as f64;
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
                    let i = sorted.partition_point(|&v| v < x - bandwidth);
                    let j = sorted.partition_point(|&v| v <= x + bandwidth);
                    let mut sum = 0.0;
                    for &v in &sorted[i..j] {
                        sum += cdf_fn((x - v) / h);
                    }
                    sum / n
                }))
            },
            None => {
                Ok(Box::new(move |x: f64| -> f64 {
                    values.iter().map(|&x_i| cdf_fn((x - x_i) / h)).sum::<f64>() / n
                }))
            }
        }
    } else {
        match support {
            Some(s) => {
                let bandwidth = h * s;
                Ok(Box::new(move |x: f64| -> f64 {
                    let i = sorted.partition_point(|&v| v < x - bandwidth);
                    let j = sorted.partition_point(|&v| v <= x + bandwidth);
                    let mut sum = 0.0;
                    for &v in &sorted[i..j] {
                        sum += kernel_fn((x - v) / h);
                    }
                    sum / (n * h)
                }))
            },
            None => {
                Ok(Box::new(move |x: f64| -> f64 {
                    values.iter().map(|&x_i| kernel_fn((x - x_i) / h)).sum::<f64>() / (n * h)
                }))
            }
        }
    }
}

pub fn kde_random<T>(data: &[T], h: f64, kernel: Kernel, seed: Option<u64>) -> Result<Box<dyn FnMut() -> f64 + Send + Sync>, StatsrustError>
where T: ToPrimitive + Copy
{
    if data.is_empty() { return Err(StatsrustError::NoDataPoints); }
    if h <= 0.0 { return Err(StatsrustError::InvalidBandwidth); }
    let inv_cdf = kernel.inv_cdf();
    let values: Vec<f64> = data.iter()
        .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
        .collect::<Result<Vec<_>, _>>()?;
    let mut rng = match seed {
        Some(seed) => SplitMix64::new(seed),
        None => SplitMix64::from_entropy(),
    };
    let values_clone = values.clone();
    Ok(Box::new(move || -> f64 {
        let index = rng.gen_range_usize(values_clone.len());
        let x_i = values_clone[index];
        x_i + h * inv_cdf(rng.next_f64())
    }))
}

#[derive(Debug, Clone, Copy)]
pub struct NormalDist {
    mu: f64,
    sigma: f64,
}

impl NormalDist {
    pub fn new(mu: f64, sigma: f64) -> Result<Self, StatsrustError> {
        if sigma < 0.0 { return Err(StatsrustError::NegativeValueNotAllowed); }
        Ok(Self { mu, sigma })
    }

    pub fn from_samples<T>(data: &[T]) -> Result<Self, StatsrustError>
    where T: ToPrimitive + Copy
    {
        if data.len() < 2 { return Err(StatsrustError::NotEnoughDataPoints); }
        let values: Vec<f64> = data.iter()
            .map(|&x| x.to_f64().ok_or(StatsrustError::NonNumericInput))
            .collect::<Result<Vec<_>, _>>()?;
        let m = mean(&values)?;
        let v = variance(&values, Some(m))?;
        Ok(Self { mu: m, sigma: v.sqrt() })
    }

    pub fn samples(&self, n: usize, seed: Option<u64>) -> Vec<f64> {
        let mut rng = match seed {
            Some(seed) => SplitMix64::new(seed),
            None => SplitMix64::from_entropy(),
        };
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n.div_ceil(2) {
            let u1: f64 = rng.next_f64().max(1e-15);
            let u2: f64 = rng.next_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            samples.push(self.mu + self.sigma * r * theta.cos());
            if samples.len() < n {
                samples.push(self.mu + self.sigma * r * theta.sin());
            }
        }
        samples.truncate(n);
        samples
    }

    pub fn pdf(&self, x: f64) -> f64 {
        if self.sigma == 0.0 {
            if (x - self.mu).abs() < 1e-10 { f64::INFINITY } else { 0.0 }
        } else {
            let norm = 1.0 / (self.sigma * (2.0 * std::f64::consts::PI).sqrt());
            let exponent = -((x - self.mu) * (x - self.mu)) / (2.0 * self.sigma * self.sigma);
            norm * exponent.exp()
        }
    }

    pub fn cdf(&self, x: f64) -> f64 {
        if self.sigma == 0.0 {
            if x < self.mu { 0.0 } else { 1.0 }
        } else {
            normal_cdf((x - self.mu) / self.sigma)
        }
    }

    pub fn inv_cdf(&self, p: f64) -> Result<f64, StatsrustError> {
        if p <= 0.0 || p >= 1.0 { return Err(StatsrustError::InvalidProbability); }
        if self.sigma == 0.0 { return Ok(self.mu); }
        Ok(self.mu + self.sigma * normal_inv_cdf(p))
    }

    pub fn quantiles(&self, n: usize) -> Vec<f64> {
        (1..n).map(|i| self.inv_cdf(i as f64 / n as f64).unwrap_or(f64::NAN)).collect()
    }

    pub fn overlap(&self, other: &Self) -> f64 {
        let (x, y) = if (other.sigma, other.mu) < (self.sigma, self.mu) {
            (other, self)
        } else {
            (self, other)
        };
        if x.sigma == 0.0 || y.sigma == 0.0 { return 0.0; }
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

    pub fn zscore(&self, x: f64) -> Result<f64, StatsrustError> {
        if self.sigma == 0.0 { return Err(StatsrustError::NegativeValueNotAllowed); }
        Ok((x - self.mu) / self.sigma)
    }

    pub fn mean(&self) -> f64 { self.mu }
    pub fn median(&self) -> f64 { self.mu }
    pub fn mode(&self) -> f64 { self.mu }
    pub fn stdev(&self) -> f64 { self.sigma }
    pub fn variance(&self) -> f64 { self.sigma * self.sigma }
}

impl std::ops::Add<f64> for NormalDist {
    type Output = NormalDist;
    fn add(self, rhs: f64) -> Self::Output { NormalDist::new(self.mu + rhs, self.sigma).unwrap() }
}
impl std::ops::Sub<f64> for NormalDist {
    type Output = NormalDist;
    fn sub(self, rhs: f64) -> Self::Output { NormalDist::new(self.mu - rhs, self.sigma).unwrap() }
}
impl std::ops::Mul<f64> for NormalDist {
    type Output = NormalDist;
    fn mul(self, rhs: f64) -> Self::Output { NormalDist::new(self.mu * rhs, self.sigma * rhs.abs()).unwrap() }
}
impl std::ops::Div<f64> for NormalDist {
    type Output = NormalDist;
    fn div(self, rhs: f64) -> Self::Output { NormalDist::new(self.mu / rhs, self.sigma / rhs.abs()).unwrap() }
}
impl std::ops::Add<NormalDist> for NormalDist {
    type Output = NormalDist;
    fn add(self, rhs: NormalDist) -> Self::Output {
        NormalDist::new(self.mu + rhs.mu, (self.sigma.powi(2) + rhs.sigma.powi(2)).sqrt()).unwrap()
    }
}
impl std::ops::Sub<NormalDist> for NormalDist {
    type Output = NormalDist;
    fn sub(self, rhs: NormalDist) -> Self::Output {
        NormalDist::new(self.mu - rhs.mu, (self.sigma.powi(2) + rhs.sigma.powi(2)).sqrt()).unwrap()
    }
}

// ==========================================
// Tests
// ==========================================

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
    fn test_harmonic_mean_basic() {
        let data = vec![1.0, 2.0, 4.0];
        let result = harmonic_mean(&data, None).unwrap();
        assert!((result - 12.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_harmonic_mean_equal_values() {
        let data = vec![5.0, 5.0, 5.0];
        assert!((harmonic_mean(&data, None).unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_harmonic_mean_single_element() {
        let data = vec![3.0];
        assert!((harmonic_mean(&data, None).unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_harmonic_mean_weighted() {
        let data = vec![2.0, 4.0];
        let weights = vec![1.0, 2.0];
        let result = harmonic_mean(&data, Some(&weights)).unwrap();
        assert!((result - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_harmonic_mean_rejects_zero() {
        let data = vec![1.0, 0.0, 3.0];
        assert!(harmonic_mean(&data, None).is_err());
    }

    #[test]
    fn test_harmonic_mean_rejects_negative() {
        let data = vec![1.0, -2.0, 3.0];
        assert!(harmonic_mean(&data, None).is_err());
    }

    #[test]
    fn test_harmonic_mean_rejects_zero_mean_path() {
        let data = vec![-1.0, 1.0];
        assert!(harmonic_mean(&data, None).is_err());
    }

    #[test]
    fn test_harmonic_mean_empty() {
        let data: Vec<f64> = vec![];
        assert!(harmonic_mean(&data, None).is_err());
    }

    #[test]
    fn test_harmonic_mean_weight_mismatch() {
        let data = vec![1.0, 2.0];
        let weights = vec![1.0];
        assert!(harmonic_mean(&data, Some(&weights)).is_err());
    }

    #[test]
    fn test_harmonic_mean_zero_weight_sum() {
        let data = vec![1.0, 2.0];
        let weights = vec![0.0, 0.0];
        assert!(harmonic_mean(&data, Some(&weights)).is_err());
    }

    #[test]
    fn test_median_grouped_empty() {
        let data: Vec<f64> = vec![];
        assert!(median_grouped(&data, 1.0).is_err());
    }

    #[test]
    fn test_median_grouped_single_element() {
        let data = vec![5.0];
        let result = median_grouped(&data, 2.0).unwrap();
        assert!((result - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_grouped_all_same() {
        let data = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let result = median_grouped(&data, 1.0).unwrap();
        assert!((result - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_grouped_two_elements() {
        let data = vec![1.0, 3.0];
        let result = median_grouped(&data, 1.0).unwrap();
        assert!((result - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_median_grouped_even_count() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = median_grouped(&data, 1.0).unwrap();
        assert!((result - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_median_grouped_odd_count() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = median_grouped(&data, 1.0).unwrap();
        assert!((result - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_grouped_negative_values() {
        let data = vec![-5.0, -3.0, -1.0];
        let result = median_grouped(&data, 1.0).unwrap();
        assert!((result - (-3.0)).abs() < 1e-12);
    }

    #[test]
    fn test_median_grouped_small_interval() {
        let data = vec![1.0, 2.0, 3.0];
        let result = median_grouped(&data, 0.1).unwrap();
        assert!((result - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_median_grouped_large_interval() {
        let data = vec![1.0, 2.0, 3.0];
        let result = median_grouped(&data, 100.0).unwrap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_median_grouped_invalid_interval() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(median_grouped(&data, 0.0).is_err());
        assert!(median_grouped(&data, -1.0).is_err());
    }

    #[test]
    fn test_median_grouped_duplicates() {
        let data = vec![1.0, 2.0, 2.0, 2.0, 3.0];
        let result = median_grouped(&data, 1.0).unwrap();
        assert!((result - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_quartic_inv_cdf_precision() {
        for &p in &[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
            let t = quartic_inv_cdf(p);
            let p_recovered = quartic_cdf(t);
            assert!(
                (p_recovered - p).abs() < 1e-10,
                "quartic_inv_cdf({}) = {}, cdf roundtrip = {}, error = {}",
                p, t, p_recovered, (p_recovered - p).abs()
            );
        }
    }

    #[test]
    fn test_triweight_inv_cdf_precision() {
        for &p in &[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
            let t = triweight_inv_cdf(p);
            let p_recovered = triweight_cdf(t);
            assert!(
                (p_recovered - p).abs() < 1e-10,
                "triweight_inv_cdf({}) = {}, cdf roundtrip = {}, error = {}",
                p, t, p_recovered, (p_recovered - p).abs()
            );
        }
    }

    #[test]
    fn test_quartic_inv_cdf_boundaries() {
        assert!((quartic_inv_cdf(0.0) - (-1.0)).abs() < 1e-12);
        assert!((quartic_inv_cdf(1.0) - 1.0).abs() < 1e-12);
        assert!((quartic_inv_cdf(0.5) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_triweight_inv_cdf_boundaries() {
        assert!((triweight_inv_cdf(0.0) - (-1.0)).abs() < 1e-12);
        assert!((triweight_inv_cdf(1.0) - 1.0).abs() < 1e-12);
        assert!((triweight_inv_cdf(0.5) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_parabolic_inv_cdf_still_exact() {
        for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let t = parabolic_inv_cdf(p);
            let p_recovered = parabolic_cdf(t);
            assert!(
                (p_recovered - p).abs() < 1e-10,
                "parabolic roundtrip error at p={}: {}", p, (p_recovered - p).abs()
            );
        }
    }

    #[test]
    fn test_all_compact_kernels_inv_cdf_roundtrip() {
        let kernels = [
            (Kernel::Rectangular, "rectangular"),
            (Kernel::Triangular, "triangular"),
            (Kernel::Parabolic, "parabolic"),
            (Kernel::Quartic, "quartic"),
            (Kernel::Triweight, "triweight"),
            (Kernel::Cosine, "cosine"),
        ];
        for &(kernel, name) in &kernels {
            let cdf_fn = kernel.cdf();
            let inv_fn = kernel.inv_cdf();
            for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
                let t = inv_fn(p);
                let p2 = cdf_fn(t);
                assert!(
                    (p2 - p).abs() < 1e-8,
                    "{} inv_cdf roundtrip failed: p={}, recovered={}", name, p, p2
                );
            }
        }
    }

    #[test]
    fn test_normal_dist() {
        let dist = NormalDist::new(0.0, 1.0).unwrap();
        assert_eq!(dist.mean(), 0.0);
        assert_eq!(dist.stdev(), 1.0);
        let samples = dist.samples(1000, Some(42));
        assert_eq!(samples.len(), 1000);
        let dist2 = NormalDist::from_samples(&samples).unwrap();
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
    fn test_box_muller() {
        let dist = NormalDist::new(0.0, 1.0).unwrap();
        let samples = dist.samples(10000, Some(42));
        let m = samples.iter().sum::<f64>() / samples.len() as f64;
        let v = samples.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;
        assert!((m - 0.0).abs() < 0.05);
        assert!((v - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_parabolic_kernel() {
        let kernel = Kernel::Parabolic;
        let inv_cdf = kernel.inv_cdf();
        assert!((inv_cdf(0.5) - 0.0).abs() < 1e-10);
        assert!((inv_cdf(0.0) + 1.0).abs() < 1e-10);
        assert!((inv_cdf(1.0) - 1.0).abs() < 1e-10);
    }
}
