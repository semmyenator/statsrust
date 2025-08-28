# statsrust Library API Documentation

![Apache 2.0 | MIT License](https://img.shields.io/badge/License-Apache%202.0%20%7C%20MIT-blue.svg)
![CC BY 4.0 License](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)

## 1. Error Handling

### `StatError` Enumeration

The library utilizes a comprehensive error enumeration for precise error reporting. All public functions return `Result<T, StatError>`.

| Error Variant | Trigger Condition | Error Code |
|---------------|-------------------|------------|
| `NoDataPoints` | Empty dataset provided | 1001 |
| `NotEnoughDataPoints` | Insufficient data points for operation | 1002 |
| `NegativeValueNotAllowed` | Negative value where positive required | 1003 |
| `ZeroWeightSum` | Sum of weights equals zero | 1004 |
| `MismatchedLengths` | Mismatched sequence lengths | 1005 |
| `UnknownKernel(String)` | Invalid kernel name provided | 1006 |
| `InvalidBandwidth` | Bandwidth parameter ≤ 0 | 1007 |
| `NonNumericInput` | Non-numeric value in numeric context | 1008 |
| `InvalidProbability` | Probability value outside [0.0, 1.0] | 1009 |
| `InsufficientPopulationData` | Population variance requires ≥ 1 data point | 1010 |
| `InsufficientSampleData` | Sample variance requires ≥ 2 data points | 1011 |
| `ConstantInput` | Constant input for correlation calculation | 1012 |

## 2. Core Statistical Functions

### 2.1 Central Tendency Measures

#### `mean<T>(data: &[T]) -> Result<f64, StatError>`

**Description**: Computes arithmetic mean of input data.

**Constraints**:
- `T: Numeric` (must implement appropriate numeric traits)
- `data` must not be empty

**Time Complexity**: O(n)
**Space Complexity**: O(1)

#### `fmean<T>(data: &[T], weights: Option<&[T]>) -> Result<f64, StatError>`

**Description**: Computes weighted arithmetic mean.

**Parameters**:
- `data`: Input data sequence
- `weights`: Optional weights sequence (must match data length)

**Constraints**:
- All weights must be non-negative
- Sum of weights must be positive
- If weights provided, lengths must match data

**Time Complexity**: O(n)
**Space Complexity**: O(1)

#### `geometric_mean<T>(data: &[T]) -> Result<f64, StatError>`

**Description**: Computes geometric mean.

**Constraints**:
- All input values must be positive
- `data` must not be empty

**Time Complexity**: O(n)
**Space Complexity**: O(1)

#### `harmonic_mean<T>(data: &[T], weights: Option<&[T]>) -> Result<f64, StatError>`

**Description**: Computes harmonic mean.

**Constraints**:
- All input values must be positive
- Weights (if provided) must be positive
- If weights provided, lengths must match data

**Time Complexity**: O(n)
**Space Complexity**: O(1)

#### `mode<T: Eq + Hash + Copy>(data: &[T]) -> Result<T, StatError>`

**Description**: Returns the most frequent value in the dataset.

**Constraints**:
- `data` must not be empty

**Time Complexity**: O(n)
**Space Complexity**: O(n)

#### `multimode<T: Eq + Hash + Copy>(data: &[T]) -> Vec<T>`

**Description**: Returns all values with maximum frequency.

**Constraints**: None (returns empty vector for empty input)

**Time Complexity**: O(n)
**Space Complexity**: O(n)

### 2.2 Position Measures

#### `median<T>(data: &[T]) -> Result<f64, StatError>`

**Description**: Computes the median value (average of middle two values for even-sized datasets).

**Constraints**:
- `data` must not be empty

**Time Complexity**: O(n log n) [due to sorting]
**Space Complexity**: O(n)

#### `median_low<T>(data: &[T]) -> Result<f64, StatError>`

**Description**: Computes the low median (smaller middle value for even-sized datasets).

**Constraints**:
- `data` must not be empty

**Time Complexity**: O(n log n)
**Space Complexity**: O(n)

#### `median_high<T>(data: &[T]) -> Result<f64, StatError>`

**Description**: Computes the high median (larger middle value for even-sized datasets).

**Constraints**:
- `data` must not be empty

**Time Complexity**: O(n log n)
**Space Complexity**: O(n)

#### `median_grouped<T>(data: &[T], interval: f64) -> Result<f64, StatError>`

**Description**: Computes the median for grouped data with specified interval.

**Parameters**:
- `interval`: Grouping interval width (> 0)

**Constraints**:
- `data` must not be empty
- `interval` must be positive

**Time Complexity**: O(n log n)
**Space Complexity**: O(n)

### 2.3 Variability Measures

#### `pvariance<T>(data: &[T], mu: Option<f64>) -> Result<f64, StatError>`

**Description**: Computes population variance (dividing by N).

**Parameters**:
- `mu`: Optional precomputed mean

**Constraints**:
- `data` must not be empty

**Time Complexity**: O(n)
**Space Complexity**: O(n) [for data copying]

#### `pstdev<T>(data: &[T], mu: Option<f64>) -> Result<f64, StatError>`

**Description**: Computes population standard deviation.

**Parameters**:
- `mu`: Optional precomputed mean

**Constraints**:
- `data` must not be empty

**Time Complexity**: O(n)
**Space Complexity**: O(n)

#### `variance<T>(data: &[T], xbar: Option<f64>) -> Result<f64, StatError>`

**Description**: Computes sample variance (dividing by n-1).

**Parameters**:
- `xbar`: Optional precomputed mean

**Constraints**:
- `data` must contain at least two elements

**Time Complexity**: O(n)
**Space Complexity**: O(n)

#### `stdev<T>(data: &[T], xbar: Option<f64>) -> Result<f64, StatError>`

**Description**: Computes sample standard deviation.

**Parameters**:
- `xbar`: Optional precomputed mean

**Constraints**:
- `data` must contain at least two elements

**Time Complexity**: O(n)
**Space Complexity**: O(n)

### 2.4 Correlation and Regression

#### `correlation<T>(x: &[T], y: &[T], method: &str) -> Result<f64, StatError>`

**Description**: Computes correlation coefficient between two datasets.

**Parameters**:
- `method`: "linear" (Pearson) or "ranked" (Spearman)

**Constraints**:
- `x` and `y` must have matching lengths
- Must contain at least two data points
- Input must not be constant

**Time Complexity**: 
- "linear": O(n)
- "ranked": O(n log n) [due to sorting]

**Space Complexity**: O(n)

#### `covariance<T>(x: &[T], y: &[T]) -> Result<f64, StatError>`

**Description**: Computes sample covariance between two datasets.

**Constraints**:
- `x` and `y` must have matching lengths
- Must contain at least two data points

**Time Complexity**: O(n)
**Space Complexity**: O(n)

#### `linear_regression<T>(x: &[T], y: &[T], proportional: bool) -> Result<(f64, f64), StatError>`

**Description**: Performs linear regression analysis.

**Parameters**:
- `proportional`: Whether to force regression line through origin

**Constraints**:
- `x` and `y` must have matching lengths
- Must contain at least two data points
- `x` must not be constant

**Time Complexity**: O(n)
**Space Complexity**: O(n)

### 2.5 Quantiles

#### `quantiles<T>(data: &[T], n: usize, method: &str) -> Result<Vec<f64>, StatError>`

**Description**: Computes n-1 quantile points dividing data into n equal-sized intervals.

**Parameters**:
- `n`: Number of intervals (must be ≥ 2)
- `method`: "inclusive" or "exclusive"

**Constraints**:
- `data` must not be empty
- `n` must be at least 2

**Time Complexity**: O(n log n) [due to sorting]
**Space Complexity**: O(n)

## 3. Kernel Density Estimation

### 3.1 Kernel Enumeration

The `Kernel` enum provides implementations for various kernel functions:

| Variant | Supported Names | Support | Characteristics |
|---------|----------------|---------|----------------|
| `Normal` | "normal", "gauss" | (-∞, ∞) | Infinite support, smoothest |
| `Parabolic` | "parabolic", "epanechnikov" | [-1, 1] | Optimal MSE, computationally efficient |
| `Triangular` | "triangular" | [-1, 1] | Linear decay, simplest bounded |
| `Quartic` | "quartic", "biweight" | [-1, 1] | Smoother than Epanechnikov |
| `Triweight` | "triweight" | [-1, 1] | Higher-order polynomial |
| `Rectangular` | "rectangular", "uniform" | [-1, 1] | Simplest kernel |
| `Cosine` | "cosine" | [-1, 1] | Alternative smooth kernel |
| `Logistic` | "logistic" | (-∞, ∞) | Heavy-tailed alternative |
| `Sigmoid` | "sigmoid" | (-∞, ∞) | Alternative smooth kernel |

#### `Kernel::from_name(name: &str) -> Result<Self, StatError>`

**Description**: Creates a kernel from a name string (case-insensitive).

**Constraints**:
- `name` must be a recognized kernel name

**Time Complexity**: O(1)
**Space Complexity**: O(1)

### 3.2 KDE Functions

#### `kde<T>(data: &[T], bandwidth: f64, kernel: &str, cdf: bool) -> Result<Box<dyn Fn(f64) -> f64 + Send + Sync>, StatError>`

**Description**: Creates a kernel density estimation function.

**Parameters**:
- `bandwidth`: Smoothing parameter (> 0)
- `kernel`: Kernel function name
- `cdf`: Whether to compute CDF instead of PDF

**Constraints**:
- `data` must not be empty
- `bandwidth` must be positive
- `kernel` must be a recognized kernel name

**Time Complexity (per evaluation)**:
- Bounded kernels: O(log n + k) where k is points in support
- Unbounded kernels: O(n)

**Space Complexity**: O(n log n) [for sorted data storage]

#### `kde_random<T>(data: &[T], bandwidth: f64, kernel: &str, seed: Option<u64>) -> Result<Box<dyn FnMut() -> f64 + Send + Sync>, StatError>`

**Description**: Creates a random sample generator from KDE.

**Parameters**:
- `seed`: Optional random number generator seed

**Constraints**:
- `data` must not be empty
- `bandwidth` must be positive
- `kernel` must be a recognized kernel name

**Time Complexity (per sample)**: O(log n)
**Space Complexity**: O(n)

## 4. Normal Distribution Model

### 4.1 `NormalDist` Struct

Represents a normal distribution with mean `mu` and standard deviation `sigma`.

#### Creation Methods

##### `NormalDist::new(mu: f64, sigma: f64) -> Result<Self, StatError>`

**Description**: Creates a new normal distribution with specified parameters.

**Constraints**:
- `sigma` must be positive

**Time Complexity**: O(1)
**Space Complexity**: O(1)

##### `NormalDist::from_samples<T>(data: &[T]) -> Result<Self, StatError>`

**Description**: Estimates normal distribution parameters from sample data.

**Constraints**:
- `data` must contain at least two elements

**Time Complexity**: O(n)
**Space Complexity**: O(n)

#### Distribution Properties

##### `pdf(&self, x: f64) -> Result<f64, StatError>`

**Description**: Computes probability density at point x.

**Constraints**:
- `sigma` must be positive

**Time Complexity**: O(1)
**Space Complexity**: O(1)

##### `cdf(&self, x: f64) -> Result<f64, StatError>`

**Description**: Computes cumulative probability up to point x.

**Constraints**:
- `sigma` must be positive

**Time Complexity**: O(1)
**Space Complexity**: O(1)

##### `inv_cdf(&self, p: f64) -> Result<f64, StatError>`

**Description**: Computes inverse CDF (quantile function).

**Parameters**:
- `p`: Probability value (0.0 < p < 1.0)

**Constraints**:
- `p` must be in (0.0, 1.0)
- `sigma` must be positive

**Time Complexity**: O(1)
**Space Complexity**: O(1)

##### `overlap(&self, other: &Self) -> f64`

**Description**: Computes overlapping area between two normal distributions.

**Time Complexity**: O(1)
**Space Complexity**: O(1)

##### `zscore(&self, x: f64) -> Result<f64, StatError>`

**Description**: Computes Z-score for value x.

**Constraints**:
- `sigma` must be positive

**Time Complexity**: O(1)
**Space Complexity**: O(1)

#### Statistical Measures

##### `mean(&self) -> f64`

**Description**: Returns the mean parameter (μ).

**Time Complexity**: O(1)
**Space Complexity**: O(1)

##### `median(&self) -> f64`

**Description**: Returns the median (equal to mean for normal distribution).

**Time Complexity**: O(1)
**Space Complexity**: O(1)

##### `mode(&self) -> f64`

**Description**: Returns the mode (equal to mean for normal distribution).

**Time Complexity**: O(1)
**Space Complexity**: O(1)

##### `stdev(&self) -> f64`

**Description**: Returns the standard deviation parameter (σ).

**Time Complexity**: O(1)
**Space Complexity**: O(1)

##### `variance(&self) -> f64`

**Description**: Returns the variance (σ²).

**Time Complexity**: O(1)
**Space Complexity**: O(1)

##### `quantiles(&self, n: usize) -> Vec<f64>`

**Description**: Computes n-1 equally spaced quantiles.

**Parameters**:
- `n`: Number of intervals (must be ≥ 2)

**Constraints**:
- `n` must be at least 2

**Time Complexity**: O(n)
**Space Complexity**: O(n)

#### Arithmetic Operations

##### Scalar Operations

| Operation | Resulting Distribution | Constraints |
|-----------|------------------------|-------------|
| `dist + c` | N(μ+c, σ) | None |
| `dist - c` | N(μ-c, σ) | None |
| `dist * c` | N(μ×c, σ×\|c\|) | c ≠ 0 |
| `dist / c` | N(μ/c, σ/\|c\|) | c ≠ 0 |

##### Distribution Operations (Independent Distributions)

| Operation | Resulting Distribution | Constraints |
|-----------|------------------------|-------------|
| `dist1 + dist2` | N(μ₁+μ₂, √(σ₁²+σ₂²)) | None |
| `dist1 - dist2` | N(μ₁-μ₂, √(σ₁²+σ₂²)) | None |

**Time Complexity**: O(1) for all operations
**Space Complexity**: O(1) for all operations

##### `samples(&self, n: usize, seed: Option<u64>) -> Vec<f64>`

**Description**: Generates random samples from the distribution.

**Parameters**:
- `n`: Number of samples to generate
- `seed`: Optional random number generator seed

**Constraints**:
- `n` must be positive
- `sigma` must be positive

**Time Complexity**: O(n)
**Space Complexity**: O(n)

## 5. Technical Specifications

### 5.1 Type Constraints

- **Numeric Types**: All statistical functions accept types implementing `num_traits::Float` and `Copy`
- **Input Requirements**: Input sequences must be convertible to `f64` values
- **Data Constraints**: Functions enforce mathematical requirements on input data

### 5.2 Numerical Stability Guarantees

- **Variance Calculation**: Uses two-pass algorithm with centered data to prevent catastrophic cancellation
- **Geometric Mean**: Uses logarithmic transformation to avoid overflow/underflow
- **Kernel Operations**: Implements support boundaries for computational efficiency
- **Distribution Operations**: Preserves mathematical properties through algebraic transformations

### 5.3 Performance Characteristics

| Function Category | Time Complexity | Space Complexity | Notes |
|-------------------|-----------------|------------------|-------|
| Central Tendency | O(n) | O(1) | |
| Position Measures | O(n log n) | O(n) | Due to sorting |
| Variability Measures | O(n) | O(n) | |
| Correlation (linear) | O(n) | O(n) | |
| Correlation (ranked) | O(n log n) | O(n) | Due to sorting |
| Quantiles | O(n log n) | O(n) | Due to sorting |
| KDE (bounded kernels) | O(log n + k) per eval | O(n) | k = points in support |
| KDE (unbounded kernels) | O(n) per eval | O(n) | |
| Normal Distribution ops | O(1) | O(1) | |

### 5.4 Error Handling Protocol

- **Error Propagation**: All errors include contextual information
- **Error Classification**: Errors categorized by type and severity
- **Error Recovery**: Functions fail gracefully without side effects
- **Error Consistency**: Consistent error conditions across similar functions

This document is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).  
Original author: statsrust Authors