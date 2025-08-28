
# Contributing to statsrust Library

![Apache 2.0 | MIT License](https://img.shields.io/badge/License-Apache%202.0%20%7C%20MIT-blue.svg)
![CC BY 4.0 License](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)

Thank you for considering contributing to the statsrust Library! This comprehensive statistical analysis toolkit for Rust developers welcomes contributions from the community. Your help is essential to making this library more robust, feature-rich, and user-friendly.

## How Can I Contribute?

### Reporting Bugs

We use GitHub issues to track bugs. Please ensure your description is clear and includes sufficient instructions to reproduce the issue.

When reporting a bug, please include:

- Your operating system name and version
- Any details about your local setup that might be helpful
- Detailed steps to reproduce the bug
- Expected behavior
- Actual behavior

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Please provide:

- A clear and descriptive title
- A step-by-step description of the suggested enhancement
- Specific examples to demonstrate the steps
- Why this enhancement would be useful to users

### Pull Requests

The process for submitting a pull request is:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contribution Guidelines

To ensure high-quality contributions, please follow these guidelines:

### Code Style

- Follow Rust community best practices and style guidelines
- Use `rustfmt` for code formatting (run `cargo fmt` before submitting)
- Pass all `cargo clippy` checks
- Maintain consistent naming conventions (`snake_case` for functions, `PascalCase` for types)
- **Include proper license headers in all new code files** (see License Agreement section below)

### Documentation

- Include comprehensive documentation for all public APIs using Rustdoc format
- Document edge cases and error conditions
- Keep examples up-to-date and testable
- Update relevant documentation files when adding new features

### Testing

- Add unit tests covering normal cases, edge cases, and error conditions
- Ensure all tests pass (`cargo test`)
- Consider adding performance benchmarks for critical algorithms
- Verify numerical stability in statistical calculations

### Numerical Considerations

- Prioritize numerical stability in statistical calculations
- Use appropriate algorithms to prevent overflow, underflow, and cancellation errors
- Document any approximations or trade-offs made for performance
- Consider edge cases like empty datasets, constant inputs, etc.

### Performance

- Consider time and space complexity of new algorithms
- Use efficient data structures and algorithms
- Profile performance-critical sections
- Document performance characteristics where relevant

## Setting Up for Development

1. Fork the repository on GitHub
2. Clone your fork locally: `git clone https://github.com/your-username/statsrust.git`
3. Install Rust and dependencies:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup update
   ```
4. Install development tools:
   ```bash
   cargo install cargo-edit cargo-watch
   ```
5. Build the project: `cargo build`
6. Run tests: `cargo test`

## Pull Request Process

- Ensure all tests pass and code is formatted correctly
- Update documentation to reflect your changes
- Include a clear description of what the PR does and why it's needed
- Reference any relevant issues in the description
- Be responsive to code review comments
- Maintain a clean commit history (consider squashing related commits)

## License Agreement for Contributions

### Software Code Contributions

By contributing code to this project, you agree that your contributions will be licensed under both:

- [Apache License 2.0](LICENSE-APACHE) ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- [MIT License](LICENSE-MIT) ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

**Important note for code contributors:**

All new source code files must include the following license header:

```rust
// Copyright (c) [current year] statsrust Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
```

This dual licensing model means your code will be available to users under either of these licenses, at their option.

### Documentation and Non-Code Content

By contributing documentation, examples, or other non-code content, you agree that your contributions will be licensed under:

- [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](LICENSE-CC) ([LICENSE-CC](LICENSE-CC) or https://creativecommons.org/licenses/by/4.0/)

### Clear Distinction Between Content Types

**Code content includes:**
- Rust source files (.rs)
- Build scripts
- Test code
- Example applications in `/examples`
- Any other executable code

**Non-code content includes:**
- Documentation files (README, CONTRIBUTING, etc.)
- Design documents
- Issue templates
- Markdown documentation

**Special case - documentation examples:** Code snippets within documentation (Rustdoc, markdown files) are considered code content and fall under the dual Apache/MIT licensing model

## Third-Party Dependencies and License Compatibility

This project depends on several external crates (including but not limited to: `thiserror`, `num-traits`, `ndarray`, `statrs`, `rand`).

### License Compatibility Assurance

- All dependencies use licenses compatible with our dual licensing model (primarily MIT and Apache 2.0)
- When adding new dependencies, contributors must verify that the dependency's license is compatible with both Apache 2.0 and MIT licenses
- The project maintainers will review all new dependencies for license compatibility before merging

### Dependency License Verification Process

1. Check the dependency's license using `cargo license`
2. Confirm it's one of the following (or compatible with): MIT, Apache-2.0, BSD, ISC
3. Document the license verification in the PR description when adding new dependencies

## NOTICE

The dual licensing for software code allows maximum flexibility for users while maintaining compatibility with both permissive and copyleft projects. When contributing, please be aware that:

- Your code submissions will be available under both Apache 2.0 and MIT license terms
- Documentation and non-code content will be available under CC-BY-4.0
- Code examples within documentation are considered code content and fall under the dual licensing model

## Additional Resources

- [Rust Style Guide](https://doc.rust-lang.org/1.0.0/style/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Rust Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)

Thank you for your interest in contributing to statsrust Library! Your efforts help make statistical analysis in Rust more accessible and powerful for everyone.

This document is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).  
Original author: statsrust Authors
