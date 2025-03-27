# AltDistributions.jl

![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![build](https://github.com/tpapp/AltDistributions.jl/workflows/CI/badge.svg)](https://github.com/tpapp/AltDistributions.jl/actions?query=workflow%3ACI)
[![codecov.io](http://codecov.io/github/tpapp/AltDistributions.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/AltDistributions.jl?branch=master)

This is a collection of some probability distributions I find useful, primarily for Bayesian estimation. Eventually, they should be considered for contributing to [Distributions.jl](https://github.com/JuliaStats/Distributions.jl), but currently they reside in this package because

1. I am experimenting with the interface,
2. I am experimenting with the implementation (making it friendly to automatic differentiation),
3. not all functionality is implemented (eg only the `logpdf`).

When the name of distributions coincides with one in `Distributions`, it is prefixed with `Alt`, eg `AltMvNormal`.

## Bibliography

- Lewandowski, Daniel, Dorota Kurowicka, and Harry Joe. "Generating random correlation matrices based on vines and extended onion method." Journal of multivariate analysis 100.9 (2009): 1989–2001.
