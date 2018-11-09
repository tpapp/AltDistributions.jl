# AltDistributions

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.org/tpapp/AltDistributions.jl.svg?branch=master)](https://travis-ci.org/tpapp/AltDistributions.jl)
[![codecov.io](http://codecov.io/github/tpapp/AltDistributions.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/AltDistributions.jl?branch=master)

This is a collection of probability distributions that I find useful. Eventually, they should be considered for contributing to [Distributions.jl](https://github.com/JuliaStats/Distributions.jl), but currently they reside in this package because

1. I am experimenting with the interface,
2. I am experimenting with the implementation (making it friendly to automatic differentiation),
3. not all functionality is implemented (eg only the `logpdf`).

When the name of distributions coincides with one in `Distributions`, it is prefixed with `Alt`, eg `AltMvNormal`.
