"""
See the docstrings of exported symbols.
"""
module AltDistributions

export Fixed, AltMvNormal, LKJL, StdCorrFactor, AltMultinomial, AltBinomial,
    logpdf_C

using ArgCheck: @argcheck
using Base.Math: cbrt
using Distributions: Multinomial, Distributions
import Distributions: logpdf
using DocStringExtensions: SIGNATURES
using LinearAlgebra
using LinearAlgebra: checksquare, AbstractTriangular
using Parameters: @unpack
using Random: SamplerTrivial, Random, AbstractRNG
import Random: rand
using LogExpFunctions: xlogy
using SpecialFunctions: logabsbinomial, logfactorial

####
#### utilities
####

"""
Julia types accepted as a factor `L` of a covariance matrix `Σ=LL'`.
"""
const CovarianceFactor = Union{UniformScaling, AbstractMatrix}

"""
$(SIGNATURES)

Check that `μ` and `L::CovarianceFactor` have conforming dimensions (eg for `AltMvNormal`).

Used internally.
"""
function conforming_μL(μ::AbstractVector, L::AbstractMatrix)
    n = length(μ)
    size(L) == (n, n)
end

conforming_μL(μ::AbstractVector, ::UniformScaling) = true

struct StdCorrFactor{V <: AbstractVector, S <: CovarianceFactor, T} <: AbstractMatrix{T}
    σ::V
    F::S
    @doc """
    $(SIGNATURES)

    A factor `L` of a covariance matrix `Σ = LL'` given as `L = Diagonal(σ) * F`. Can be
    used in place of `L`, without performing the multiplication.
    """
    function StdCorrFactor(σ::V, F::S) where {V <: AbstractVector, S <: CovarianceFactor}
        T = typeof(one(eltype(F)) * one(eltype(σ)))
        @argcheck conforming_μL(σ, F)
        new{V,S,T}(σ, F)
    end
end

Base.:\(L::StdCorrFactor, y::Union{AbstractVector,AbstractMatrix}) = L.F \ (L.σ .\ y)

Base.size(L::StdCorrFactor) = (n = length(L.σ); (n, n))

Base.getindex(L::StdCorrFactor, I::Vararg{Int,2}) = getindex(Diagonal(L.σ) * L.F, I...) # just for printing

LinearAlgebra.logdet(L::StdCorrFactor) = sum(log, L.σ) + logdet(L.F)

"""
    Fixed(value)

Wrapper type to signal that `value` is "fixed" for the purposes of a log density
calculation. Formally,

```julia
logpdf(d, Fixed(v)) == logpdf(d, v) + C
```

where `C` is a constant term that only depends on `v`. In other words,

```julia
logpdf(distribution(θ), Fixed(v)) - logpdf(distribution(θ′), Fixed(v))
```

should always be correctly calculated. Use `get(Fixed(value)` to access the value.
"""
struct Fixed{T}
    value::T
end

get(f::Fixed) = f.value

####
#### AltMvNormal
####

Base.@kwdef struct AltMvNormal{M <: AbstractVector,T <: CovarianceFactor, S <: Real}
    "mean"
    μ::M
    "Cholesky factor, `L*L'` is the variance matrix. `L` can be *any* conformable matrix
    (or matrix-like object, eg UniformScaling), triangularity etc are not imposed."
    L::T
    "logdet(L), cached"
    logdet_L::S
    @doc """
    $(SIGNATURES)

    Inner constructor used internally, for specifying `L` directly when the first argument
    is `Val{:L}`.

    You **don't want to use this unless you obtain `L` directly**. Use a `Cholesky`
    factorization instead.
    """
    function AltMvNormal(::Val{:L}, μ::M, L::T) where {M <: AbstractVector,
                                                       T <: CovarianceFactor}
        @argcheck conforming_μL(μ, L) "Non-conformable mean and variance factor."
        logdet_L = logdet(L)
        new{M, T, typeof(logdet_L)}(μ, L, logdet_L)
    end
end

AltMvNormal(μ::AbstractVector, Σ::Cholesky) = AltMvNormal(Val{:L}(), μ, Σ.L)

"""
$(SIGNATURES)

Multivariate normal distribution with mean `μ` and covariance matrix `Σ`, which can be an
abstract matrix (eg a factorization) or `I`. If `Σ` is not symetric because of numerical
error, wrap in `LinearAlgebra.Symmetric`.

Use the `AltMvNormal(Val(:L), μ, L)` constructor for using `LL'=Σ` directly.

Also, see [`StdCorrFactor`](@ref) for formulating `L` from standard deviations and a
Cholesky factor of a *correlation* matrix:

```julia
AltMvNormal(Val(:L), μ, StdCorrFactor(σ, S))
```
"""
function AltMvNormal(μ::AbstractVector, Σ::AbstractMatrix)
    @argcheck issymmetric(Σ) "Σ is not symmetric."
    AltMvNormal(μ, cholesky(Σ))
end

AltMvNormal(μ::AbstractVector, Σ::Diagonal) = AltMvNormal(Val{:L}(), μ, Diagonal(.√diag(Σ)))

AltMvNormal(μ::AbstractVector, ::UniformScaling) = AltMvNormal(Val{:L}(), μ, I)

function logpdf(d::AltMvNormal, x::AbstractVector)
    @unpack μ, L, logdet_L = d
    -0.5*length(μ)*log(2*π) - logdet_L - 0.5*sum(abs2, L \ (x .- μ))
end

function rand(rng::AbstractRNG, sampler::SamplerTrivial{<:AltMvNormal})
    @unpack μ, L = sampler[]
    L * randn(rng, length(μ)) .+ μ
end

function Base.eltype(::Type{<:AltMvNormal{M,T}}) where {M,T}
    Vector{promote_type(eltype(M), eltype(T))}
end

####
#### LKJL
####

struct LKJL{T <: Real}
    η::T
    @doc """
        $(SIGNATURES)

    The LKJ distribution (Lewandowski et al 2009) for the Cholesky factor L of correlation
    matrices.

    A correlation matrix ``Ω=LL'`` has the density ``|Ω|^{η-1}``. However, it is usually not
    necessary to construct ``Ω``, so this distribution is formulated for the Cholesky
    decomposition `L*L'`, and takes `L` directly.

    Note that the methods **does not check if `L` yields a valid correlation matrix**.

    Valid values are ``η > 0``. When ``η > 1``, the distribution is unimodal at `Ω=I`, while
    ``0 < η < 1`` has a trough. ``η = 2`` is recommended as a vague prior.

    When ``η = 1``, the density is uniform in `Ω`, but not in `L`, because of the Jacobian
    correction of the transformation.
    """
    function LKJL(η::T) where T <: Real
        @argcheck η > 0
        new{T}(η)
    end
end

"""
$(SIGNATURES)

Calculate logdpf + constant for the LKJL distribution for a Cholesky factor of a correlation
matrix. Note that

1. `L` is not checked to be a valid Cholesky factor of a correlation matrix,

2. the constant is not calculated, to speed up computation --- the intended usage is MCMC
sampling.
"""
function logpdf_C(d::LKJL, L::Union{AbstractTriangular, Diagonal})
    @argcheck !Base.has_offset_axes(L)
    @unpack η = d
    n = size(L, 1)
    n ≤ 1 && return zero(eltype(L))
    a = 2*(η-1)
    sum(i -> log(L[i,i]) * (n - i + a), 2:n)
end

####
#### AltBinomial and AltMultinomial
####

struct AltMultinomial{T <: Integer, V <: AbstractVector{<:Real}}
    total_count::T
    partial_probabilities::V
    @doc """
    $(SIGNATURES)

    Multinomial distribution for the given `total_count`. The last probability should not be
    specified, as it is calculated as a residual. Small numerical error is tolerated,
    negative probabilities are not.
    """
    function AltMultinomial(total_count::T, partial_probabilities::V
                            ) where {T <: Integer, V <: AbstractVector{<:Real}}
        @argcheck all(partial_probabilities .≥ 0)
        new{T,V}(total_count, partial_probabilities)
    end
end

function logpdf(distribution::AltMultinomial, fixed_counts::Fixed)
    @unpack partial_probabilities = distribution
    counts = get(fixed_counts)
    @argcheck length(counts) == length(partial_probabilities) + 1
    P = eltype(partial_probabilities)
    total_p = zero(P)
    ℓ = xlogy(zero(eltype(counts)), zero(P))
    for (c, p) in zip(counts, partial_probabilities)
        ℓ += xlogy(c, p)
        total_p += p
    end
    rem_p = 1 - total_p
    if rem_p ≥ 0
        ℓ += xlogy(counts[end], rem_p)
    elseif rem_p ≤ -cbrt(eps(P))
        # otherwise we quietly ignore
        throw(DomainError(rem_p, "remainder probability is negative"))
    end
    ℓ
end

function logpdf(distribution::AltMultinomial, counts)
    @unpack total_count = distribution
    @argcheck total_count == sum(counts)
    ℓ = logfactorial(total_count)
    for c in counts
        ℓ -= logfactorial(c)
    end
    ℓ + logpdf(distribution, Fixed(counts))
end

function rand(rng::AbstractRNG, sampler::SamplerTrivial{<:AltMultinomial})
    distribution = sampler[]
    @unpack total_count, partial_probabilities = distribution
    x = Vector{Int}(undef, length(partial_probabilities) + 1)
    probabilities = vcat(partial_probabilities, 1 - sum(partial_probabilities))
    Distributions.multinom_rand!(rng, total_count, probabilities, x)
    x
end

struct AltBinomial{T <: Integer, P <: Real}
    total_count::T
    probability::P
    @doc """
    $(SIGNATURES)

    Binomial distribution for the given `total_count` and `probability`.
    """
    function AltBinomial(total_count::T, probability::P) where {T <: Integer, P <: Real}
        @argcheck 0 ≤ probability ≤ 1
        new{T, P}(total_count, probability)
    end
end

function logpdf(distribution::AltBinomial, fixed_count::Fixed{<:Integer})
    @unpack total_count, probability = distribution
    k = get(fixed_count)
    xlogy(k, probability) + xlogy(total_count - k, 1 - probability)
end

function logpdf(distribution::AltBinomial, count::Integer)
    first(logabsbinomial(distribution.total_count, count)) + logpdf(distribution, Fixed(count))
end

function rand(rng::AbstractRNG, sampler::SamplerTrivial{<:AltBinomial})
    @unpack total_count, probability = sampler[]
    # this is inefficient, but cheap to implement
    first(rand(rng, AltMultinomial(total_count, [probability])))
end

end # module
