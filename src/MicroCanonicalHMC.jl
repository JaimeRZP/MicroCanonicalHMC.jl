module MicroCanonicalHMC

export Settings, MCHMC, Sample
export Summarize
export TuringTarget, GaussianTarget, RosenbrockTarget, CustomTarget
export ParallelTarget

using LinearAlgebra, Statistics, Random, DataFrames
using DynamicPPL, LogDensityProblemsAD, LogDensityProblems, ForwardDiff
using AbstractMCMC, MCMCChains, MCMCDiagnosticTools, Distributed
using Distributions, DistributionsAD, ProgressMeter

abstract type Target <: AbstractMCMC.AbstractModel end

include("hamiltonian.jl")
include("targets.jl")
include("sampler.jl")
include("integrators.jl")
include("tuning.jl")
include("abstractmcmc.jl")

include("ensemble/targets.jl")
include("ensemble/sampler.jl")
include("ensemble/integrators.jl")
include("ensemble/tuning.jl")
include("ensemble/abstractmcmc.jl")

end
