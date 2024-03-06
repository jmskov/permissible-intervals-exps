using Revise

using Distributions
using Random
using TransitionIntervals
import TransitionIntervals.cdf

include(joinpath(@__DIR__, "..", "gp_utilities.jl"))
include(joinpath(@__DIR__, "..", "conversion_utils.jl"))

# ARG 0. System Parameters
results_dir = "$(@__DIR__)/results/test-mew"
mkpath(results_dir)

""" Dynamics function for thermostat
"""

Random.seed!(11)
mt = MersenneTwister(11)
n = 200; 
input_range = [20.,23.]
x_train = [rand(mt, Uniform(input_range[1],input_range[2]), 1, n);   rand(mt, 1, n)];
σ_noise = 0.005  
obs_σ2_noise = σ_noise^2
logObsNoise = log10(obs_σ2_noise)

τ = 5.   
αe = 8.0e-3
αH = 3.6e-3
Te = 15.0
Th = 55.0
process_noise_dist = Normal(0.0, σ_noise)
f(x) = x[1] + τ*(αe*(Te-x[1]) + αH*(Th-x[1])*x[2]) + σ_noise*randn(mt) # standard normal dist. with var σ^2 = 0.01
y = zeros(1, n)
for (i, c) in enumerate(eachcol(x_train))
    y[i] = f(c)
end
dataset = (x_train, y)

state_delta = 0.5
control_delta = 0.5
full_state = [20.0 23.0; 0.0 1.0]
spacing = [state_delta, control_delta]
discretization = UniformDiscretization(DiscreteState([20.0, 0.0], [23.0, 1.0]), spacing)

# ARG 2. Specification File 
spec_filename = "/workspace/src/specifications/2_bounded_until.toml"
threshold = 0.9 # add this to the spec file
process_distribution = Normal(0.0, σ_noise)

gps, image_fcn, sigma_fcn = regress_and_build(dataset, process_distribution, rkhs_flag=true) 

# RHKS Definitions
mutable struct UniformRKHSError <: UniformError 
    sigma::Float64
    info_bound::Float64
    f_sup::Float64
    kernel_length::Float64
    norm_bound::Float64
    log_noise::Float64
end

function cdf(d::UniformRKHSError, x::Real)
    if x < 0.0
        return 0.0
    end
    R = exp(d.log_noise)
    frac = x/(d.sigma)
    dbound = exp(-0.5*(1/R*(frac - d.norm_bound))^2 + d.info_bound + 1.)
    return 1. - min(dbound, 1.) # is this min needed?
end

# gps are same, just do first
rkhs_error_dist = UniformRKHSError(
    -1.0, # sigma, set later
    info_gain_bound(gps[1]), # info_bound
    1.0, # f_sup
    sqrt(gps[1].kernel.ℓ2),
    -1.0, # norm_bound, calculated later
    gps[1].logNoise.value # log_noise
)

rkhs_copies = [deepcopy(rkhs_error_dist) for _ in 1:Threads.nthreads()]

sigma_reuse_dict = Dict()
function uncertainty_fcn(lower, upper; thread_idx=1)
    # get uncertainty bounds
    if haskey(sigma_reuse_dict, (lower, upper))
        sigma = sigma_reuse_dict[(lower, upper)]
    else 
        σ_bound = sigma_fcn(lower, upper)
        sigma = σ_bound[1]
        sigma_reuse_dict[(lower, upper)] = σ_bound[1] 
    end
    dist = rkhs_copies[thread_idx]
    dist.sigma = sigma
    radius = sqrt(sum((upper - lower).^2))
    dist.norm_bound = RKHS_norm_bound(gps[1], dist.f_sup, radius)
    # do other calculations here
    return dist
end

abstraction = transition_intervals(discretization, image_fcn, process_noise_dist, uncertainty_fcn)

# abstraction = abstract(problem, results_dir)

# here, post-process to remove redunant rows from the transition matrices
using SparseArrays
using DimensionalData

num_control_partitions = Int(1/control_delta) # todo: not manual
num_states = Int(size(abstraction.states, 1)/num_control_partitions)
Plows, Phighs = get_transition_matrices_subset(abstraction, num_states, num_control_partitions)

# now save as CDF in appropriate format
using LazySets
using Serialization

states = Vector{Hyperrectangle}(undef, num_states)
## convert states into Vector of Hyperrectangles
for i=1:num_states
    states[i] = Hyperrectangle(low=abstraction.states[i].lower[1:1], high=abstraction.states[i].upper[1:1])
end

# make sure the transition matrices are valid
for i=1:num_control_partitions
    @info "validating the $(i)th transition matrix"
    TransitionIntervals.validate_transition_matrices(Plows[i], Phighs[i])
end

@assert size(states,1) == size(Plows[1], 2) == size(Phighs[1], 2)

for i=1:num_control_partitions
    save_dir = joinpath(results_dir, "thermostat_test_$(control_delta)")
    mkpath(save_dir)
    filename = joinpath(save_dir, "region_data_thermostat-test_$(num_states)-interval-$i.bin")
    serialize(filename, Dict("states"=>states, "Plow"=>Plows[i], "Phigh"=>Phighs[i]))
end
