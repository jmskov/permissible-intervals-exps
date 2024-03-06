using Revise

using Distributions
using Random
using TransitionIntervals
import TransitionIntervals.cdf

include(joinpath(@__DIR__, "../../shared", "gp_utilities.jl"))
include(joinpath(@__DIR__, "..", "conversion_utils.jl"))

# ARG 0. System Parameters
results_dir = "$(@__DIR__)/results/test"
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
process_distribution = Normal(0.0, σ_noise)

gps, image_fcn, sigma_fcn = regress_and_build(dataset, process_distribution, rkhs_flag=true) 

# RHKS Definitions
rkhs_error_dist = UniformRKHSError(
    -1.0, # sigma, set later
    info_gain_bound(gps[1]), # info_bound
    1.0, # f_sup
    sqrt(gps[1].kernel.ℓ2),
    -1.0, # norm_bound, calculated later
    gps[1].logNoise.value # log_noise
)

rkhs_copies = [deepcopy(rkhs_error_dist) for _ in 1:Threads.nthreads()]

# get the states and bound the uncertainties.
states = discretize(discretization)
function bound_sigmas(states, sigma_fcn)
    sigma_vec = Vector{Float64}(undef, length(states))
    Threads.@threads for i in 1:length(states)
        @views sigma_vec[i] = sigma_fcn(states[i].lower, states[i].upper) 
    end
    return sigma_vec
end
sigma_vec = bound_sigmas(states, sigma_fcn)

# create a dict we can actually use
function make_sigma_dict(states, sigma_vec)
    sigma_dict = Dict()
    for (i, state) in enumerate(states)
        sigma_dict[[state.lower, state.upper]] = sigma_vec[i]
    end
    return sigma_dict
end
sigma_dict = make_sigma_dict(states, sigma_vec)

function uncertainty_fcn(lower, upper; thread_idx=1)
    sigma = sigma_dict[[lower, upper]]
    dist = rkhs_copies[thread_idx]
    dist.sigma = sigma
    radius = sqrt(sum((upper - lower).^2))
    dist.norm_bound = RKHS_norm_bound(gps[1], dist.f_sup, radius)
    return dist
end

abstraction = transition_intervals(states, discretization, image_fcn, process_noise_dist, uncertainty_fcn)

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
