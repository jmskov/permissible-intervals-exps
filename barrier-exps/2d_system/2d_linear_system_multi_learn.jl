# here, post-process to remove redunant rows from the transition matrices
using TransitionIntervals
import TransitionIntervals.cdf

using Distributions
using SparseArrays
using DimensionalData

# todo: these includes are not ideal
include(joinpath(@__DIR__, "..", "conversion_utils.jl"))
include(joinpath(@__DIR__, "../../shared", "gp_utilities.jl"))

noise_sigma = 0.01
process_noise_dist = Normal(0.0, noise_sigma) 
control_delta = 0.1
state_delta = 0.1
discretization = UniformDiscretization(DiscreteState([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.5, 0.5]), [state_delta, state_delta, control_delta, control_delta])

# add GP stuff here
f_parse(x) = [0.5 0; 0 0.5]*x[1:2] + x[3:4]
N_data = 3000
data_range = [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.5, 0.5]]
@info "Generating dataset with $N_data data points"
dataset = DataSandbox.sample_function(f_parse, data_range, N_data, process_noise_dist=process_noise_dist) 
delta_flag = false 
rkhs_flag = true
kernel_ls = 0.8
gps, image_fcn, sigma_fcn = regress_and_build(dataset, process_noise_dist, rkhs_flag=rkhs_flag, delta_flag=delta_flag, kernel_ls=kernel_ls) 

# gps are same, just do first
rkhs_error_dist = UniformRKHSError(
    -1.0, # sigma, set later
    info_gain_bound(gps[1]), # info_bound
    # 0.1,
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

# parse out the transition matrices
# num_control_partitions = Int(0.5/control_delta) # todo: not manual
num_control_partitions = Int((size(abstraction.Plow, 2)-1)/(size(abstraction.Plow, 1)-1))
num_states = Int(size(abstraction.states, 1)/num_control_partitions)
Plows, Phighs = convert_matrices_barriers(abstraction)

# now save states and matrices in appropriate format
using LazySets
using Serialization

states = Vector{Hyperrectangle}(undef, num_states)
## convert states into Vector of Hyperrectangles
for i=1:num_states
    states[i] = Hyperrectangle(low=abstraction.states[i].lower[1:2], high=abstraction.states[i].upper[1:2])
end

# make sure the transition matrices are valid
for i=1:num_control_partitions
    @info "validating the $(i)th transition matrix"
    TransitionIntervals.validate_transition_matrices(Plows[i], Phighs[i])
end

@assert size(states,1) == size(Plows[1], 2) == size(Phighs[1], 2)

save_dir = joinpath(@__DIR__, "multi_system_learned_$(control_delta)ctl_$(noise_sigma)std_$N_data-data")
mkpath(save_dir)
for i=1:num_control_partitions
    filename = joinpath(save_dir, "region_data-learned_2d_system_$(num_states)-interval-$i.bin")
    serialize(filename, Dict("states"=>states, "Plow"=>Plows[i], "Phigh"=>Phighs[i]))
end

serialize(joinpath(save_dir, "states_$(num_states).bin"), Dict("states"=>abstraction.states))