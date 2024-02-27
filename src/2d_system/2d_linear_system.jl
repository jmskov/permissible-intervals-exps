# here, post-process to remove redunant rows from the transition matrices
using Distributions
using SparseArrays
using DimensionalData

using TransitionIntervals
using StochasticBarrierFunctions

include(joinpath(@__DIR__, "..", "conversion_utils.jl"))

# abstraction
function system_image(lower, upper; thread_idx=1)
    # x' = Ax + Bu where B = [1 1]
    A = [0.5 0; 0 0.5]

    c1 = A * lower[1:2]
    c2 = A * upper[1:2]
    c3 = A * [lower[1]; upper[2]]
    c4 = A * [upper[1]; lower[2]]

    res = zeros(2,2)
    u_lb = lower[3]
    u_ub = upper[3]
    for i=1:2
        res[i,1] = min(c1[i], c2[i], c3[i], c4[i]) + u_lb
        res[i,2] = max(c1[i], c2[i], c3[i], c4[i]) + u_ub
    end

    return res[:,1], res[:,2]
end

process_noise_dist = Normal(0.0, 0.01) 
control_delta = 0.1
state_delta = 0.1
discretization = UniformDiscretization(DiscreteState([0.0, 0.0, -1.0], [1.0, 1.0, 1.0]), [state_delta, state_delta, control_delta])
abstraction = transition_intervals(discretization, system_image, process_noise_dist)

# parse out the transition matrices
num_control_partitions = Int(2/control_delta) # todo: not manual
num_states = Int(size(abstraction.states, 1)/num_control_partitions)

Plows, Phighs = get_transition_matrices_subset(abstraction, num_states, num_control_partitions)

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

for i=1:num_control_partitions
    save_dir = joinpath(@__DIR__, "simple_system_$(control_delta)")
    mkpath(save_dir)
    filename = joinpath(save_dir, "region_data_simple-system_$(num_states)-interval-$i.bin")
    serialize(filename, Dict("states"=>states, "Plow"=>Plows[i], "Phigh"=>Phighs[i]))
end