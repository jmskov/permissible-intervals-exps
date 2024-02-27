# here, post-process to remove redunant rows from the transition matrices
using Distributions
using SparseArrays
using DimensionalData

using TransitionIntervals
using StochasticBarrierFunctions

include(joinpath(@__DIR__, "..", "conversion_utils.jl"))

# abstraction
linear_system_mat = [0.5 1.0]
process_noise_dist = Normal(0.0, 0.01) 
control_delta = 0.1
state_delta = 0.1
discretization = UniformDiscretization(DiscreteState([0.0, 0.0], [1.0, 1.0]), [state_delta, control_delta])
abstraction = transition_intervals(discretization, linear_system_mat, process_noise_dist)

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
    save_dir = joinpath(@__DIR__, "simple_system_$(control_delta)")
    mkpath(save_dir)
    filename = joinpath(save_dir, "region_data_simple-system_$(num_states)-interval-$i.bin")
    serialize(filename, Dict("states"=>states, "Plow"=>Plows[i], "Phigh"=>Phighs[i]))
end