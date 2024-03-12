# here, post-process to remove redunant rows from the transition matrices
using Distributions
using SparseArrays
using DimensionalData

using TransitionIntervals

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

noise_sigma = 0.01
process_noise_dist = Normal(0.0, noise_sigma) 
control_delta = 0.1
state_delta = 0.1
discretization = UniformDiscretization(DiscreteState([0.0, 0.0, 0.0], [1.0, 1.0, 0.5]), [state_delta, state_delta, control_delta])
abstraction = transition_intervals(discretization, system_image, process_noise_dist)

# verify the symmetry of the transition matrices
function find_state_idx(states, state_mean)
    for (i, state) in enumerate(states)
        foo = TransitionIntervals.mean(state)
        if state_mean ≈ foo
            return i
        end
    end
end

# now, assert that all possible targets are mirrored
verify_mirror_flag = false 
if verify_mirror_flag
    for (j, source_state) in enumerate(abstraction.states)
        state_mean = TransitionIntervals.mean(source_state)
        state_idx = j
        state_mirror = [state_mean[2]; state_mean[1]; state_mean[3]]
        mirror_idx = find_state_idx(abstraction.states, state_mirror)
        for (i, target_state) in enumerate(abstraction.states)
            target_state_mean = TransitionIntervals.mean(target_state)
            mirrored_target_state_mean = [target_state_mean[2]; target_state_mean[1]; target_state_mean[3]]
            mirror_target_idx = find_state_idx(abstraction.states, mirrored_target_state_mean) 
            @assert abstraction.Plow[i, state_idx] == abstraction.Plow[mirror_target_idx, mirror_idx]
            @assert all(abstraction.Phigh[i, state_idx] .≈ abstraction.Phigh[mirror_target_idx, mirror_idx])
        end
    end
end

# parse out the transition matrices
num_control_partitions = Int(0.5/control_delta) # todo: not manual
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

# todo: save data in a clean way
save_dir = joinpath(@__DIR__, "simple_system_$(control_delta)ctl_$(noise_sigma)std")
mkpath(save_dir)
for i=1:num_control_partitions
    filename = joinpath(save_dir, "region_data_simple-system_$(num_states)-interval-$i.bin")
    serialize(filename, Dict("states"=>states, "Plow"=>Plows[i], "Phigh"=>Phighs[i]))
end

serialize(joinpath(save_dir, "states_$(num_states).bin"), Dict("states"=>abstraction.states))