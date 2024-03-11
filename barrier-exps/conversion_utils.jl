"""
    This parses a large NxN transition matrix that has been generated for a state-space control product space, and returns a set of transition matrices for each control interval.
"""
function get_transition_matrices_subset(abstraction, num_states, num_control_partitions)
    # get the transition matrices
    Plows = [] 
    Phighs = []
    L = size(abstraction.Plow, 1)
    size_new = div(L-1, num_control_partitions) + 1
    axlist = (Dim{:to}(1:size_new), Dim{:from}(1:size_new - 1))

    for i=1:num_control_partitions
        @info "partition: $i"
        Plow = spzeros(size_new, size_new-1)
        Phigh = spzeros(size_new, size_new-1)
        base_idx = 1+(i-1)*(size_new-1)
        source_idxs = base_idx:1:base_idx+size_new-2
        target_idxs = 1:1:size_new-1

        Plow[1:end-1, 1:end] = abstraction.Plow[target_idxs, source_idxs]
        Plow[end, 1:end] = abstraction.Plow[end, source_idxs]
        Phigh[1:end-1, 1:end] = abstraction.Phigh[target_idxs, source_idxs]
        Phigh[end, 1:end] = abstraction.Phigh[end, source_idxs]

        # n_states = Int((discretization.compact_space.upper[1] - discretization.compact_space.lower[1])/state_delta)*10
        start_idx = (i-1)*num_states + 1
        end_idx = i*num_states

        Pl_sub = abstraction.Plow[1:num_states, start_idx:end_idx] 
        @assert size(Pl_sub) == size(Plow[1:end-1, 1:end]) 
        @assert Plow[1:end-1, 1:end] == abstraction.Plow[1:num_states, start_idx:end_idx]

        Plow = DimArray(Plow, axlist)
        Phigh = DimArray(Phigh, axlist)

        push!(Plows, Plow)
        push!(Phighs, Phigh)
    end
    
    # get the subset of the transition matrices
    return Plows, Phighs 
end

"""
    This function saves the matrices and states in the appropriate format for the barrier function
"""
function convert_matrices_barriers(abstraction)
    # get the transition matrices
    Plows = [] 
    Phighs = []
    num_states = size(abstraction.Plow, 1)
    num_control_partitions = Int((size(abstraction.Plow, 2)-1)/(num_states-1))
    @info "num_control_partitions: $num_control_partitions"
    axlist = (Dim{:to}(1:num_states), Dim{:from}(1:num_states - 1))

    for i=1:num_control_partitions
        @info "partition: $i"

        Plow_sub = abstraction.Plow[:, 1+(i-1)*(num_states-1):i*(num_states-1)]
        Phigh_sub = abstraction.Phigh[:, 1+(i-1)*(num_states-1):i*(num_states-1)]

        Plow = DimArray(Plow_sub, axlist)
        Phigh = DimArray(Phigh_sub, axlist)

        push!(Plows, Plow)
        push!(Phighs, Phigh)
    end
    
    # get the subset of the transition matrices
    return Plows, Phighs 
end