using MAT

beta_filename = joinpath(@__DIR__, "beta_dual196.mat")
indeces_filename = joinpath(@__DIR__, "indices196.mat")

beta = matread(beta_filename)["beta_dual"] # 5x100 matrix... what is the order of the cols? will it follow the order of the states?

indeces = matread(indeces_filename)["index_matrix"] # indeces is column of controls, with and the corresponding state index. these are the removed indeces

# create a new dict of state to admissible control indeces
admissible_controls = Dict{Int, Vector{Int}}
# add all the control indeces to the dict (1:5 for each state)
for i=1:100
    admissible_controls[i] =  collect(1:5)
end
# at the same time, create a dict version of the list
removed_controls = Dict{Int, Vector{Int}}()

num_zero = 0
for i=1:size(indeces, 1)
    global num_zero
    deleteat!(admissible_controls[indeces[i,2]], admissible_controls[indeces[i,2]] .== indeces[i,1])
    if isempty(admissible_controls[indeces[i,2]])
        num_zero += 1
    end

    if indeces[i,2] ∈ keys(removed_controls)
        push!(removed_controls[indeces[i,2]], indeces[i,1])
    else
        removed_controls[indeces[i,2]] = [indeces[i,1]]
    end
end
@assert num_zero == 0

zero_states = []
for (k, v) in admissible_controls
    if isempty(v)
        push!(zero_states, k)
    end
end

# now define the initial state range
# initial_state_range = [0.4:0.5, 0.4:0.5]

# get the max beta
beta_max = maximum(beta)
@info "The max beta is $beta_max"

N = 100
eta = 1e-6  # always manual?
P_safe_bound = eta + N*beta_max
@info "The P_safe_bound is $P_safe_bound"