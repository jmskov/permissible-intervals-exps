using Distributions
using MAT
using Serialization
using Plots
using Printf

using TransitionIntervals

#==
# todo: this should not be manual
START Manual Stuff
==#
indeces_filename = "/Users/john/Projects/PhD/code/GaussianProcessBarrier.jl/StochasticBarrierFunctions/synthesize/data/final/4d_linear/known/indices_ceg.mat"
states_filename = "/Users/john/Projects/PhD/code/projects/barriers-gps/barrier-exps/2d_system/simple_system_0.1_multi/states.bin" 

savedir = dirname(indeces_filename)
num_states = 100 # number of state-space regions, not the product space
title_tag = "Dual Batch"

# barrier stuff
N = 100     # horizon for Psafe
eta = 1e-6  # always manual?

# simulation stuff
init_state = [0.4 0.5; 0.4 0.5] # init region of barrier
# init_state = [0.0 1.0; 0.0 1.0] # init region of barrier
# the control intervals to choose from
all_intervals = [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]

control_intervals = []
for int2 in all_intervals
    for int1 in all_intervals
        push!(control_intervals, [int1, int2])
    end
end


num_controls = length(control_intervals)
N_sim = 1000
process_dist = Normal(0, 0.01)
f_true(x) = [0.5 0; 0 0.5]*x[1:2] + [1; 1]*x[3] + rand(process_dist, (2,1))
#==
END Manual Stuff
==#

# > by construction, the product space iterates over the state regions before the control regions
states = deserialize(states_filename)["states"][1:num_states]
# parse out the states from the control 
parsed_states = Vector{DiscreteState}(undef, num_states)
for i=1:num_states
    parsed_states[i] = DiscreteState(states[i].lower[1:2], states[i].upper[1:2])
end

indeces = matread(indeces_filename)["index_matrix"] # indeces is column of controls, with and the corresponding state index. these are the removed indeces
beta_max = matread(indeces_filename)["max_beta"]

# create a new dict of state to admissible control indeces
admissible_controls = Dict{Int, Vector{Int}}()
all_controls = Dict{Int, Vector{Int}}()
# add all the control indeces to the dict (1:5 for each state)
for i=1:num_states
    admissible_controls[i] =  collect(1:num_controls)
    all_controls[i] = collect(1:num_controls)
end
# at the same time, create a dict version of the list
removed_controls = Dict{Int, Vector{Int}}()

num_zero = 0
for i=1:size(indeces, 1)
    global num_zero
    deleteat!(admissible_controls[indeces[i,1]], admissible_controls[indeces[i,1]] .== indeces[i,2])
    if isempty(admissible_controls[indeces[i,1]])
        num_zero += 1
    end

    if indeces[i,1] ∈ keys(removed_controls)
        push!(removed_controls[indeces[i,1]], indeces[i,2])
    else
        removed_controls[indeces[i,1]] = [indeces[i,2]]
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
# todo: load barrier file and get max beta
@info "The max beta is $beta_max"

P_safe_bound = -1.
if title_tag == "IMDP"
    global P_safe_bound
    P_safe_bound = 1.0 - (1-beta_max)^N
else
    P_safe_bound = eta + N*beta_max
end
@info "The P_safe_bound is $P_safe_bound"

# now, simulate!
using Distributions
process_dist = Normal(0, 0.01)
f_true(x) = [0.5 0; 0 0.5]*x[1:2] + [x[3]; x[4]] + rand(process_dist, (2,1))

init_state = [0.4 0.5; 0.4 0.5]
init_state_ds = DiscreteState(init_state[:,1], init_state[:,2])
function get_init_state(range)
    new_state = zeros(size(range, 1), 1)
    for i=1:size(range, 1)
        dist = Uniform(range[i, 1], range[i,2])
        new_state[i] = rand(dist, 1)[1]
    end
    return new_state
end

function get_state_idx(states, state)
    for (i, s) in enumerate(states)
        if all([s.lower[i] <= state[i] <= s.upper[i] for i in eachindex(state)]) 
            return i
        end
    end
end

function get_discrete_control(admissible_dict, state_idx)
    return rand(admissible_dict[state_idx])
end

function sample_control(control_intervals, control_idx)

    control = zeros(2,1)
    for i=1:2
        control[i] = rand(Uniform(control_intervals[control_idx][i][1], control_intervals[control_idx][i][2]), 1)[1]
    end
    return control 
end

num_violate = 0
trajectories = []

for i=1:N_sim
    state = get_init_state(init_state)
    next_state_idx = get_state_idx(parsed_states, state)
    new_traj = zeros(2, N)
    new_traj[:,1] = state
    next_state = state
    for j=2:N
        control_idx1 = get_discrete_control(admissible_controls, next_state_idx)
        control1 = sample_control(control_intervals, control_idx1)
        next_state = f_true([next_state; control1])
        next_state_idx = get_state_idx(parsed_states, next_state)
        new_traj[:,j] = next_state
        if next_state_idx in zero_states || isnothing(next_state_idx)
            global num_violate
            @info "Violation at iteration $i"
            num_violate += 1
            break
        end
    end
    push!(trajectories, new_traj)
end

function adv_trajectory(init_state, admissible_controls, control_intervals, f_true, N, zero_states)
    state = get_init_state(init_state) 
    adv_traj = zeros(2, N)
    adv_traj[:,1] = state
    next_state = state
    next_state_idx = get_state_idx(parsed_states, state)
    num_good = 1
    for j=2:N
        control_idx = maximum(admissible_controls[next_state_idx]) 
        control = maximum(control_intervals[control_idx]) 
        next_state = f_true([next_state; control])
        next_state_idx = get_state_idx(parsed_states, next_state)
        adv_traj[:,j] = next_state
        num_good += 1
        if next_state_idx in zero_states || isnothing(next_state_idx)
            global num_violate
            @info "Violation at iteration $j"
            break
        end
    end
    return adv_traj[:, 1:num_good]
end

adv_traj_test = adv_trajectory(init_state, admissible_controls, control_intervals, f_true, N, zero_states)
adv_traj_nonadmissible = adv_trajectory(init_state, all_controls, control_intervals, f_true, N, zero_states)

# get formatted string to 5 decimal places for P_safe_bound
P_safe_bound_str = @sprintf("%.5f", P_safe_bound)
# plt = Plots.plot(title="$title_tag - $(N_sim) Trajectories (N=$N) - $(num_violate) Violations - Pviolate: $(P_safe_bound_str)", xlabel="x1", ylabel="x2", size=(800, 800), dpi=200) 
plt = Plots.plot(title="", xlabel="", ylabel="", size=(600, 600), dpi=200) 
full_state = DiscreteState([0.0, 0.0], [1.0, 1.0])
TransitionIntervals.plot!(plt, full_state, color=:green, alpha=0.05, linewidth=2, linecolor=:black, label="Full State")

for traj in trajectories
    Plots.plot!(plt, traj[1,:], traj[2,:], label="", alpha=0.1)
end

Plots.plot!(plt, adv_traj_nonadmissible[1,:], adv_traj_nonadmissible[2,:], label="Adv., non-admiss.", color=:black, linewidth=2, linestyle=:dash)
Plots.plot!(plt, adv_traj_test[1,:], adv_traj_test[2,:], label="Adv., adm.", color=:black, linewidth=2)

TransitionIntervals.plot!(plt, init_state_ds, color=:black, alpha=0.3, label="Initial State")
mkpath(savedir)
savefig(plt, "$savedir/trajectories.png")
serialize("$savedir/trajectories.bin", plt)

for key in keys(removed_controls)
    # TransitionIntervals.annotate!(plt, states[key], "$(length(removed_controls[key]))/$(length(control_intervals))", color=:blue, fontsize=5)
    TransitionIntervals.annotate!(plt, states[key], "$(length(removed_controls[key]))", color=:black, fontsize=12)
    TransitionIntervals.plot!(plt, states[key], color=:black, fillalpha=0.0, linewidth=2, label="")
end
Plots.plot!(plt, legend=:topleft)
savefig(plt, "$savedir/trajectories_annotate.png")
serialize("$savedir/trajectories_annotate.bin", plt)

# save the max beta as plain text
filename = joinpath(savedir, "psafe.txt")
open(filename, "w") do io
    # print max beta
    println(io, "beta_max: $(beta_max)")
    println(io, "P_safe_bound: $(P_safe_bound)")
end
