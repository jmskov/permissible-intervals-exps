using IntervalMDP
using TOML
using Colors

using TransitionIntervals

VERIFY_PLOT_COLORS = [
    colorant"#D6FAFF"
    colorant"#00AFF5"
    colorant"#D55672"
]

function verify_and_plot(abstraction, spec_filename; p_threshold=0.99)
    reach, avoid = terminal_states(spec_filename, abstraction.states)
    result = verify(abstraction, reach, avoid)
    sat_states = findall(result[1:end-1,3] .>= p_threshold)
    plt = plot(abstraction.states[sat_states]; fillcolor=VERIFY_PLOT_COLORS[2], label="", aspect_ratio=1, linewidth=0, fillalpha=0.7)
    unsat_states = findall(result[1:end-1,4] .< p_threshold)
    plot!(plt, abstraction.states[unsat_states]; fillcolor=VERIFY_PLOT_COLORS[3], label="", linewidth=0, fillalpha=0.7)
    amb_states = setdiff(1:length(result[:,1])-1, union(sat_states, unsat_states))
    plot!(plt, abstraction.states[amb_states]; fillcolor=VERIFY_PLOT_COLORS[1], label="", linewidth=0, fillalpha=0.5)
    display(plt)
    return plt, amb_states
end

function verify_and_plot_bounds(abstraction, spec_filename)
    reach, avoid = terminal_states(spec_filename, abstraction.states)
    result = verify(abstraction, reach, avoid)
    plt1 = plot(abstraction.states, result[1:end-1, 3]; fillcolor=:blue, label="", aspect_ratio=1, linewidth=0)
    plt2 = plot(abstraction.states, result[1:end-1, 4]; fillcolor=:blue, label="", aspect_ratio=1, linewidth=0)
    plt = plot(plt1, plt2, layout=(1,2))
    display(plt)
    return plt 
end

function plot_verification(abstraction, result; p_threshold=0.99)
    sat_states = findall(result[1:end-1,3] .>= p_threshold)
    plt = plot(abstraction.states[sat_states]; fillcolor=VERIFY_PLOT_COLORS[2], label="", aspect_ratio=1, linewidth=0, fillalpha=0.7)
    unsat_states = findall(result[1:end-1,4] .< p_threshold)
    plot!(plt, abstraction.states[unsat_states]; fillcolor=VERIFY_PLOT_COLORS[3], label="", linewidth=0, fillalpha=0.7)
    amb_states = setdiff(1:length(result[:,1])-1, union(sat_states, unsat_states))
    plot!(plt, abstraction.states[amb_states]; fillcolor=VERIFY_PLOT_COLORS[1], label="", linewidth=0, fillalpha=0.5)
    display(plt)
    return plt
end

# outer function for IMC verification
function verify(abstraction::TransitionIntervals.Abstraction, terminal_states::Vector{Int}, avoid_set::Vector{Int}, filename::String = ""; tolerance=1e-6)
    
    prob = IntervalProbabilities(;
        lower = abstraction.Plow,
        upper = abstraction.Phigh,
    )
    mc = IntervalMarkovChain(prob, [-1])
    prop = InfiniteTimeReachAvoid(terminal_states, avoid_set, tolerance) 
    spec_low = Specification(prop, Pessimistic, Minimize)
    problem_low = Problem(mc, spec_low)
    Vlow, k, residual = value_iteration(problem_low)
    spec_high = Specification(prop, Optimistic, Maximize)
    problem_high = Problem(mc, spec_high)
    Vupp, k, residual = value_iteration(problem_high)

    # create the result matrix that I so depend on
    result_matrix = zeros(length(Vlow), 4)
    # todo: actions
    result_matrix[:,1] = collect(1:length(Vlow))
    result_matrix[:,3] = Vlow
    result_matrix[:,4] = Vupp

    if length(filename) > 0
        serialize(filename, result_matrix) 
    end

    return result_matrix
end

function verify(abstraction::TransitionIntervals.Abstraction, spec_filename::String; tolerance=1e-6)
    reach, avoid = terminal_states(spec_filename, abstraction.states)
    result = verify(abstraction, reach, avoid; tolerance=tolerance)
    return result 
end

function get_1_steppers(result_matrix, P_high, threshold)
    # get satisfying states
    sat_states = findall(result_matrix[1:end-1,3] .>= threshold)
    # get unsat states
    unsat_states = findall(result_matrix[1:end-1,4] .< threshold)
    # get the indeterminate states
    amb_states = setdiff(1:length(result_matrix[:,1])-1, union(sat_states, unsat_states))
    # find all amb states that possibly transition to sat states

    targets = Vector{Int}()
    for amb in amb_states
        # get the states that the amb state transitions to
        if !isempty(findall(P_high[sat_states, amb] .> 0.0))
            push!(targets, amb)
        end 
    end
    return targets 
end

function classify_results(result_matrix, threshold)
    # get satisfying states
    sat_states = findall(result_matrix[1:end-1,3] .>= threshold)
    # get unsat states
    unsat_states = findall(result_matrix[1:end-1,4] .< threshold)
    # get the indeterminate states
    amb_states = setdiff(1:length(result_matrix[:,1])-1, union(sat_states, unsat_states))
    state_classifications = zeros(Int, length(result_matrix[:,1])-1)
    state_classifications[sat_states] .= 1
    state_classifications[unsat_states] .= -1
    return state_classifications
end

function get_reachability_labels(state_label_fcn, state_means, target_label, unsafe_label)
    terminal_states = Vector{Int64}()
    unsafe_states = Vector{Int64}()

    for (i, pt) in enumerate(state_means)
        if state_label_fcn(pt) == target_label 
            push!(terminal_states, i)
        end
        if state_label_fcn(pt) == unsafe_label 
            push!(unsafe_states, i)
        end
    end
    return terminal_states, unsafe_states
end

function terminal_states(spec_file::String, states::Vector{DiscreteState})
    # get the terminal states
    state_label_fcn, _, _, target_label, unsafe_label, _, _ = load_PCTL_specification(spec_file) 
    state_means = TransitionIntervals.mean.(states)
    terminal_state_idxs, unsafe_state_idxs = get_reachability_labels(state_label_fcn, state_means, target_label, unsafe_label)
    return terminal_state_idxs, unsafe_state_idxs
end

#==
# TODO: The following are remnants of code to load a PCTL specification from a TOML file. This should be generalized somewhere, someday.
==#

"""
    is_point_in_rectangle

Checks whether the given point is inside the hyperrectangle.
"""
function is_point_in_rectangle(pt, rectangle)
	dim = length(pt)
	# Format of rectange: [lowers, uppers]
	res = true
	for i=1:dim
		res = rectangle[i] <= pt[i] <= rectangle[i+dim]
		if !res 
			return false
		end
	end
	return true
end

"""
    general_label_fcn

Function prototype to label IMDP states.
"""
function general_label_fcn(point, default_label::String, unsafe_label::String, labels_dict::Dict; unsafe=false, unsafe_default=false)
    if unsafe 
        # Hacky workaround
        if unsafe_default
            return default_label
        end
        return unsafe_label 
    end
    state_label = default_label
    for label in keys(labels_dict) 
        for region in labels_dict[label]
            if is_point_in_rectangle(point, region) 
                state_label = label
                break
            end
        end
    end
    return state_label
end

"""
    load_PCTL_specification

Load a PCTL specification from a TOML file.
"""
function load_PCTL_specification(spec_filename::String)
    f = open(spec_filename)
    spec_data = TOML.parse(f)
    close(f)

    ϕ1 = spec_data["phi1"] == false ? nothing : spec_data["phi1"] 
    ϕ2 = spec_data["phi2"]
    default_label = spec_data["default"]
    unsafe_label = spec_data["unsafe"]

    labels_dict = Dict(ϕ1 => [], ϕ2 => [], unsafe_label => [])
    dims = spec_data["dims"]
    for geometry in spec_data["labels"]["phi1"]
        push!(labels_dict[ϕ1], geometry)
    end

    for geometry in spec_data["labels"]["phi2"]
        push!(labels_dict[ϕ2], geometry)
    end

    for geometry in spec_data["labels"]["unsafe"]
        push!(labels_dict[unsafe_label], geometry)
    end

    unsafe_default_flag = spec_data["default_outside_compact"]

    lbl_fcn = (point; unsafe=false) -> general_label_fcn(point, default_label, unsafe_label, labels_dict, unsafe=unsafe, unsafe_default=unsafe_default_flag)
    return lbl_fcn, labels_dict, ϕ1, ϕ2, unsafe_label, spec_data["steps"], spec_data["name"]
end

function get_state_labels_plotting(spec_filename::String)
    f = open(spec_filename)
    spec_data = TOML.parse(f)
    close(f)

    labels_map = spec_data["labels"]["map"]
    labels_dict = Dict(labels_map[1] => [], labels_map[2]  => [], labels_map[3]  => [])
    dims = spec_data["dims"]
    for geometry in spec_data["labels"]["phi1"]
        push!(labels_dict[labels_map[1]], reshape(geometry, dims, 2))
    end

    for geometry in spec_data["labels"]["phi2"]
        push!(labels_dict[labels_map[2]], reshape(geometry, dims, 2))
    end

    for geometry in spec_data["labels"]["unsafe"]
        push!(labels_dict[labels_map[3]], reshape(geometry, dims, 2))
    end

    return labels_dict
end