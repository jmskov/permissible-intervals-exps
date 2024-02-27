using Random

Random.seed!(11)
mt = MersenneTwister(11)
input_range = [20.,23.]
σ_noise = 0.005  

τ = 5.   
αe = 8.0e-3
αH = 3.6e-3
Te = 15.0
Th = 55.0
f(x) = x[1] + τ*(αe*(Te-x[1]) + αH*(Th-x[1])*x[2]) + σ_noise*randn(mt) 

state_step = 0.5
x_input = input_range[1]: state_step : input_range[2]
control_step = 0.1
control_input = 0:control_step:1

# create a grid of states and control inputs
states = [[x, u] for x in x_input for u in control_input]
outputs = [f(x) for x in states]

# create a vector field from these observations
using Plots

plt = Plots.plot(xlabel="Temp", ylabel="Control", dpi=300)
ep = 0.02
for (state, output) in zip(states, outputs)
    # draw a line between them
    Plots.plot!(plt, [state[1], output], [state[2], state[2]], color="black", lw=0.5, label="") 
    Plots.plot!(plt, [state[1], state[1]], [state[2]-ep, state[2]+ep], color="black", lw=0.5, label="")
end

display(plt)
savefig(plt, joinpath(@__DIR__, "thermostat.png"))
