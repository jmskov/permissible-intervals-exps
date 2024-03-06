using Serialization
using SparseArrays
using LazySets
using UnicodePlots

learned_filename = joinpath(@__DIR__, "learned_2d_system_0.1/region_data-learned_2d_system_100-interval-12.bin")
known_filename = joinpath(@__DIR__, "simple_system_0.1/region_data_simple-system_100-interval-12.bin")

learn_res = deserialize(learned_filename)
known_res = deserialize(known_filename)

histogram([learn_res["Phigh"] - known_res["Phigh"]...])