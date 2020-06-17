using Flux: Chain, Dense, σ, softmax, params, Conv, BatchNorm, relu
using Test
using Yama: flatten, unflatten_set

@testset "Yama.jl" begin
    @testset "Trajectory" begin
        include("test_trajectory.jl")
    end
end
