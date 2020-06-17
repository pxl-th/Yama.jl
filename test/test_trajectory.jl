@testset "Flatten/unflatten" begin
    mlp_parameters = params(Chain(Dense(4, 16, σ), Dense(16, 4), softmax))
    mlp_parameters_count = mapreduce(length, +, mlp_parameters)

    flattened = mlp_parameters |> flatten
    @test length(flattened) == mlp_parameters_count

    unflattened = unflatten_set(flattened, mlp_parameters)
    for (p, u) in zip(mlp_parameters, unflattened)
        @test p == u
    end

    mlp_parameters_smol = params(Dense(4, 16, σ))
    @test_throws AssertionError unflatten_set(flattened, mlp_parameters_smol)

    conv_parameters = params(Chain(
        Conv((3, 3), 3=>16, pad=(1,1), relu), BatchNorm(16),
        Conv((3, 3), 16=>32, pad=(1,1), relu), BatchNorm(32),
    ))
    conv_parameters_count = mapreduce(length, +, conv_parameters)
    flattened = conv_parameters |> flatten
    @test length(flattened) == conv_parameters_count

    unflattened = unflatten_set(flattened, conv_parameters)
    for (p, u) in zip(conv_parameters, unflattened)
        @test p == u
    end
end
