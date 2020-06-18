using BSON: @save, @load

using CuArrays
using Flux:
    Chain, Dense, relu, σ, softmax, logitcrossentropy, onehotbatch, flatten,
    params, gradient, gpu, cpu, loadparams!, trainmode!, testmode!,
    Conv, MaxPool
using Flux.Optimise: ADAM, update!
using Flux.Data.MNIST: images, labels
using Flux.Data: DataLoader

using Printf: @sprintf
using Parameters: @with_kw
using Plots

using Yama: create_surface, SurfaceArgs, create_trajectory

pyplot()
Plots.PyPlotBackend()

function make_minibatch(data, labels)
    batch_size = length(data)
    batch = Array{Float32}(undef, size(data[1])..., 1, batch_size)
    @inbounds for i in 1:batch_size
        batch[:, :, :, i] = Float32.(data[i])
    end

    batch, Float32.(onehotbatch(labels, 0:9))
end

@with_kw struct Args
    lr::Float32 = 3e-4
    epochs::Int32 = 20
    batch_size::Int32 = 32
    save_dir::String = "./cnn-checkpoints"
    use_gpu::Bool = true
end

function build_model(imgsize = (28, 28, 1))
    Chain(
        Conv((3, 3), 1 => 16, pad=(1, 1), relu), MaxPool((2, 2)),
        Conv((3, 3), 16 => 32, pad=(1, 1), relu), MaxPool((2, 2)),
        Conv((3, 3), 32 => 32, pad=(1, 1), relu), MaxPool((2, 2)),
        flatten,
        Dense(3 * 3 * 32, 10),
    )
end

function evaluation_step(model, batch, args::SurfaceArgs)
    x, y = make_minibatch(batch...)
    if args.use_gpu
        x, y = x |> gpu, y |> gpu
    end
    logitcrossentropy(model(x), y)
end

function save_model(model, path::String)
    checkpoint_weights = params(model) .|> cpu
    @save path checkpoint_weights
end

function train(; kws...)
    args = Args(; kws...)
    mkpath(args.save_dir)

    model = build_model()
    if args.use_gpu
        model = model |> gpu
    end
    model_parameters = params(model)

    optimizer = ADAM(args.lr)
    train_loader = DataLoader(
        images(), labels(), batchsize=args.batch_size, shuffle=false,
    )
    val_loader = DataLoader(
        images(:test), labels(:test), batchsize=args.batch_size,
    )

    save_model(model, joinpath(args.save_dir, "checkpoint-epoch=00.bson"))

    for epoch in 1:args.epochs
        trainmode!(model)
        for (i, (data, labels)) in enumerate(train_loader)
            x, y = make_minibatch(data, labels)
            if args.use_gpu
                x, y = x |> gpu, y |> gpu
            end
            gradients = gradient(model_parameters) do
                logitcrossentropy(model(x), y)
            end
            update!(optimizer, model_parameters, gradients)
        end

        testmode!(model)
        validation_loss = 0.0f0
        for (i, (data, labels)) in enumerate(val_loader)
            x, y = make_minibatch(data, labels)
            if args.use_gpu
                x, y = x |> gpu, y |> gpu
            end
            validation_loss += logitcrossentropy(model(x), y) |> cpu
        end
        validation_loss = validation_loss / length(val_loader)
        println("Epoch $epoch, Validation loss $(validation_loss)")

        save_model(
            model,
            joinpath(args.save_dir, "checkpoint-epoch=$(@sprintf("%02d", epoch)).bson"),
        )
    end
end

function main_surface()
    model = build_model()
    args = SurfaceArgs(
        batch_size=32,
        x_directions_file="./cnn-checkpoints/checkpoint-epoch=00.bson",
    )
    save_file = "./cnn-surface.bson"
    model_file = "./cnn-checkpoints/checkpoint-epoch=10.bson"

    @load model_file checkpoint_weights
    loadparams!(model, checkpoint_weights)

    # loader = DataLoader(images(:test), labels(:test), batchsize=batch_size)
    loader = DataLoader(images(), labels(), batchsize=args.batch_size)
    coordinates, loss_surface = create_surface(
        model, loader, evaluation_step, args,
    )

    @save save_file coordinates loss_surface
    # @load save_file coordinates loss_surface
    surface(coordinates..., loss_surface, linewidth=0, antialiased=false)
    gui()
end

function get_losses(position, coordinates, loss_surface)::Array{Float32, 1}
    x_coordinates, y_coordinates = coordinates
    x_nearest = argmin(abs.(x_coordinates .- position[1]))
    y_nearest = argmin(abs.(y_coordinates .- position[2]))
    [x_coordinates[x_nearest], y_coordinates[y_nearest], loss_surface[x_nearest, y_nearest]]
end

function main_trajectory()
    checkpoints = readdir("./cnn-checkpoints", join=true)
    checkpoint_file = checkpoints[end]
    checkpoints = checkpoints[1:end - 1]
    @load checkpoint_file checkpoint_weights
    target_weights = params(checkpoint_weights)

    positions = create_trajectory(target_weights, checkpoints)[2:end, :]
    positions ./= maximum(abs.(positions), dims=1)

    surface_file = "./cnn-surface.bson"
    @load surface_file coordinates loss_surface

    positions_losses = Array{Float32, 2}(undef, 3, size(positions, 1))
    for i in 1:size(positions, 1)
        positions_losses[:, i] = get_losses(
            @view(positions[i, :]), coordinates, loss_surface,
        )
    end

    surface(coordinates..., loss_surface, linewidth=0, antialiased=false)
    plot!(
        positions_losses[2, :], positions_losses[1, :], positions_losses[3, :],
        w=2, label="Optimizer path", color="white", linestyle=:dashdot,
    )
    gui()

    # contourf(coordinates..., loss_surface)
    # plot!(positions[:, 2], positions[:, 1], color="white", linestyle=:dashdot, w=2)
    # gui()
end
