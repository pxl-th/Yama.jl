using BSON: @save, @load, load

using CuArrays
using Flux:
    Chain, Dense, relu, σ, softmax, logitcrossentropy, onehotbatch, flatten,
    params, gradient, gpu, cpu, loadparams!, trainmode!, testmode!
using Flux.Optimise: ADAM, update!
using Flux.Data.MNIST: images, labels
using Flux.Data: DataLoader

using Parameters: @with_kw
using Plots

using Yama: create_surface, SurfaceArgs

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
    batch_size::Int32 = 64
    save_dir::String = "./"
    use_gpu::Bool = true
end

function build_model()
    Chain(
        Dense(28 * 28, 64, σ),
        Dense(64, 10), softmax,
    )
end

function evaluation_step(model, batch, args::SurfaceArgs)
    x, y = make_minibatch(batch...)
    x = flatten(x)
    if args.use_gpu
        x = x |> gpu
        y = y |> gpu
    end
    logitcrossentropy(model(x), y)
end

function train(; kws...)
    args = Args(; kws...)

    model = build_model()
    save_file = joinpath(args.save_dir, "mlp-random.bson")
    directions_weights = params(model) .|> cpu
    @save save_file directions_weights

    if args.use_gpu
        model = model |> gpu
    end
    model_parameters = params(model)

    optimizer = ADAM(args.lr)
    train_loader = DataLoader(
        images(), labels(), batchsize=args.batch_size, shuffle=true,
    )
    val_loader = DataLoader(
        images(:test), labels(:test), batchsize=args.batch_size, shuffle=false,
    )

    for epoch in 1:args.epochs
        trainmode!(model)
        for (i, (data, labels)) in enumerate(train_loader)
            x, y = make_minibatch(data, labels)
            x = flatten(x)
            if args.use_gpu
                x = x |> gpu
                y = y |> gpu
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
            x = flatten(x)
            if args.use_gpu
                x = x |> gpu
                y = y |> gpu
            end
            validation_loss += logitcrossentropy(model(x), y) |> cpu
        end
        validation_loss = validation_loss / length(val_loader)
        println("Epoch $epoch, Validation loss $(validation_loss)")
    end

    model_parameters = model_parameters .|> cpu
    save_file = joinpath(args.save_dir, "mpl-mnist.bson")
    @save save_file model_parameters
end

function main()
    model = build_model()
    args = SurfaceArgs(x_directions_file="./mlp-random.bson")
    save_file = "./surface.bson"
    model_file = "./mpl-mnist.bson"
    batch_size = 64

    if isfile(model_file)
        @load model_file model_parameters
        loadparams!(model, model_parameters)
    end

    loader = DataLoader(images(:test), labels(:test), batchsize=batch_size)
    coordinates, loss_surface = create_surface(
        model, loader, evaluation_step, args,
    )

    @save save_file coordinates loss_surface
    # @load save_file coordinates loss_surface
    surface(coordinates..., loss_surface, linewidth=0, antialiased=false)
    gui()
end
