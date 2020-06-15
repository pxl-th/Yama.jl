using BSON: @save, @load

using CuArrays
using Flux:
    Chain, Dense, relu, logitcrossentropy, onehotbatch, flatten,
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
    Chain(Dense(28 * 28, 10))
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
    if args.use_gpu
        model = model |> gpu
    end
    model_parameters = params(model)

    train_data = images()
    train_labels = labels()
    train_loader = DataLoader(
        train_data, train_labels, batchsize=args.batch_size, shuffle=true,
    )

    val_data = images(:test)
    val_labels = labels(:test)
    val_loader = DataLoader(
        val_data, val_labels, batchsize=args.batch_size, shuffle=false,
    )

    optimizer = ADAM(args.lr)

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
    args = SurfaceArgs(
        model_file="./mpl-mnist.bson",
        save_file="./surface.bson",
    )

    model = build_model()
    if !isa(args.model_file, Nothing)
        @load args.model_file model_parameters
        loadparams!(model, model_parameters)
    end

    data = images(:test)
    label = labels(:test)
    loader = DataLoader(data, label, batchsize=args.batch_size)

    coordinates, loss_surface = create_surface(
        model, loader, evaluation_step, args,
    )
    if !isa(args.save_file, Nothing)
        @save args.save_file coordinates loss_surface
    end

    # @load args.save_file coordinates loss_surface
    surface(coordinates..., loss_surface, linewidth=0, antialiased=false)
    gui()
end

main()
