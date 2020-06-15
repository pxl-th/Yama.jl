@with_kw struct SurfaceArgs
    xmin::Float32 = -1
    xmax::Float32 = 1
    xnum::Int32 = 20

    ymin::Float32 = -1
    ymax::Float32 = 1
    ynum::Int32 = 20

    batch_size::Int32 = 64
    use_gpu::Bool = true

    model_file::String
    save_file::String
end

function evaluation_loop(
    model, dataloader::DataLoader, evaluation_step, args::SurfaceArgs,
)::Float32
    total = Float32(length(dataloader) * dataloader.batchsize)
    total_loss = 0.0f0
    for batch in dataloader
        batch_loss = evaluation_step(model, batch, args) * dataloader.batchsize
        total_loss += batch_loss |> cpu
    end
    total_loss / total
end

function create_surface(
    model, dataloader::DataLoader, evaluation_step, args::SurfaceArgs,
)
    model_parameters = params(model)
    if args.use_gpu
        model = model |> gpu
    end
    testmode!(model)

    x_coordinates = range(args.xmin, stop=args.xmax, length=args.xnum) |> collect
    y_coordinates = range(args.ymin, stop=args.ymax, length=args.ynum) |> collect
    directions = (
        create_random_directions(model_parameters),
        create_random_directions(model_parameters),
    )

    loss_surface = fill(-1.0f0, args.xnum, args.ynum)
    it = product(enumerate(y_coordinates), enumerate(x_coordinates))
    @showprogress 1 "Creating surface " for ((j, y), (i, x)) in it
        shifted = shift_weights(model_parameters, directions, (x, y))
        loadparams!(model, shifted)

        eval_loss = evaluation_loop(model, dataloader, evaluation_step, args)
        loss_surface[i, j] = eval_loss
    end

    (x_coordinates, y_coordinates), loss_surface
end
