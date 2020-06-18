"""
Arguments for configuring loss surface computation.

# Parameters
- `xmin::Float32`, `xmax::Float32`, `xnum::Int32`: define span of the surface
and amount of point in the `x` direction.
- `ymin::Float32`, `ymax::Float32`, `ynum::Int32`: define span of the surface
and amount of point in the `y` direction.
- `use_gpu::Bool`: Whether to use gpu. If `true` then it is up to you to
transfer mini-batch in `evaluation_step` function to the gpu.
- `x_directions_file::Union{Nothing, String}`: If provided, directions
for `x` axis will be loaded from it. Otherwise, random initialized.
Should match in shape with model's weights.
- `y_directions_file::Union{Nothing, String}`: If provided, directions
for `y` axis will be loaded from it. Otherwise, random initialized.
Should match in shape with model's weights.

!!! note
    If `use_gpu = true` then it is up to you to transfer
    mini-batch in `evaluation_step` function to the gpu.

!!! note
    Both x- and y- directions files in should contain directions
    under `checkpoint_weights` key.

    Saving weights like this in code would look like
    `BSON.@save file checkpoint_weights`.
"""
@with_kw struct SurfaceArgs
    xmin::Float32 = -1
    xmax::Float32 = 1
    xnum::Int32 = 20

    ymin::Float32 = -1
    ymax::Float32 = 1
    ynum::Int32 = 20

    batch_size::Int32 = 64
    use_gpu::Bool = true

    x_directions_file::Union{Nothing, String} = nothing
    y_directions_file::Union{Nothing, String} = nothing
end

function evaluation_loop(
    model, dataloader::DataLoader, evaluation_step::Function, args::SurfaceArgs,
)::Float32
    total = Float32(length(dataloader) * dataloader.batchsize)
    total_loss = 0.0f0
    for batch in dataloader
        batch_loss = evaluation_step(model, batch, args) * dataloader.batchsize
        total_loss += batch_loss |> cpu
    end
    total_loss / total
end

function get_directions(
    model_parameters::Params, args::SurfaceArgs,
)::Tuple{Params, Params}
    directions = Array{Params}(undef, 2)

    directions_files = (args.x_directions_file, args.y_directions_file)
    is_targets = [isa(df, String) && isfile(df) for df in directions_files]

    @inbounds for (i, (is_target, target_file)) in enumerate(zip(is_targets, directions_files))
        if is_target
            @load target_file checkpoint_weights
            i_directions = create_directions(model_parameters, params(checkpoint_weights))
        else
            i_directions = create_directions(model_parameters)
        end
        directions[i] = i_directions
    end
    tuple(directions...)
end

"""
```julia
function create_surface(
    model, dataloader::DataLoader, evaluation_step::Function, args::SurfaceArgs,
)
```

Create loss surface.

# Parameters
- `model`: Model to use in loss function.
- `dataloader::DataLoader`: Dataset on which to evaluate loss function.
- `evaluation_step`:
    Custom-defined function which given model, mini-batch and args,
    computes loss on that mini-batch.
- `args::SurfaceArgs`: Parameters used when computing surface.

!!! note
    If you specified `use_gpu` in `args`, then it is up to you,
    to transfer mini-batch in `evaluation_step` function to gpu.
"""
function create_surface(
    model, dataloader::DataLoader, evaluation_step::Function, args::SurfaceArgs,
)
    model_parameters = params(model)
    directions = get_directions(model_parameters, args)

    if args.use_gpu
        model = model |> gpu
    end
    testmode!(model)

    x_coordinates = range(args.xmin, stop=args.xmax, length=args.xnum) |> collect
    y_coordinates = range(args.ymin, stop=args.ymax, length=args.ynum) |> collect

    loss_surface = fill(-1.0f0, args.xnum, args.ynum)
    it = product(enumerate(y_coordinates), enumerate(x_coordinates))
    @showprogress 2 "Creating surface " for ((j, y), (i, x)) in it
        shifted = shift_weights(model_parameters, directions, (x, y))
        loadparams!(model, shifted)

        eval_loss = evaluation_loop(model, dataloader, evaluation_step, args)
        loss_surface[i, j] = eval_loss
    end

    (x_coordinates, y_coordinates), loss_surface
end
