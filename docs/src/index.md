```@meta
CurrentModule = Yama
```

# Yama 山

Visualize loss landscape.

```@raw html
<p align="center">
  <img src="https://raw.githubusercontent.com/pxl-th/Yama.jl/master/res/mnist.png" width=250>
  <img src="https://raw.githubusercontent.com/pxl-th/Yama.jl/master/res/mnist-log.png" width=250>
  <img src="https://raw.githubusercontent.com/pxl-th/Yama.jl/master/res/custom.png" width=250>
</p>
```

## MNIST Example

Import necessary stuff.

```julia
using Flux.Data.MNIST: images, labels
using Yama: create_surface, SurfaceArgs
using Plots

pyplot()
Plots.PyPlotBackend()
```

Define parameters used when creating surface.

```julia
args = SurfaceArgs(
    model_file="./mpl-mnist.bson",
    save_file="./surface.bson",
)
```

Create and load model if specified `model_file` in `SurfaceArgs`.

```julia
model = Chain(Dense(28 * 28, 10))
if !isa(args.model_file, Nothing)
    @load args.model_file model_parameters
    loadparams!(model, model_parameters)
end
```

Define function that computes loss on mini-batch.
**Note** that if you set `use_gpu` in `SurfaceArgs` to `true`,
then it is up to you to transfer mini-batch to gpu as done in this example.

```julia
function make_minibatch(data, labels)
    batch_size = length(data)
    batch = Array{Float32}(undef, size(data[1])..., 1, batch_size)
    @inbounds for i in 1:batch_size
        batch[:, :, :, i] = Float32.(data[i])
    end

    batch, Float32.(onehotbatch(labels, 0:9))
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
```

Compute, save and plot loss surface.

```julia
coordinates, loss_surface = create_surface(
    model, loader, evaluation_step, args,
)
if !isa(args.save_file, Nothing)
    @save args.save_file coordinates loss_surface
end

surface(coordinates..., loss_surface, linewidth=0, antialiased=false)
gui()
```

You can later load already computed surface, without the need to re-compute
from scratch.

```julia
@load args.save_file coordinates loss_surface
```
