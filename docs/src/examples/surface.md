## Create surface

Create surface of a model using MNIST dataset.

Import necessary stuff.

```julia
using BSON: @save, @load

using CuArrays
using Flux: Chain, Dense, logitcrossentropy, onehotbatch, params, gpu, loadparams!
using Flux.Data.MNIST: images, labels
using Yama: create_surface, SurfaceArgs
using Plots

pyplot()
Plots.PyPlotBackend()
```

Define parameters used when creating surface.

```julia
args = SurfaceArgs(use_gpu=true)
```

Create and load model.

```julia
model = Chain(Dense(28 * 28, 10))
@load model_file checkpoint_weights
loadparams!(model, checkpoint_weights)
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
    if args.use_gpu
        x, y = x |> gpu, y |> gpu
    end
    logitcrossentropy(model(x), y)
end
```

Compute, save and plot loss surface.

```julia
loader = DataLoader(images(), labels(), batchsize=args.batch_size)
coordinates, loss_surface = create_surface(
    model, loader, evaluation_step, args,
)
@save args.save_file coordinates loss_surface
surface(coordinates..., loss_surface)
gui()
```

You can later load already computed surface, without the need to re-compute
from scratch.

```julia
@load args.save_file coordinates loss_surface
```
