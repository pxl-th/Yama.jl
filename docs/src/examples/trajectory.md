## Create trajectory

Create trajectory between model checkpoints that were created during training.
This way we can visualize path that optimizer took.

Import necessary stuff.

```julia
using Yama: create_trajectory
using Plots

pyplot()
Plots.PyPlotBackend()
```

Load target model and create list of checkpoints.

```julia
checkpoints = readdir("./cnn-checkpoints", join=true)

target_checkpoint_file = checkpoints[end]
@load target_checkpoint_file checkpoint_weights
target_weights = params(checkpoint_weights)

checkpoints = checkpoints[1:end - 1] # Remove target model from checkpoints
```

Compute positions of the trajectory and plot.

```julia
positions = create_trajectory(target_weights, checkpoints)
plot(positions[:, 1], positions[:, 2])
gui()
```

In the [examples directory](https://github.com/pxl-th/Yama.jl/tree/master/examples)
you can find how to plot this trajectory onto the surface
that was created in the surface example.
