```@meta
CurrentModule = Yama
```

# Yama 山

[GitHub](https://github.com/pxl-th/Yama.jl)

Visualizer loss surface in random directions (1st & 2nd picture)
or between model checkpoints (3rd picture).

```@raw html
<p align="center">
  <img src="https://raw.githubusercontent.com/pxl-th/Yama.jl/master/res/mnist.png" width=300>
  <img src="https://raw.githubusercontent.com/pxl-th/Yama.jl/master/res/mnist-log.png" width=300>
</p>
```

Visualize path that optimizer took during training (4th picture).

```@raw html
<p align="center">
  <img src="https://raw.githubusercontent.com/pxl-th/Yama.jl/master/res/mnist-two-checkpoints.png" width=300>
  <img src="https://raw.githubusercontent.com/pxl-th/Yama.jl/master/res/optimizer-path-cnn-mnist-log.png" width=300>
</p>
```

## What can you do with it

- Create surface in random directions around current model parameters.
- Create surface between model checkpoints in x- and y-directions.
- Create trajectory between model checkpoints. This way you can visualize
  path that optimizer took during training, for example.

!!! note
    All models that you provide via checkpoint files, should contain
    parameters under `checkpoint_weights` key.

    Saving weights like this in code would look like
    `BSON.@save file checkpoint_weights`.

Visit **Examples** section for short guides or
[examples directory](https://github.com/pxl-th/Yama.jl/tree/master/examples)
for self-sufficient examples.

## References

- [Hao Li et.al. "Visualizing the Loss Landscape of Neural Nets"](https://arxiv.org/abs/1712.09913)
- [Original code in Python](https://github.com/tomgoldstein/loss-landscape)
