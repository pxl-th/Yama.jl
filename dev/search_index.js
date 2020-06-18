var documenterSearchIndex = {"docs":
[{"location":"examples/surface/#Create-surface-1","page":"Surface","title":"Create surface","text":"","category":"section"},{"location":"examples/surface/#","page":"Surface","title":"Surface","text":"Create surface of a model using MNIST dataset.","category":"page"},{"location":"examples/surface/#","page":"Surface","title":"Surface","text":"Import necessary stuff.","category":"page"},{"location":"examples/surface/#","page":"Surface","title":"Surface","text":"using BSON: @save, @load\n\nusing CuArrays\nusing Flux: Chain, Dense, logitcrossentropy, onehotbatch, params, gpu, loadparams!\nusing Flux.Data.MNIST: images, labels\nusing Yama: create_surface, SurfaceArgs\nusing Plots\n\npyplot()\nPlots.PyPlotBackend()","category":"page"},{"location":"examples/surface/#","page":"Surface","title":"Surface","text":"Define parameters used when creating surface.","category":"page"},{"location":"examples/surface/#","page":"Surface","title":"Surface","text":"args = SurfaceArgs(use_gpu=true)","category":"page"},{"location":"examples/surface/#","page":"Surface","title":"Surface","text":"Create and load model.","category":"page"},{"location":"examples/surface/#","page":"Surface","title":"Surface","text":"model = Chain(Dense(28 * 28, 10))\n@load model_file checkpoint_weights\nloadparams!(model, checkpoint_weights)","category":"page"},{"location":"examples/surface/#","page":"Surface","title":"Surface","text":"Define function that computes loss on mini-batch. Note that if you set use_gpu in SurfaceArgs to true, then it is up to you to transfer mini-batch to gpu as done in this example.","category":"page"},{"location":"examples/surface/#","page":"Surface","title":"Surface","text":"function make_minibatch(data, labels)\n    batch_size = length(data)\n    batch = Array{Float32}(undef, size(data[1])..., 1, batch_size)\n    @inbounds for i in 1:batch_size\n        batch[:, :, :, i] = Float32.(data[i])\n    end\n    batch, Float32.(onehotbatch(labels, 0:9))\nend\n\nfunction evaluation_step(model, batch, args::SurfaceArgs)\n    x, y = make_minibatch(batch...)\n    if args.use_gpu\n        x, y = x |> gpu, y |> gpu\n    end\n    logitcrossentropy(model(x), y)\nend","category":"page"},{"location":"examples/surface/#","page":"Surface","title":"Surface","text":"Compute, save and plot loss surface.","category":"page"},{"location":"examples/surface/#","page":"Surface","title":"Surface","text":"loader = DataLoader(images(), labels(), batchsize=args.batch_size)\ncoordinates, loss_surface = create_surface(\n    model, loader, evaluation_step, args,\n)\n@save args.save_file coordinates loss_surface\nsurface(coordinates..., loss_surface)\ngui()","category":"page"},{"location":"examples/surface/#","page":"Surface","title":"Surface","text":"You can later load already computed surface, without the need to re-compute from scratch.","category":"page"},{"location":"examples/surface/#","page":"Surface","title":"Surface","text":"@load args.save_file coordinates loss_surface","category":"page"},{"location":"examples/trajectory/#Create-trajectory-1","page":"Trajectory","title":"Create trajectory","text":"","category":"section"},{"location":"examples/trajectory/#","page":"Trajectory","title":"Trajectory","text":"Create trajectory between model checkpoints that were created during training. This way we can visualize path that optimizer took.","category":"page"},{"location":"examples/trajectory/#","page":"Trajectory","title":"Trajectory","text":"Import necessary stuff.","category":"page"},{"location":"examples/trajectory/#","page":"Trajectory","title":"Trajectory","text":"using Yama: create_trajectory\nusing Plots\n\npyplot()\nPlots.PyPlotBackend()","category":"page"},{"location":"examples/trajectory/#","page":"Trajectory","title":"Trajectory","text":"Load target model and create list of checkpoints.","category":"page"},{"location":"examples/trajectory/#","page":"Trajectory","title":"Trajectory","text":"checkpoints = readdir(\"./cnn-checkpoints\", join=true)\n\ntarget_checkpoint_file = checkpoints[end]\n@load target_checkpoint_file checkpoint_weights\ntarget_weights = params(checkpoint_weights)\n\ncheckpoints = checkpoints[1:end - 1] # Remove target model from checkpoints","category":"page"},{"location":"examples/trajectory/#","page":"Trajectory","title":"Trajectory","text":"Compute positions of the trajectory and plot.","category":"page"},{"location":"examples/trajectory/#","page":"Trajectory","title":"Trajectory","text":"positions = create_trajectory(target_weights, checkpoints)\nplot(positions[:, 1], positions[:, 2])\ngui()","category":"page"},{"location":"examples/trajectory/#","page":"Trajectory","title":"Trajectory","text":"In the examples directory you can find how to plot this trajectory onto the surface that was created in the surface example.","category":"page"},{"location":"#","page":"Yama 山","title":"Yama 山","text":"CurrentModule = Yama","category":"page"},{"location":"#Yama-山-1","page":"Yama 山","title":"Yama 山","text":"","category":"section"},{"location":"#","page":"Yama 山","title":"Yama 山","text":"GitHub","category":"page"},{"location":"#","page":"Yama 山","title":"Yama 山","text":"Visualizer loss surface in random directions (1st & 2nd picture) or between model checkpoints (3rd picture).","category":"page"},{"location":"#","page":"Yama 山","title":"Yama 山","text":"<p align=\"center\">\n  <img src=\"https://raw.githubusercontent.com/pxl-th/Yama.jl/master/res/mnist.png\" width=300>\n  <img src=\"https://raw.githubusercontent.com/pxl-th/Yama.jl/master/res/mnist-log.png\" width=300>\n</p>","category":"page"},{"location":"#","page":"Yama 山","title":"Yama 山","text":"Visualize path that optimizer took during training (4th picture).","category":"page"},{"location":"#","page":"Yama 山","title":"Yama 山","text":"<p align=\"center\">\n  <img src=\"https://raw.githubusercontent.com/pxl-th/Yama.jl/master/res/mnist-two-checkpoints.png\" width=300>\n  <img src=\"https://raw.githubusercontent.com/pxl-th/Yama.jl/master/res/optimizer-path-cnn-mnist-log.png\" width=300>\n</p>","category":"page"},{"location":"#What-can-you-do-with-it-1","page":"Yama 山","title":"What can you do with it","text":"","category":"section"},{"location":"#","page":"Yama 山","title":"Yama 山","text":"Create surface in random directions around current model parameters.\nCreate surface between model checkpoints in x- and y-directions.\nCreate trajectory between model checkpoints. This way you can visualize path that optimizer took during training, for example.","category":"page"},{"location":"#","page":"Yama 山","title":"Yama 山","text":"note: Note\nAll models that you provide via checkpoint files, should contain parameters under checkpoint_weights key.Saving weights like this in code would look like BSON.@save file checkpoint_weights.","category":"page"},{"location":"#","page":"Yama 山","title":"Yama 山","text":"Visit Examples section for short guides or examples directory for self-sufficient examples.","category":"page"},{"location":"#References-1","page":"Yama 山","title":"References","text":"","category":"section"},{"location":"#","page":"Yama 山","title":"Yama 山","text":"Hao Li et.al. \"Visualizing the Loss Landscape of Neural Nets\"\nOriginal code in Python","category":"page"},{"location":"docs/#API-1","page":"User API","title":"API","text":"","category":"section"},{"location":"docs/#","page":"User API","title":"User API","text":"Modules = [Yama]\nPrivate = false","category":"page"},{"location":"docs/#Yama.SurfaceArgs","page":"User API","title":"Yama.SurfaceArgs","text":"Arguments for configuring loss surface computation.\n\nParameters\n\nxmin::Float32, xmax::Float32, xnum::Int32: define span of the surface\n\nand amount of point in the x direction.\n\nymin::Float32, ymax::Float32, ynum::Int32: define span of the surface\n\nand amount of point in the y direction.\n\nuse_gpu::Bool: Whether to use gpu. If true then it is up to you to\n\ntransfer mini-batch in evaluation_step function to the gpu.\n\nx_directions_file::Union{Nothing, String}: If provided, directions\n\nfor x axis will be loaded from it. Otherwise, random initialized. Should match in shape with model's weights.\n\ny_directions_file::Union{Nothing, String}: If provided, directions\n\nfor y axis will be loaded from it. Otherwise, random initialized. Should match in shape with model's weights.\n\nnote: Note\nIf use_gpu = true then it is up to you to transfer mini-batch in evaluation_step function to the gpu.\n\nnote: Note\nBoth x- and y- directions files in should contain directions under checkpoint_weights key.Saving weights like this in code would look like BSON.@save file checkpoint_weights.\n\n\n\n\n\n","category":"type"},{"location":"docs/#Yama.create_surface-Tuple{Any,Flux.Data.DataLoader,Function,SurfaceArgs}","page":"User API","title":"Yama.create_surface","text":"function create_surface(\n    model, dataloader::DataLoader, evaluation_step::Function, args::SurfaceArgs,\n)\n\nCreate loss surface.\n\nParameters\n\nmodel: Model to use in loss function.\ndataloader::DataLoader: Dataset on which to evaluate loss function.\nevaluation_step:   Custom-defined function which given model, mini-batch and args,   computes loss on that mini-batch.\nargs::SurfaceArgs: Parameters used when computing surface.\n\nnote: Note\nIf you specified use_gpu in args, then it is up to you, to transfer mini-batch in evaluation_step function to gpu.\n\n\n\n\n\n","category":"method"},{"location":"docs/#Yama.create_trajectory","page":"User API","title":"Yama.create_trajectory","text":"function create_trajectory(\n    weights::Params, checkpoints::Vector{String}, projection::Symbol = :cos,\n)::Array{Float32, 2}\n\nCreate trajectory between model checkpoints. This allows for path visualization that optimizer took during training, for example.\n\nArguments\n\nweights::Params: Target weights of the model. E.g. from last epoch.\ncheckpoints::Vector{String}: Checkpoints of the model from previous epochs.\n\nThey will be loaded in sorted order and should contain weights under checpoint_weights key.\n\nprojection::Symbol: Projection method. Either :lstsq or :cos.\n\nnote: Note\nCheckpoint files should contain weights under checpoint_weights key.\n\n\n\n\n\n","category":"function"}]
}