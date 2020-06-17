function flatten(weights::Params)::Array{Float32, 1}
    cat([ndims(w) > 1 ? reshape(w, length(w)) : w for w in weights]..., dims=1)
end

function unflatten_set(directions::Array{Float32, 1}, weights::Params)::Params
    unflattened = deepcopy(weights)
    idx = 1
    for uf in unflattened
        uf[:] = reshape(directions[idx:idx + length(uf) - 1], size(uf))
        idx += length(uf)
    end
    @assert idx - 1 == length(directions) "Failed to correctly unflatten, please report this issue."
    unflattened
end

function pca_directions(
    weights::Params, checkpoints::Vector{String},
)::Array{Float32, 2}
    directions_matrix = []
    for checkpoint in checkpoints
        @load checkpoint checkpoint_weights
        directions = create_directions(weights, params(checkpoint_weights)) |> flatten
        push!(directions_matrix, directions)
    end
    directions_matrix = cat(directions_matrix..., dims=2)
    model = fit(PCA, directions_matrix, maxoutdim=2)
    projection(model)
end

function project(
    directions::Array{Float32, 1}, pca_projection::Array{Float32, 2},
    method::Symbol = :cos,
)::Array{Float32, 1}
    if method == :lstsq
        return pca_projection \ directions
    end
    normalizer = [norm(pca_projection[:, i]) for i in 1:size(pca_projection, 2)]
    (pca_projection' * directions) ./ normalizer
end

function trajectory(
    weights::Params, pca_projection::Array{Float32, 2},
    checkpoints::Vector{String},
)::Array{Float32, 2}
    coordinates = zeros(Float32, length(checkpoints), 2)
    @inbounds for (i, checkpoint) in enumerate(checkpoints)
        @load checkpoint checkpoint_weights
        directions = create_directions(weights, params(checkpoint_weights)) |> flatten
        projection = project(directions, pca_projection, :cos)
        coordinates[i, :] = projection
    end
    coordinates
end
