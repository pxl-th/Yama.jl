function get_random_weights(weights::Params)::Params
    params([Float32.(randn(size(w))) for w in weights])
end

function normalize_direction!(direction::Array{Float32}, weight::Array{Float32})
    if ndims(direction) == 1
        direction[:] .= weight
    else
        @inbounds @views for i in 1:size(direction, 1)
            direction[i, :] .*= (
                norm(weight[i, :]) / (norm(direction[i, :]) + 1e-6)
            )
        end
    end
end

function normalize_directions!(directions::Params, weights::Params)
    for (direction, weight) in zip(directions, weights)
        normalize_direction!(direction, weight)
    end
end

function create_random_directions(weights::Params)::Params
    directions = get_random_weights(weights)
    normalize_directions!(directions, weights)
    directions
end

function shift_weights(
    weights::Params, directions::Tuple{Params, Params},
    step::Tuple{Float32, Float32},
)::Params
    direction_x, direction_y = directions
    step_x, step_y = step
    params([
        @. weight + dx * step_x + dy * step_y
        for (weight, dx, dy) in zip(weights, direction_x, direction_y)
   ])
end
