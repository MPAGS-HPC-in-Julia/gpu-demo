using CUDA
using Random
function random_walk(T)
    x = Int32(0)
    for _ in 1:T
        x += (rand(Float32) < 0.5f0) * Int32(2) - Int32(1)
    end

    return x
end

function walks(N, T)
    walks = Vector{Int32}(undef, N);
    walks .= random_walk.(T)
    return walks
end

N = 2048;
T = 100;
xs = walks(N, T);

avg_final_pos = reduce(+, xs) / length(xs)


function walks_gpu(N, T)
    walks = CuArray{Int32}(undef, N);
    walks .= random_walk.(T)
    return walks
end

xs_gpu = walks_gpu(N, T)

avg_final_pos = reduce(+, xs_gpu) / length(xs_gpu)
