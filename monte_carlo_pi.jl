using CUDA

function throw_dart()
    x = rand() * 2 - 1
    y = rand() * 2 - 1
    return (x^2+y^2<=1)
end
function est_pi(N)
    hits = mapreduce(_->throw_dart(), +, 1:N);
    return 4 * hits / N
end
function est_pi_gpu(N)
    darts = CuArray{Bool}(undef, N)
    darts .= (_->throw_dart()).(nothing)
    est = 4 * reduce(+, darts, init=0) / N 
    CUDA.unsafe_free!(darts)
    return est
end

using BenchmarkTools
@benchmark est_pi($(2^20))
@benchmark CUDA.@sync est_pi_gpu($(2^20))