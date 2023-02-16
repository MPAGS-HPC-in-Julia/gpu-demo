using CUDA
using BenchmarkTools
function _est_pi_gpu!(totals, n)
    block_counts = CUDA.@cuStaticSharedMem(UInt16, 256)
    idx = threadIdx().x
    hit_count = UInt16(0)
    for _ in 1:n
        x = rand(Float32)
        y = rand(Float32)
        is_hit = (x*x+y*y<=1)

        hit_count += UInt16(is_hit)
    end
    block_counts[idx] = hit_count

    step = blockDim().x ÷ 2
    while (step != 0)
        CUDA.sync_threads()
        if idx <= step
            block_counts[idx] += block_counts[idx + step]
        end
        step ÷= 2
    end

    if idx == 1
        CUDA.@atomic totals[1] += block_counts[1]
    end
    
    nothing
end

function est_pi_gpu(n)
    threads = 256
    n_per_thread = 16
    blocks = cld(n, threads*n_per_thread)
    total_num = threads * n_per_thread * blocks
    total = CuArray{UInt32}([0])
    @cuda threads=threads blocks=blocks _est_pi_gpu!(total, n_per_thread)
    gpu_total = UInt32(0)
    CUDA.@allowscalar begin
        gpu_total = total[]
    end
    CUDA.unsafe_free!(total)
    return 4 * gpu_total / total_num
end

function throw_dart()
    x = rand() * 2 - 1
    y = rand() * 2 - 1
    return (x^2+y^2<=1)
end
function est_pi_gpu_array(N)
    darts = CuArray{Bool}(undef, N)
    darts .= (_->throw_dart()).(nothing)
    est = 4 * reduce(+, darts, init=0) / N 
    CUDA.unsafe_free!(darts)
    return est
end

# @benchmark CUDA.@sync est_pi_gpu($(2^20))
# @benchmark CUDA.@sync est_pi_gpu_array($(2^20))