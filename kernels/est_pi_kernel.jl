using CUDA
using BenchmarkTools

function _est_pi_gpu!(totals)
    block_counts = CUDA.@cuStaticSharedMem(UInt16, 256)
    x = rand(Float32)
    y = rand(Float32)
    is_hit = (x*x+y*y<=1)
    idx = threadIdx().x

    block_counts[idx] = UInt16(is_hit)

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
    blocks = cld(n, threads)
    total = CuArray{UInt32}([0])
    @cuda threads=threads blocks=blocks _est_pi_gpu!(total)
    gpu_total = UInt32(0)
    CUDA.@allowscalar begin
        gpu_total = total[]
    end
    CUDA.unsafe_free!(total)
    return 4 * gpu_total / (threads * blocks)
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