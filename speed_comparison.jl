using BSON
using CUDA
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

using BenchmarkTools


N = 2048
A = rand(Float32, N, N);
B = rand(Float32, N, N);
C = similar(A);

@benchmark mul!($C, $A, $B)

A_gpu = cu(A);
B_gpu = cu(B);
C_gpu = similar(A_gpu);

@benchmark mul!($C_gpu, $A_gpu, $B_gpu)













function benchmarks(ns)
    function bench_cpu(n)
        a = rand(Float32, n, n)
        b = rand(Float32, n, n)
        c = similar(a)
        return @belapsed mul!($c, $a, $b)
    end
    function bench_gpu(n)
        a = CUDA.rand(Float32, n, n)
        b = CUDA.rand(Float32, n, n)
        c = similar(a)
        return @belapsed CUDA.@sync mul!($c, $a, $b)
    end
    cpu_times_task = Threads.@spawn bench_cpu.(ns)
    gpu_times_task = Threads.@spawn bench_gpu.(ns)
    cpu_times = fetch(cpu_times_task)
    gpu_times = fetch(gpu_times_task)
    return (ns=ns, cpu_times=cpu_times, gpu_times=gpu_times)
end

using BSON: @save, @load
@save "results/gpu_vs_cpu_matmul.bson" results
@load "results/gpu_vs_cpu_matmul.bson" results
using Plots
function plot_results(results)
    ns = results.ns
    cpu_times = results.cpu_times
    gpu_times = results.gpu_times

    plt = plot(ns, cpu_times./cpu_times, label="CPU", xscale=:log10, yscale=:log10, linestyle=:dash, markershape=:square)
    plot!(plt, ns, cpu_times./gpu_times, label="GPU", xscale=:log10, yscale=:log10, linestyle=:solid, markershape=:circle)

    xlabel!(plt, "n")
    ylabel!(plt, "Relative Speedup")
    plot!(plt; legend=:topleft)
    return plt
end
plot_results(results)