using CUDA
CUDA.allowscalar(false)
N = 2^15
threads = 1024
a = CUDA.rand(N)
b = CUDA.rand(N)
c = similar(a)

function cu_add!(c, a, b)
    for i in eachindex(c)
        c[i] = a[i]+b[i]
    end
    nothing
end

cu_add!(c, a, b)

function cu_add!(c, a, b)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(c)
        c[i] = a[i] + b[i]
    end
    nothing
end
function cu_add!(c, a, b)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(c)
        c[i] = a[i] + b[i]
    end
    nothing
end
@cuda blocks=cld(N, threads) threads=threads cu_add!(c, a, b)

isapprox(c, a.+b)