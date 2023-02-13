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

@benchmark CUDA.@sync mul!($C_gpu, $A_gpu, $B_gpu)