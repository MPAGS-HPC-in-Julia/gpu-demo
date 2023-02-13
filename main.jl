using CUDA
# Check CUDA is compatible
CUDA.versioninfo()

N = 2048
a = rand(Float32, N);
b = rand(Float32, N);
c = similar(a)

c .= a .+ b

# Copy arrays to the GPU
a_gpu = cu(a)
b_gpu = cu(b)
c_gpu = similar(a_gpu)

c_gpu .= a_gpu .+ b_gpu

# Copy back from the GPU
c_from_gpu = Array(c_gpu)

# Check whether they are equal
isapprox(c, c_from_gpu)

