using CUDA


N = 1024
a = CUDA.rand(N)

function type_unstable(x)
    if x < 0.5
        return 0 # Int
    else
        return 1.0 # Float
    end
end
@code_warntype type_unstable(2.0)

b = type_unstable.(a)


function type_stable(x)
    if x < 0.5
        return 0.0
    else
        return 1.0
    end
end
@code_warntype type_stable(2.0)

b = type_stable.(a)

# Scalar indexing
function custom_sum(arr)
    a = zero(eltype(arr))
    for i in eachindex(arr)
        a += x[i]
    end

    a
end
# CUDA.allowscalar(false)

custom_sum(b)