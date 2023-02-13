using CUDA


N = 1024
a = CUDA.rand(N)

function type_unstable(x)
    if x < 0.5
        return 0
    else
        return 1.0
    end
end

b = type_unstable.(a)

function type_stable(x)
    if x < 0.5
        return 0.0
    else
        return 1.0
    end
end

b = type_stable.(a)

function custom_sum(arr)
    a = zero(eltype(arr))
    for x in arr
        a += x
    end

    a
end

custom_sum(b)