using GraphBLASInterface, SuiteSparseGraphBLAS
using SparseArrays
using Base.Threads

const trans = GrB_Descriptor()
GrB_Descriptor_new(trans)
GrB_Descriptor_set(trans, GrB_INP0, GrB_TRAN)
GrB_Descriptor_set(trans, GrB_INP1, GrB_TRAN)

GrB_init(GrB_NONBLOCKING)

function sm2gbm(A::SparseMatrixCSC{Int64, Int64})
    res = GrB_Matrix{Int64}()
    GrB_Matrix_new(res, GrB_INT64, size(A, 1), size(A, 2))
    I, J, X = SparseArrays.findnz(A)
    GrB_Matrix_build(res, ZeroBasedIndex.(I.-1), ZeroBasedIndex.(J.-1), X, size(X,1), GrB_FIRST_INT64)
    return res
end

function sm2gbm(A::SparseMatrixCSC{Int8, Int64})
    res = GrB_Matrix{Int8}()
    GrB_Matrix_new(res, GrB_INT8, size(A, 1), size(A, 2))
    I, J, X = SparseArrays.findnz(A)
    GrB_Matrix_build(res, ZeroBasedIndex.(I.-1), ZeroBasedIndex.(J.-1), X, size(X,1), GrB_FIRST_INT8)
    return res
end

function gbm2sm(A::GrB_Matrix{Int64})
    I, J, X = GrB_Matrix_extractTuples(A)
    I = map(e -> e.x+1, I)
    J = map(e -> e.x+1, J)
    return SparseArrays.sparse(I, J, X)
end

function gbm_new_int64(r, c)
    C = GrB_Matrix{Int64}()
    GrB_Matrix_new(C, GrB_INT64, r, c)
    return C
end

function gbm_new_int8(r, c)
    C = GrB_Matrix{Int8}()
    GrB_Matrix_new(C, GrB_INT8, r, c)
    return C
end

function gbv_new_int64(l)
    V = GrB_Vector{Int64}()
    GrB_Vector_new(V, GrB_INT64, l)
    return V
end

function gbv_new_int8(l)
    V = GrB_Vector{Int8}()
    GrB_Vector_new(V, GrB_INT8, l)
    return V
end

function mm(A::GrB_Matrix{Int64}, B::GrB_Matrix{Int64})
    C = gbm_new_int64(GrB_Matrix_nrows(A), GrB_Matrix_ncols(B))

    #GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, A, B, desc)
    GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, A, B, GrB_NULL)

    return C
end

function mm(A::GrB_Matrix{Int8}, B::GrB_Matrix{Int64})
    C = gbm_new_int8(GrB_Matrix_nrows(A), GrB_Matrix_ncols(B))

    #GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, A, B, desc)
    GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT8, A, B, GrB_NULL)

    return C
end

function sm(A::GrB_Matrix{Int64})
    V = gbv_new_int64(size(A, 1))
    GrB_reduce(V, GrB_NULL, GrB_NULL, GxB_PLUS_INT64_MONOID, A, GrB_NULL)
    return V
end

function sm(A::GrB_Matrix{Int8})
    V = gbv_new_int8(size(A, 1))
    GrB_reduce(V, GrB_NULL, GrB_NULL, GxB_PLUS_INT8_MONOID, A, GrB_NULL)
    return V
end

function dmv(A::GrB_Matrix{Int64}, B::GrB_Vector{Int64})
    @assert GrB_Matrix_ncols(A) == GrB_Vector_size(B)
    res = gbm_new_int64(GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
    #tmp = [gbv_new_int64(GrB_Vector_size(B)) for i in 1:nthreads()]

    cache = Array{GrB_Vector{Int64}}(undef, GrB_Matrix_ncols(A));

    @inbounds @threads for j in 1:GrB_Matrix_ncols(A)

        cache[j] = gbv_new_int64(GrB_Vector_size(B))
        GrB_Col_extract(cache[j], GrB_NULL, GrB_NULL, A, GrB_ALL, 0, ZeroBasedIndex(j-1), trans)
        GrB_eWiseMult(cache[j], GrB_NULL, GrB_NULL, GxB_TIMES_DIV_UINT64, cache[j], B, GrB_NULL)

        #cache[j] = gbv_new_int64(GrB_Vector_size(B))
        #GrB_Vector_dup(cache[j], tmp[threadid()])

    end

    @inbounds for j in eachindex(cache)
        GrB_Col_assign(res, GrB_NULL, GrB_NULL, cache[j], GrB_ALL, 0, ZeroBasedIndex(j-1), GrB_NULL)
    end
    #GrB_Col_assign(res, GrB_NULL, GrB_NULL, cache[1], GrB_ALL, 0, ZeroBasedIndex(0), trans)
    res2 = gbm_new_int64(GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
    GrB_transpose(res2,GrB_NULL,GrB_NULL,res,GrB_NULL)
    GrB_wait()  # flush pending transitions
    #for e in tmp
    #    GrB_Vector_free(e)
    #end
    for e in cache
        GrB_Vector_free(e)
    end
    #GrB_Matrix_free(res)
    return res
end

function dmv(A::GrB_Matrix{Int8}, B::GrB_Vector{Int64})
    @assert GrB_Matrix_ncols(A) == GrB_Vector_size(B)
    res = gbm_new_int8(GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
    tmp = gbv_new_int8(GrB_Vector_size(B))
    asdhasdjhaskd
    for j in 0:GrB_Matrix_ncols(A)-1
        # select col j from A -> tmp
        GrB_Col_extract(tmp, GrB_NULL, GrB_NULL, A, GrB_ALL, 0, ZeroBasedIndex(j), trans)
        # q .// v -> tmp
        GrB_eWiseMult(tmp, GrB_NULL, GrB_NULL, GxB_TIMES_DIV_UINT8, tmp, B, GrB_NULL)
        # copy tmp in res[j]
        GrB_Col_assign(res, GrB_NULL, GrB_NULL, tmp, GrB_ALL, 0, ZeroBasedIndex(j), GrB_NULL)
    end
    res2 = gbm_new_int64(GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
    GrB_transpose(res2,GrB_NULL,GrB_NULL,res,GrB_NULL)
    GrB_wait()  # flush pending transitions
    GrB_Vector_free(tmp)
    GrB_Matrix_free(res)

    return res2
end

function d(A::GrB_Matrix{Int64}, B::GrB_Matrix{Int64})
    #B_T = gbm_new_int64(size(B, 1), size(B, 2))
    #GrB_transpose(B_T,GrB_NULL,GrB_NULL,B,GrB_NULL)
    C = mm(A,B)
    V = sm(A)
    res = dmv(C, V)
    return res
end

function d(A::GrB_Matrix{Int8}, B::GrB_Matrix{Int8})
    #B_T = gbm_new_int8(size(B, 1), size(B, 2))
    #GrB_transpose(B_T,GrB_NULL,GrB_NULL,B,GrB_NULL)
    C = mm(A,B)
    V = sm(A)
    res = dmv(C, V)
    return res
end
