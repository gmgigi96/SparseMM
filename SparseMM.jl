using GraphBLASInterface, SuiteSparseGraphBLAS
using SparseArrays

GrB_init(GrB_NONBLOCKING)

function sm2gbm(A::SparseMatrixCSC)
    res = GrB_Matrix{Int64}()
    GrB_Matrix_new(res, GrB_INT64, size(A, 1), size(A, 2))
    I, J, X = SparseArrays.findnz(A)
    GrB_Matrix_build(res, ZeroBasedIndex.(I.-1), ZeroBasedIndex.(J.-1), X, size(X,1), GrB_FIRST_INT64)
    return res
end

function gbm2sm(A::GrB_Matrix{Int64})
    I, J, X = GrB_Matrix_extractTuples(A)
    I = map(e -> e.x+1, I)
    J = map(e -> e.x+1, J)
    return SparseArrays.sparse(I, J, X)
end

function v2m(V::GrB_Vector{Int64})
    I, X = GrB_Vector_extractTuples(V)
    cols = GrB_Vector_size(V)
    J = ZeroBasedIndex.(zeros(Int64, cols))
    res = gbm_new(cols, 1)
    GrB_Matrix_build(res, I, J, X, cols, GrB_FIRST_INT64)
    return res
end

function gbm_new(r, c)
    C = GrB_Matrix{Int64}()
    GrB_Matrix_new(C, GrB_INT64, r, c)
    return C
end

function gbv_new(l)
    V = GrB_Vector{Int64}()
    GrB_Vector_new(V, GrB_INT64, l)
    return V
end

function mm(A::GrB_Matrix{Int64}, B::GrB_Matrix{Int64})
    C = gbm_new(GrB_Matrix_nrows(A), GrB_Matrix_ncols(B))

    #GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, A, B, desc)
    GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, A, B, GrB_NULL)

    return C
end

function mm!(A::GrB_Matrix{Int64}, B::GrB_Matrix{Int64}, C::GrB_Matrix{Int64})
    #GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, A, B, desc)
    GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, A, B, GrB_NULL)
end

function sm(A::GrB_Matrix{Int64})
    V = gbv_new(size(A, 1))
    GrB_reduce(V, GrB_NULL, GrB_NULL, GxB_PLUS_INT64_MONOID, A, GrB_NULL)
    return V
end

function dmv(A::GrB_Matrix{Int64}, B::GrB_Vector{Int64})
    @assert GrB_Matrix_ncols(A) == GrB_Vector_size(B)
    res = gbm_new(GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
    tmp = gbv_new(GrB_Vector_size(B))

    for j in 0:GrB_Matrix_ncols(A)-1
        # select col j from A -> tmp
        GrB_Col_extract(tmp, GrB_NULL, GrB_NULL, A, GrB_ALL, 0, ZeroBasedIndex(j), GrB_NULL)
        # q .// v -> tmp
        GrB_eWiseMult(tmp, GrB_NULL, GrB_NULL, GxB_TIMES_DIV_UINT64, tmp, B, GrB_NULL)
        # copy tmp in res[j]
        GrB_Col_assign(res, GrB_NULL, GrB_NULL, tmp, GrB_ALL, 0, ZeroBasedIndex(j), GrB_NULL)
    end

    GrB_wait()  # flush pending transitions
    GrB_Vector_free(tmp)

    return res
end

function d(A::GrB_Matrix{Int64}, B::GrB_Matrix{Int64})
    B_T = gbm_new(size(1,B), size(2, B))
    GrB_transpose(B_T,GrB_NULL,GrB_NULL,B,GrB_NULL)
    C = mm(A,B_T)
    V = SM(A)
    V_M = v2m(V)
    res = dmv(A, V_M)
    return res
end
