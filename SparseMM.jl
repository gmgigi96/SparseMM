using GraphBLASInterface, SuiteSparseGraphBLAS
using SparseArrays

GrB_init(GrB_NONBLOCKING)

function sm2gbm(A::SparseMatrixCSC{Int64, Int64})
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

function gbm_new(r, c)
    C = GrB_Matrix{Int64}()
    GrB_Matrix_new(C, GrB_INT64, r, c)
    return C
end

function mm(A::GrB_Matrix{Int64}, B::GrB_Matrix{Int64})

    #C = GrB_Matrix{Int64}()
    #GrB_Matrix_new(C, GrB_INT64, GrB_Matrix_nrows(A), GrB_Matrix_ncols(B))

    C = gbm_new(GrB_Matrix_nrows(A), GrB_Matrix_ncols(B))

    #GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, A, B, desc)
    GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, A, B, GrB_NULL)

    return C
end

function mm!(A::GrB_Matrix{Int64}, B::GrB_Matrix{Int64}, C::GrB_Matrix{Int64})

    #GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, A, B, desc)
    GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, A, B, GrB_NULL)

    return C
end
