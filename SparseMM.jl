using GraphBLASInterface, SuiteSparseGraphBLAS
using SparseArrays

GrB_init(GrB_NONBLOCKING)

const INTDIV = GrB_BinaryOp()
GrB_BinaryOp_new(INTDIV, (//), GrB_INT64, GrB_INT64, GrB_INT64)

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


function SM(A::GrB_Matrix{Int64})
    V = gbv_new(size(A, 1))
    GrB_reduce(V, GrB_NULL, GrB_NULL, GxB_PLUS_INT64_MONOID, A, GrB_NULL)

    return V
end


function dmv(A::GrB_Matrix{Int64}, B::GrB_Matrix{Int64})
    @assert GrB_Matrix_ncols(A) == GrB_Vector_size(B)
    res = gbm_new(GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
    GrB_eWiseMult(res, GrB_NULL, GrB_NULL, INTDIV, A, B, GrB_NULL)
    return res
end
