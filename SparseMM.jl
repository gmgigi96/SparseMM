using GraphBLASInterface, SuiteSparseGraphBLAS
using SparseArrays
using Base.Threads

const TRAN = GrB_Descriptor()
GrB_Descriptor_new(TRAN)
GrB_Descriptor_set(TRAN, GrB_INP0, GrB_TRAN)
GrB_Descriptor_set(TRAN, GrB_INP1, GrB_TRAN)

const TRAN_INP1 = GrB_Descriptor()
GrB_Descriptor_new(TRAN_INP1)
GrB_Descriptor_set(TRAN_INP1, GrB_INP1, GrB_TRAN)

const DIV_INT64 = GrB_BinaryOp()
GrB_BinaryOp_new(DIV_INT64, div, GrB_INT64, GrB_INT64, GrB_INT64)

const DIV_INT8 = GrB_BinaryOp()
GrB_BinaryOp_new(DIV_INT8, div, GrB_INT8, GrB_INT8, GrB_INT8)

const db2 = x -> x÷2

const DIV_BY_TWO_INT64 = GrB_UnaryOp()
GrB_UnaryOp_new(DIV_BY_TWO_INT64, db2, GrB_INT64, GrB_INT64)

const DIV_BY_TWO_INT8 = GrB_UnaryOp()
GrB_UnaryOp_new(DIV_BY_TWO_INT8, db2, GrB_INT8, GrB_INT8)

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

function gbm2sm(A::GrB_Matrix)
    I, J, X = GrB_Matrix_extractTuples(A)
    I = map(e -> e.x+1, I)
    J = map(e -> e.x+1, J)
    return SparseArrays.sparse(I, J, X, GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
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

function mm(A::GrB_Matrix{Int64}, B::GrB_Matrix{Int64}, tran_b = false)
    C = gbm_new_int64(GrB_Matrix_nrows(A), GrB_Matrix_ncols(B))

    if tran_b
        GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, A, B, TRAN_INP1)
    else
        GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT64, A, B, GrB_NULL)
    end

    return C
end

function mm(A::GrB_Matrix{Int8}, B::GrB_Matrix{Int8}, tran_b = false)
    C = gbm_new_int8(GrB_Matrix_nrows(A), GrB_Matrix_ncols(B))

    if tran_b
        GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT8, A, B, TRAN_INP1)
    else
        GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_PLUS_TIMES_INT8, A, B, GrB_NULL)
    end

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

function v2m_int64(V::GrB_Vector{Int64}, j, M::GrB_Matrix{Int64})
    GrB_Matrix_clear(M)
    I, X = GrB_Vector_extractTuples(V)
    J = ZeroBasedIndex.(fill(j, GrB_Matrix_ncols(M)))
    GrB_Matrix_build(M, I, J, X, GrB_Matrix_ncols(M), GrB_FIRST_INT64)
    return M
end

function dmv(A::GrB_Matrix{Int64}, B::GrB_Vector{Int64})
    res = gbm_new_int64(GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
    I, J, X = GrB_Matrix_extractTuples(A)

    function _dmv0(i)
        x = GrB_Vector_extractElement(B, i)
        if x != GrB_NO_VALUE; return Int64(x); end
        return 1
    end

    X1 = map(_dmv0, I)


    I = vcat(I, I)
    J = vcat(J, J)
    X = vcat(X, X1)

    GrB_Matrix_build(res, I, J, X, length(X), DIV_INT64)

    return res
end

function dmv(A::GrB_Matrix{Int8}, B::GrB_Vector{Int8})
    res = gbm_new_int8(GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
    I, J, X = GrB_Matrix_extractTuples(A)

    function _dmv0(i)
        x = GrB_Vector_extractElement(B, i)
        if x != GrB_NO_VALUE; return Int8(x); end
        return 1
    end

    X1 = map(_dmv0, I)


    I = vcat(I, I)
    J = vcat(J, J)
    X = vcat(X, X1)

    GrB_Matrix_build(res, I, J, X, length(X), DIV_INT8)

    return res
end

function dmv_old(A::GrB_Matrix{Int64}, B::GrB_Vector{Int64})
    @assert GrB_Matrix_ncols(A) == GrB_Vector_size(B)
    res = gbm_new_int64(GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
    tmp = gbv_new_int64(GrB_Vector_size(B))

    for j in 0:GrB_Matrix_ncols(A)-1
        # select col j from A -> tmp
        GrB_Col_extract(tmp, GrB_NULL, GrB_NULL, A, GrB_ALL, 0, ZeroBasedIndex(j), TRAN)
        # q .// v -> tmp
        GrB_eWiseMult(tmp, GrB_NULL, GrB_NULL, GxB_TIMES_DIV_UINT64, tmp, B, GrB_NULL)
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

function dmv_old(A::GrB_Matrix{Int8}, B::GrB_Vector{Int64})
    @assert GrB_Matrix_ncols(A) == GrB_Vector_size(B)
    res = gbm_new_int8(GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
    tmp = gbv_new_int8(GrB_Vector_size(B))

    for j in 0:GrB_Matrix_ncols(A)-1
        # select col j from A -> tmp
        GrB_Col_extract(tmp, GrB_NULL, GrB_NULL, A, GrB_ALL, 0, ZeroBasedIndex(j), TRAN)
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

function div_by_two(A::GrB_Matrix{Int64})
    res = gbm_new_int64(GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
    GrB_apply(res, GrB_NULL, GrB_NULL, DIV_BY_TWO_INT64, A, GrB_NULL)
    return res
end

function div_by_two(A::GrB_Matrix{Int8})
    res = gbm_new_int8(GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
    GrB_apply(res, GrB_NULL, GrB_NULL, DIV_BY_TWO_INT8, A, GrB_NULL)
    return res
end

function div_by_two!(A::GrB_Matrix{Int64})
    GrB_apply(A, GrB_NULL, GrB_NULL, DIV_BY_TWO_INT64, A, GrB_NULL)
end

function div_by_two!(A::GrB_Matrix{Int8})
    GrB_apply(A, GrB_NULL, GrB_NULL, DIV_BY_TWO_INT8, A, GrB_NULL)
end

function d2(A::GrB_Matrix, B::GrB_Matrix)
    C = mm(A,B, true)
    res = div_by_two(C)
    return res
end

function d3(A::GrB_Matrix, B::GrB_Matrix)
    C = mm(A,B,true)
    V = sm(A)
    res = dmv(C, V)
    return res
end
