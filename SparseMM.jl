using GraphBLASInterface, SuiteSparseGraphBLAS
using SparseArrays
using Base.Threads

const TRAN = GrB_Descriptor()
GrB_Descriptor_new(TRAN)
GrB_Descriptor_set(TRAN, GrB_INP0, GrB_TRAN)
GrB_Descriptor_set(TRAN, GrB_INP1, GrB_TRAN)

const int_div = (a, b) -> if a == 0 || b == 0 0 else a ÷ b end

const DIV_INT64 = GrB_BinaryOp()
GrB_BinaryOp_new(DIV_INT64, int_div, GrB_INT64, GrB_INT64, GrB_INT64)

const DIV_INT8 = GrB_BinaryOp()
GrB_BinaryOp_new(DIV_INT8, int_div, GrB_INT8, GrB_INT8, GrB_INT8)

const db2 = x -> x÷2

const DIV_BY_TWO_INT64 = GrB_UnaryOp()
GrB_UnaryOp_new(DIV_BY_TWO_INT64, db2, GrB_INT64, GrB_INT64)

const DIV_BY_TWO_INT8 = GrB_UnaryOp()
GrB_UnaryOp_new(DIV_BY_TWO_INT8, db2, GrB_INT8, GrB_INT8)

GrB_init(GrB_NONBLOCKING)

@inline function GrB_type(T)
    return eval(Symbol("GrB_", uppercase(string(typeof(zero(T))))))
end

@inline function GrB_op(fun_name, T)
    return eval(Symbol("GrB_", fun_name, "_", uppercase(string(typeof(zero(T))))))
end

@inline function GxB_op(fun_name, T)
    return eval(Symbol("GxB_", fun_name, "_", uppercase(string(typeof(zero(T))))))
end

@inline function GxB_monoid(fun_name, T)
    return eval(Symbol("GxB_", fun_name, "_", uppercase(string(typeof(zero(T)))), "_MONOID"))
end

function sm2gbm(A::SparseMatrixCSC{TE, TI}, tran=false) where {TE, TI}
    res = GrB_Matrix{TE}()
    if tran
        GrB_Matrix_new(res, GrB_type(TE), size(A, 2), size(A, 1))
        J, I, X = SparseArrays.findnz(A)
    else
        GrB_Matrix_new(res, GrB_type(TE), size(A, 1), size(A, 2))
        I, J, X = SparseArrays.findnz(A)
    end
    GrB_Matrix_build(res, ZeroBasedIndex.(I.-1), ZeroBasedIndex.(J.-1), X, size(X,1), GrB_op("FIRST", TE))
    return res
end

function gbm2sm(A::GrB_Matrix)
    I, J, X = GrB_Matrix_extractTuples(A)
    I = map(e -> e.x+1, I)
    J = map(e -> e.x+1, J)
    return SparseArrays.sparse(I, J, X, GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
end

function gbm_new(T, r, c)
    C = GrB_Matrix{T}()
    GrB_Matrix_new(C, GrB_type(T), r, c)
    return C
end

function gbv_new(T, l)
    V = GrB_Vector{T}()
    GrB_Vector_new(V, GrB_type(T), l)
    return V
end

function mm(A::GrB_Matrix{T}, B::GrB_Matrix{T}) where T
    @assert GrB_Matrix_ncols(A) == GrB_Matrix_nrows(B)
    C = gbm_new(T, GrB_Matrix_nrows(A), GrB_Matrix_ncols(B))

    GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_op("PLUS_TIMES", T), A, B, GrB_NULL)

    return C
end

function sm(A::GrB_Matrix{T}) where T
    V = gbv_new(T, size(A, 1))
    GrB_reduce(V, GrB_NULL, GrB_NULL, GxB_monoid("PLUS", T), A, GrB_NULL)
    return V
end

function v2m(V::GrB_Vector{T}, j, M::GrB_Matrix{T}) where T
    GrB_Matrix_clear(M)
    I, X = GrB_Vector_extractTuples(V)
    J = ZeroBasedIndex.(fill(j, GrB_Matrix_ncols(M)))
    GrB_Matrix_build(M, I, J, X, GrB_Matrix_ncols(M), GrB_op("FIRST", T))
    return M
end

function dmv(A::GrB_Matrix{T}, B::GrB_Vector{T}) where T
    res = gbm_new(T, GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
    I, J, X = GrB_Matrix_extractTuples(A)

    function _dmv0(i)
        x = GrB_Vector_extractElement(B, i)
        if x != GrB_NO_VALUE; return T(x); end
        return 1
    end

    X1 = map(_dmv0, I)

    I = vcat(I, I)
    J = vcat(J, J)
    X = vcat(X, X1)

    GrB_Matrix_build(res, I, J, X, length(X), DIV_INT64)    # GUARDA QUI

    return res
end

function dmv_old(A::GrB_Matrix{T}, B::GrB_Vector{T}) where T
    @assert GrB_Matrix_ncols(A) == GrB_Vector_size(B)
    res = gbm_new(GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
    tmp = gbv_new(GrB_Vector_size(B))

    for j in 0:GrB_Matrix_ncols(A)-1
        # select col j from A -> tmp
        GrB_Col_extract(tmp, GrB_NULL, GrB_NULL, A, GrB_ALL, 0, ZeroBasedIndex(j), TRAN)
        # q .// v -> tmp
        GrB_eWiseMult(tmp, GrB_NULL, GrB_NULL, GxB_op("TIMES_DIV", T), tmp, B, GrB_NULL)
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

function div_by_two(A::GrB_Matrix{T}) where T
    res = gbm_new(T, GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
    GrB_apply(res, GrB_NULL, GrB_NULL, DIV_BY_TWO_INT64, A, GrB_NULL)           # GUARDA QUI
    return res
end

function div_by_two!(A::GrB_Matrix{T}) where T
    GrB_apply(A, GrB_NULL, GrB_NULL, DIV_BY_TWO_INT64, A, GrB_NULL)             # GUARDA QUI
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
