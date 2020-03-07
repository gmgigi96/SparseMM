module SparseMM

using GraphBLASInterface, SuiteSparseGraphBLAS
using SparseArrays
using Base.Threads
using Utils

export sm2gbm, gbm2sm, gbm_new, gbv_new, mm, sm, v2m, dmv, div_by_two, div_by_two!, msc

GrB_init(GrB_NONBLOCKING)

"""
    sm2gbm(A, tran=false)

Convert a SparseMatrix into a GraphBLAS matrix.

# Arguments
- `A::SparseMatrixCSC`  : Sparse Matrix
- `tran::Bool`          : if true trasposes the matrix before convertion
"""
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

"""
    gbm2sm(A)

Convert a GraphBLAS matrix into a SparseMatrix.

# Arguments
- `A::GrB_Matrix` : the GraphBLAS matrix
"""
function gbm2sm(A::GrB_Matrix)
    I, J, X = GrB_Matrix_extractTuples(A)
    I = map(e -> e.x+1, I)
    J = map(e -> e.x+1, J)
    return SparseArrays.sparse(I, J, X, GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
end

"""
    gbm_new(T, r, c)

Create a GraphBLAS matrix with `r` row and `c` columns, with element of type `T`.
"""
function gbm_new(T, r, c)
    C = GrB_Matrix{T}()
    GrB_Matrix_new(C, GrB_type(T), r, c)
    return C
end

"""
    gbv_new(T, l)

Create a GraphBLAS vector with `l` element of type `T`.
"""
function gbv_new(T, l)
    V = GrB_Vector{T}()
    GrB_Vector_new(V, GrB_type(T), l)
    return V
end

"""
    mm(A, B)

Multiply the GraphBLAS matrix A by the GraphBLAS matrix B.
"""
function mm(A::GrB_Matrix{T}, B::GrB_Matrix{T}) where T
    @assert GrB_Matrix_ncols(A) == GrB_Matrix_nrows(B)
    C = gbm_new(T, GrB_Matrix_nrows(A), GrB_Matrix_ncols(B))

    GrB_mxm(C, GrB_NULL, GrB_NULL, GxB_op("PLUS_TIMES", T), A, B, GrB_NULL)

    return C
end

"""
    sm(A)

Sum rows of GraphBLAS matrix A.
"""
function sm(A::GrB_Matrix{T}) where T
    V = gbv_new(T, size(A, 1))
    GrB_reduce(V, GrB_NULL, GrB_NULL, GxB_monoid("PLUS", T), A, GrB_NULL)
    return V
end

"""
    v2m(V, M)

Multiply vector V of type T by the matrix M of type T.
"""
function v2m(V::GrB_Vector{T}, j, M::GrB_Matrix{T}) where T
    GrB_Matrix_clear(M)
    I, X = GrB_Vector_extractTuples(V)
    J = ZeroBasedIndex.(fill(j, GrB_Matrix_ncols(M)))
    GrB_Matrix_build(M, I, J, X, GrB_Matrix_ncols(M), GrB_op("FIRST", T))
    return M
end

"""
    dmv(A,B)

Divide the matrix A of type T by vector B of type T columnwise.
"""
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

    GrB_Matrix_build(res, I, J, X, length(X), DIV(T))

    return res
end

"""
    dmv_old(A, B)

Same as `dmv` but much slower (Deprecated).
"""
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

"""
    div_by_two(A)

Element-wise division by two of matrix A, creating a new matrix as result.
"""
function div_by_two(A::GrB_Matrix{T}) where T
    res = gbm_new(T, GrB_Matrix_nrows(A), GrB_Matrix_ncols(A))
    GrB_apply(res, GrB_NULL, GrB_NULL, DIV_BY_TWO(T), A, GrB_NULL)
    return res
end

"""
    div_by_two!(A)

Element-wise division by two of matrix A.
"""
function div_by_two!(A::GrB_Matrix{T}) where T
    GrB_apply(A, GrB_NULL, GrB_NULL, DIV_BY_TWO(T), A, GrB_NULL)
end

function __msc0(D::GrB_Matrix{T}, cols, r) where T
    Ires, Xres = ZeroBasedIndex[], T[]
    vec = gbv_new(T, r)

    for j in cols
        GrB_Col_extract(vec, GrB_NULL, GrB_NULL, D, GrB_ALL, 0, ZeroBasedIndex(j-1), GrB_NULL)
        I, X = GrB_Vector_extractTuples(vec)
        append!(Ires, I)
        append!(Xres, X)
        GrB_Vector_clear(vec)
    end

    GrB_free(vec)
    return Ires, Xres

end

function msc(D::GrB_Matrix{T}, v) where T
    r = GrB_Matrix_nrows(D)
    res = gbm_new(T, r, length(v))    

    X = T[]
    J = ZeroBasedIndex[]
    I = ZeroBasedIndex[]

    for (i, c) in enumerate(v)
        _I, _X = __msc0(D, c, r)

        append!(I, _I)
        append!(X, _X)
        append!(J, fill(i-1, length(_X)))
    end

    GrB_Matrix_build(res, I, J, X, length(X), GrB_op("PLUS", T))

    return res

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

end

