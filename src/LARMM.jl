include("SparseMM.jl")

"""
    d1(A,B)

Compute ``∂_1 = M_0 * M_1'``

# Arguments
- `A` : Sparse Matrix
- `B` : Sparse Matrix
"""
function d1(A, B)
    # Create GraphBLAS Matrix from A and B
    A_GB, B_GB = sm2gbm(A), sm2gbm(B, true)

    # C = A * B'
    C_GB = mm(A_GB, B_GB)

    res = gbm2sm(C_GB)

    # Free all stuffs
    GrB_Matrix_free(A_GB)
    GrB_Matrix_free(B_GB)
    GrB_Matrix_free(C_GB)

    return res
end

"""
    d2(A,B)

Compute ``∂_2 = (A * B') .÷ sum(A, dims=2)``

# Arguments
- `A` : Sparse Matrix
- `B` : Sparse Matrix
"""
function d2(A, B)
    # Create GraphBLAS Matrix from A and B
    A_GB, B_GB = sm2gbm(A), sm2gbm(B, true)

    # C = A * B'
    C_GB = mm(A_GB, B_GB)

    # C .÷ 2
    div_by_two!(C_GB)

    res = gbm2sm(C_GB)

    # Free all stuffs
    GrB_Matrix_free(A_GB)
    GrB_Matrix_free(B_GB)
    GrB_Matrix_free(C_GB)

    return res
end

"""
    d3(A,B)

Compute ``∂_3 = (A * B') .÷ sum(A, dims=2)``

# Arguments
- `A` : Sparse Matrix
- `B` : Sparse Matrix
"""
function d3(A, B)
    # Create GraphBLAS Matrix from A and B
    A_GB, B_GB = sm2gbm(A), sm2gbm(B, true)

    # C = A * B'
    C_GB = mm(A_GB, B_GB)

    # sum(A, dims=2)
    V_GB = sm(A_GB)

    # C .÷ V
    R_GB = dmv(C_GB, V_GB)

    res = gbm2sm(R_GB)

    # Free all stuffs
    GrB_Matrix_free(A_GB)
    GrB_Matrix_free(B_GB)
    GrB_Matrix_free(C_GB)
    GrB_Matrix_free(R_GB)
    GrB_Vector_free(V_GB)

    return res
end
