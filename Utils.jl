using GraphBLASInterface, SuiteSparseGraphBLAS

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
