using BenchmarkTools
using Random

sparseN(N) = sparse(randperm(N), randperm(N), ones(Int64, N), N, N)

A = sparseN(100000)
B = sparseN(100000)

m1 = @btime $A*$B'

As = sm2gbm(A)
Bs = sm2gbm(sparse(B'))
mgb = @btime mm($As, $Bs)

m2 = gbm2sm(mgb)

@assert m1 == m2

m4 = @btime $A .// sum($B, dims=2)

V = SM(Bs)
m3 = @btime dmv($As, $V)

@assert m3 == m4
