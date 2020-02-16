using BenchmarkTools
using Random

sparseN(N) = sparse(randperm(N), randperm(N), ones(Int64, N), N, N)

A = sparseN(1000)
B = sparseN(1000)

m1 = A*B'

As = sm2gbm(A)
Bs = sm2gbm(B)
mgb = mm(As, Bs)

m2 = gbm2sm(mgb)

@assert m1 == m2

s = sum(A, dims=2)
Ss = sm(As)

r1 = collect(m1) .÷ s
r2 = dmv(mgb, Ss)



m1 = (collect(A*B') .÷ sum(A, dims=2))

m2 = d3(As, Bs)
m2 = gbm2sm(m2)

@assert m1 == m2



B = sparse([0 1 0 1 0 0 0;
     1 0 0 1 1 0 1;
     0 0 0 1 0 1 1;
     1 1 1 0 0 1 1;
     0 1 0 0 0 1 1;
     0 0 1 1 1 0 0;
     0 1 1 1 1 0 0])

BS = sm2gbm(B)
V = sm(BS)
RES = dmv(BS, V)

println(collect(gbm2sm(RES)))

# ************************************************************************************** #

B = sparse([2 0 4;
            2 2 0;
            0 0 2])

VM = sparse([1 1;
             0 1;
             0 2])
Bs = sm2gbm(B)
VMs = sm2gbm(VM)
V = sm(VMs)

@GxB_fprint(V, GxB_COMPLETE)

R = dmv(Bs, V)

@assert gbm2sm(R) == [1 0 2; 2 2 0; 0 0 1]

# ************************************************************************************ #

randSparse(N) = sparse(randperm(N), randperm(N), rand(0:N, N), N, N)

A = randSparse(1000)
B = randSparse(1000)

function delta_3(M_2, M_3)
	s = sum(M_2,dims=2)
	d = (M_2 * M_3')
	res = d ./ s
	return res .÷ 1
end

m1 = delta_3(A, B)

As = sm2gbm(A)
Bs = sm2gbm(B)

m2 = d3(As, Bs)
m2 = gbm2sm(m2)

@assert m1 == m2
# ************************************************************************************ #

B = sparse([2 0 4;
            2 2 0;
            1 0 0])

VM = sparse([1 1;
             0 1;
             0 0])
Bs = sm2gbm(B)
VMs = sm2gbm(VM)
V = sm(VMs)

@GxB_fprint(V, GxB_COMPLETE)

R = dmv(Bs, V)

@assert gbm2sm(R) == [1 0 2; 2 2 0; 1 0 0]

# ************************************************************************************ #

B = sparse([2 0 4;
            2 2 0;
            1 0 0])


Bs = sm2gbm(B)
B_div_2 = div_by_two(Bs)

@assert  gbm2sm(B_div_2) == [1 0 2; 1 1 0; 0 0 0]


# ************************************************************************************ #
