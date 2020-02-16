using Random
using Test

include("LARMM.jl")

# ************************************************************************************** #

sparseN(N) = sparse(randperm(N), randperm(N), ones(Int64, N), N, N)

const A = sparseN(1000)
const B = sparseN(1000)

function delta_3(M_2, M_3)
	s = sum(M_2,dims=2)
	d = (M_2 * M_3')
	res = d ./ s
	res = res .÷ 1

	res = map(x -> if isnan(x) 0 else x end, res)

	return res
end

function test_d1_random_matrix()
	global A, B
	return d1(A,B)
end

function test_d2_random_matrix()
	global A, B
	return d2(A,B)
end

function test_d3_random_matrix()
	global A, B
	return d3(A,B)
end

# ************************************************************************************** #
const A1 = sparse([	2 0 4;
            		2 2 0;
            		0 0 2])
const VM = sparse([	1 1;
             		0 1;
             		0 2])

function test_dmv_small_matrix()
	global A1, VM
	Bs = sm2gbm(A1)
	VMs = sm2gbm(VM)
	R = dmv(Bs, sm(VMs))
	return gbm2sm(R)
end

# ************************************************************************************ #

function test_dmv_small_matrix()
	global A1, VM
	Bs = sm2gbm(A1)
	VMs = sm2gbm(VM)
	R = dmv(Bs, sm(VMs))
	return gbm2sm(R)
end
# ************************************************************************************ #

randSparse(N) = sparse(randperm(N), randperm(N), rand(0:N, N), N, N)

const A2 = randSparse(1000)
const B2 = randSparse(1000)

function test_d1_random_values()
	global A2, B2
	return d1(A2,B2)
end

function test_d2_random_values()
	global A2, B2
	return d2(A2,B2)
end

function test_d3_random_values()
	global A2, B2
	return d3(A2,B2)
end

# ************************************************************************************ #
const M = sparse([ 2 0 4;
            	   2 1 0;
            	   1 0 0])

function test_div_by_two()
	global M
	Bs = sm2gbm(M)
	B_div_2 = div_by_two(Bs)
	return gbm2sm(B_div_2)
end

# ************************************************************************************ #
randSparse(N) = sparse(randperm(N), randperm(N), rand(0:N, N), N, N)

const A3 = sparse([0 3 0; 2 0 0; 0 0 0])
const B3 = sparse([0 0 0; 0 1 0; 0 0 0])

function test_d3_random_values_2()
	global A3, B3
	return d3(A3,B3)
end

@testset "SparseMM test" begin
	@test test_d3_random_values_2() == delta_3(A3, B3)
	@test test_d1_random_matrix() == A*B'
	@test test_d2_random_matrix() == (A * B') .÷ 2
	@test test_d3_random_matrix() == delta_3(A, B)
	@test test_dmv_small_matrix() == [1 0 2; 2 2 0; 0 0 1]
	@test test_d1_random_values() == A2*B2'
	@test test_d2_random_values() == (A2 * B2') .÷ 2
	@test test_d3_random_values() == delta_3(A2, B2)
	@test test_div_by_two() == [1 0 2; 1 0 0; 0 0 0]
end
