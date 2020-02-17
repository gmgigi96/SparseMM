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

sparseNM(N,M) = if N < M; sparse(randperm(N), randperm(N), ones(Int64, N), N, M); else
	sparse(randperm(M), randperm(M), ones(Int64, M), N, M); end

const A1r = sparseNM(3, 4)
const B1r = sparseNM(5, 4)

function delta_3(M_2, M_3)
	s = sum(M_2,dims=2)
	d = (M_2 * M_3')
	res = d ./ s
	res = res .÷ 1

	res = map(x -> if isnan(x) 0 else x end, res)

	return res
end

function test_d1_random_matrix_rect()
	global A1r, B1r
	return d1(A1r,B1r)
end

function test_d2_random_matrix_rect()
	global A1r, B1r
	return d2(A1r,B1r)
end

function test_d3_random_matrix_rect()
	global A1r, B1r
	return d3(A1r,B1r)
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

const A1r2 = sparse([2 0 4 1;
            		 2 2 0 2;
            		 0 0 2 0])

function test_dmv_small_matrix_rect()
	global A1r2, VM
	Bs = sm2gbm(A1r2)
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

randSparseNM(N,M) = if N < M; sparse(randperm(N), randperm(N), rand(0:N, N), N, M); else
	sparse(randperm(M), randperm(M), rand(0:N, M), N, M); end

const A2r = randSparseNM(100, 50)
const B2r = randSparseNM(100, 50)

function test_d1_random_values_rect()
	global A2r, B2r
	return d1(A2r,B2r)
end

function test_d2_random_values_rect()
	global A2r, B2r
	return d2(A2r,B2r)
end

function test_d3_random_values_rect()
	global A2r, B2r
	return d3(A2r,B2r)
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

const Mr = sparse([ 2 0 4 3;
            	    2 1 0 0;
            	    1 0 0 5])

function test_div_by_two_rect()
	global Mr
	Bs = sm2gbm(Mr)
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

@testset "Test" begin
	@testset "SparseMM squared matrices" begin
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
	@testset "SparseMM not squared matrices" begin
		@test test_d3_random_values_2() == delta_3(A3, B3)
		@test test_d1_random_matrix_rect() == A1r*B1r'
		@test test_d2_random_matrix_rect() == (A1r * B1r') .÷ 2
		@test test_d3_random_matrix_rect() == delta_3(A1r, B1r)
		@test test_dmv_small_matrix_rect() == [1 0 2 0; 2 2 0 2; 0 0 1 0]
		@test test_d1_random_values_rect() == A2r*B2r'
		@test test_d2_random_values_rect() == (A2r * B2r') .÷ 2
		@test test_d3_random_values_rect() == delta_3(A2r, B2r)
	end
end
