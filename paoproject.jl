using LinearAlgebraicRepresentation
Lar = LinearAlgebraicRepresentation
using SparseArrays, DataStructures
using Test

include("LARMM.jl")


# Dati random
function random3cells(shape,npoints)
	pointcloud = rand(3,npoints).*shape
	grid = DataStructures.DefaultDict{Array{Int,1},Int}(0)

	for k = 1:size(pointcloud,2)
		v = map(Int∘trunc,pointcloud[:,k])
		if grid[v] == 0 # do not exists
			grid[v] = 1
		else
			grid[v] += 1
		end
	end

	out = Array{Lar.Struct,1}()
	for (k,v) in grid
		V = k .+ [
		 0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0;
		 0.0  0.0  1.0  1.0  0.0  0.0  1.0  1.0;
		 0.0  1.0  0.0  1.0  0.0  1.0  0.0  1.0]
		cell = (V,[[1,2,3,4,5,6,7,8]])
		push!(out, Lar.Struct([cell]))
	end
	out = Lar.Struct( out )
	V,CV = Lar.struct2lar(out)
end

function CV2FV( v::Array{Int64} )
	faces = [
		[v[1], v[2], v[3], v[4]], [v[5], v[6], v[7], v[8]],
		[v[1], v[2], v[5], v[6]],	[v[3], v[4], v[7], v[8]],
		[v[1], v[3], v[5], v[7]], [v[2], v[4], v[6], v[8]]]
end

function CV2EV( v::Array{Int64} )
	edges = [
		[v[1], v[2]], [v[3], v[4]], [v[5], v[6]], [v[7], v[8]], [v[1], v[3]], [v[2], v[4]],
		[v[5], v[7]], [v[6], v[8]], [v[1], v[5]], [v[2], v[6]], [v[3], v[7]], [v[4], v[8]]]
end

function cuda_K( CV )
	I = vcat( [ [k for h in CV[k]] for k=1:length(CV) ]...)
	J = vcat(CV...)
	Vals = [1 for k=1:length(I)]
	return CuArrays.CUSPARSE.sparse(I,J,Vals)
end

function cpu_K( CV )
	I = vcat( [ [k for h in CV[k]] for k=1:length(CV) ]...)
	J = vcat(CV...)
	Vals = [1 for k=1:length(I)]
	return SparseArrays.sparse(I,J,Vals)
end

function K( CV )
	I = vcat( [ [k for h in CV[k]] for k=1:length(CV) ]...)
	J = vcat(CV...)
	Vals = [1 for k=1:length(I)]
	return SparseArrays.sparse(I,J,Vals)
end

function delta_1(A, B)
	return A * B'
end

function delta_2(A, B)
	return (A * B') .÷ 2
end

function delta_3(A, B)
	s = sum(A,dims=2)
	d = (A * B')
	res = d ./ s
	return res .÷ 1
end


# **************************************************************************************************************** #
# **************************************************************************************************************** #
# *************************************************** TEST ******************************************************* #
# **************************************************************************************************************** #
# **************************************************************************************************************** #

Vg, CVg = Lar.cuboidGrid([30,20,10])
#V,CV = random3cells([40,20,20],4_000)

VVg = [[v] for v=1:size(Vg,2)]
FVg = collect(Set{Array{Int64,1}}(vcat(map(CV2FV,CVg)...)))
EVg = collect(Set{Array{Int64,1}}(vcat(map(CV2EV,CVg)...)))

M_0g = K(VVg)
M_1g = K(EVg)
M_2g = K(FVg)
M_3g = K(CVg)

function test_time_julia()
	println("***** TEST TIME JULIA *****")

	V, CV = Lar.cuboidGrid([30,20,10])
	#V,CV = random3cells([40,20,20],4_000)

	VV = [[v] for v=1:size(V,2)]
	FV = collect(Set{Array{Int64,1}}(vcat(map(CV2FV,CV)...)))
	EV = collect(Set{Array{Int64,1}}(vcat(map(CV2EV,CV)...)))

	M_0 = K(VV)
	M_1 = K(EV)

	M_2 = K(FV)
	M_3 = K(CV)

	println("Calcolo ∂_1...")
	d_1 = @time delta_1(M_0, M_1)

	println("Calcolo ∂_2...")
	d_2 = @time delta_2(M_1, M_2)

	println("Calcolo ∂_3...")
	d_3 = @time delta_3(M_2, M_3)

	return true
end


function test_time_sparse()
	println("***** TEST TIME GBLIB *****")

	V, CV = Lar.cuboidGrid([30,20,10])
	#V,CV = random3cells([40,20,20],4_000)

	VV = [[v] for v=1:size(V,2)]
	FV = collect(Set{Array{Int64,1}}(vcat(map(CV2FV,CV)...)))
	EV = collect(Set{Array{Int64,1}}(vcat(map(CV2EV,CV)...)))

	M_0 = K(VV)
	M_1 = K(EV)
	M_2 = K(FV)
	M_3 = K(CV)

	println("Calcolo ∂_1...")
	d_1 = @time d1(M_0, M_1)

	println("Calcolo ∂_2...")
	d_2 = @time d2(M_1, M_2)

	println("Calcolo ∂_3...")
	d_3 = @time d3(M_2, M_3)

	return true
end

function d1_correctness()
	global M_0g, M_1g
	return d1(M_0g, M_1g)
end

function d2_correctness()
	global M_1g, M_2g
	return d2(M_1g, M_2g)
end

function d3_correctness()
	global M_2g, M_3g
	return d3(M_2g, M_3g)
end


@testset "LAR tests" begin
	@testset "time" begin
		@test test_time_julia() == true
		@test test_time_sparse() == true
	end
	@testset "correctness" begin
		@test d1_correctness() == delta_1(M_0g, M_1g)
		@test d2_correctness() == delta_2(M_1g, M_2g)
		@test d3_correctness() == delta_3(M_2g, M_3g)
	end
end

#S2 = sum(∂_3,dims=2)
#inner = [k for k=1:length(S2) if S2[k]==2]
#outer = setdiff(collect(1:length(FV)), inner)


# REMARK for the paper: VIP VIP VIP VIP VIP VIP VIP VIP VIP VIP VIP VIP
  #V - #E + #F = #holes + 1
#=
	for V,CV = Lar.cuboidGrid([30,20,10]) we get
	V - E + F = 7161 - 20260 + 19100 == 6001 (i.e. 6000 voxels + the exterior cell !!)
=#

#using ViewerGL
#GL = ViewerGL

#GL.VIEW([ GL.GLGrid(V,EV,GL.Point4d(1,1,1,1))
#         GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

#GL.VIEW([ GL.GLGrid(V,FV[inner],GL.Point4d(1,1,1,1))
#         GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);

#GL.VIEW([ GL.GLGrid(V,FV[outer],GL.Point4d(1,1,1,1))
#         GL.GLAxis(GL.Point3d(-1,-1,-1),GL.Point3d(1,1,1)) ]);
