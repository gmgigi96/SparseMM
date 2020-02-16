include("SparseMM.jl")

using LinearAlgebraicRepresentation
Lar = LinearAlgebraicRepresentation
using SparseArrays, DataStructures
using BenchmarkTools

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

M_0 = []
M_1 = []
M_2 = []
M_3 = []

function delta_3(M_2, M_3)
	s = sum(M_2,dims=2)
	d = (M_2 * M_3')
	res = d ./ s
	return res .÷ 1
end

function test_time_julia()
	global M_0, M_1, M_2, M_3

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
	d_1 = @btime ($M_0 * $M_1')


	println("Calcolo ∂_2...")
	d_2 = @btime(($M_1 * $M_2') .÷ 2)

	println("Calcolo ∂_3...")
	d_3 = @btime (delta_3($M_2, $M_3))

	# free memory
	M_0 = []
	M_1 = []
	M_2 = []
	M_3 = []
	println("Freeing memory...")
	GC.gc()

end

M0s  = []
M1s  = []
M1ts = []
M2s  = []
M2ts = []
M3ts = []

function test_time_sparse()
	global M_0, M_1, M_2, M_3, M0s, M1s, M1ts, M2s, M2ts, M3ts

	V, CV = Lar.cuboidGrid([30,20,10])
	#V,CV = random3cells([40,20,20],4_000)

	VV = [[v] for v=1:size(V,2)]
	FV = collect(Set{Array{Int64,1}}(vcat(map(CV2FV,CV)...)))
	EV = collect(Set{Array{Int64,1}}(vcat(map(CV2EV,CV)...)))

	M_0 = K(VV)
	M_1 = K(EV)
	M_2 = K(FV)
	M_3 = K(CV)


	M0s  = sm2gbm(M_0)
	M1s  = sm2gbm(M_1)
	M1ts = sm2gbm(sparse(M_1'))
	M2s  = sm2gbm(M_2)
	M2ts = sm2gbm(sparse(M_2'))
	M3ts = sm2gbm(sparse(M_3'))

	println("Calcolo ∂_1...")
	d1 = @btime mm($M0s, $M1ts)

	println("Calcolo ∂_2...")
	d2 = @btime d2($M1s, $M2ts)

	println("Calcolo ∂_3...")
	d3 = @btime d3($M2s, $M3ts)

	M_0 = []
	M_1 = []
	M_2 = []
	M_3 = []

	M0s  = []
	M1s  = []
	M1ts = []
	M2s  = []
	M2ts = []
	M3ts = []

	println("Freeing memory...")
	Gc.gc()
end


test_time_julia()
test_time_sparse()




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
