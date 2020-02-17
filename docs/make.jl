push!(LOAD_PATH, "../src/")
using Documenter, LARMM, SparseMM

makedocs(
    format = Documenter.HTML(),
    sitename = "SparseMatrixMoltiplication",
    modules = [LARMM. SparseMM],
)
