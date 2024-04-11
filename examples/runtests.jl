using PhysicsInformedRegression
using Test

@testset "PhysicsInformedRegression.jl" begin
    # Write your tests here.
    # These tests will run automatically for control during push/pull requests.
    println("Testing lotka_volterra.jl")
    include("lotka_volterra.jl")
    @test paramsest[a] ≈ parameterdict[a] rtol = 1e-1
    @test paramsest[b] ≈ parameterdict[b] rtol = 1e-1
    @test paramsest[c] ≈ parameterdict[c] rtol = 1e-1
    @test paramsest[d] ≈ parameterdict[d] rtol = 1e-1
end
