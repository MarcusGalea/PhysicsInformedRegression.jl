using ModelingToolkit, DifferentialEquations,PhysicsInformedRegression
using Test


pkgpath = dirname(dirname(pathof(PhysicsInformedRegression)))

@testset "PhysicsInformedRegression" begin
    @testset "Lorenz Attractor" begin
        include(joinpath(pkgpath, "examples", "lorenz_attractor.jl"))
        export paramsest, p
        # Compare the estimated parameters to the true parameters
        parameterdict = Dict(p)
        for (i, param) in enumerate(parameters(sys))
            @test parameterdict[param] ≈ paramsest[param] atol = 1e-1
        end
    end
end
