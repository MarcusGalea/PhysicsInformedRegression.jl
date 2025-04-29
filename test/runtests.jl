using ModelingToolkit, DifferentialEquations,PhysicsInformedRegression
using Test

@testset "PhysicsInformedRegression" begin
    ## LORENZ ATTRACTOR
    @parameters σ ρ β
    @variables t x(t) y(t) z(t)
    D = Differential(t)

    eqs = [D(x) ~ σ * (y - x),
        D(y) ~ x * (ρ - z) - y,
        D(z) ~ x * y - β * z]

    # Define the system
    @named sys = ODESystem(eqs)

    # Define the initial conditions and parameters
    u0 = [D(x) => 2.0,
        x => 1.0,
        y => 0.0,
        z => 0.0]

    p = [σ => 10.0,
        ρ => 28.0,
        β => 8/3]
        # Define the time span
    timesteps = collect(0.0:0.01:10.0)

    # Simulate the system
    prob = ODEProblem(sys, u0,(timesteps[1], timesteps[end]) ,p, saveat = timesteps)
    sol = solve(prob)
    #test finite difference function
    du_approx = finite_diff(sol.u, sol.t)
    @test du_approx[1] ≈ [-8.207555137226253, 26.633998839581725, 0.1263896161024725] atol = 1e-1

    #test setup_linear_system function
    A,b = setup_linear_system(sys)
    @test size(A) == (3, 3)
    @test size(b) == (3,)

    #test physics_informed_regression function
    paramsest = PhysicsInformedRegression.physics_informed_regression(sys, sol.u, du_approx, A, b)
    #compare the estimated parameters to the true parameters
    parameterdict = Dict(p)
    for (i, param) in enumerate(parameters(sys))
        @test parameterdict[param] ≈ paramsest[param] atol = 1e-1
    end
end
