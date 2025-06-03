using Pkg
Pkg.activate(@__DIR__)
using ModelingToolkit, DifferentialEquations,PhysicsInformedRegression

## LORENZ ATTRACTOR
@parameters σ ρ β
@independent_variables t
@variables t x(t) y(t) z(t)
D = Differential(t)

eqs = [D(x) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

# Define the system
@named sys = ODESystem(eqs, t)
sys = complete(sys)


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

# Compute the derivatives
du_approx = finite_diff(sol.u, sol.t)

# Setup model for regression
A,b = setup_linear_system(sys)

# Estimate the parameters
paramsest = PhysicsInformedRegression.physics_informed_regression(sys, sol.u, du_approx, A, b)

#compare the estimated parameters to the true parameters
parameterdict = Dict(p)
for (i, param) in enumerate(parameters(sys))
    println("Parameter $(param) = $(parameterdict[param]) estimated as $(paramsest[param])")
end

# Plot the results
# using Plots

# sol_est = solve(ODEProblem(sys, u0,(timesteps[1], timesteps[end]) ,paramsest), Tsit5(), saveat = timesteps)
# plot(sol,label = "True", title = "Lorenz Attractor", lw = 2, dpi = 600, idxs = (1,2,3))
# plot!(sol_est, label = "Estimated", lw = 1, ls = :dash, dpi = 600, idxs = (1,2,3))
# savefig("plots/Lorenz.png")


