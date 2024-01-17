using ModelingToolkit
using DifferentialEquations
using LinearAlgebra
using Interpolations
using PhysicsInformedRegression

## LORENZ ATTRACTOR
@parameters σ ρ β
@variables t x(t) y(t) z(t)
D = Differential(t)

eqs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

# Define the system
@named sys = ODESystem(eqs)
equations(sys)
sys = structural_simplify(sys)

# Define the initial conditions and parameters
u0 = [D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0]

p = [σ => 13.0,
    ρ => 14.0,
    β => 3.0]

# Define the time span
timesteps = collect(0.0:0.01:100.0)

# Simulate the system
prob = ODEProblem(sys, u0,(timesteps[1], timesteps[end]) ,p, saveat = timesteps)
sol = solve(prob)

# Compute the derivatives
du_approx = finite_diff(sol.u, sol.t)

# Setup model for regression
A,b = setup_linear_system(sys)
print("A = ")
display(A)
print("b = ")
display(b)

# Estimate the parameters
paramsest = physics_informed_regression(sys, sol.u, du_approx, A, b)

#compare the estimated parameters to the true parameters
parameterdict = Dict(p)
for (i, param) in enumerate(parameters(sys))
    println("Parameter $(param) = $(parameterdict[param]) estimated as $(paramsest[i])")
end
