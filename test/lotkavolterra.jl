using ModelingToolkit
using DifferentialEquations
using LinearAlgebra
using Interpolations
using PhysicsInformedRegression


## LOTKA VOLTERA
@parameters a b c d
@variables t x(t) y(t)
D = Differential(t)
eqs = [D(x) ~ a*x - b*x*y,
    D(y) ~ -c*y + d*x*y]


# Define the system
@named sys = ODESystem(eqs)
equations(sys)
sys = structural_simplify(sys)

# Define the initial conditions and parameters
u0 = [x => 1.0,
    y => 1.0]

p = [a => 1.5,
    b => 1.0,
    c => 3.0,
    d => 1.0]

# Define the time span
start = 0; stop = 10; len = 100
timesteps = collect(range(start, stop, length = len))

# Simulate the system
prob = ODEProblem(sys, u0,(timesteps[1], timesteps[end]) ,p, saveat = timesteps)
sol = solve(prob)

# Compute the derivatives
du_finite_approx =  finite_diff(sol.u, sol.t)
#du_spline_approx = spline_derivatives(sol.u, sol.t)

# Estimate the parameters
paramsest = physics_informed_regression(sys, sol.u, du_finite_approx)

#compare the estimated parameters to the true parameters
parameterdict = Dict(p)
for (i, param) in enumerate(parameters(sys))
    println("Parameter $(param) = $(parameterdict[param]) estimated as $(paramsest[i])")
end
