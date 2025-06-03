using Pkg
Pkg.activate(@__DIR__)
using ModelingToolkit
using DifferentialEquations
using PhysicsInformedRegression

### SIR MODEL
@parameters β γ
@independent_variables t
@variables S(t) I(t) R(t)
D = Differential(t)
eqs = [D(S) ~ -β*S*I,
    D(I) ~ β*S*I - γ*I,
    D(R) ~ γ*I]

u0 = [S => 0.99,
    I => 0.01,
    R => 0.0]

p = [β => 0.5,
    γ => 1/3]

# Define the system
@named sys = ODESystem(eqs, t)
sys = structural_simplify(sys)

# Define the time span
start = 0
stop = 80
len = 80
timesteps = collect(range(start, stop, length = len))

# Simulate the system
prob = ODEProblem(sys, u0,(start,stop) ,p, saveat = timesteps)
sol = solve(prob)

# Compute the derivatives
du_finite_approx =  finite_diff(sol.u, sol.t)
#du_spline_approx =  spline_derivatives(sol.u, sol.t)

# Estimate the parameters
paramsest = physics_informed_regression(sys, sol.u, du_finite_approx)

#compare the estimated parameters to the true parameters
parameterdict = Dict(p)
for (i, param) in enumerate(parameters(sys))
    println("Parameter $(param) = $(parameterdict[param]) estimated as $(paramsest[param])")
end

# Plot the results
using Plots
estimated_sol = solve(ODEProblem(sys, u0,(start, stop) ,paramsest), Tsit5(), saveat = timesteps)
plot(sol, label = ["S" "I" "R"], title = "SIR Model", lw = 2, dpi = 600)
plot!(estimated_sol, label = ["S_est" "I_est" "R_est"], lw = 2, ls = :dash, dpi = 600)
savefig("plots/SIR.png")