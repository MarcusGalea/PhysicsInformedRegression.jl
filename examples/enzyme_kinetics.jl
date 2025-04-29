using ModelingToolkit,DifferentialEquations,PhysicsInformedRegression,Catalyst

# Define the reaction network
model = @reaction_network begin
    k_a, E + S --> ES
    k_d, ES --> S + E
    k_c, ES --> E + P
end

# Generate ODEs from the reaction network
odesys = convert(ODESystem, model)

# simulate the ODEs
u₀ = [1.0, 1.0, 0.0, 0.0]
tspan = (0.0, 10.0)
p = [0.3, 0.2, 0.1]
prob = ODEProblem(odesys, u₀, tspan, p)
sol = solve(prob, Tsit5(), saveat=0.2)

#approximate derivatives
du_dt = PhysicsInformedRegression.finite_diff(sol.u, sol.t)

#estimate parameters
p_est = physics_informed_regression(odesys,sol.u,du_dt)

#compare the estimated parameters to the true parameters
using Plots
sol_est = solve(ODEProblem(odesys, u₀, tspan, p_est), Tsit5(), saveat=0.1)
plot(sol, title = "Enzyme Reaction", lw = 2, dpi = 600)
plot!(sol_est, label = ["E_est" "S_est" "ES_est" "P_est"], lw = 1, ls = :dash, dpi = 600)