using Pkg
Pkg.activate(@__DIR__)
using Revise,ModelingToolkit,OrdinaryDiffEqTsit5,PhysicsInformedRegression,Catalyst

# Define the reaction network
model = @reaction_network begin
    k_a, E + S --> ES
    k_d, ES --> S + E
    k_c, ES --> E + P
end

# Generate ODEs from the reaction network
odesys = complete(ode_model(model))

# simulate the ODEs
u₀ = Dict(:E => 1.0, :S => 1.0, :ES => 0.0, :P => 0.0)
tspan = (0.0, 10.0)
p = Dict(:k_a => 1.0, :k_d => 0.5, :k_c => 0.2)
prob = ODEProblem(odesys, merge(u₀,p), tspan)
sol = solve(prob, Tsit5(), saveat=0.1)

#approximate derivatives
du_dt = PhysicsInformedRegression.finite_diff(sol.u, sol.t)

#estimate parameters
p_est = physics_informed_regression(odesys,sol.u,du_dt)

#compare the estimated parameters to the true parameters
using Plots
sol_est = solve(ODEProblem(odesys, merge(u₀,p_est), tspan), Tsit5(), saveat=0.1)
plot(sol, title = "Enzyme Reaction", lw = 2, dpi = 600)
plot!(sol_est, label = ["E_est" "S_est" "ES_est" "P_est"], lw = 1, ls = :dash, dpi = 600) 