using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, DomainSets, Plots,PhysicsInformedRegression

@parameters C2
@independent_variables t x
@variables u(..)
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)


eq  = Dtt(u(t,x)) ~ C2*Dxx(u(t,x))

# Initial and boundary conditions
bcs = [u(t,0) ~ 0.,# for all t > 0. this is a Dirichlet BC
        u(t,1) ~ 0.,# for all t > 0. this is a Dirichlet BC
        u(0,x) ~ x*(1. - x), #for all 0 < x < 1. this is an initial condition
        Dt(u(0,x)) ~ 0. ] #for all  0 < x < 1]. This is an initial condition

# Space and time domains
domains = [t ∈ (0.0,1.0),
            x ∈ (0.0,1.0)]

@named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)], [C2], defaults = Dict(C2 => 1.0))


# Method of lines discretization
dx = 0.01
dt = 0.01
order = 4

discretization = MOLFiniteDifference([x => dx], t, approx_order = order)
# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)
sol = solve(prob, Tsit5(), saveat=dt)

discrete_x = sol[x]
discrete_t = sol[t]

solu = sol[u(t, x)]

plt = plot()

for i in eachindex(discrete_t)
    plot!(discrete_x, solu[i, :], label="t=$(discrete_t[i])", title = "Burgers Equation", lw = 2, dpi = 600)
    plot!(xlabel = "x", ylabel = "u(t,x)", legend = true)
end
plt

paramsest = physics_informed_regression(pdesys, sol; interp_fun = cubic_spline_interpolation, lambda = 0.0)
