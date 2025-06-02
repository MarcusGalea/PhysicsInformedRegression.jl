using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, DomainSets, Plots,PhysicsInformedRegression

@parameters ν
@independent_variables t x
@variables u(..)
Dx = Differential(x)
Dxx = Differential(x)^2
Dt = Differential(t)

parameterdict = Dict(ν => 1.0) #ground truth

eq  = Dt(u(t,x)) + u(t,x)*Dx(u(t,x))~ ν*Dxx(u(t,x))

# Initial and boundary conditions
bcs = [u(0,x) ~ -sin(π*x), # initial condition (PINN PAPER)
        u(t,-1) ~ 0.0, # left boundary condition
        u(t,1) ~ 0.0] # right boundary condition
# Note: The initial condition is a sine wave, which is a common choice for testing the Burgers' equation.

       
        

# Space and time domains
domains = [t ∈ (0.0,1.0),
            x ∈ (-1.0,1.0)]
            

@named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)], [ν], defaults = parameterdict)


# Method of lines discretization
dx = 0.1
dt = 0.1
order = 2

discretization = MOLFiniteDifference([x => dx], t, approx_order = order)
# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization, p = parameterdict)
sol = solve(prob, Tsit5(), saveat=dt)

discrete_x = sol[x]
discrete_t = sol[t]

solu = sol[u(t, x)]


### Parameter estimation
using Interpolations
#ridge coefficient
λ = 0.0
paramsest = physics_informed_regression(pdesys, sol; interp_fun = BSpline(Cubic(Line(OnGrid()))), lambda = λ)


dom = [sol[iv] for iv in sol.ivs]
redef_dom = PhysicsInformedRegression.uniform_domain(dom)

# Compare the estimated parameters to the true parameters
for (i, param) in enumerate(parameters(pdesys))
    println("Parameter $(param) = $(parameterdict[param]) estimated as $(paramsest[param])")
end



### Recreated parameters
@named pdesys_recovered = PDESystem(eq,bcs,domains,[t,x],[u(t,x)], [ν], defaults = paramsest)
prob_est = discretize(pdesys_recovered,discretization, p = paramsest)
sol_est = solve(prob_est, Tsit5(), saveat=dt)


# anim = @animate for i in 1:length(discrete_t)
#     plot(discrete_x, solu[i, :], label="t=$(discrete_t[i])", title = "Burgers Equation", lw = 2, dpi = 600)
#     plot!(xlabel = "x", ylabel = "u(t,x)", legend = true)
#     plot!(discrete_x, sol_est[u(t, x)][i, :], label="est t=$(discrete_t[i])", lw = 2, ls = :dash, dpi = 600, ylims = (-0.3,0.3))
# end

p1 = plot()
for i in 1:length(discrete_t)
    plot!(p1, discrete_x, solu[i, :], label="t=$(discrete_t[i])", title = "Burgers Equation", lw = 2, dpi = 600)
    plot!(p1, xlabel = "x", ylabel = "u(t,x)", legend = true)
    plot!(p1, discrete_x, sol_est[u(t, x)][i, :], label="est t=$(discrete_t[i])", lw = 2, ls = :dash, dpi = 600, ylims = (-0.3,0.3))
end
p1
# gif(anim, "plots/burgers_eq.gif", fps = 10)