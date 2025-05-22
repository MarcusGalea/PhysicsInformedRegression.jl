using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, DomainSets, Plots,PhysicsInformedRegression

@parameters C2
@independent_variables t x
@variables u(..)
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)

parameterdict = Dict(C2 => 1.0)

eq  = Dtt(u(t,x)) ~ C2*Dxx(u(t,x))

# Initial and boundary conditions
bcs = [u(t,0) ~ 0.,# for all t > 0. this is a Dirichlet BC
        u(t,1) ~ 0.,# for all t > 0. this is a Dirichlet BC
        u(0,x) ~ x*(1. - x), #for all 0 < x < 1. this is an initial condition
        Dt(u(0,x)) ~ 0. ] #for all  0 < x < 1]. This is an initial condition

# Space and time domains
domains = [t ∈ (0.0,1.0),
            x ∈ (0.0,1.0)]

@named pdesys = PDESystem(eq,bcs,domains,[t,x],[u(t,x)], [C2], defaults = parameterdict)


# Method of lines discretization
dx = 0.1
dt = 0.01
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
paramsest = physics_informed_regression(pdesys, sol; interp_fun = BSpline(Cubic(Line(OnGrid()))), lambda = 0.0)


neqs = length(equations(pdesys))
nparams = length(parameters(pdesys))
Afun = [eval(build_function(Atemp[i,j], _U, _dU, _ddU, expression=Val{false})) for i=1:neqs, j=1:nparams]
bfun = [eval(build_function(btemp[i], _U, _dU, _ddU, expression=Val{false})) for i=1:neqs]

neqs = length(equations(pdesys))
nparams = length(parameters(pdesys))
ndat = prod(size(states))
Atotal = Matrix{Any}(undef, neqs*ndat, nparams)
btotal = Vector{Any}(undef, neqs*ndat)
for (i,c) in enumerate(CartesianIndices(states))
    idx = (i-1)*neqs+1:i*neqs
    Atotal[idx,:] = [Afun[i,j](states[c], gradients[c], hessians[c]) for i=1:neqs, j=1:nparams]
    btotal[idx] = [bfun[i](states[c], gradients[c], hessians[c]) for i=1:neqs]
end


# Compare the estimated parameters to the true parameters
for (i, param) in enumerate(parameters(pdesys))
    println("Parameter $(param) = $(parameterdict[param]) estimated as $(paramsest[param])")
end



# ### Recreated parameters
# @named pdesys_recovered = PDESystem(eq,bcs,domains,[t,x],[u(t,x)], [C2], defaults = paramsest)
# prob_est = discretize(pdesys_recovered,discretization, p = paramsest)
# sol_est = solve(prob_est, Tsit5(), saveat=dt)


# anim = @animate for i in 1:length(discrete_t)
#     plot(discrete_x, solu[i, :], label="t=$(discrete_t[i])", title = "Burgers Equation", lw = 2, dpi = 600)
#     plot!(xlabel = "x", ylabel = "u(t,x)", legend = true)
#     plot!(discrete_x, sol_est[u(t, x)][i, :], label="est t=$(discrete_t[i])", lw = 2, ls = :dash, dpi = 600, ylims = (-0.3,0.3))
# end

# gif(anim, "plots/burgers_eq.gif", fps = 10)