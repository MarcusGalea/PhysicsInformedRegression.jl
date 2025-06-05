using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets,PhysicsInformedRegression
# Method of Manufactured Solutions: exact solution
u_exact = (t,x) -> exp.(-t) * cos.(x)

# Parameters, variables, and derivatives
@parameters α
@independent_variables t x
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2

parameterdict = Dict(α => 10.0)


# 1D PDE and boundary conditions
eq  = Dt(u(t, x)) ~ α * Dxx(u(t, x))
bcs = [u(0, x) ~ cos(x),
        u(t, 0) ~ exp(-t),
        u(t, 1) ~ exp(-t) * cos(1)]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0)]

# PDE system
@named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)], [α]; defaults = parameterdict)


# Method of lines discretization
dx = 0.01
dt = 0.01
order = 4
discretization = MOLFiniteDifference([x => dx], t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)

# Solve ODE problem
sol = solve(prob, Tsit5(), saveat=dt)

import Interpolations:cubic_spline_interpolation

### PARAMETER ESTIMATION
import PhysicsInformedRegression:physics_informed_regression
using Interpolations
paramsest = physics_informed_regression(pdesys, sol; interp_fun = BSpline(Cubic(Line(OnGrid()))), lambda = 0.0)

# Compare the estimated parameters to the true parameters
for (i, param) in enumerate(parameters(pdesys))
    println("Parameter $(param) = $(parameterdict[param]) estimated as $(paramsest[param])")
end

#### simulate retrieved parameters
@named pdesys_retrieved = PDESystem(eq, bcs, domains, [t, x], [u(t, x)], [α]; defaults = paramsest)

# Convert the PDE problem into an ODE problem
prob_est = discretize(pdesys_retrieved,discretization)

# Solve ODE problem
sol_est = solve(prob_est, Tsit5(), saveat=dt)
#### Check results

plt = plot()

for i in eachindex(sol[t])
    plot!(sol[t], sol[u(t,x)][i, :], label="t=$(discrete_t[i]) true")
    plot!(sol_est[t], sol_est[u(t,x)][i, :], label="t=$(discrete_t[i]) est", ls=:dash, lw=2)
    plot!(xlabel="x", ylabel="u(t,x)", title="heat equation", legend=:topright)
end
plt
# savefig(plt, "plots/heat_eq.png")
#save in plots folder

# A,b = setup_linear_system(pdesys)

# #get the independent and dependent variables
# ivs =  pdesys.ivs
# dvs = pdesys.dvs
# #get the domain of the system
# dom = sol.ivdomain

# @variables _U[1:length(dvs)] _dU[1:length(dvs),1:length(ivs)] _ddU[1:length(dvs), 1:length(ivs), 1:length(ivs)]


# dom = uniform_domain(dom) #check if the domain is uniformly spaced 

# #compute the gradient
# u_gradient_exact =  (t,x) -> [-exp(-t) *cos(x), -exp(-t) * sin(x)]


# #evaluate the exact gradients and hessians
# gradients_exact = [u_gradient_exact(x,t) for x in collect(dom[2]), t in collect(dom[1])]

# u_hessian_exact = (t,x) -> [[exp(-t) * cos(x), exp(-t) * sin(x) ]; [exp(-t) * sin(x), -exp(-t) * cos(x)]]
# hessian_exact = [u_hessian_exact(x,t) for x in collect(dom[2]), t in collect(dom[1])]



# state_maps, gradient_maps, hessian_maps = symbolic_maps(A, b, _U, _dU, _ddU, ivs, dvs)
# states, gradients_approx, hessians_approx = compute_gradients_hessians(sol, dvs, ivs, dom, gradient_maps, hessian_maps; interp_fun = cubic_spline_interpolation)

# #get the squared error between the exact and approximate gradients
# for (i,c) in enumerate(CartesianIndices(gradients_exact))
#     println("Gradient $(i) = $(gradients_exact[c]) estimated as $(gradients_approx[c])")
#     println("Squared error = $(sum((gradients_exact[c] .- gradients_approx[c]).^2))")
# end

# t_range, x_range = dom
# uvals = [u_exact(t, x) for t in collect(t_range), x in collect(x_range)]
# itp = interpolate(uvals, interp_fun)

# sitp = scale(itp, t_range, x_range)
# hessian_approx_2 = [Interpolations.hessian(sitp, t, x) for t in collect(t_range), x in collect(x_range)]


