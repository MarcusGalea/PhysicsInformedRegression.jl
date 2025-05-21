using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets,PhysicsInformedRegression
# Method of Manufactured Solutions: exact solution
u_exact = (x,t) -> exp.(-t) * cos.(x)

# Parameters, variables, and derivatives
@parameters α
@independent_variables t x
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2

# 1D PDE and boundary conditions
eq  = Dt(u(t, x)) ~ α * Dxx(u(t, x))
bcs = [u(0, x) ~ cos(x),
        u(t, 0) ~ exp(-t),
        u(t, 1) ~ exp(-t) * cos(1)]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0)]

# PDE system
@named pdesys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)], [α]; defaults = Dict(α => 1.0))


# Method of lines discretization
dx = 0.1
order = 2
discretization = MOLFiniteDifference([x => dx], t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)

# Solve ODE problem
sol = solve(prob, Tsit5(), saveat=0.1)

import Interpolations:cubic_spline_interpolation

paramsest = physics_informed_regression(pdesys, sol; interp_fun = cubic_spline_interpolation, lambda = 0.0)
