# PhysicsInformedRegression

[![Build Status](https://github.com/MarcusGalea/PhysicsInformedRegression.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/MarcusGalea/PhysicsInformedRegression.jl/actions/workflows/CI.yml?query=branch%3Amaster)

This package provides a method for solving inverse problems using physics informed regression, and serves as an alternative to [DiffEqParamEstim.jl](https://docs.sciml.ai/DiffEqParamEstim/stable/). The advantage of `physics_informed_regression` is that it computes least squares estimates fast, compared to gradient based methods.
# Initial setup

```julia
using Pkg
Pkg.activate(".") # activate the current directory as the project environment.
Pkg.instantiate() # install the dependencies.

#The steps above will only need to be run once. 
#They change when the module is verified and added to the Julia registry.
```

# Walkthrough
This package is intended as an extension to the [SciML](https://sciml.ai/) ecosystem, and is designed to be used in conjunction with [ModelingToolkit.jl](https://mtk.sciml.ai/dev/). The following example demonstrates how to use this package to estimate the parameters of the Lorenz attractor.
## Setting up model and data for regression
The first step is to define the symbolic model and generate some data
```julia
using ModelingToolkit
using DifferentialEquations

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
timesteps = collect(0.0:0.001:100.0)

# Simulate the system
prob = ODEProblem(sys, u0,(timesteps[1], timesteps[end]) ,p, saveat = timesteps)
sol = solve(prob)
sol.u
```
Which outputs
```
100001-element Vector{Vector{Float64}}:
 [2.0, 0.0, 0.0, 1.0]
 [1.987078058301959, 0.014006965095118491, 7.004635254071749e-6, 1.0019935260145856]
 [1.974312464914167, 0.028027721536687127, 2.8036859965505627e-5, 1.00397420821669]
 ⋮
 [-35.19367350059237, -11.172904290010658, 18.92328345950416, -11.086333049515464]
 [-35.19414176028773, -11.10672575260858, 18.990109725284057, -11.121527127377504]
```
## using PhysicsInformedRegression to estimate the parameters
The inverse problem is solved fast using the following block of code
```julia
using PhysicsInformedRegression

# Compute the derivatives with finite differences
du_approx = finite_diff(sol.u, sol.t)

# Solve the inverse problem
paramsest = physics_informed_regression(sys, sol.u, du_approx)

#compare the estimated parameters to the true parameters
parameterdict = Dict(p)
for (i, param) in enumerate(parameters(sys))
    println("Parameter $(param) = $(parameterdict[param]) estimated as $(paramsest[param])")
end
```
Which outputs
```
Parameter σ = 13.0 estimated as 13.00028735839472
Parameter ρ = 14.0 estimated as 13.999688469148941
Parameter β = 3.0 estimated as 2.99933724874079
```

## Details of the regression
The regression method rewrites the ODE equations as a matrix equation. For the Lorenz attractor, that would be
```julia
using Latexify

# Setup model for regression
A,b = PhysicsInformedRegression.setup_linear_system(sys)
println(latexify(A)) 
println(latexify(b))
```
```math
A = \begin{align}
\left[
\begin{array}{ccc}
 - x\left( t \right) + y\left( t \right) & 0.0 & 0.0 \\
0.0 & x\left( t \right) & 0.0 \\
0.0 & 0.0 &  - z\left( t \right) \\
0.0 & 0.0 & 0.0 \\
\end{array}
\right]
\quad
b = 
\left[
\begin{array}{c}
\frac{\mathrm{d} xˍt\left( t \right)}{\mathrm{d}t} \\
\frac{\mathrm{d} y\left( t \right)}{\mathrm{d}t} + y\left( t \right) + x\left( t \right) z\left( t \right) \\
\frac{\mathrm{d} z\left( t \right)}{\mathrm{d}t} - x\left( t \right) y\left( t \right) \\
 - xˍt\left( t \right) + \frac{\mathrm{d} x\left( t \right)}{\mathrm{d}t} \\
\end{array}
\right]
\end{align}
```
Assuming the system is linear in terms of the parameters, the matrix and vector are used to rewrite the ODE equations as
```math
\begin{align}
A \cdot \begin{bmatrix} \sigma \\ \rho \\ \beta \end{bmatrix} = b
\end{align}
```

$A$ and $b$ are evaluated for each time step, which allows for the construction of an overdetermined system. The parameters are then computed in `physics_informed_regression` using ordinary least squares (Details can be found in the paper [PAPER_URL]()).

