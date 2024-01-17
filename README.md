# PhysicsInformedRegression

[![Build Status](https://github.com/MarcusGalea/PhysicsInformedRegression.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/MarcusGalea/PhysicsInformedRegression.jl/actions/workflows/CI.yml?query=branch%3Amaster)

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
using Latexify

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
timesteps = collect(0.0:0.01:100.0)

# Simulate the system
prob = ODEProblem(sys, u0,(timesteps[1], timesteps[end]) ,p, saveat = timesteps)
sol = solve(prob)
```
## using PhysicsInformedRegression to estimate the parameters
The inverse problem is solved fast using the following block of code
```julia
using PhysicsInformedRegression

# Compute the derivatives with finite differences
du_approx = PhysicsInformedRegression.finite_diff(sol.u, sol.t)

# Solve the inverse problem
paramsest = PhysicsInformedRegression.physics_informed_regression(sys, sol.u, du_approx)

#compare the estimated parameters to the true parameters
parameterdict = Dict(p)
for (i, param) in enumerate(parameters(sys))
    println("Parameter $(param) = $(parameterdict[param]) estimated as $(paramsest[i])")
end
```
Which outputs
```
Parameter σ = 13.0 estimated as 12.991401084075799
Parameter ρ = 14.0 estimated as 13.999745096023222
Parameter β = 3.0 estimated as 2.9993130127413283
```

## Details of the regression
The regression method rewrites the ODE equations as a matrix equation. For the Lorenz attractor, that would be
```julia
# Setup model for regression
A,b = PhysicsInformedRegression.setup_linear_system(sys)
latexify(A); latexify(b)
```
$$
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
$$
Assuming the system is linear in terms of the parameters, the matrix and vector are used to rewrite the ODE equations as
$$
\begin{align}
A \cdot \begin{bmatrix} \sigma \\ \rho \\ \beta \end{bmatrix} = b
\end{align}
$$
$A$ and $b$ are evaluated for each time step, which allows for the construction of an overdetermined system. This system is solved in `physics_informed_regression` using ordinary least squares (Details can be found in the paper [PAPER_URL]()).

