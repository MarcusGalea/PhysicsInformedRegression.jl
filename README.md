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
This package is intended as an extension to the [SciML](https://sciml.ai/) ecosystem, and is designed to be used in conjunction with [ModelingToolkit.jl](https://mtk.sciml.ai/dev/). The following example demonstrates how to use this package to estimate the parameters of the Lotka Volterra equations.
## Setting up model and data for regression
The first step is to define the symbolic model and generate some data using numeric integration.
```julia
using ModelingToolkit
using DifferentialEquations

## LOTKA VOLTERA
@parameters a b c d
@variables t x(t) y(t)
D = Differential(t)
eqs = [D(x) ~ a*x - b*x*y,
    D(y) ~ -c*y + d*x*y]


# Define the system
@named sys = ODESystem(eqs)

# Define the initial conditions and parameters
u0 = [x => 1.0,
    y => 1.0]

p = [a => 1.5,
    b => 1.0,
    c => 3.0,
    d => 1.0]

# Define the time span
start = 0; stop = 10; len = 1000 
timesteps = collect(range(start, stop, length = len))

# Simulate the system
prob = ODEProblem(sys, u0,(timesteps[1], timesteps[end]) ,p, saveat = timesteps)
sol = solve(prob)
sol.u
```
Which computes the solution curves at 1000 time steps
```
1000-element Vector{Vector{Float64}}:
 [1.0, 1.0]
 [1.0051174993735024, 0.9802039752764311]
 [1.0104591544379558, 0.9608501307635154]
 ⋮
 [1.0277257470932304, 0.9244136106150449]
 [1.0337581256020607, 0.9063703842886133]
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
And prints the parameters
```
Parameter a = 1.5 estimated as 1.4989786370553717
Parameter b = 1.0 estimated as 0.9991910171105692
Parameter d = 1.0 estimated as 0.9993763917081405
Parameter c = 3.0 estimated as 2.99943001337657
```
The results can also be visualised
```julia
using Plots
estimated_sol = solve(ODEProblem(sys, u0,(start, stop) ,paramsest), Tsit5(), saveat = timesteps)
plot(sol, label = ["x" "y"], title = "Lotka Volterra", lw = 2, dpi = 600)
plot!(estimated_sol, label = ["x_est" "y_est"], lw = 2, ls = :dash, dpi = 600)
```
which outputs
![Lotka Volterra](https://github.com/MarcusGalea/PhysicsInformedRegression.jl/blob/main/plots/lotka_volterra.png)


## Details of the regression
The regression method rewrites the ODE equations as a matrix equation. For the Lotka Volterra equations, that would be
```julia
using Latexify

# Setup model for regression
A,b = PhysicsInformedRegression.setup_linear_system(sys)
println(latexify(A)) 
println(latexify(b))
```
```math
\bm{A} =
\begin{align}
\left[
\begin{array}{cccc}
x\left( t \right) &  - x\left( t \right) y\left( t \right) & 0.0 & 0.0 \\
0.0 & 0.0 & x\left( t \right) y\left( t \right) &  - y\left( t \right) \\
\end{array}
\right]
\quad
\bm{b} = 
\left[
\begin{array}{c}
\frac{\mathrm{d} x\left( t \right)}{\mathrm{d}t} \\
\frac{\mathrm{d} y\left( t \right)}{\mathrm{d}t} \\
\end{array}
\right]
\end{align}
```
Assuming the system is linear in terms of the parameters, the matrix and vector are used to rewrite the ODE equations as
```math
\begin{align}
\bm{A} \cdot \begin{bmatrix} a \\ b \\ c \\ d \end{bmatrix} = \bm{b}
\end{align}
```

$\bm{A}$ and $\bm{b}$ are evaluated for each time step, which allows for the construction of an overdetermined system. The parameters are then computed in `physics_informed_regression` using ordinary least squares (Details can be found in the paper [PAPER_URL]()).

