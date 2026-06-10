
using Pkg
Pkg.activate("./examples//NN_examples/")
using Revise, ModelingToolkit, OrdinaryDiffEqTsit5, Lux, Random, ReverseDiff
using OptimizationOptimJL, Plots, Optimization
## LOTKA VOLTERA


seed = 1234
rng = MersenneTwister(seed)
rbf(x) = exp.(-(x.^2))  
NN = Lux.Chain(
    Lux.Dense(2,5,rbf), Lux.Dense(5,5, rbf), Lux.Dense(5,5, rbf), Lux.Dense(5,2)
)
p_nn, _ = Lux.setup(rng, NN)
U_fun = (x,y)-> LuxCore.stateless_apply(NN, [x,y], p_nn)


@parameters a b c d
@independent_variables t
@variables x(t) y(t)
D = Differential(t)
@parameters (U::Function)(..)[1:2] [tunable = false]
eqs = [D(x) ~ a*x + U(x,y)[1],
    D(y) ~ -c*y + U(x,y)[2]]


# Define the system
@mtkcompile  sys = ODESystem(eqs, t)


# Define the initial conditions and parameters
u0 = Dict([x => 1.0,
    y => 1.0])


params = Dict([
    a => 1.5,
    # b => 1.0,
    c => 3.0,
    # d => 1.0
    U => U_fun
])  

# Define the time span
start = 0; stop = 10; len = 1000 
timesteps = collect(range(start, stop, length = len))

# Simulate the system
prob = ODEProblem(sys, merge(u0, params), (timesteps[1], timesteps[end]), saveat = timesteps)
sol = solve(prob, Tsit5())  

import PhysicsInformedRegression as PIR
du_dx = PIR.finite_diff(sol.u, sol.t)
A, b = PIR.setup_linear_system(sys)
Afun, bfun = PIR.create_callable_function(A, b, sys)

    iv = ModelingToolkitBase.get_iv(sys)
    non_tunable_params = [param for param in parameters(sys) if !ModelingToolkitBase.istunable(param)] 
    Atemp = Symbolics.diff2term.(A)
    btemp = Symbolics.diff2term.(b)

    Dt = Differential(iv)
    vars = ModelingToolkitBase.get_unknowns(sys)
    diffvars = Symbolics.diff2term.(Dt.(vars))

bfun(1.0,1.0,0.0, U_fun)
params_est,mse = PIR.physics_informed_regression(sys, sol.u, du_dx, Afun, bfun, U_fun; verbose = true) 


# ts = collect(0.0:0.1:10.0)
# spline = LinearInterpolation(ts .^ 2, ts)
# Tspline = typeof(spline)
# @variables x(t)
# @parameters (interp::Tspline)(..)

# @mtkcompile sys = System(D(x) ~ interp(t), t)
# prob = ODEProblem(sys, [x => 0.0, interp => spline], (0.0, 1.0))
