using Pkg
Pkg.activate("./examples//NN_examples/")
using Revise, ModelingToolkit,PhysicsInformedRegression, OrdinaryDiffEqTsit5, Interpolations
using NeuralPDE, Lux, Random, OptimizationOptimJL, LineSearches, Plots, ReverseDiff, Optimization


## LOTKA VOLTERA
## LOTKA VOLTERA
@parameters α=1.5 β=1.0 γ=3.0 δ=1.0
@independent_variables t    
@variables x(t) y(t)
D = Differential(t)
eqs = [D(x) ~ α * x - β * x * y,
       D(y) ~ δ * x * y - γ * y]    

# Define the system
@mtkcompile sys = ODESystem(eqs, t, [x, y], [α, β, γ, δ])

tspan = (0.0, 5.0) |> f64
# u0 = [5.0, 5.0]
# prob = ODEProblem{false}(sys, merge(Dict(u0), Dict(p)), tspan)

u0 = [1.0, 1.0] 
p_true = [1.5, 1.0, 3.0, 1.0] |> f64    
odefunc = ODEFunction(sys)
prob  = ODEProblem{false}(odefunc, u0, tspan, p_true)
sol = solve(prob, Tsit5(), saveat = 0.1)
plot(sol)

using PhysicsInformedRegression 
du_dx = finite_diff(sol.u, sol.t)
A,b = setup_linear_system(sys)
Afun, bfun = PhysicsInformedRegression.create_callable_function(A, b, sys)
bfun(1.0,1.0,0.0)
params_est,mse = physics_informed_regression(sys, sol.u, du_dx, Afun, bfun; verbose = false)   

Afun(sol.u[1], du_dx[1], sol.t[1])
bfun(sol.u[1], du_dx[1], sol.t[1])

function loss_function(scalar)
    # println(typeof(scalar))
    du_approx = du_dx*first(scalar)
    # println(typeof(du_approx))*
    uvals = sol.u.*one(typeof(first(scalar)))
    # println(typeof(uvals))*
    params_est,mse = physics_informed_regression(sys, uvals, du_approx, 
            Afun, bfun; verbose = true)
    println(" I reach this MSE: ", mse)
    return mse
end
using Zygote  
using ForwardDiff   

p = [0.5]

# Test gradients using ForwardDiff, Zygote, and ReverseDiff  
gd_FD = ForwardDiff.gradient(loss_function, p)
# gd_Z = Zygote.gradient(loss_function, p)
gd_RD = ReverseDiff.gradient(loss_function, p)   


# Atest = rand(10,10)
# btest = rand(10)
# gd_Z_test = Zygote.gradient(x -> sum(x.*(Atest \ btest)), p)

# opt_algo = AutoReverseDiff()    
# optf = OptimizationFunction((x,p) -> loss_function(x), opt_algo)
# optprob = OptimizationProblem(optf, [0.5])


# optres = solve(optprob, ADAM(), maxiters = 200)