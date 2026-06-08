using Pkg
Pkg.activate("./examples//NN_examples/")
using Revise, ModelingToolkit,PhysicsInformedRegression, OrdinaryDiffEqTsit5, Interpolations
using NeuralPDE, Lux, Random, Printf, OptimizationOptimJL, LineSearches, ComponentArrays, Plots, ReverseDiff, Optimization

include("Neural_PDE_methods.jl")
## LOTKA VOLTERA
## LOTKA VOLTERA
@parameters α=1.5 β=1.0 γ=3.0 δ=1.0
@independent_variables t    
@variables x(t) y(t)
D = Differential(t)
eqs = [D(x) ~ α * x - β * x * y,
       D(y) ~ δ * x * y - γ * y]    

# Define the system
@named sys = ODESystem(eqs, t, [x, y], [α, β, γ, δ])
sys = complete(sys)

tspan = (0.0, 5.0) |> f64
# u0 = [5.0, 5.0]
# prob = ODEProblem{false}(sys, merge(Dict(u0), Dict(p)), tspan)

u0 = [1.0, 1.0] 
p_guess = [1.0, 1.0, 1.0, 1.0]
odefunc = ODEFunction(sys)
prob  = ODEProblem{false}(odefunc, u0, tspan, p_guess)



p_true = [1.5, 1.0, 3.0, 1.0] |> f64
prob_data = remake(prob, p = p_true)
sol_data = solve(prob_data, Tsit5(), saveat = 0.1)
t_ = sol_data.t
u_ = reduce(hcat, sol_data.u)


# PINN
rng = Random.default_rng()
Random.seed!(rng, 0)
n = 15
chain = Chain(Dense(1, n, σ), Dense(n, n, σ), Dense(n, n, σ), Dense(n, 2))
ps, st = Lux.setup(rng, chain) |> f64


callback = function (p, l)
        @printf("[NNODE]\tIter: [%*d]\tLoss: %g\n", p.iter, l)
return false
end

additional_loss(phi, θ) = sum(abs2, phi(t_, θ) .- u_) / size(u_, 2)
# opt = LBFGS(linesearch = BackTracking())
using OptimizationOptimisers
opt = ADAM()
n_collocation_points = 500
alg = NNODE(chain, opt, ps; strategy = WeightedIntervalTraining([0.2, 0.6, 0.2], n_collocation_points),
    param_estim = true, additional_loss)#, callback = callback)

# sol = solve(prob, alg, verbose = true, abstol = 1e-8, maxiters = 2000, saveat = t_)
 
using PhysicsInformedRegression

# Compute the derivatives
# du_approx =  PhysicsInformedRegression.spline_derivatives(sol.u, sol.t)

A, b = PhysicsInformedRegression.setup_linear_system(sys)
Afun, bfun = PhysicsInformedRegression.create_callable_function(A, b, sys)
import NeuralPDE: inner_loss
function NeuralPDE.inner_loss(
        phi::NeuralPDE.ODEPhi, f, autodiff::Bool, t::AbstractVector, θ, p, param_estim::Bool
    )
    # p_ = param_estim ? θ.p : p
    nn_out = phi(t, θ)
    dnn_out_dt = NeuralPDE.ode_dfdx(phi, t, θ, false)
    p_est, mse = PhysicsInformedRegression.physics_informed_regression(sys, nn_out, dnn_out_dt, Afun, bfun; lambda = 0.0)
    return mse
end


maxiters = 2000 
verbose = true

total_loss, init_params = generate_total_loss(prob, alg; dt = 0.01)

opt_algo = AutoReverseDiff()
optf = OptimizationFunction(total_loss, opt_algo)

plen = maxiters === nothing ? 6 : ndigits(maxiters)
callback = function (p, l)
    if verbose
        if maxiters === nothing
            @printf("[NNODE]\tIter: [%*d]\tLoss: %g\n", plen, p.iter, l)
        else
            @printf("[NNODE]\tIter: [%*d/%d]\tLoss: %g\n", plen, p.iter, maxiters, l)
        end
    end
    return false
end

optprob = OptimizationProblem(optf, init_params)
res = solve(optprob, opt; callback, maxiters, alg.kwargs...)




# sol_PIR = solve(prob, alg, verbose = true, abstol = 1e-8, maxiters = 2000, saveat = t_)


# (; u0, tspan, f, p) = prob
# t0 = tspan[1]
# # add estim_collocate, dataset (or nothing) in NNODE


# phi, init_params = NeuralPDE.generate_phi_θ(chain, t0, u0, alg.init_params)
# θ = ComponentArray(; depvar = init_params, prob.p)

# nn_out = phi(sol.t, θ)
# dnn_out_dt = NeuralPDE.ode_dfdx(phi, sol.t, θ, false)


# p_est, mse = PhysicsInformedRegression.physics_informed_regression(sys, nn_out, dnn_out_dt, Afun, bfun; lambda = 0.0)






# nn_out = phi(t, θ)
# umatrix = hcat(sol.u...)
# du_approx = hcat(PhysicsInformedRegression.spline_derivatives(sol.u, sol.t)...)
# paramsest,mse = PhysicsInformedRegression.physics_informed_regression(sys, umatrix, du_approx, Afun, bfun; lambda = 0.0)
