using ModelingToolkit
using DifferentialEquations
#using Interpolations
using PhysicsInformedRegression


## LOTKA VOLTERA
@independent_variables t
@parameters α β γ δ
@variables x(t) y(t)
D = Differential(t)
eqs = [D(x) ~ α*x - β*x*y,
    D(y) ~ -γ*y + δ*x*y]


# Define the system
@named sys = ODESystem(eqs, t)
sys = complete(sys)

# Define the initial conditions and parameters
u0 = [x => 1.0,
    y => 1.0]

p = Dict([α => 1.5,
    β => 1.0,
    γ => 3.0,
    δ => 0.5])

# Define the time span
start = 0; stop = 10; len = 1000
timesteps = collect(range(start, stop, length = len))

# Simulate the system
prob = ODEProblem(sys, u0,(timesteps[1], timesteps[end]) ,p, saveat = timesteps)
sol = solve(prob)


total_n_data_points = 100
parameter_estimates = Dict{Tuple, Vector}()
max_noise_level = 0.1
max_u_val = maximum(sol.u)
n_iter = 20 # Number of iterations for averaging the estimates
for noise in [0.0, 0.01, 0.05, 0.1]
    for n_data_points in [5,10,50,100]
        # Select a subset of the solution

        param_ests = zeros(length(parameters(sys)))
        for _ in 1:n_iter
            grid_step_size = ceil(Int, total_n_data_points / n_data_points)

            selected_t = sol.t[1:grid_step_size:end]
            selected_u = (hcat(sol.u[1:grid_step_size:end, :]...)+randn(length(u0),length(selected_t)).* max_u_val .* noise)'

            #reshape
            selected_u = [selected_u[i,:] for i in 1:size(selected_u,1)]

            # Compute the derivatives using spline interpolation
            du_approx = PhysicsInformedRegression.finite_diff(selected_u, selected_t)

            # Estimate the parameters
            paramsest = PhysicsInformedRegression.physics_informed_regression(sys, selected_u, du_approx)
            param_ests .+= [paramsest[param] for param in parameters(sys)]
        end
        param_ests ./= n_iter # Average the estimates over the 20 iterations

        #compute relative error for each parameter
        relative_errors = [abs.((param_ests[i] - p[parameters(sys)[i]]) / p[parameters(sys)[i]]) for i in 1:length(parameters(sys))]

        # Store the estimates
        parameter_estimates[(n_data_points,noise)] = relative_errors
    end
end


# Collect unique values for rows and columns
n_data_points_vals = sort(unique([k[1] for k in keys(parameter_estimates)]))
noise_vals = sort(unique([k[2] for k in keys(parameter_estimates)]))

# Build a matrix of relative errors for α
err_matrix = round.([parameter_estimates[(n, noise)][4] for n in n_data_points_vals, noise in noise_vals]*100, sigdigits=3)

# Create DataFrame
df = DataFrame(err_matrix, :auto)
rename!(df, Symbol.(string.("noise_", noise_vals)))
df.n_data_points = n_data_points_vals
select!(df, :n_data_points, Not(:n_data_points))  # Move n_data_points to first column


latexify(df; env = :table, booktabs = true, latex = false) |> print

#plot convergence of parameter estimates
using Plots
p1 = plot()



## NO NOISE


parameter_estimates = Dict{Int, Vector}()
total_n_data_points = 50
for n_data_points in 1:2:total_n_data_points
    # Select a subset of the solution
    grid_step_size = ceil(Int, total_n_data_points / n_data_points)

    selected_t = sol.t[1:grid_step_size:end]
    selected_u = (hcat(sol.u[1:grid_step_size:end, :]...))'

    #reshape
    selected_u = [selected_u[i,:] for i in 1:size(selected_u,1)]

    # Compute the derivatives using spline interpolation
    du_approx = PhysicsInformedRegression.finite_diff(selected_u, selected_t)

    # Estimate the parameters
    paramsest = PhysicsInformedRegression.physics_informed_regression(sys, selected_u, du_approx)
    # Store the estimates
    parameter_estimates[n_data_points] = [paramsest[param] for param in parameters(sys)]
end


using Plots
marker_shapes = [:circle, :square, :diamond, :utriangle]
p1 = plot()
parameter_names = [string(param) for param in parameters(sys)]
x_values = collect(keys(parameter_estimates))
parameter_values = hcat([collect(values(parameter_estimates[n_data_points])) for n_data_points in x_values]...)
scatter!(p1, x_values, parameter_values', label = hcat(parameter_names...), xlabel = "Number of Data Points", ylabel = "Parameter Estimates",
     title = "Convergence without noise", lw = 2, dpi = 1200, markershape = hcat(marker_shapes...), markersize = 6)
#plot horizontal lines for true parameter values
for (i, param) in enumerate(parameters(sys))
    plot!(p1, [1, total_n_data_points], repeat([p[param]], 2), label = "$(param) true value", ls = :dash, lw = 3, color = i)
end
p1
savefig("plots/lotka_volterra_convergence_no_noise.png")