module PhysicsInformedRegression

@info "Loading dependencies for PhysicsInformedRegression.jl..."
@info "Importing ModelingToolkit..."
using ModelingToolkit
@info "Importing DifferentialEquations..."
using DifferentialEquations
@info "Importing LinearAlgebra..."
using LinearAlgebra


export setup_linear_system, physics_informed_regression
include("regression_functions.jl")

@info "Importing Interpolations..."
using Interpolations
export spline_derivatives, finite_diff
include("derivative_functions.jl") 

end