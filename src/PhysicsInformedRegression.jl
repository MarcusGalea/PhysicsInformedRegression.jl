__precompile__(true)
module PhysicsInformedRegression

@info "Loading dependencies for PhysicsInformedRegression.jl..."
@info "Importing ModelingToolkit..."
using ModelingToolkit
@info "Importing DifferentialEquations..."
#using DifferentialEquations
@info "Importing LinearAlgebra..."
#using LinearAlgebra
#using LinearSolve

@info "Importing Interpolations..."
using Interpolations
export spline_derivatives, finite_diff
include("derivative_functions.jl") 

#using StaticArrays
export setup_linear_system, physics_informed_regression
include("regression_functions.jl")

using Interpolations,SciMLBase
#include PDE methods
include("pde_regression_functions.jl")
export physics_informed_regression
end