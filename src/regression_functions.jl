import ModelingToolkitBase:t_nounits as t
"""
This function is used to setup the linear system for the regression problem. It returns the matrix A and the vector b, such that A*x = b, where x is the vector of parameters.\n
    $(TYPEDSIGNATURES)
# Arguments \n
- `sys`: The system of equations to be solved for the parameters (ODESystem or DAESystem)
# Returns \n
- `A`: The symbolic matrix A in the equation A*x = b
- `b`: The symbolic vector b in the equation A*x = b
"""
function setup_linear_system(sys::T) where T<:Union{ModelingToolkitBase.System, ModelingToolkitBase.PDESystem}
    # Get the equations and parameters
    eqs = equations(sys)
    params = parameters(sys)
    # Initialize an empty matrix with zeros
    A = Array{Any}(undef, length(eqs), length(params)) #{Union{SymbolicUtils.BasicSymbolic,AbstractFloat}} for more efficiency, but doesn't work with dual numbers
    b = Array{Any}(undef, length(eqs)) #{Union{SymbolicUtils.BasicSymbolic,AbstractFloat}} for more efficiency, but doesn't work with dual numbers
    # Fill the matrix
    for (i, eq) in enumerate(eqs)
        isolated_expr = Symbolics.simplify(eq.rhs - eq.lhs, expand = true)
        for (j, param) in enumerate(params)

            # Fix non-tunable parameters with default values
            if !ModelingToolkitBase.istunable(param) && ModelingToolkitBase.hasdefault(param)
                param_value = ModelingToolkitBase.getdefault(param)
                #substitute the default value of the parameter into the equation
                isolated_expr = Symbolics.simplify(substitute(isolated_expr, param => param_value), expand = true)
                A[i, j] = 0.0
                continue
            end   

            # Get the coefficient of the parameter in the equation
            coeff = Symbolics.coeff(isolated_expr, param)
            # Set the matrix element
            A[i, j] = isequal(coeff, 0//1) ? 0.0 : coeff
            isolated_expr = Symbolics.simplify(isolated_expr-coeff * param, expand = true)
            @assert isequal(Symbolics.degree(isolated_expr, param), 0) "Error: parameter $param not isolated in equation $eq. Ensure model is linear in terms of parameters."
        end
        b[i] = -isolated_expr
    end
    return A, b
end
"""
This function is used to solve the regression problem. It returns the vector of parameters.\n
    $(TYPEDSIGNATURES)
# Arguments \n
- `sys`: The system of equations to be solved for the parameters (ODESystem or DAESystem)
- `u`: The vector of states
- `du`: The vector of derivatives
- `A`: The symbolic matrix A in the equation A*x = b
- `b`: The symbolic vector b in the equation A*x = b
# Returns \n
- `paramsest`: The vector of estimated parameters
"""
function physics_informed_regression(sys::ModelingToolkitBase.System, u::Array, du::Array; verbose = false, kwargs...)
    A,b = setup_linear_system(sys)
    if verbose
        println("The linear system is setup with")
        println("A = ")
        display(A)
        println("b = ")
        display(b)
    end
    Afun, bfun = create_callable_function(A, b, sys)
    paramsest = physics_informed_regression(sys, u, du, Afun, bfun; kwargs...)
    return paramsest
end

# function physics_informed_regression(sys::ModelingToolkitBase.System,
#                                     u:: Array,
#                                     du::Array,
#                                     A,
#                                     b;
#                                     kwargs...)
#     #Convert u and du to vectors if they are not already
#     n,m = size(u)
#     if m == length(unknowns(sys))
#         if !(u isa Vector)
#             #convert Matrix to vector of vector
#             u = [u[i,:] for i in 1:size(u,1)]
#         end
#         if !(du isa Vector)
#             #convert Matrix to vector of vector
#             du = [du[i,:] for i in 1:size(du,1)]
#         end
#     elseif n == length(unknowns(sys))
#         if !(u isa Vector)
#             #convert Matrix to vector of vector
#             u = [u[i,:] for i in 1:size(u,2)]
#         end
#         if !(du isa Vector)
#             #convert Matrix to vector of vector
#             du = [du[i,:] for i in 1:size(du,2)]
#         end 
#     else
#         @error "Error: u and du must be either vectors of states and derivatives, or matrices with dimensions (n, m) where n is the number of time points and m is the number of states. Currently, size(u) = $(size(u)) and size(du) = $(size(du))."
#     end
#     return physics_informed_regression(sys, u, du, A, b; kwargs...)
# end

Symbolics.diff2term(x::Float64) = x #if the expression is a number, return it as is

"""
This function is used to create callable functions from the symbolic matrix A and vector b. 
    It returns two functions, Afun and bfun, which can be called with the states, derivatives, and time 
    (if time is used in the system of equations) to get the numerical values of A and b.\n
    $(TYPEDSIGNATURES)
"""
function create_callable_function(A, b, sys::ModelingToolkitBase.System)
    iv = ModelingToolkitBase.get_iv(sys)
    Atemp = Symbolics.diff2term.(A)
    btemp = Symbolics.diff2term.(b)

    Dt = Differential(iv)
    vars = ModelingToolkitBase.get_unknowns(sys)
    diffvars = Symbolics.diff2term.(Dt.(vars))
    Afun = first(eval(build_function(Atemp, vars, diffvars, iv, expression = Val{false})))
    bfun = first(eval(build_function(btemp, vars, diffvars, iv, expression = Val{false})))
    return Afun, bfun
end

function create_callable_function(sys::ModelingToolkitBase.System)
    A, b = setup_linear_system(sys)
    return create_callable_function(A, b, sys)
end

function physics_informed_regression(sys::ModelingToolkitBase.System, 
                                        u::Any, 
                                        du::Any, 
                                        Afun, 
                                        bfun; 
                                        lambda = 0.0,
                                        verbose = false,
                                        tvals = nothing, #only needed if not autonomous system,
                                        # alg = LinearSolve.DefaultLinearSolver,
                                        )
    # Define the variables and derivatives as indexed symbols

    neqs = length(equations(sys))
    ndat = u isa Vector ? length(u) : size(u, 2)
    nparams = length(parameters(sys))


    # Build the total matrix and vector of type Any to allow for dual numbers
    Atotal = Matrix{eltype(first(first(u)))}(undef, neqs*ndat, nparams)
    btotal = Vector{eltype(first(first(u)))}(undef, neqs*ndat)
    for timeidx in 1:ndat
        idx = (timeidx-1)*neqs+1:timeidx*neqs
        tval = tvals === nothing ? nothing : tvals[timeidx]
        uval = u isa Vector ? u[timeidx] : u[:,timeidx]
        duval = du isa Vector ? du[timeidx] : du[:,timeidx]
        Atotal[idx,:] = Afun(uval, duval, tval)
        btotal[idx] = bfun(uval, duval, tval)
    end

    Atotaltranspose = transpose(Atotal)
    covariance_matrix = Atotaltranspose*Atotal

    if lambda > 0.0 #add regularization to the regression problem
        Lambda = zeros(nparams,nparams) #regularization matrix
        for i in 1:nparams
            Lambda[i,i] = lambda
        end
        covariance_matrix += Lambda
    end

    # Solve the linear system using the normal equations with regularization
    paramest = covariance_matrix \ (Atotaltranspose*btotal)

    r = btotal - Atotal*paramest
    mse = sum(r.^2)/length(r)
    if verbose
        println("Successfully estimated parameters, with mean squared error $(mse)")
    end
    return Dict(zip(parameters(sys),paramest)), mse
end
