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
    tunable_params = [param for param in params if ModelingToolkitBase.istunable(param)]
    non_tunable_params = [param for param in params if !ModelingToolkitBase.istunable(param)]
    # Initialize an empty matrix with zeros
    A = Array{Any}(undef, length(eqs), length(tunable_params)) #{Union{SymbolicUtils.BasicSymbolic,AbstractFloat}} for more efficiency, but doesn't work with dual numbers
    b = Array{Any}(undef, length(eqs)) #{Union{SymbolicUtils.BasicSymbolic,AbstractFloat}} for more efficiency, but doesn't work with dual numbers
    # Fill the matrix
    for (i, eq) in enumerate(eqs)
        isolated_expr = Symbolics.simplify(eq.rhs - eq.lhs, expand = true)
        for (j, param) in enumerate(tunable_params)
            # Get the coefficient of the parameter in the equation
            coeff = Symbolics.coeff(isolated_expr, param)
            # Set the matrix element
            A[i, j] = isequal(coeff, 0//1) ? 0.0 : coeff
            isolated_expr = Symbolics.simplify(isolated_expr-coeff * param, expand = true)
            @assert isequal(Symbolics.degree(isolated_expr, param), 0) "Error: parameter $param not isolated in equation $eq. Ensure model is linear in terms of parameters."
        end
        for param in non_tunable_params
            # Fix non-tunable parameters with default values
            if ModelingToolkitBase.hasdefault(param)
                param_value = ModelingToolkitBase.getdefault(param)
                #substitute the default value of the parameter into the equation
                isolated_expr = Symbolics.simplify(substitute(isolated_expr, param => param_value), expand = true)
            end
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

# Optional keyword arguments \n      
- `lambda` = 0.0: The L2 regularization parameter for the regression problem. Default is 0.0 (no regularization).    
- `verbose` = false: If true, prints the matrix A and vector b, and the mean squared error of the regression. Default is false. 
- `tvals` = nothing: The vector of time values corresponding to the states and derivatives, only needed if the system of equations is non-autonomous. Default is nothing (assumes autonomous system).

# Returns \n
- `paramsest`: The vector of estimated parameters
"""
function physics_informed_regression(sys::ModelingToolkitBase.System, u::Array, du::Array, args...; verbose = false, kwargs...)
    A,b = setup_linear_system(sys)
    if verbose
        println("The linear system is setup with")
        println("A = ")
        display(A)
        println("b = ")
        display(b)
    end
    Afun, bfun = create_callable_function(A, b, sys)
    paramsest = physics_informed_regression(sys, u, du, Afun, bfun, args...; kwargs...)
    return paramsest
end


Symbolics.diff2term(x::Float64) = x #if the expression is a number, return it as is

"""
This function is used to create callable functions from the symbolic matrix A and vector b. 
    It returns two functions, Afun and bfun, which can be called with the states, derivatives, and time 
    (if time is used in the system of equations) to get the numerical values of A and b.\n
    $(TYPEDSIGNATURES)
"""
function create_callable_function(A, b, sys::ModelingToolkitBase.System)
    iv = ModelingToolkitBase.get_iv(sys)
    non_tunable_params = [param for param in parameters(sys) if !ModelingToolkitBase.istunable(param)] 
    Atemp = Symbolics.diff2term.(A)
    btemp = Symbolics.diff2term.(b)

    Dt = Differential(iv)
    vars = ModelingToolkitBase.get_unknowns(sys)
    diffvars = Symbolics.diff2term.(Dt.(vars))
    Afun = first(eval(build_function(Atemp, vars, diffvars, iv, non_tunable_params..., expression = Val{false})))
    bfun = first(eval(build_function(btemp, vars, diffvars, iv, non_tunable_params..., expression = Val{false})))
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
                                        bfun,
                                        args...
                                        ; 
                                        lambda = 0.0,
                                        verbose = false,
                                        tvals = 0, #only needed if not autonomous system,
                                        )
    # Define the variables and derivatives as indexed symbols

    neqs = length(equations(sys))
    ndat = u isa Vector ? length(u) : size(u, 2)
    nparams = length([param for param in parameters(sys) if ModelingToolkitBase.istunable(param)])
    # Build the total matrix and vector of type Any to allow for dual numbers

    Atotal = Array{eltype(first(first(u)))}(undef, neqs*ndat, nparams)  
    btotal = Array{eltype(first(first(u)))}(undef, neqs*ndat)   


    for timeidx in 1:ndat
        idx = (timeidx-1)*neqs+1:timeidx*neqs
        uval = u isa Vector ? u[timeidx] : u[:,timeidx]
        tval = isequal(tvals, 0) ? 0*one(eltype(uval)) : tvals[timeidx]
        duval = du isa Vector ? du[timeidx] : du[:,timeidx]
        Atotal[idx,:] = Afun(uval, duval, tval, args...)
        btotal[idx] = bfun(uval, duval, tval, args...)
    end
    #Convert Atotal and btotal to the narrowest type possible for efficiency in the linear algebra operations
    Atotaltranspose = transpose(Atotal)
    covariance_matrix = Atotaltranspose*Atotal
    # println("Covariance matrix: " * string(typeof(covariance_matrix)), " with size: ", size(covariance_matrix))
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
    return Dict(zip([param for param in parameters(sys) if ModelingToolkitBase.istunable(param)],paramest)), mse
end
 