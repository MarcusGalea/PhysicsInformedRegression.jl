"""
This function is used to setup the linear system for the regression problem. It returns the matrix A and the vector b, such that A*x = b, where x is the vector of parameters.\n
    setup_linear_system(sys::AbstractTimeDependentSystem)
# Arguments \n
- `sys`: The system of equations to be solved for the parameters (ODESystem or DAESystem)
# Returns \n
- `A`: The symbolic matrix A in the equation A*x = b
- `b`: The symbolic vector b in the equation A*x = b
"""
function setup_linear_system(sys::AbstractTimeDependentSystem)
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
    physics_informed_regression(sys::AbstractTimeDependentSystem, u::Vector, du::Vector)
    physics_informed_regression(sys::AbstractTimeDependentSystem, u::Vector, du::Vector, A::Matrix, b::Vector)
# Arguments \n
- `sys`: The system of equations to be solved for the parameters (ODESystem or DAESystem)
- `u`: The vector of states
- `du`: The vector of derivatives
- `A`: The symbolic matrix A in the equation A*x = b
- `b`: The symbolic vector b in the equation A*x = b
# Returns \n
- `paramsest`: The vector of estimated parameters
"""
function physics_informed_regression(sys, u, du)
    A,b = setup_linear_system(sys)
    paramsest = physics_informed_regression(sys, u, du, A, b)
    return paramsest
end

function physics_informed_regression(sys, u, du, A, b)
    # Define the variables and derivatives as indexed symbols
    @variables U[1:length(states(sys))] dU[1:length(states(sys))]
    neqs = length(equations(sys))
    ndat = length(u)
    nparams = length(parameters(sys))

    # Define the substitutions for the derivatives and states as a dictionary (hashmap)
    derivative_substitutions = Dict()
    state_substitutions = Dict()
    for (i, state) in enumerate(states(sys))
        derivative_substitutions[Differential(sys.iv)(U[i])] = dU[i]
        state_substitutions[state] = U[i]
    end

    substitute_derivatives(expr) = substitute(expr, derivative_substitutions)
    substitute_states(expr) = substitute(expr, state_substitutions)

    # rewrite the equations in terms of the indexed variables and derivatives
    Atemp = substitute_states.(A)
    btemp = substitute_states.(b)

    Atemp = substitute_derivatives.(Atemp)
    btemp = substitute_derivatives.(btemp)

    Afun = [eval(build_function(Atemp[i,j], U, dU, expression=Val{false})) for i=1:neqs, j=1:nparams]
    bfun = [eval(build_function(btemp[i], U, dU, expression=Val{false})) for i=1:neqs]
    
    # Build the total matrix and vector
    Atotal = zeros(neqs*ndat, length(parameters(sys)))
    btotal = zeros(neqs*ndat)
    for timeidx in 1:ndat
        idx = (timeidx-1)*neqs+1:timeidx*neqs

        Atotal[idx,:] = [Afun[i,j](u[timeidx], du[timeidx]) for i=1:neqs, j=1:nparams]
        btotal[idx] = [bfun[i](u[timeidx], du[timeidx]) for i=1:neqs]
    end

    #setup equation for parameter estimation (Ordinary Least Squares)
    Atotaltranspose = transpose(Atotal)
    paramest = (Atotaltranspose*Atotal) \ (Atotaltranspose*btotal)
    return paramest
end
