"""
This function is used to setup the linear system for the regression problem. It returns the matrix A and the vector b, such that A*x = b, where x is the vector of parameters.\n
    setup_linear_system(sys::AbstractTimeDependentSystem)
# Arguments \n
- `sys`: The system of equations to be solved for the parameters (ODESystem or DAESystem)
# Returns \n
- `A`: The symbolic matrix A in the equation A*x = b
- `b`: The symbolic vector b in the equation A*x = b
"""
function setup_linear_system(sys::T) where T<:Union{AbstractTimeDependentSystem, PDESystem}
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
    kwargs: lambda = 0.0 (regularization parameter)
# Arguments \n
- `sys`: The system of equations to be solved for the parameters (ODESystem or DAESystem)
- `u`: The vector of states
- `du`: The vector of derivatives
- `A`: The symbolic matrix A in the equation A*x = b
- `b`: The symbolic vector b in the equation A*x = b
# Returns \n
- `paramsest`: The vector of estimated parameters
"""
function physics_informed_regression(sys::AbstractTimeDependentSystem, u, du; verbose = false, kwargs...)
    A,b = setup_linear_system(sys)
    if verbose
        println("The linear system is setup with")
        println("A = ")
        display(A)
        println("b = ")
        display(b)
    end
    paramsest = physics_informed_regression(sys, u, du, A, b; kwargs...)
    return paramsest
end

function physics_informed_regression(sys::AbstractTimeDependentSystem,
                                    u:: Array,
                                    du::Array,
                                    A,
                                    b;
                                    kwargs...)
    #Convert u and du to vectors if they are not already
    if !(u isa Vector)
        #convert Matrix to vector of vector
        u = [collect(u[i,:]) for i in 1:size(u,1)]
    end
    if !(du isa Vector)
        #convert Matrix to vector of vector
        du = [collect(du[i,:]) for i in 1:size(du,1)]
    end
end

    

function physics_informed_regression(sys::AbstractTimeDependentSystem, 
                                        u::Vector, 
                                        du::Vector, 
                                        A, 
                                        b; 
                                        lambda = 0.0,
                                        verbose = false,
                                        tvals = nothing, #only needed if time is used in system of equations
                                        )
    # Define the variables and derivatives as indexed symbols
    @independent_variables t
    @variables U[1:length(ModelingToolkit.get_unknowns(sys))] dU[1:length(ModelingToolkit.get_unknowns(sys))]
    neqs = length(equations(sys))
    ndat = length(u)
    nparams = length(parameters(sys))

    # Define the substitutions for the derivatives and states as a dictionary (hashmap)
    derivative_substitutions = Dict()
    state_substitutions = Dict()
    for (i, state) in enumerate(ModelingToolkit.get_unknowns(sys))
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

    Afun = [eval(build_function(Atemp[i,j], U, dU, t, expression=Val{false})) for i=1:neqs, j=1:nparams]
    bfun = [eval(build_function(btemp[i], U, dU, t, expression=Val{false})) for i=1:neqs]

    # Build the total matrix and vector of type Any to allow for dual numbers
    Atotal = Matrix{Any}(undef, neqs*ndat, nparams)
    btotal = Vector{Any}(undef, neqs*ndat)
    #convert to static arrays
    # u = SVector{length(states(sys))}.(u)
    # du = SVector{length(states(sys))}.(du)
    for timeidx in 1:ndat
        idx = (timeidx-1)*neqs+1:timeidx*neqs
        tval = tvals === nothing ? nothing : tvals[timeidx]

        Atotal[idx,:] = [Afun[i,j](u[timeidx], du[timeidx], tval) for i=1:neqs, j=1:nparams]
        btotal[idx] = [bfun[i](u[timeidx], du[timeidx], tval) for i=1:neqs]
        #Atotal[idx,:] = Afun(u[timeidx], du[timeidx])
        #btotal[idx] = bfun(u[timeidx], du[timeidx])
    end

    #setup equation for parameter estimation (Ordinary Least Squares)
    #convert to narrower type
    Atotal = collect((x for x in Atotal))
    btotal = collect(x for x in btotal)
    Atotaltranspose = transpose(Atotal)
    Lambda = zeros(nparams,nparams)
    for i in 1:nparams
        Lambda[i,i] = lambda
    end
    paramest = (Atotaltranspose*Atotal + Lambda) \ (Atotaltranspose*btotal)
    #prob = LinearProblem(Atotaltranspose*Atotal, Atotaltranspose*btotal)
    #linsolve = init(prob)
    #paramest = solve(linsolve).u   
    if verbose
        r = btotal - Atotal*paramest
        println("Successfully estimated parameters, with root mean squared error $(sqrt(sum(r.^2)/length(r)))")
    end
    return Dict{SymbolicUtils.BasicSymbolic{Real}, Any}(zip(parameters(sys),paramest))
end
