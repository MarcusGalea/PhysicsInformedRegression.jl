"""
physics_informed_regression(pdesys:: ModelingToolkit.PDESystem,
                                    sol::Union{SciMLBase.PDETimeSeriesSolution, Dict};
                                    interp_fun = cubic_spline_interpolation,
                                    lambda = 0.0)
This function computes the physics informed regression for the system of equations.

# Args
    pdesys: The PDE system to be solved for the parameters (PDESystem)
    sol: The PDE solution object or data Dict mapping the dependent variables to their values at each grid point
    interp_fun: The interpolation function to use (default is cubic_spline_interpolation)
    lambda: The regularization parameter (default is 0.0)

# Returns
    paramsest: The vector of estimated parameters
"""
function physics_informed_regression(pdesys:: ModelingToolkit.PDESystem,
                                    sol::Union{SciMLBase.PDETimeSeriesSolution, Dict};
                                    interp_fun = cubic_spline_interpolation,
                                    lambda = 0.0)
    A,b = PhysicsInformedRegression.setup_linear_system(pdesys)

    #get the independent and dependent variables
    ivs =  pdesys.ivs
    dvs = pdesys.dvs
    #get the domain of the system
    dom = sol.ivdomain

    @variables _U[1:length(dvs)] _dU[1:length(dvs),1:length(ivs)] _ddU[1:length(dvs), 1:length(ivs), 1:length(ivs)]


    dom = uniform_domain(dom) #check if the domain is uniformly spaced 

    state_maps, gradient_maps, hessian_maps = symbolic_maps(A, b, _U, _dU, _ddU, ivs, dvs)
    states, gradients, hessians = compute_gradients_hessians(sol, dvs, ivs, dom; interp_fun = interp_fun)


    substitute_hessians(expr) = substitute(expr, hessian_maps)
    substitute_gradients(expr) = substitute(expr, gradient_maps)
    substitute_states(expr) = substitute(expr, state_maps)

    Atemp = substitute_hessians.(A)
    btemp = substitute_hessians.(b)

    Atemp = substitute_gradients.(Atemp)
    btemp = substitute_gradients.(btemp)

    Atemp = substitute_states.(Atemp)
    btemp = substitute_states.(btemp)

    neqs = length(equations(pdesys))
    nparams = length(parameters(pdesys))
    Afun = [eval(build_function(Atemp[i,j], _U, _dU, _ddU, expression=Val{false})) for i=1:neqs, j=1:nparams]
    bfun = [eval(build_function(btemp[i], _U, _dU, _ddU, expression=Val{false})) for i=1:neqs]

    neqs = length(equations(pdesys))
    nparams = length(parameters(pdesys))
    ndat = prod(size(states))
    Atotal = Matrix{Any}(undef, neqs*ndat, nparams)
    btotal = Vector{Any}(undef, neqs*ndat)
    for (i,c) in enumerate(CartesianIndices(states))
        idx = (i-1)*neqs+1:i*neqs
        Atotal[idx,:] = [Afun[i,j](states[c], gradients[c], hessians[c]) for i=1:neqs, j=1:nparams]
        btotal[idx] = [bfun[i](states[c], gradients[c], hessians[c]) for i=1:neqs]
    end

    #convert to narrower type
    Atotal = collect((x for x in Atotal))
    btotal = collect(x for x in btotal)
    Atotaltranspose = transpose(Atotal)
    Lambda = zeros(nparams,nparams)
    for i in 1:nparams
        Lambda[i,i] = lambda
    end

    paramest = (Atotaltranspose*Atotal + Lambda) \ (Atotaltranspose*btotal)
    return Dict(zip(parameters(pdesys), paramest))
end



"""
compute_gradients_hessians(sol, dvs, ivs, dom)
This function computes the gradients and hessians of the dependent variables in the system of equations.

# Args
    sol: The PDE solution object or data Dict mapping the dependent variables to their values at each grid point
    dvs: The vector of dependent variables (indexed symbolic variables)
    ivs: The vector of independent variables (symbolic variables)
    dom: The domain of the system of equations

# Returns
    vals: The values of the dependent variables
    gradients: The gradients of the dependent variables
    hessians: The hessians of the dependent variables
"""
function compute_gradients_hessians(sol::Union{SciMLBase.PDETimeSeriesSolution, Dict}, 
                                    dvs::Vector, 
                                    ivs::Vector, 
                                    dom::Union{Tuple, Vector};
                                    interp_fun = cubic_spline_interpolation)
    datashape = size(sol[dvs[1]]) # get the shape of the data

    vals = Matrix{Any}(undef, datashape...)
    gradients = Matrix{Any}(undef, datashape...)
    hessians = Matrix{Any}(undef, datashape...)

    interp = Dict()
    for dv in dvs #compute the interpolation for each dependent variable
        interp[dv] = interp_fun(dom, sol[dv])
    end
    dom_vals = collect.(dom)
    for c in CartesianIndices(datashape)
        #loop over indexes in c
        iv_vals = []
        for i in 1:length(datashape)
            #get the index of the current dimension
            push!(iv_vals, dom_vals[i][c[i]])
        end
        vals[c] = [sol[dv][c] for dv in dvs] #store the values of the dependent variables
        gradients[c] = zeros(length(dvs), length(ivs))
        if length(gradient_maps) > 0
            gradients[c][:,:] = hcat([Interpolations.gradient(interp[dv], iv_vals...) for dv in dvs]...) #approximate the gradient
        end
        hessians[c] = zeros(length(dvs), length(ivs), length(ivs))
        if length(hessian_maps) > 0
            hessians[c][:,:,:] = hcat([Interpolations.hessian(interp[dv], iv_vals...) for dv in dvs]...) #approximate the hessian
        end
    end
    return vals, gradients, hessians
end

"""
symbolic_maps(A, b, U, dU, ddU)
This function computes the symbolic maps for the derivatives and states in the system of equations.

# Args
    A: The matrix of equations (Vector of expressions)
    b: The vector of equations (Vector of expressions)
    U: The vector of new dependent variables (indexed symbolic variables)
    dU: The matrix of new first derivatives (indexed symbolic variables)
    ddU: The matrix of new second derivatives (indexed symbolic variables)
    ivs: The vector of independent variables (symbolic variables)
    dvs: The vector of old dependent variables (indexed symbolic variables)

# Returns
    state_maps: A dictionary of the state maps
    gradient_maps: A dictionary of the gradient maps
    hessian_maps: A dictionary of the hessian maps
"""
function symbolic_maps( A::Matrix, 
                        b::Vector,
                        U::Symbolics.Arr,
                        dU:: Symbolics.Arr,
                        ddU:: Symbolics.Arr,
                        ivs::Vector,
                        dvs::Vector)

    
    #initialize the maps

    state_maps = Dict([dv => U[i] for (i, dv) in enumerate(dvs)]) #create dependent variable maps
    gradient_maps = Dict()
    hessian_maps = Dict()


    depidcs = Dict([(x, i) for (i, x) in enumerate(dvs)]) #get the index of the dependent variables
    indepidcs = Dict([(x,i) for (i, x) in enumerate(ivs)]) #get the index of the independent variables


    #get all differential operators in the equations
    diffexprs = [vcat(Symbolics.filterchildren.(Symbolics.is_derivative, A)...); vcat(Symbolics.filterchildren.(Symbolics.is_derivative, b)...)]

    for diffex in diffexprs #loop over differential operators
        O = diffex
        indepvar = Num[] # get variables in the derivative
        while is_derivative(O) 
            push!(indepvar, operation(O).x) # get the independent variable
            O = arguments(O)[1]
        end
        depvar = O # get the dependent variable which is differentiated

        if length(indepvar) == 1 #first order derivative
            depidx = depidcs[depvar]
            indepidx = indepidcs[indepvar[1]]
            gradient_maps[diffex] = dU[depidx, indepidx] #create the gradient map
        elseif  length(indepvar) == 2 #second order derivative
            depidx = depidcs[depvar]
            indepidx1 = indepidcs[indepvar[1]]
            indepidx2 = indepidcs[indepvar[2]]
            hessian_maps[diffex] = ddU[depidx, indepidx1, indepidx2] #create the hessian map
        else
            error("Can't compute higher than 2nd order derivatives")
        end
    end
    return state_maps, gradient_maps, hessian_maps
end


"""
Checks if the domain is uniformly spaced

This function checks if the domain is uniformly spaced by checking if the difference between

# Args 
    x: The domain to check
    abstol: The absolute tolerance for the difference
"""
function is_uniformly_spaced(x, abstol = 1e-4)
    dx = x[2] - x[1]
    return all(abs.(diff(x) .- dx) .< abstol)
end


"""
This function reconstructs the domain checking whether the domain is uniformly spaced
# Args
    dom: The domain to check
    kwargs: The keyword arguments to pass to the is_uniformly_spaced function

    # Returns
        redef_dom: The reconstructed domain as a Range
"""
function uniform_domain(dom, kwargs...)
    redef_dom = []
    #reconstruct the domain checking whether the domain is uniformly spaced
    for (i, d) in enumerate(dom)
        if d isa Vector
            if !is_uniformly_spaced(d, kwargs...)
                error("The domain is not uniformly spaced")
            end
            d = d[1]:d[2]-d[1]:d[end]
        end
        push!(redef_dom, d)
    end
    redef_dom = tuple(redef_dom...)
    return redef_dom
end
