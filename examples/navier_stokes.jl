using Pkg

Pkg.activate("./private//")
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, DomainSets
using PhysicsInformedRegression,MAT,Plots, Interpolations



#### Real data

data = matread("data/CYLINDER_ALL.mat")
X = data["VORTALL"][:,1:end-1]
Y = data["VORTALL"][:,end]


#reshape the data
ny = Int(data["ny"]) #number of points in the y direction
nx = Int(data["nx"]) #number of points in the x direction
N = 151 #number of time steps

# Parameters
Lx = 1.0
Ly = 1.0
ST = 0.173 #Strouhal number at Re = 100
T = 1/ST  #period of the vortex shedding
dt = T / (N - 1) #time step size

@parameters Re
@independent_variables t x y
@variables ω(..) u(..) v(..)

# Domain
x_dom = Interval(0.0, Lx)
y_dom = Interval(0.0, Ly)
t_dom = Interval(0.0, T)
domains = [x ∈ x_dom, y ∈ y_dom, t ∈ t_dom]

# Differential operators
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2



# Vorticity equation (2D Navier-Stokes)
eq = Dt(ω(x,y,t)) + u(x,y,t) * Dx(ω(x,y,t)) + v(x,y,t) * Dy(ω(x,y,t)) ~ Re * (Dxx(ω(x,y,t)) + Dyy(ω(x,y,t)))

bcs = [
] #irrelevant for the PhysicsInformedRegression, but can be added if needed

@named pdesys = PDESystem(eq, bcs, domains, [x,y,t], [ω(x,y,t), u(x,y,t), v(x,y,t)], [Re])




ω_data = reshape(data["VORTALL"], nx, ny, N)
u_data = reshape(data["UALL"], nx, ny, N)
v_data = reshape(data["VALL"], nx, ny, N)

ivs = [x,y,t]
dvs = [ω(x,y,t), u(x,y,t), v(x,y,t)]

#show the data at a specific time step
n = 70
heatmap(ω_data[:,:,n], aspect_ratio = 1, title = "Vorticity at t=$(n*dt)", xlabel = "y", ylabel = "x", colorbar_title = "ω", c=:viridis, xlim = (0, ny), ylim = (0, nx), size = (600, 600))

##### Physics-informed regression
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
datainfo = Dict(
            t => 0:dt:T, #LinRange(0, T, N), #
            x => 0:dx:Lx, #LinRange(0, Lx, W), #
            y => 0:dy:Ly, #LinRange(0, Lx, H), #
            ω(x,y,t) => ω_data,
            u(x,y,t) => u_data,
            v(x,y,t) => v_data,
) #map symbolic variables to data

using Random
Ns = 10000
Random.seed!(3) #for reproducibility
#size of v_data is (nx, ny, N)
x_samples,y_samples,t_samples = rand(50:149, Ns), rand(100:150, Ns), rand(1:150, Ns) 

#, n*Int.(ones(Ns))
#convert to Cartesian indices
samples = CartesianIndex.(x_samples, y_samples, t_samples)
interp_fun = BSpline(Cubic(Line(OnGrid())))
# paramsest = PhysicsInformedRegression.physics_informed_regression(pdesys, datainfo; samples = samples, interp_fun = interp_fun)



function compute_jacobian(
                        ivs::Vector, dvs::Vector, 
                        datainfo::Union{Dict, SciMLBase.PDETimeSeriesSolution},     
                        coordinate::CartesianIndex;
                        dom = [collect.(datainfo[iv]) for iv in ivs]
                        )
    # Get the independent variables from the system
    n_ivs = length(ivs)
    n_dvs = length(dvs)

    Jacobian = zeros(n_dvs, n_ivs)
    for (i, dv) in enumerate(dvs)
        for j in eachindex(ivs)
            # get adjacent indices in the domain
            idx_prev = coordinate - CartesianIndex([j==i for i in 1:n_ivs]...)
            idx_next = coordinate + CartesianIndex([j==i for i in 1:n_ivs]...)
            idx_prev = checkbounds(Bool,datainfo[dv], idx_prev) ? idx_prev : coordinate
            idx_next = checkbounds(Bool, datainfo[dv], idx_next) ? idx_next : coordinate

            # compute the distance between the previous and next indices
            dist = first(dom[j][idx_next[j]] - dom[j][idx_prev[j]])

            # get the values at the prev, current and next indices
            y_prev = datainfo[dv][idx_prev]
            y_next = datainfo[dv][idx_next]

            # compute the finite difference approximation
            Jacobian[i,j] = (y_next - y_prev) / dist
        end
    end
    return Jacobian
end

using DocStringExtensions
"""
Compute the Hessian matrix at a given coordinate in the data grid.
    $(DocStringExtensions.TYPEDSIGNATURES)
# Arguments \n
- `ivs::Vector` : A vector of independent variables.
- `dvs::Vector` : A vector of dependent variables.
- `datainfo::Union{Dict, SciMLBase.PDETimeSeriesSolution}` : A dictionary or PDE time series solution containing the data information.
- `coordinate::CartesianIndex` : The Cartesian index at which to compute the Hessian.
# Keyword Arguments \n
- `jacobians::Union{Nothing, Dict{CartesianIndex, Matrix{Float64}}}` : A dictionary of precomputed Jacobians indexed by Cartesian indices. If `nothing`, the Jacobians will be computed on-the-fly.
- `dom::Vector{Vector}` : A vector of vectors representing the domain for each independent variable. Defaults to the values in `datainfo` for each independent variable.
# Returns \n
- `Hessian::Array{Float64}` : A 3D array representing the Hessian matrix at the specified coordinate, with dimensions (number of dependent variables, number of independent variables, number of independent variables).
"""
function compute_hessian(
                        ivs::Vector, dvs::Vector, 
                        datainfo::Union{Dict, SciMLBase.PDETimeSeriesSolution},
                        coordinate::CartesianIndex;
                        jacobians = nothing,
                        dom = [collect.(datainfo[iv]) for iv in ivs]
                        )

    # Get the independent variables from the system
    n_ivs = length(ivs)
    n_dvs = length(dvs)

    Hessian = zeros(n_dvs, n_ivs, n_ivs)
    for i in eachindex(dvs)
        for j in eachindex(ivs)
            # get adjacent indices in the domain
            idx_prev = coordinate - CartesianIndex([j==k for k in 1:n_ivs]...)
            idx_next = coordinate + CartesianIndex([j==k for k in 1:n_ivs]...)
            # check if the indices are within bounds of the data
            idx_prev = checkbounds(Bool, datainfo[dvs[i]], idx_prev) ? idx_prev : coordinate
            idx_next = checkbounds(Bool, datainfo[dvs[i]], idx_next) ? idx_next : coordinate

            # compute the distance between the previous and next indices
            dist = first(dom[j][idx_next[j]] - dom[j][idx_prev[j]])

            if !isnothing(jacobians) # use the precomputed Jacobians if available
                jacobian_prev = compute_jacobian(ivs, dvs, datainfo, idx_prev)
                jacobian_next = compute_jacobian(ivs, dvs, datainfo, idx_next)
            else
                jacobian_prev = jacobians[idx_prev]
                jacobian_next = jacobians[idx_next]
            end

            # compute the finite difference approximation
            for k in 1:n_ivs
                Hessian[i,j,k] = (jacobian_next[i,k] - jacobian_prev[i,k]) / dist
            end
        end
    end
    return Hessian
end

using DocStringExtensions
"""
Observation is a mutable struct that holds the data and computed values for a specific observation in a physics-informed regression problem.
    $(DocStringExtensions.TYPEDFIELDS)
    """
mutable struct Observation
    """ Cartesian index of the observation in the data grid. """
    coordinate::CartesianIndex
    """ Dependent variable values at the observation coordinate. """
    iv_values::Vector
    """ Computed state at the observation coordinate. """
    dv_values::Vector
    """ Computed gradient at the observation coordinate. """
    jacobian::Array
    """ Computed Hessian at the observation coordinate. """
    hessian::Array
    """ Data information dictionary containing the independent and dependent variable mappings from symbol to value. """
    data_info::Dict


    """
    Constructor for Observation.
        $(DocStringExtensions.TYPEDSIGNATURES)
    """
    Observation(coordinate::CartesianIndex, 
                        iv_values::Vector, 
                        dv_values::Vector, 
                        jacobian::Array, 
                        hessian::Array, 
                        data_info::Dict) = new(coordinate, iv_values, dv_values, jacobian, hessian, data_info)
end


"""
Observations is a function that creates a collection of Observation objects for a given set of coordinates, independent variables (ivs), dependent variables (dvs), and data information (datainfo).
    $(DocStringExtensions.TYPEDSIGNATURES)
# Arguments
- `coordinates::Array` : An array of Cartesian indices representing the coordinates for which observations are to be created.
- `ivs::Vector` : A vector of independent variables.
- `dvs::Vector` : A vector of dependent variables.
- `datainfo::Dict` : A dictionary containing the data information for the observations.
- `data_structure::Dict` : A dictionary to store the Observation objects, indexed by their Cartesian indices. Defaults to an empty dictionary.
# Returns
- `data_structure::Dict` : A dictionary containing the Observation objects, indexed by their Cartesian indices.
"""
function Observations(
                    coordinates::Array,
                    ivs::Vector, 
                    dvs::Vector,
                    datainfo::Dict;
                    data_structure = Dict{CartesianIndex, Observation}()
                    )
    dom = collect.([datainfo[iv] for iv in keys(datainfo)])
    jacobians = Dict{CartesianIndex, Matrix{Float64}}()
    function jacobian_fun(coordinate::CartesianIndex)
        if haskey(jacobians, coordinate)
            return jacobians[coordinate]
        else
            jacobian = compute_jacobian(ivs, dvs, datainfo, coordinate; dom = dom)
            jacobians[coordinate] = jacobian
            return jacobian
        end
    end

    # Create a Observation for each coordinate
    for coordinate in coordinates
        # Compute the values for the independent and dependent variables at the coordinate
        iv_values = [datainfo[iv][coordinate[idx]] for (idx, iv) in enumerate(ivs)]
        dv_values = [datainfo[dv][coordinate[idx]] for (idx, dv) in enumerate(dvs)]

        # Compute the Jacobian and Hessian at the coordinate
        jacobian = jacobian_fun(coordinate)

        hessian = compute_hessian(ivs, dvs, datainfo, coordinate; jacobians = jacobian_fun, dom = dom)
        
        #string representation
        stringrep = join(["$(substitute(dv, Dict(zip(ivs, round.(iv_values, sigdigits=2))))) = $(round(dv_value, sigdigits=2))\n" for (dv, dv_value) in zip(dvs, dv_values)])

        data_info = Dict(zip([ivs; dvs; :name],[iv_values; dv_values; stringrep]))

        # Create a data structure entry for the coordinate
        data_structure[coordinate] = Observation(coordinate, iv_values, dv_values, jacobian, hessian, data_info)
    end
    return data_structure
end

Base.show(io::IO, ddo::Observation) = print(io, ddo.data_info[:name])

observations = Observations(samples, 
                            [x,y,t], 
                            [ω(x,y,t), u(x,y,t), v(x,y,t)],
                            datainfo;
                            data_structure = Dict{CartesianIndex, Observation}())

observations[samples[1]].iv_values

using Base
#index Array of Observation by CartesianIndex
Base.getindex(ddo::Observation, idx::CartesianIndex) = ddo.coordinate == idx ? ddo : throw(ArgumentError("Index $(idx) does not match the Observation coordinate $(ddo.coordinate)"))
coordinate = CartesianIndex(50, 100, 1) #example coordinate to compute the Jacobian atcoordinate[2] 
coordinate[1]
#sample random indices from v_data
j = 1 #index to increment
#convert j to boolean index vector



A,b = setup_linear_system(pdesys)

#get the independent and dependent variables
ivs =  pdesys.ivs
dvs = pdesys.dvs

#get the domain of the system
dom = [sol[iv] for iv in ivs]

shapesizes = [size(sol[dv]) for dv in dvs]
@assert all(x -> x == shapesizes[1], shapesizes) "All dependent variables must have the same shape." 

domain_size = first(shapesizes)
for (i,n_elements) in enumerate(first.(collect(size.(collect(dom)))))
    @assert n_elements == domain_size[i] "Dependent variable $(dvs[i]) has a different number of elements in the predefined domain $(dom[i]) than the expected $(domain_size[i])"
end

#indexed symbolic dependent variables
@variables _U[1:length(dvs)] _dU[1:length(dvs),1:length(ivs)] _ddU[1:length(dvs), 1:length(ivs), 1:length(ivs)]
#indexed symbolic independent variables
@independent_variables _X[1:length(ivs)]



redef_dom = PhysicsInformedRegression.uniform_domain(dom) #check if the domain is uniformly spaced 

indepvar_maps = Dict(zip(ivs, _X))
depvar_maps, gradient_maps, hessian_maps = PhysicsInformedRegression.symbolic_maps(A, b, _U, _dU, _ddU, ivs, dvs)


states, gradients, hessians = PhysicsInformedRegression.compute_gradients_hessians(sol, dvs, ivs, redef_dom, gradient_maps, hessian_maps; interp_fun = interp_fun, samples = samples)

##### Test gradient and hessian substitution


fun_to_test = (x,y,t) -> x^2 + y^2 + t^2
grad_to_test = (x,y,t) -> [2*x, 2*y, 2*t]
hess_to_test = (x,y,t) -> [2 0 0; 0 2 0; 0 0 2]


idvs_test = [x,y,t]
dvs_test = [u(x,y,t)]

dom = (1:10, 1:10, 1:10) #domain for the test function

Nsamples = 10
samples_test = CartesianIndex.(1:10, 1:10, 1:10) #sample Cartesian indices for the test function
u_vals_test = [fun_to_test(x,y,t) for x in dom[1], y in dom[2], t in dom[3]]
data_test_info = Dict(
    x => dom[1],
    y => dom[2],
    t => dom[3],
    u(x,y,t) => u_vals_test
)
test_observations = DataDrivenObservations(samples_test, 
                            idvs_test, 
                            dvs_test,
                            data_test_info;
                            data_structure = Dict{CartesianIndex, DataDrivenObservation}())






substitute_hessians(expr) = substitute(expr, hessian_maps)
substitute_gradients(expr) = substitute(expr, gradient_maps)
substitute_states(expr) = substitute(expr, depvar_maps)
substitute_indepvars(expr) = substitute(expr, indepvar_maps)

Atemp = substitute_indepvars.(substitute_states.((substitute_gradients.(substitute_hessians.(A)))))

btemp = substitute_indepvars.(substitute_states.((substitute_gradients.(substitute_hessians.(b)))))

neqs = length(equations(pdesys))
nparams = length(parameters(pdesys))
Afun = [eval(build_function(Atemp[i,j], _U, _dU, _ddU, _X, expression=Val{false})) for i=1:neqs, j=1:nparams]
bfun = [eval(build_function(btemp[i], _U, _dU, _ddU, _X, expression=Val{false})) for i=1:neqs]

neqs = length(equations(pdesys))
nparams = length(parameters(pdesys))
ndat = length(samples)
Atotal = Matrix{Any}(undef, neqs*ndat, nparams)
btotal = Vector{Any}(undef, neqs*ndat)

n_ivs = length(ivs)
iv_values = collect.(dom)
current_iv_val = zeros(n_ivs)
for (i,c) in enumerate(samples)
    for j=1:n_ivs
        current_iv_val[j] = iv_values[j][c[j]]
    end
    idx = (i-1)*neqs+1:i*neqs
    Atotal[idx,:] = [Afun[i,j](states[c], gradients[c], hessians[c], current_iv_val) for i=1:neqs, j=1:nparams]
    btotal[idx] = [bfun[i](states[c], gradients[c], hessians[c], current_iv_val) for i=1:neqs]
end

