
using DocStringExtensions
using Interpolations

mutable struct FiniteDifference
    """
    FiniteDifference is a mutable struct that holds the finite difference approximation of a function.
        $(DocStringExtensions.TYPEDFIELDS)
    """

end


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
                    coordinates::Union{Array, CartesianIndices{T}},
                    ivs::Vector, 
                    dvs::Vector,
                    datainfo::Union{Dict, SciMLBase.PDETimeSeriesSolution};
                    data_structure = Dict{CartesianIndex, Observation}()
                    ) where T <: Any
    dom = collect.([datainfo[iv] for iv in ivs]) #domain for each independent variable
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
        dv_values = [datainfo[dv][coordinate] for dv in dvs]

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
            iv_prev = first(dom[j][idx_prev[j]])
            iv_next = first(dom[j][idx_next[j]])
            dist = iv_next - iv_prev

            # get the values at the prev, current and next indices
            y_prev = datainfo[dv][idx_prev]
            y_next = datainfo[dv][idx_next]

            # compute the finite difference approximation
            Jacobian[i,j] = (y_next - y_prev) / dist
        end
    end
    return Jacobian
end

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
    for (i, dv) in enumerate(dvs)
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
                # Use the finite difference approximation for the cross derivatives
                Hessian[i,j,k] = (jacobian_next[i,k] - jacobian_prev[i,k]) / dist
                # end
            end
        end
    end
    return Hessian
end


