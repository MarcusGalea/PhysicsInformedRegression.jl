"""
This function computes the derivatives of states using splines.\n
    finite_diff(f::Vector, t::Vector)
# Arguments \n
- `f`: The vector of states
- `t`: The vector of time points
# Returns \n
- `df`: The vector of derivatives
"""
function spline_derivatives(f::Vector, t::Vector)
    uvals = hcat(f...)
    xs = t[1]:(t[end]-t[1])/(length(t)-1):t[end]
    scaled_itp = [Interpolations.scale(Interpolations.interpolate(uvals[i,:], BSpline(Linear())),xs) for i in 1:size(uvals,1)]
    du_splines = [[first(Interpolations.gradient(scaled_itp[i], tval)) for i =1:length(scaled_itp)] for tval in t]
    return du_splines
end


"""
This function computes the derivatives of states using finite differences.\n
    finite_diff(f::Vector, t::Vector)
# Arguments \n
- `f`: The vector of states
- `t`: The vector of time points
# Returns \n
- `df`: The vector of derivatives
"""
function finite_diff(f::Vector, t::Vector)
    df = zeros(length(t), length(first(f)))
    for i in 2:length(t)-1
        df[i,:] = (f[i+1] - f[i-1])./ (t[i+1] .- t[i-1])
    end
    df[1,:] = (f[2] - f[1])./ (t[2] - t[1])
    df[end,:] = (f[end] - f[end-1])./ (t[end] - t[end-1])
    df = [df[i,:] for i in 1:size(df,1)]
    return df        
end

"""
This function computes the derivatives of states using finite differences.\n
    finite_diff(f::Matrix, t::Vector)
# Arguments \n
- `f`: The Matrix of states
- `t`: The vector of time points
# Returns \n
- `df`: The vector of derivatives
"""
function finite_diff(f::Matrix, t::Vector)
    df = zeros(length(t), size(f, 2))
    for i in 2:length(t)-1
        df[i,:] = (f[i+1,:] - f[i-1,:])./ (t[i+1] .- t[i-1])
    end
    df[1,:] = (f[2,:] - f[1,:])./ (t[2] - t[1])
    df[end,:] = (f[end,:] - f[end-1,:])./ (t[end] - t[end-1])
    return df        
end



