using Pkg
Pkg.activate("./examples//")
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, DomainSets
using PhysicsInformedRegression,MAT,Plots, Interpolations



#### Real data

data = matread("data/CYLINDER_ALL.mat")


#reshape the data (reordered for some reason)
nx = Int(data["ny"]) #number of points in the x direction (449)
ny = Int(data["nx"]) #number of points in the y direction (199)
N = 151 #number of time steps

# Parameters
Lx = 8.96
Ly = 3.96
ST = 0.173 #Strouhal number at Re = 100
T = 1/ST  #period of the vortex shedding
dt = T / (N - 1) #time step size

@parameters ν_inv
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
eq = Dt(ω(x,y,t)) + u(x,y,t) * Dx(ω(x,y,t)) + v(x,y,t) * Dy(ω(x,y,t)) ~ ν_inv * (Dxx(ω(x,y,t)) + Dyy(ω(x,y,t)))

bcs = [
] #irrelevant for the PhysicsInformedRegression, but can be added if needed

@named pdesys = PDESystem(eq, bcs, domains, [x,y,t], [ω(x,y,t), u(x,y,t), v(x,y,t)], [ν_inv])




ω_data = reshape(data["VORTALL"], ny, nx, N)
u_data = reshape(data["UALL"], ny, nx, N)
v_data = reshape(data["VALL"], ny, nx, N)

#reorder indices to match symbolic model
ω_data = permutedims(ω_data, (2, 1, 3)) #from (y,x,t) to (x,y,t)
u_data = permutedims(u_data, (2, 1, 3)) #from (y,x,t) to (x,y,t)
v_data = permutedims(v_data, (2, 1, 3)) #from (y,x,t) to (x,y,t)



ivs = [x,y,t]
dvs = [ω(x,y,t), u(x,y,t), v(x,y,t)]

#show the data at a specific time step
n = 70
heatmap(ω_data[:,:,n]', aspect_ratio = 1, title = "Vorticity at t=$(n*dt)", colorbar_title = "ω", c=:viridis, xlim = (0, nx), ylim = (0, ny), size = (600, 600))

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
Ns = 1000

Random.seed!(3) #for reproducibility
# #size of v_data is (nx, ny, N)
x_samples,y_samples,t_samples = rand(100:150, Ns), rand(50:149, Ns), collect(1:N) #sample time steps (1 to N)


# y_samples = [134,  94,  67, 114, 118,  73, 139, 120,  60,  79,  53, 117, 137,
#         85,  79,  84,  73, 137,  54,  84,  79,  66, 126,  73, 119,  55,
#         59,  83,  79,  63,  90,  54,  88, 137,  69,  77, 134,  63, 115,
#         82,  94, 111,  96, 133,  89, 124,  81,  50, 136,  85].+1
# x_samples = [ 90, 114, 137, 101,  96, 128, 135, 127, 149, 122,  89,  80, 124,
#        107, 102, 131,  81, 122,  96, 110,  87, 100, 122, 120, 102, 140,
#        116,  99, 119, 136, 100, 125, 103, 123, 125, 100, 100,  90, 143,
#        128,  92, 147,  99, 103, 120,  84, 122, 124,  95, 101].+1
# t_samples = collect(1:N) #sample time steps (1 to N)

#convert to Cartesian indices
samples = vcat([CartesianIndex.(x_samples,y_samples, t*ones(Int, Ns)) for t in t_samples]...)
# samples = CartesianIndex.(x_samples, y_samples, t_samples)

interp_fun = BSpline(Cubic(Line(OnGrid())))
paramsest = PhysicsInformedRegression.physics_informed_regression(pdesys, datainfo; samples = samples, interp_fun = interp_fun)



#
observations = PhysicsInformedRegression.Observations(samples, 
                            [x,y,t], 
                            [ω(x,y,t), u(x,y,t), v(x,y,t)],
                            datainfo;
                            data_structure = Dict{CartesianIndex, Observation}())

dvortdx = zeros(Float64, N) #initialize the Jacobian vector for ∂ω/∂x
vort = zeros(Float64, N*Ns)
dvortdx = zeros(Float64, N*Ns) #initialize the Jacobian matrix for ∂ω/∂x
for (i, sample) in enumerate(samples) #iterate over the last Ns samples
    observation = observations[sample]
    vort[i] = observation.dv_values[1] #get the vorticity value at the sample coordinate
    dvortdx[i] = observation.jacobian[1,1] #get the Jacobian value at the sample coordinate
end
vort = reshape(vort, Ns, N) #reshape to (Ns, N) for plotting
dvortdx = reshape(dvortdx, Ns, N) #reshape to (Ns, N) for plotting
idcs = reshape(samples, Ns, N)
dvortdx[12,:]

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
dom = [datainfo[iv] for iv in ivs]

shapesizes = [size(datainfo[dv]) for dv in dvs]
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

for (i,c) in enumerate(samples)
    idx = (i-1)*neqs+1:i*neqs
    observation = observations[c]
    Atotal[idx,:] = [Afun[i,j](observation.dv_values, observation.jacobian, observation.hessian, observation.iv_values) for i=1:neqs, j=1:nparams]
    btotal[idx] = [bfun[i](observation.dv_values, observation.jacobian, observation.hessian, observation.iv_values) for i=1:neqs]
end
##### Test gradient and hessian substitution


fun_to_test = (x,y,t) -> x^2 -y^2 + t + x*y
grad_to_test = (x,y,t) -> [2*x + y, -2*y + x, 1]
hess_to_test = (x,y,t) -> [
    [2, 1, 0],
    [1, -2, 0],
    [0, 0, 0]
]


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
dom_new = [collect.(dom)...]
test_observations = Observations([CartesianIndex(1,1,1)], 
                            idvs_test, 
                            dvs_test,
                            data_test_info;
                            data_structure = Dict{CartesianIndex, Observation}())
test = test_observations[CartesianIndex(1,1,1)] #get the test observation
test.coordinate, test.iv_values, test.jacobian, test.hessian
grad_to_test(3,3,3), hess_to_test(3,3,3)

#indexed symbolic dependent variables
@variables _U[1:length(dvs)] _dU[1:length(dvs),1:length(ivs)] _ddU[1:length(dvs), 1:length(ivs), 1:length(ivs)]
#indexed symbolic independent variables
@independent_variables _X[1:length(ivs)]



redef_dom = PhysicsInformedRegression.uniform_domain(dom) #check if the domain is uniformly spaced 

indepvar_maps = Dict(zip(ivs, _X))
depvar_maps, gradient_maps, hessian_maps = PhysicsInformedRegression.symbolic_maps(A, b, _U, _dU, _ddU, ivs, dvs)

A,b= setup_linear_system(pdesys)
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

