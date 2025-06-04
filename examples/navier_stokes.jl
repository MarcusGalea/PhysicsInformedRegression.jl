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
dom_new = [collect.(dom)...]
test_observations = Observations(samples_test, 
                            idvs_test, 
                            dvs_test,
                            data_test_info;
                            data_structure = Dict{CartesianIndex, Observation}())
test = test_observations[samples_test[4]]
test.coordinate, test.jacobian, test.hessian






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

