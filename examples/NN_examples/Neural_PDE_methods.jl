function generate_total_loss(
        prob::SciMLBase.AbstractODEProblem,
        alg::NNODE,
        args...;
        dt = nothing,
        abstol = 1.0f-6,
        reltol = 1.0f-3,
        maxiters = nothing,
        tstops = nothing
    )
    (; u0, tspan, f, p) = prob
    t0 = tspan[1]
    # add estim_collocate, dataset (or nothing) in NNODE
    (;
        param_estim, estim_collocate, dataset, chain, opt, autodiff,
        init_params, batch, additional_loss, estim_collocate,
    ) = alg

    phi, init_params = NeuralPDE.generate_phi_θ(chain, t0, u0, init_params)

    (NeuralPDE.recursive_eltype(init_params) <: Complex && alg.strategy isa QuadratureTraining) &&
        error("QuadratureTraining cannot be used with complex parameters. Use other strategies.")

    init_params = if alg.param_estim
        ComponentArray(; depvar = init_params, p)
    else
        ComponentArray(; depvar = init_params)
    end

    @assert !isinplace(prob) "The NNODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."

    strategy = if alg.strategy === nothing
        if dt !== nothing
            GridTraining(dt)
        else
            QuadratureTraining(;
                quadrature_alg = QuadGKJL(),
                reltol = convert(eltype(u0), reltol), abstol = convert(eltype(u0), abstol),
                maxiters, batch = 0
            )
        end
    else
        alg.strategy
    end

    inner_f = NeuralPDE.generate_loss(strategy, phi, f, autodiff, tspan, p, batch, param_estim)

    if !isempty(dataset) &&
            (length(dataset) < 3 || !(dataset isa Vector{<:Vector{<:AbstractFloat}}))
        error("Invalid dataset. The dataset would be a timeseries (x̂,t,W) with type: Vector{Vector{AbstractFloat}")
    end

    if isempty(dataset) && param_estim && isnothing(additional_loss)
        error("Dataset or an additional loss is required for Inverse problems performing Parameter Estimation.")
    elseif isempty(dataset) && estim_collocate
        error("Dataset is required for Inverse problems performing Parameter Estimation using the Data Quadrature loss function.")
    end

    n_output = length(u0)
    L2lossData = NeuralPDE.generate_L2lossData(dataset, phi, n_output)
    L2loss2 = NeuralPDE.generate_L2loss2(f, autodiff, dataset, phi, n_output)

    # Creates OptimizationFunction Object from total_loss
    function total_loss(θ, _)
        L2_loss = inner_f(θ, phi)

        if param_estim && estim_collocate
            L2_loss = L2_loss + L2lossData(θ, phi) + L2loss2(θ, phi)
        elseif param_estim && !isempty(dataset)
            L2_loss = L2_loss + L2lossData(θ, phi)
        end
        if additional_loss !== nothing
            L2_loss = L2_loss + additional_loss(phi, θ)
        end
        if tstops !== nothing
            num_tstops_points = length(tstops)
            tstops_loss_func = NeuralPDE.evaluate_tstops_loss(
                phi, f, autodiff, tstops, p, batch, param_estim
            )
            tstops_loss = tstops_loss_func(θ, phi)
            if strategy isa GridTraining
                num_original_points = length(tspan[1]:(strategy.dx):tspan[2])
            elseif strategy isa Union{WeightedIntervalTraining, StochasticTraining}
                num_original_points = strategy.points
            else
                return L2_loss + tstops_loss
            end
            total_original_loss = L2_loss * num_original_points
            total_tstops_loss = tstops_loss * num_tstops_points
            total_points = num_original_points + num_tstops_points
            L2_loss = (total_original_loss + total_tstops_loss) / total_points
        end
        return L2_loss
    end
    return deepcopy(total_loss), init_params
end
