import Symbolics: ≲, ~

struct AnalysisResults
    objective::Union{Nothing, AnalysisResult}
    constraints::Union{Nothing, Vector{AnalysisResult}}
end

struct OptimizationCache{F, RC, LB, UB, LC, UC, S, O, D, P, C, M} <:
       SciMLBase.AbstractOptimizationCache
    f::F
    reinit_cache::RC
    lb::LB
    ub::UB
    lcons::LC
    ucons::UC
    sense::S
    opt::O
    data::D
    progress::P
    callback::C
    manifold::M
    analysis_results::AnalysisResults
    solver_args::NamedTuple
end

function OptimizationCache(prob::SciMLBase.OptimizationProblem, opt, data = DEFAULT_DATA;
        callback = DEFAULT_CALLBACK,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        progress = false,
        structural_analysis = false,
        manifold = nothing,
        kwargs...)
    reinit_cache = OptimizationBase.ReInitCache(prob.u0, prob.p)
    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)
    f = OptimizationBase.instantiate_function(
        prob.f, reinit_cache, prob.f.adtype, num_cons;
        g = SciMLBase.requiresgradient(opt), h = SciMLBase.requireshessian(opt),
        hv = SciMLBase.requireshessian(opt), fg = SciMLBase.allowsfg(opt),
        fgh = SciMLBase.allowsfgh(opt), cons_j = SciMLBase.requiresconsjac(opt), cons_h = SciMLBase.requiresconshess(opt),
        cons_vjp = SciMLBase.allowsconsjvp(opt), cons_jvp = SciMLBase.allowsconsjvp(opt), lag_h = SciMLBase.requireslagh(opt))

    if (f.sys === nothing ||
        f.sys isa SymbolicIndexingInterface.SymbolCache{Nothing, Nothing, Nothing}) &&
       structural_analysis
        try
            vars = if prob.u0 isa Matrix
                @variables X[1:size(prob.u0, 1), 1:size(prob.u0, 2)]
            else
                ArrayInterface.restructure(
                    prob.u0, [variable(:x, i) for i in eachindex(prob.u0)])
            end
            params = if prob.p isa SciMLBase.NullParameters
                []
            elseif prob.p isa MTK.MTKParameters
                [variable(:α, i) for i in eachindex(vcat(p...))]
            else
                ArrayInterface.restructure(p, [variable(:α, i) for i in eachindex(p)])
            end

            if prob.u0 isa Matrix
                vars = vars[1]
            end

            obj_expr = f.f(vars, params)

            if SciMLBase.isinplace(prob) && !isnothing(prob.f.cons)
                lhs = Array{Symbolics.Num}(undef, num_cons)
                f.cons(lhs, vars)
                cons = Union{Equation, Inequality}[]

                if !isnothing(prob.lcons)
                    for i in 1:num_cons
                        if !isinf(prob.lcons[i])
                            if prob.lcons[i] != prob.ucons[i]
                                push!(cons, prob.lcons[i] ≲ lhs[i])
                            else
                                push!(cons, lhs[i] ~ prob.ucons[i])
                            end
                        end
                    end
                end

                if !isnothing(prob.ucons)
                    for i in 1:num_cons
                        if !isinf(prob.ucons[i]) && prob.lcons[i] != prob.ucons[i]
                            push!(cons, lhs[i] ≲ prob.ucons[i])
                        end
                    end
                end
                if (isnothing(prob.lcons) || all(isinf, prob.lcons)) &&
                   (isnothing(prob.ucons) || all(isinf, prob.ucons))
                    throw(ArgumentError("Constraints passed have no proper bounds defined.
                    Ensure you pass equal bounds (the scalar that the constraint should evaluate to) for equality constraints
                    or pass the lower and upper bounds for inequality constraints."))
                end
                cons_expr = lhs
            elseif !isnothing(prob.f.cons)
                cons_expr = f.cons(vars, params)
            else
                cons_expr = nothing
            end
        catch err
            throw(ArgumentError("Automatic symbolic expression generation with failed with error: $err.
            Try by setting `structural_analysis = false` instead if the solver doesn't require symbolic expressions."))
        end
    else
        sys = f.sys isa SymbolicIndexingInterface.SymbolCache{Nothing, Nothing, Nothing} ?
              nothing : f.sys
        obj_expr = f.expr
        cons_expr = f.cons_expr === nothing ? nothing : getfield.(f.cons_expr, Ref(:lhs))
    end

    if obj_expr !== nothing && structural_analysis
        obj_expr = obj_expr |> Symbolics.unwrap
        if manifold === nothing
            obj_res = analyze(obj_expr)
        else
            obj_res = analyze(obj_expr, manifold)
        end

        @info "Objective Euclidean curvature: $(obj_res.curvature)"

        if obj_res.gcurvature !== nothing
            @info "Objective Geodesic curvature: $(obj_res.gcurvature)"
        end
    else
        obj_res = nothing
    end

    if cons_expr !== nothing && structural_analysis
        cons_expr = cons_expr .|> Symbolics.unwrap
        if manifold === nothing
            cons_res = analyze.(cons_expr)
        else
            cons_res = analyze.(cons_expr, Ref(manifold))
        end
        for i in 1:num_cons
            @info "Constraints Euclidean curvature: $(cons_res[i].curvature)"

            if cons_res[i].gcurvature !== nothing
                @info "Constraints Geodesic curvature: $(cons_res[i].gcurvature)"
            end
        end
    else
        cons_res = nothing
    end

    return OptimizationCache(f, reinit_cache, prob.lb, prob.ub, prob.lcons,
        prob.ucons, prob.sense,
        opt, data, progress, callback, manifold, AnalysisResults(obj_res, cons_res),
        merge((; maxiters, maxtime, abstol, reltol),
            NamedTuple(kwargs)))
end

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt,
        data = DEFAULT_DATA;
        callback = DEFAULT_CALLBACK,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        progress = false,
        kwargs...)
    return OptimizationCache(prob, opt, data; maxiters, maxtime, abstol, callback,
        reltol, progress,
        kwargs...)
end

# Wrapper for fields that may change in `reinit!(cache)` of a cache.
mutable struct ReInitCache{uType, P}
    u0::uType
    p::P
end
