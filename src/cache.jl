import Symbolics: ≲, ~

isa_dataiterator(data) = false

struct AnalysisResults
    objective::Union{Nothing, AnalysisResult}
    constraints::Union{Nothing, Vector{AnalysisResult}}
end

struct OptimizationCache{F, RC, LB, UB, LC, UC, S, O, P, C, M} <:
       SciMLBase.AbstractOptimizationCache
    f::F
    reinit_cache::RC
    lb::LB
    ub::UB
    lcons::LC
    ucons::UC
    sense::S
    opt::O
    progress::P
    callback::C
    manifold::M
    analysis_results::AnalysisResults
    solver_args::NamedTuple
end

function OptimizationCache(prob::SciMLBase.OptimizationProblem, opt;
        callback = DEFAULT_CALLBACK,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        progress = false,
        structural_analysis = false,
        manifold = nothing,
        kwargs...)

    if isa_dataiterator(prob.p)
        reinit_cache = OptimizationBase.ReInitCache(prob.u0, iterate(prob.p)[1])
        reinit_cache_passedon = OptimizationBase.ReInitCache(prob.u0, prob.p)
    else
        reinit_cache = OptimizationBase.ReInitCache(prob.u0, iterate(prob.p)[1])
        reinit_cache_passedon = reinit_cache
    end

    num_cons = prob.ucons === nothing ? 0 : length(prob.ucons)

    f = OptimizationBase.instantiate_function(
        prob.f, reinit_cache, prob.f.adtype, num_cons;
        g = SciMLBase.requiresgradient(opt), h = SciMLBase.requireshessian(opt),
        hv = SciMLBase.requireshessian(opt), fg = SciMLBase.allowsfg(opt),
        fgh = SciMLBase.allowsfgh(opt), cons_j = SciMLBase.requiresconsjac(opt), cons_h = SciMLBase.requiresconshess(opt),
        cons_vjp = SciMLBase.allowsconsjvp(opt), cons_jvp = SciMLBase.allowsconsjvp(opt), lag_h = SciMLBase.requireslagh(opt))

    if structural_analysis
        obj_expr, cons_expr = symify_cache(f, prob)
        try
            obj_res, cons_res = analysis(obj_expr, cons_expr)
        catch err
            throw("Structural analysis requires SymbolicAnalysis.jl to be loaded, either add `using SymbolicAnalysis` to your script or set `structural_analysis = false`.")
        end
    else
        obj_res = nothing
        cons_res = nothing
    end

    return OptimizationCache(f, reinit_cache_passedon, prob.lb, prob.ub, prob.lcons,
        prob.ucons, prob.sense,
        opt, progress, callback, manifold, AnalysisResults(obj_res, cons_res),
        merge((; maxiters, maxtime, abstol, reltol),
            NamedTuple(kwargs)))
end

function SciMLBase.__init(prob::SciMLBase.OptimizationProblem, opt;
        callback = DEFAULT_CALLBACK,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        progress = false,
        kwargs...)
    return OptimizationCache(prob, opt; maxiters, maxtime, abstol, callback,
        reltol, progress,
        kwargs...)
end

# Wrapper for fields that may change in `reinit!(cache)` of a cache.
mutable struct ReInitCache{uType, P}
    u0::uType
    p::P
end
