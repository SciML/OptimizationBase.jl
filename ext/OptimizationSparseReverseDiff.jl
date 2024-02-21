function OptimizationBase.instantiate_function(f::OptimizationFunction{true}, x,
        adtype::AutoSparseReverseDiff,
        p = SciMLBase.NullParameters(),
        num_cons = 0)
    _f = (θ, args...) -> first(f.f(θ, p, args...))

    chunksize = default_chunk_size(length(x))

    if f.grad === nothing
        if adtype.compile
            _tape = ReverseDiff.GradientTape(_f, x)
            tape = ReverseDiff.compile(_tape)
            grad = function (res, θ, args...)
                ReverseDiff.gradient!(res, tape, θ)
            end
        else
            cfg = ReverseDiff.GradientConfig(x)
            grad = (res, θ, args...) -> ReverseDiff.gradient!(res,
                x -> _f(x, args...),
                θ,
                cfg)
        end
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if f.hess === nothing
        hess_sparsity = Symbolics.hessian_sparsity(_f, x)
        hess_colors = SparseDiffTools.matrix_colors(tril(hess_sparsity))
        if adtype.compile
            T = ForwardDiff.Tag(OptimizationSparseReverseTag(), eltype(x))
            xdual = ForwardDiff.Dual{
                typeof(T),
                eltype(x),
                min(chunksize, maximum(hess_colors)),
            }.(x,
                Ref(ForwardDiff.Partials((ones(eltype(x),
                    min(chunksize, maximum(hess_colors)))...,))))
            h_tape = ReverseDiff.GradientTape(_f, xdual)
            htape = ReverseDiff.compile(h_tape)
            function g(res1, θ)
                ReverseDiff.gradient!(res1, htape, θ)
            end
            jaccfg = ForwardColorJacCache(g,
                x;
                tag = typeof(T),
                colorvec = hess_colors,
                sparsity = hess_sparsity)
            hess = function (res, θ, args...)
                SparseDiffTools.forwarddiff_color_jacobian!(res, g, θ, jaccfg)
            end
        else
            hess = function (res, θ, args...)
                res .= SparseDiffTools.forwarddiff_color_jacobian(θ,
                    colorvec = hess_colors,
                    sparsity = hess_sparsity) do θ
                    ReverseDiff.gradient(x -> _f(x, args...), θ)
                end
            end
        end
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            # _θ = ForwardDiff.Dual.(θ, v)
            # res = similar(_θ)
            # grad(res, _θ, args...)
            # H .= getindex.(ForwardDiff.partials.(res), 1)
            res = zeros(length(θ), length(θ))
            hess(res, θ, args...)
            H .= res * v
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> f.cons(res, θ, p)
        cons_oop = (x) -> (_res = zeros(eltype(x), num_cons); cons(_res, x); _res)
    end

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if cons !== nothing && f.cons_j === nothing
        jaccache = SparseDiffTools.sparse_jacobian_cache(AutoSparseForwardDiff(),
            SparseDiffTools.SymbolicsSparsityDetection(),
            cons_oop,
            x,
            fx = zeros(eltype(x), num_cons))
        # let cons = cons, θ = cache.u0, cons_jac_colorvec = cons_jac_colorvec, cons_jac_prototype = cons_jac_prototype, num_cons = num_cons
        #     ForwardColorJacCache(cons, θ;
        #     colorvec = cons_jac_colorvec,
        #     sparsity = cons_jac_prototype,
        #     dx = zeros(eltype(θ), num_cons))
        # end
        cons_jac_prototype = jaccache.jac_prototype
        cons_jac_colorvec = jaccache.coloring
        cons_j = function (J, θ, args...; cons = cons, cache = jaccache.cache)
            forwarddiff_color_jacobian!(J, cons, θ, cache)
            return
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec
    if cons !== nothing && f.cons_h === nothing
        fncs = [(x) -> cons_oop(x)[i] for i in 1:num_cons]
        conshess_sparsity = Symbolics.hessian_sparsity.(fncs, Ref(x))
        conshess_colors = SparseDiffTools.matrix_colors.(conshess_sparsity)
        if adtype.compile
            T = ForwardDiff.Tag(OptimizationSparseReverseTag(), eltype(x))
            xduals = [ForwardDiff.Dual{
                typeof(T),
                eltype(x),
                min(chunksize, maximum(conshess_colors[i])),
            }.(x,
                Ref(ForwardDiff.Partials((ones(eltype(x),
                    min(chunksize, maximum(conshess_colors[i])))...,)))) for i in 1:num_cons]
            consh_tapes = [ReverseDiff.GradientTape(fncs[i], xduals[i]) for i in 1:num_cons]
            conshtapes = ReverseDiff.compile.(consh_tapes)
            function grad_cons(res1, θ, htape)
                ReverseDiff.gradient!(res1, htape, θ)
            end
            gs = [(res1, x) -> grad_cons(res1, x, conshtapes[i]) for i in 1:num_cons]
            jaccfgs = [ForwardColorJacCache(gs[i],
                x;
                tag = typeof(T),
                colorvec = conshess_colors[i],
                sparsity = conshess_sparsity[i]) for i in 1:num_cons]
            cons_h = function (res, θ, args...)
                for i in 1:num_cons
                    SparseDiffTools.forwarddiff_color_jacobian!(res[i],
                        gs[i],
                        θ,
                        jaccfgs[i])
                end
            end
        else
            cons_h = function (res, θ)
                for i in 1:num_cons
                    res[i] .= SparseDiffTools.forwarddiff_color_jacobian(θ,
                        colorvec = conshess_colors[i],
                        sparsity = conshess_sparsity[i]) do θ
                        ReverseDiff.gradient(fncs[i], θ)
                    end
                end
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, p)
    end

    if f.lag_h === nothing
        lag_h = nothing # Consider implementing this
    else
        lag_h = (res, θ, σ, μ) -> f.lag_h(res, θ, σ, μ, p)
    end
    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = hess_sparsity,
        hess_colorvec = hess_colors,
        cons_jac_prototype = cons_jac_prototype,
        cons_jac_colorvec = cons_jac_colorvec,
        cons_hess_prototype = conshess_sparsity,
        cons_hess_colorvec = conshess_colors,
        lag_h, f.lag_hess_prototype)
end

function OptimizationBase.instantiate_function(f::OptimizationFunction{true},
        cache::OptimizationBase.ReInitCache,
        adtype::AutoSparseReverseDiff, num_cons = 0)
    _f = (θ, args...) -> first(f.f(θ, cache.p, args...))

    chunksize = default_chunk_size(length(cache.u0))

    if f.grad === nothing
        if adtype.compile
            _tape = ReverseDiff.GradientTape(_f, cache.u0)
            tape = ReverseDiff.compile(_tape)
            grad = function (res, θ, args...)
                ReverseDiff.gradient!(res, tape, θ)
            end
        else
            cfg = ReverseDiff.GradientConfig(cache.u0)
            grad = (res, θ, args...) -> ReverseDiff.gradient!(res,
                x -> _f(x, args...),
                θ,
                cfg)
        end
    else
        grad = (G, θ, args...) -> f.grad(G, θ, cache.p, args...)
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if f.hess === nothing
        hess_sparsity = Symbolics.hessian_sparsity(_f, cache.u0)
        hess_colors = SparseDiffTools.matrix_colors(tril(hess_sparsity))
        if adtype.compile
            T = ForwardDiff.Tag(OptimizationSparseReverseTag(), eltype(cache.u0))
            xdual = ForwardDiff.Dual{
                typeof(T),
                eltype(cache.u0),
                min(chunksize, maximum(hess_colors)),
            }.(cache.u0,
                Ref(ForwardDiff.Partials((ones(eltype(cache.u0),
                    min(chunksize, maximum(hess_colors)))...,))))
            h_tape = ReverseDiff.GradientTape(_f, xdual)
            htape = ReverseDiff.compile(h_tape)
            function g(res1, θ)
                ReverseDiff.gradient!(res1, htape, θ)
            end
            jaccfg = ForwardColorJacCache(g,
                cache.u0;
                tag = typeof(T),
                colorvec = hess_colors,
                sparsity = hess_sparsity)
            hess = function (res, θ, args...)
                SparseDiffTools.forwarddiff_color_jacobian!(res, g, θ, jaccfg)
            end
        else
            hess = function (res, θ, args...)
                res .= SparseDiffTools.forwarddiff_color_jacobian(θ,
                    colorvec = hess_colors,
                    sparsity = hess_sparsity) do θ
                    ReverseDiff.gradient(x -> _f(x, args...), θ)
                end
            end
        end
    else
        hess = (H, θ, args...) -> f.hess(H, θ, cache.p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            # _θ = ForwardDiff.Dual.(θ, v)
            # res = similar(_θ)
            # grad(res, _θ, args...)
            # H .= getindex.(ForwardDiff.partials.(res), 1)
            res = zeros(length(θ), length(θ))
            hess(res, θ, args...)
            H .= res * v
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = function (res, θ)
            f.cons(res, θ, cache.p)
            return
        end
        cons_oop = (x) -> (_res = zeros(eltype(x), num_cons); cons(_res, x); _res)
    end

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if cons !== nothing && f.cons_j === nothing
        # cons_jac_prototype = Symbolics.jacobian_sparsity(cons,
        # zeros(eltype(cache.u0), num_cons),
        # cache.u0)
        # cons_jac_colorvec = matrix_colors(cons_jac_prototype)
        jaccache = SparseDiffTools.sparse_jacobian_cache(AutoSparseForwardDiff(),
            SparseDiffTools.SymbolicsSparsityDetection(),
            cons_oop,
            cache.u0,
            fx = zeros(eltype(cache.u0), num_cons))
        # let cons = cons, θ = cache.u0, cons_jac_colorvec = cons_jac_colorvec, cons_jac_prototype = cons_jac_prototype, num_cons = num_cons
        #     ForwardColorJacCache(cons, θ;
        #     colorvec = cons_jac_colorvec,
        #     sparsity = cons_jac_prototype,
        #     dx = zeros(eltype(θ), num_cons))
        # end
        cons_jac_prototype = jaccache.jac_prototype
        cons_jac_colorvec = jaccache.coloring
        cons_j = function (J, θ)
            forwarddiff_color_jacobian!(J, cons, θ, jaccache.cache)
            return
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, cache.p)
    end

    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec
    if cons !== nothing && f.cons_h === nothing
        fncs = map(1:num_cons) do i
            function (x)
                res = zeros(eltype(x), num_cons)
                f.cons(res, x, cache.p)
                return res[i]
            end
        end
        conshess_sparsity = map(1:num_cons) do i
            let fnc = fncs[i], θ = cache.u0
                Symbolics.hessian_sparsity(fnc, θ)
            end
        end
        conshess_colors = SparseDiffTools.matrix_colors.(conshess_sparsity)
        if adtype.compile
            T = ForwardDiff.Tag(OptimizationSparseReverseTag(), eltype(cache.u0))
            xduals = [ForwardDiff.Dual{
                typeof(T),
                eltype(cache.u0),
                min(chunksize, maximum(conshess_colors[i])),
            }.(cache.u0,
                Ref(ForwardDiff.Partials((ones(eltype(cache.u0),
                    min(chunksize, maximum(conshess_colors[i])))...,)))) for i in 1:num_cons]
            consh_tapes = [ReverseDiff.GradientTape(fncs[i], xduals[i]) for i in 1:num_cons]
            conshtapes = ReverseDiff.compile.(consh_tapes)
            function grad_cons(res1, θ, htape)
                ReverseDiff.gradient!(res1, htape, θ)
            end
            gs = let conshtapes = conshtapes
                map(1:num_cons) do i
                    function (res1, x)
                        grad_cons(res1, x, conshtapes[i])
                    end
                end
            end
            jaccfgs = [ForwardColorJacCache(gs[i],
                cache.u0;
                tag = typeof(T),
                colorvec = conshess_colors[i],
                sparsity = conshess_sparsity[i]) for i in 1:num_cons]
            cons_h = function (res, θ)
                for i in 1:num_cons
                    SparseDiffTools.forwarddiff_color_jacobian!(res[i],
                        gs[i],
                        θ,
                        jaccfgs[i])
                end
            end
        else
            cons_h = function (res, θ)
                for i in 1:num_cons
                    res[i] .= SparseDiffTools.forwarddiff_color_jacobian(θ,
                        colorvec = conshess_colors[i],
                        sparsity = conshess_sparsity[i]) do θ
                        ReverseDiff.gradient(fncs[i], θ)
                    end
                end
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, cache.p)
    end

    if f.lag_h === nothing
        lag_h = nothing # Consider implementing this
    else
        lag_h = (res, θ, σ, μ) -> f.lag_h(res, θ, σ, μ, cache.p)
    end

    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = hess_sparsity,
        hess_colorvec = hess_colors,
        cons_jac_prototype = cons_jac_prototype,
        cons_jac_colorvec = cons_jac_colorvec,
        cons_hess_prototype = conshess_sparsity,
        cons_hess_colorvec = conshess_colors,
        lag_h, f.lag_hess_prototype)
end

function OptimizationBase.instantiate_function(f::OptimizationFunction{false}, x,
        adtype::AutoSparseReverseDiff,
        p = SciMLBase.NullParameters(),
        num_cons = 0)
    _f = (θ, args...) -> first(f.f(θ, p, args...))

    chunksize = default_chunk_size(length(x))

    if f.grad === nothing
        if adtype.compile
            _tape = ReverseDiff.GradientTape(_f, x)
            tape = ReverseDiff.compile(_tape)
            grad = function (θ, args...)
                ReverseDiff.gradient!(tape, θ)
            end
        else
            cfg = ReverseDiff.GradientConfig(x)
            grad = (θ, args...) -> ReverseDiff.gradient(x -> _f(x, args...),
                θ,
                cfg)
        end
    else
        grad = (θ, args...) -> f.grad(θ, p, args...)
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if f.hess === nothing
        hess_sparsity = Symbolics.hessian_sparsity(_f, x)
        hess_colors = SparseDiffTools.matrix_colors(tril(hess_sparsity))
        if adtype.compile
            T = ForwardDiff.Tag(OptimizationSparseReverseTag(), eltype(x))
            xdual = ForwardDiff.Dual{
                typeof(T),
                eltype(x),
                min(chunksize, maximum(hess_colors)),
            }.(x,
                Ref(ForwardDiff.Partials((ones(eltype(x),
                    min(chunksize, maximum(hess_colors)))...,))))
            h_tape = ReverseDiff.GradientTape(_f, xdual)
            htape = ReverseDiff.compile(h_tape)
            function g(θ)
                ReverseDiff.gradient!(htape, θ)
            end
            jaccfg = ForwardColorJacCache(g,
                x;
                tag = typeof(T),
                colorvec = hess_colors,
                sparsity = hess_sparsity)
            hess = function (θ, args...)
                return SparseDiffTools.forwarddiff_color_jacobian(g, θ, jaccfg)
            end
        else
            hess = function (θ, args...)
                return SparseDiffTools.forwarddiff_color_jacobian(θ,
                    colorvec = hess_colors,
                    sparsity = hess_sparsity) do θ
                    ReverseDiff.gradient(x -> _f(x, args...), θ)
                end
            end
        end
    else
        hess = (θ, args...) -> f.hess(θ, p, args...)
    end

    if f.hv === nothing
        hv = function (θ, v, args...)
            # _θ = ForwardDiff.Dual.(θ, v)
            # res = similar(_θ)
            # grad(res, _θ, args...)
            # H .= getindex.(ForwardDiff.partials.(res), 1)
            res = zeros(length(θ), length(θ))
            hess(θ, args...) * v
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (θ) -> f.cons(θ, p)
        cons_oop = cons
    end

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if cons !== nothing && f.cons_j === nothing
        jaccache = SparseDiffTools.sparse_jacobian_cache(AutoSparseForwardDiff(),
            SparseDiffTools.SymbolicsSparsityDetection(),
            cons_oop,
            x,
            fx = zeros(eltype(x), num_cons))
        # let cons = cons, θ = cache.u0, cons_jac_colorvec = cons_jac_colorvec, cons_jac_prototype = cons_jac_prototype, num_cons = num_cons
        #     ForwardColorJacCache(cons, θ;
        #     colorvec = cons_jac_colorvec,
        #     sparsity = cons_jac_prototype,
        #     dx = zeros(eltype(θ), num_cons))
        # end
        cons_jac_prototype = jaccache.jac_prototype
        cons_jac_colorvec = jaccache.coloring
        cons_j = function (θ, args...; cons = cons, cache = jaccache.cache)
            if num_cons > 1
                return forwarddiff_color_jacobian(cons, θ, cache)
            else
                return forwarddiff_color_jacobian(cons, θ, cache)[1, :]
            end
        end
    else
        cons_j = (θ) -> f.cons_j(θ, p)
    end

    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec
    if cons !== nothing && f.cons_h === nothing
        fncs = [(x) -> cons_oop(x)[i] for i in 1:num_cons]
        conshess_sparsity = Symbolics.hessian_sparsity.(fncs, Ref(x))
        conshess_colors = SparseDiffTools.matrix_colors.(conshess_sparsity)
        if adtype.compile
            T = ForwardDiff.Tag(OptimizationSparseReverseTag(), eltype(x))
            xduals = [ForwardDiff.Dual{
                typeof(T),
                eltype(x),
                min(chunksize, maximum(conshess_colors[i])),
            }.(x,
                Ref(ForwardDiff.Partials((ones(eltype(x),
                    min(chunksize, maximum(conshess_colors[i])))...,)))) for i in 1:num_cons]
            consh_tapes = [ReverseDiff.GradientTape(fncs[i], xduals[i]) for i in 1:num_cons]
            conshtapes = ReverseDiff.compile.(consh_tapes)
            function grad_cons(θ, htape)
                ReverseDiff.gradient!(htape, θ)
            end
            gs = [(x) -> grad_cons(x, conshtapes[i]) for i in 1:num_cons]
            jaccfgs = [ForwardColorJacCache(gs[i],
                x;
                tag = typeof(T),
                colorvec = conshess_colors[i],
                sparsity = conshess_sparsity[i]) for i in 1:num_cons]
            cons_h = function (θ, args...)
                map(1:num_cons) do i
                    SparseDiffTools.forwarddiff_color_jacobian(gs[i],
                        θ,
                        jaccfgs[i])
                end
            end
        else
            cons_h = function (θ)
                map(1:num_cons) do i
                    SparseDiffTools.forwarddiff_color_jacobian(θ,
                        colorvec = conshess_colors[i],
                        sparsity = conshess_sparsity[i]) do θ
                        ReverseDiff.gradient(fncs[i], θ)
                    end
                end
            end
        end
    else
        cons_h = (θ) -> f.cons_h(θ, p)
    end

    if f.lag_h === nothing
        lag_h = nothing # Consider implementing this
    else
        lag_h = (θ, σ, μ) -> f.lag_h(θ, σ, μ, p)
    end
    return OptimizationFunction{false}(f.f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = hess_sparsity,
        hess_colorvec = hess_colors,
        cons_jac_prototype = cons_jac_prototype,
        cons_jac_colorvec = cons_jac_colorvec,
        cons_hess_prototype = conshess_sparsity,
        cons_hess_colorvec = conshess_colors,
        lag_h, f.lag_hess_prototype)
end

function OptimizationBase.instantiate_function(f::OptimizationFunction{false},
        cache::OptimizationBase.ReInitCache,
        adtype::AutoSparseReverseDiff, num_cons = 0)
    _f = (θ, args...) -> first(f.f(θ, cache.p, args...))

    chunksize = default_chunk_size(length(cache.u0))
    p = cache.p
    x = cache.u0

    if f.grad === nothing
        if adtype.compile
            _tape = ReverseDiff.GradientTape(_f, x)
            tape = ReverseDiff.compile(_tape)
            grad = function (θ, args...)
                ReverseDiff.gradient!(tape, θ)
            end
        else
            cfg = ReverseDiff.GradientConfig(x)
            grad = (θ, args...) -> ReverseDiff.gradient(x -> _f(x, args...),
                θ,
                cfg)
        end
    else
        grad = (θ, args...) -> f.grad(θ, p, args...)
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if f.hess === nothing
        hess_sparsity = Symbolics.hessian_sparsity(_f, x)
        hess_colors = SparseDiffTools.matrix_colors(tril(hess_sparsity))
        if adtype.compile
            T = ForwardDiff.Tag(OptimizationSparseReverseTag(), eltype(x))
            xdual = ForwardDiff.Dual{
                typeof(T),
                eltype(x),
                min(chunksize, maximum(hess_colors)),
            }.(x,
                Ref(ForwardDiff.Partials((ones(eltype(x),
                    min(chunksize, maximum(hess_colors)))...,))))
            h_tape = ReverseDiff.GradientTape(_f, xdual)
            htape = ReverseDiff.compile(h_tape)
            function g(θ)
                ReverseDiff.gradient!(htape, θ)
            end
            jaccfg = ForwardColorJacCache(g,
                x;
                tag = typeof(T),
                colorvec = hess_colors,
                sparsity = hess_sparsity)
            hess = function (θ, args...)
                return SparseDiffTools.forwarddiff_color_jacobian(g, θ, jaccfg)
            end
        else
            hess = function (θ, args...)
                return SparseDiffTools.forwarddiff_color_jacobian(θ,
                    colorvec = hess_colors,
                    sparsity = hess_sparsity) do θ
                    ReverseDiff.gradient(x -> _f(x, args...), θ)
                end
            end
        end
    else
        hess = (θ, args...) -> f.hess(θ, p, args...)
    end

    if f.hv === nothing
        hv = function (θ, v, args...)
            # _θ = ForwardDiff.Dual.(θ, v)
            # res = similar(_θ)
            # grad(res, _θ, args...)
            # H .= getindex.(ForwardDiff.partials.(res), 1)
            res = zeros(length(θ), length(θ))
            hess(θ, args...) * v
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (θ) -> f.cons(θ, p)
        cons_oop = cons
    end

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if cons !== nothing && f.cons_j === nothing
        jaccache = SparseDiffTools.sparse_jacobian_cache(AutoSparseForwardDiff(),
            SparseDiffTools.SymbolicsSparsityDetection(),
            cons_oop,
            x,
            fx = zeros(eltype(x), num_cons))
        # let cons = cons, θ = cache.u0, cons_jac_colorvec = cons_jac_colorvec, cons_jac_prototype = cons_jac_prototype, num_cons = num_cons
        #     ForwardColorJacCache(cons, θ;
        #     colorvec = cons_jac_colorvec,
        #     sparsity = cons_jac_prototype,
        #     dx = zeros(eltype(θ), num_cons))
        # end
        cons_jac_prototype = jaccache.jac_prototype
        cons_jac_colorvec = jaccache.coloring
        cons_j = function (θ, args...; cons = cons, cache = jaccache.cache)
            if num_cons > 1
                return forwarddiff_color_jacobian(cons, θ, cache)
            else
                return forwarddiff_color_jacobian(cons, θ, cache)[1, :]
            end
        end
    else
        cons_j = (θ) -> f.cons_j(θ, p)
    end

    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec
    if cons !== nothing && f.cons_h === nothing
        fncs = [(x) -> cons_oop(x)[i] for i in 1:num_cons]
        conshess_sparsity = Symbolics.hessian_sparsity.(fncs, Ref(x))
        conshess_colors = SparseDiffTools.matrix_colors.(conshess_sparsity)
        if adtype.compile
            T = ForwardDiff.Tag(OptimizationSparseReverseTag(), eltype(x))
            xduals = [ForwardDiff.Dual{
                typeof(T),
                eltype(x),
                min(chunksize, maximum(conshess_colors[i])),
            }.(x,
                Ref(ForwardDiff.Partials((ones(eltype(x),
                    min(chunksize, maximum(conshess_colors[i])))...,)))) for i in 1:num_cons]
            consh_tapes = [ReverseDiff.GradientTape(fncs[i], xduals[i]) for i in 1:num_cons]
            conshtapes = ReverseDiff.compile.(consh_tapes)
            function grad_cons(θ, htape)
                ReverseDiff.gradient!(htape, θ)
            end
            gs = [(x) -> grad_cons(x, conshtapes[i]) for i in 1:num_cons]
            jaccfgs = [ForwardColorJacCache(gs[i],
                x;
                tag = typeof(T),
                colorvec = conshess_colors[i],
                sparsity = conshess_sparsity[i]) for i in 1:num_cons]
            cons_h = function (θ, args...)
                map(1:num_cons) do i
                    SparseDiffTools.forwarddiff_color_jacobian(gs[i],
                        θ,
                        jaccfgs[i])
                end
            end
        else
            cons_h = function (θ)
                map(1:num_cons) do i
                    SparseDiffTools.forwarddiff_color_jacobian(θ,
                        colorvec = conshess_colors[i],
                        sparsity = conshess_sparsity[i]) do θ
                        ReverseDiff.gradient(fncs[i], θ)
                    end
                end
            end
        end
    else
        cons_h = (θ) -> f.cons_h(θ, p)
    end

    if f.lag_h === nothing
        lag_h = nothing # Consider implementing this
    else
        lag_h = (θ, σ, μ) -> f.lag_h(θ, σ, μ, p)
    end
    return OptimizationFunction{false}(f.f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = hess_sparsity,
        hess_colorvec = hess_colors,
        cons_jac_prototype = cons_jac_prototype,
        cons_jac_colorvec = cons_jac_colorvec,
        cons_hess_prototype = conshess_sparsity,
        cons_hess_colorvec = conshess_colors,
        lag_h, f.lag_hess_prototype)
end
