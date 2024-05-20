function OptimizationBase.instantiate_function(f::OptimizationFunction{true}, x,
        adtype::AutoSparse{<:AutoForwardDiff{_chunksize}}, p,
        num_cons = 0) where {_chunksize}
    if maximum(getfield.(methods(f.f), :nargs)) > 3
        error("$(string(adtype)) with SparseDiffTools does not support functions with more than 2 arguments")
    end
    chunksize = _chunksize === nothing ? default_chunk_size(length(x)) : _chunksize

    _f = (θ, args...) -> first(f.f(θ, p, args...))

    if f.grad === nothing
        gradcfg = ForwardDiff.GradientConfig(_f, x, ForwardDiff.Chunk{chunksize}())
        grad = (res, θ, args...) -> ForwardDiff.gradient!(res, x -> _f(x, args...), θ,
            gradcfg, Val{false}())
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if f.hess === nothing
        hess_sparsity = isnothing(f.hess_prototype) ? Symbolics.hessian_sparsity(_f, x) : f.hess_prototype
        hess_colors = matrix_colors(hess_sparsity)
        hess = (res, θ, args...) -> numauto_color_hessian!(res, x -> _f(x, args...), θ,
            ForwardColorHesCache(_f, x,
                hess_colors,
                hess_sparsity,
                (res, θ) -> grad(res,
                    θ,
                    args...)))
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            num_hesvecgrad!(H, (res, x) -> grad(res, x, args...), θ, v)
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
        cons_jac_prototype = isnothing(f.cons_jac_prototype) ? Symbolics.jacobian_sparsity(cons, zeros(eltype(x), num_cons),
            x) : f.cons_jac_prototype
        cons_jac_colorvec = matrix_colors(cons_jac_prototype)
        jaccache = ForwardColorJacCache(cons,
            x,
            chunksize;
            colorvec = cons_jac_colorvec,
            sparsity = cons_jac_prototype,
            dx = zeros(eltype(x), num_cons))
        cons_j = function (J, θ)
            forwarddiff_color_jacobian!(J, cons, θ, jaccache)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    cons_hess_caches = [(; sparsity = f.cons_hess_prototype, colors = f.cons_hess_colorvec)]
    if cons !== nothing && f.cons_h === nothing
        function gen_conshess_cache(_f, x, i)
            conshess_sparsity = isnothing(f.cons_hess_prototype) ? copy(Symbolics.hessian_sparsity(_f, x)) : f.cons_hess_prototype[i]
            conshess_colors = matrix_colors(conshess_sparsity)
            hesscache = ForwardColorHesCache(_f, x, conshess_colors,
                conshess_sparsity)
            return hesscache
        end

        fcons = [(x) -> (_res = zeros(eltype(x), num_cons);
        cons(_res, x);
        _res[i]) for i in 1:num_cons]
        cons_hess_caches = [gen_conshess_cache(fcons[i], x, i) for i in 1:num_cons]
        cons_h = function (res, θ)
            fetch.([Threads.@spawn numauto_color_hessian!(res[i], fcons[i], θ, cons_hess_caches[i]) for i in 1:num_cons])
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
        cons_jac_colorvec = cons_jac_colorvec,
        cons_jac_prototype = cons_jac_prototype,
        cons_hess_prototype = getfield.(cons_hess_caches, :sparsity),
        cons_hess_colorvec = getfield.(cons_hess_caches, :colors),
        lag_h, f.lag_hess_prototype)
end

function OptimizationBase.instantiate_function(f::OptimizationFunction{true},
        cache::OptimizationBase.ReInitCache,
        adtype::AutoSparse{<:AutoForwardDiff{_chunksize}},
        num_cons = 0) where {_chunksize}
    if maximum(getfield.(methods(f.f), :nargs)) > 3
        error("$(string(adtype)) with SparseDiffTools does not support functions with more than 2 arguments")
    end
    chunksize = _chunksize === nothing ? default_chunk_size(length(cache.u0)) : _chunksize
    
    x = cache.u0
    p = cache.p

    _f = (θ, args...) -> first(f.f(θ, cache.p, args...))

    if f.grad === nothing
        gradcfg = ForwardDiff.GradientConfig(_f, cache.u0, ForwardDiff.Chunk{chunksize}())
        grad = (res, θ, args...) -> ForwardDiff.gradient!(res, x -> _f(x, args...), θ,
            gradcfg, Val{false}())
    else
        grad = (G, θ, args...) -> f.grad(G, θ, cache.p, args...)
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if f.hess === nothing
        hess_sparsity = isnothing(f.hess_prototype) ? Symbolics.hessian_sparsity(_f, x) : f.hess_prototype
        hess_colors = matrix_colors(hess_sparsity)
        hess = (res, θ, args...) -> numauto_color_hessian!(res, x -> _f(x, args...), θ,
            ForwardColorHesCache(_f, x,
                hess_colors,
                hess_sparsity,
                (res, θ) -> grad(res,
                    θ,
                    args...)))
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            num_hesvecgrad!(H, (res, x) -> grad(res, x, args...), θ, v)
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
        cons_jac_prototype = isnothing(f.cons_jac_prototype) ? Symbolics.jacobian_sparsity(cons, zeros(eltype(x), num_cons),
            x) : f.cons_jac_prototype
        cons_jac_colorvec = matrix_colors(cons_jac_prototype)
        jaccache = ForwardColorJacCache(cons,
            x,
            chunksize;
            colorvec = cons_jac_colorvec,
            sparsity = cons_jac_prototype,
            dx = zeros(eltype(x), num_cons))
        cons_j = function (J, θ)
            forwarddiff_color_jacobian!(J, cons, θ, jaccache)
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    cons_hess_caches = [(; sparsity = f.cons_hess_prototype, colors = f.cons_hess_colorvec)]
    if cons !== nothing && f.cons_h === nothing
        function gen_conshess_cache(_f, x, i)
            conshess_sparsity = isnothing(f.cons_hess_prototype) ? copy(Symbolics.hessian_sparsity(_f, x)) : f.cons_hess_prototype[i]
            conshess_colors = matrix_colors(conshess_sparsity)
            hesscache = ForwardColorHesCache(_f, x, conshess_colors,
                conshess_sparsity)
            return hesscache
        end

        fcons = [(x) -> (_res = zeros(eltype(x), num_cons);
        cons(_res, x);
        _res[i]) for i in 1:num_cons]
        cons_hess_caches = [gen_conshess_cache(fcons[i], x, i) for i in 1:num_cons]
        cons_h = function (res, θ)
            fetch.([Threads.@spawn numauto_color_hessian!(res[i], fcons[i], θ, cons_hess_caches[i]) for i in 1:num_cons])
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, p)
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
        cons_hess_prototype = getfield.(cons_hess_caches, :sparsity),
        cons_hess_colorvec = getfield.(cons_hess_caches, :colors),
        lag_h, f.lag_hess_prototype)
end

function OptimizationBase.instantiate_function(f::OptimizationFunction{false}, x,
        adtype::AutoSparse{<:AutoForwardDiff{_chunksize}}, p,
        num_cons = 0) where {_chunksize}
    if maximum(getfield.(methods(f.f), :nargs)) > 3
        error("$(string(adtype)) with SparseDiffTools does not support functions with more than 2 arguments")
    end
    chunksize = _chunksize === nothing ? default_chunk_size(length(x)) : _chunksize

    _f = (θ, args...) -> first(f.f(θ, p, args...))

    if f.grad === nothing
        gradcfg = ForwardDiff.GradientConfig(_f, x, ForwardDiff.Chunk{chunksize}())
        grad = (θ, args...) -> ForwardDiff.gradient(x -> _f(x, args...), θ,
            gradcfg, Val{false}())
    else
        grad = (θ, args...) -> f.grad(θ, p, args...)
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if f.hess === nothing
        hess_sparsity = Symbolics.hessian_sparsity(_f, x)
        hess_colors = matrix_colors(tril(hess_sparsity))
        hess = (θ, args...) -> numauto_color_hessian(x -> _f(x, args...), θ,
            ForwardColorHesCache(_f, x,
                hess_colors,
                hess_sparsity,
                (G, θ) -> (G .= grad(θ,
                    args...))))
    else
        hess = (θ, args...) -> f.hess(θ, p, args...)
    end

    if f.hv === nothing
        hv = function (θ, v, args...)
            num_hesvecgrad((x) -> grad(x, args...), θ, v)
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (θ) -> f.cons(θ, p)
    end

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if cons !== nothing && f.cons_j === nothing
        res = zeros(eltype(x), num_cons)
        cons_jac_prototype = Symbolics.jacobian_sparsity((res, x) -> (res .= cons(x)),
            res,
            x)
        cons_jac_colorvec = matrix_colors(cons_jac_prototype)
        jaccache = ForwardColorJacCache(cons,
            x,
            chunksize;
            colorvec = cons_jac_colorvec,
            sparsity = cons_jac_prototype)
        cons_j = function (θ)
            if num_cons > 1
                return forwarddiff_color_jacobian(cons, θ, jaccache)
            else
                return forwarddiff_color_jacobian(cons, θ, jaccache)[1, :]
            end
        end
    else
        cons_j = (θ) -> f.cons_j(θ, p)
    end

    cons_hess_caches = [(; sparsity = f.cons_hess_prototype, colors = f.cons_hess_colorvec)]
    if cons !== nothing && f.cons_h === nothing
        function gen_conshess_cache(_f, x)
            conshess_sparsity = copy(Symbolics.hessian_sparsity(_f, x))
            conshess_colors = matrix_colors(conshess_sparsity)
            hesscache = ForwardColorHesCache(_f, x, conshess_colors,
                conshess_sparsity)
            return hesscache
        end

        fcons = [(x) -> cons(x)[i] for i in 1:num_cons]
        cons_hess_caches = gen_conshess_cache.(fcons, Ref(x))
        cons_h = function (θ)
            map(1:num_cons) do i
                numauto_color_hessian(fcons[i], θ, cons_hess_caches[i])
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
        cons_jac_colorvec = cons_jac_colorvec,
        cons_jac_prototype = cons_jac_prototype,
        cons_hess_prototype = getfield.(cons_hess_caches, :sparsity),
        cons_hess_colorvec = getfield.(cons_hess_caches, :colors),
        lag_h, f.lag_hess_prototype)
end

function OptimizationBase.instantiate_function(f::OptimizationFunction{false},
        cache::OptimizationBase.ReInitCache,
        adtype::AutoSparse{<:AutoForwardDiff{_chunksize}},
        num_cons = 0) where {_chunksize}
    if maximum(getfield.(methods(f.f), :nargs)) > 3
        error("$(string(adtype)) with SparseDiffTools does not support functions with more than 2 arguments")
    end
    chunksize = _chunksize === nothing ? default_chunk_size(length(cache.u0)) : _chunksize

    _f = (θ, args...) -> first(f.f(θ, cache.p, args...))

    p = cache.p

    if f.grad === nothing
        gradcfg = ForwardDiff.GradientConfig(_f, x, ForwardDiff.Chunk{chunksize}())
        grad = (θ, args...) -> ForwardDiff.gradient(x -> _f(x, args...), θ,
            gradcfg, Val{false}())
    else
        grad = (θ, args...) -> f.grad(θ, p, args...)
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if f.hess === nothing
        hess_sparsity = Symbolics.hessian_sparsity(_f, x)
        hess_colors = matrix_colors(tril(hess_sparsity))
        hess = (θ, args...) -> numauto_color_hessian(x -> _f(x, args...), θ,
            ForwardColorHesCache(_f, x,
                hess_colors,
                hess_sparsity,
                (G, θ) -> (G .= grad(θ,
                    args...))))
    else
        hess = (θ, args...) -> f.hess(θ, p, args...)
    end

    if f.hv === nothing
        hv = function (θ, v, args...)
            num_hesvecgrad((x) -> grad(res, x, args...), θ, v)
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (θ) -> f.cons(θ, p)
    end

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if cons !== nothing && f.cons_j === nothing
        res = zeros(eltype(x), num_cons)
        cons_jac_prototype = Symbolics.jacobian_sparsity((res, x) -> (res .= cons(x)),
            res,
            x)
        cons_jac_colorvec = matrix_colors(cons_jac_prototype)
        jaccache = ForwardColorJacCache(cons,
            x,
            chunksize;
            colorvec = cons_jac_colorvec,
            sparsity = cons_jac_prototype)
        cons_j = function (θ)
            if num_cons > 1
                return forwarddiff_color_jacobian(cons, θ, jaccache)
            else
                return forwarddiff_color_jacobian(cons, θ, jaccache)[1, :]
            end
        end
    else
        cons_j = (θ) -> f.cons_j(θ, p)
    end

    cons_hess_caches = [(; sparsity = f.cons_hess_prototype, colors = f.cons_hess_colorvec)]
    if cons !== nothing && f.cons_h === nothing
        function gen_conshess_cache(_f, x)
            conshess_sparsity = copy(Symbolics.hessian_sparsity(_f, x))
            conshess_colors = matrix_colors(conshess_sparsity)
            hesscache = ForwardColorHesCache(_f, x, conshess_colors,
                conshess_sparsity)
            return hesscache
        end

        fcons = [(x) -> cons(x)[i] for i in 1:num_cons]
        cons_hess_caches = gen_conshess_cache.(fcons, Ref(x))
        cons_h = function (θ)
            map(1:num_cons) do i
                numauto_color_hessian(fcons[i], θ, cons_hess_caches[i])
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
        cons_jac_colorvec = cons_jac_colorvec,
        cons_jac_prototype = cons_jac_prototype,
        cons_hess_prototype = getfield.(cons_hess_caches, :sparsity),
        cons_hess_colorvec = getfield.(cons_hess_caches, :colors),
        lag_h, f.lag_hess_prototype)
end
