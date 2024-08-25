module OptimizationZygoteExt

using OptimizationBase, SparseArrays
using OptimizationBase.FastClosures
import OptimizationBase.ArrayInterface
import OptimizationBase.SciMLBase: OptimizationFunction
import OptimizationBase.LinearAlgebra: I, dot
import DifferentiationInterface
import DifferentiationInterface: prepare_gradient, prepare_hessian, prepare_hvp,
                                 prepare_pullback, prepare_pushforward, pullback!,
                                 pushforward!,
                                 pullback, pushforward,
                                 prepare_jacobian, value_and_gradient!, value_and_gradient,
                                 value_derivative_and_second_derivative!,
                                 value_derivative_and_second_derivative,
                                 gradient!, hessian!, hvp!, jacobian!, gradient, hessian,
                                 hvp, jacobian
using ADTypes, SciMLBase
import Zygote

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, x, adtype::ADTypes.AutoZygote,
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false)
    global _p = p
    function _f(θ)
        return f(θ, _p)[1]
    end

    adtype, soadtype = OptimizationBase.generate_adtype(adtype)

    if g == true && f.grad === nothing
        extras_grad = prepare_gradient(_f, adtype, x)
        function grad(res, θ)
            gradient!(_f, res, adtype, θ, extras_grad)
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function grad(res, θ, p)
                global _p = p
                gradient!(_f, res, adtype, θ)
            end
        end
    elseif g == true
        grad = (G, θ) -> f.grad(G, θ, p)
        if p !== SciMLBase.NullParameters() && p !== nothing
            grad = (G, θ, p) -> f.grad(G, θ, p)
        end
    else
        grad = nothing
    end

    if fg == true && f.fg === nothing
        if g == false
            extras_grad = prepare_gradient(_f, adtype, x)
        end
        function fg!(res, θ)
            (y, _) = value_and_gradient!(_f, res, adtype, θ, extras_grad)
            return y
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function fg!(res, θ, p)
                global _p = p
                (y, _) = value_and_gradient!(_f, res, adtype, θ)
                return y
            end
        end
    elseif fg == true
        fg! = (G, θ) -> f.fg(G, θ, p)
        if p !== SciMLBase.NullParameters() && p !== nothing
            fg! = (G, θ, p) -> f.fg(G, θ, p)
        end
    else
        fg! = nothing
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if h == true && f.hess === nothing
        extras_hess = prepare_hessian(_f, soadtype, x)
        function hess(res, θ)
            hessian!(_f, res, soadtype, θ, extras_hess)
        end
    elseif h == true
        hess = (H, θ) -> f.hess(H, θ, p)
    else
        hess = nothing
    end

    if fgh == true && f.fgh === nothing
        function fgh!(G, H, θ)
            (y, _, _) = value_derivative_and_second_derivative!(_f, G, H, θ, extras_hess)
            return y
        end
    elseif fgh == true
        fgh! = (G, H, θ) -> f.fgh(G, H, θ, p)
    else
        fgh! = nothing
    end

    if hv == true && f.hv === nothing
        extras_hvp = prepare_hvp(_f, soadtype, x, zeros(eltype(x), size(x)))
        function hv!(H, θ, v)
            hvp!(_f, H, soadtype, θ, v, extras_hvp)
        end
    elseif hv == true
        hv! = (H, θ, v) -> f.hv(H, θ, v, p)
    else
        hv! = nothing
    end

    if f.cons === nothing
        cons = nothing
    else
        function cons(res, θ)
            return f.cons(res, θ, p)
        end

        function cons_oop(x)
            _res = Zygote.Buffer(x, num_cons)
            cons(_res, x)
            return copy(_res)
        end

        function lagrangian(augvars)
            θ = augvars[1:length(x)]
            σ = augvars[length(x) + 1]
            λ = augvars[(length(x) + 2):end]
            return σ * _f(θ) + dot(λ, cons_oop(θ))
        end
    end

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if cons !== nothing && cons_j == true && f.cons_j === nothing
        extras_jac = prepare_jacobian(cons_oop, adtype, x)
        function cons_j!(J, θ)
            jacobian!(cons_oop, J, adtype, θ, extras_jac)
            if size(J, 1) == 1
                J = vec(J)
            end
        end
    elseif cons !== nothing && cons_j == true
        cons_j! = (J, θ) -> f.cons_j(J, θ, p)
    else
        cons_j! = nothing
    end

    if f.cons_vjp === nothing && cons_vjp == true && cons !== nothing
        extras_pullback = prepare_pullback(cons_oop, adtype, x, ones(eltype(x), num_cons))
        function cons_vjp!(J, θ, v)
            pullback!(cons_oop, J, adtype, θ, v, extras_pullback)
        end
    elseif cons_vjp == true
        cons_vjp! = (J, θ, v) -> f.cons_vjp(J, θ, v, p)
    else
        cons_vjp! = nothing
    end

    if cons !== nothing && f.cons_jvp === nothing && cons_jvp == true
        extras_pushforward = prepare_pushforward(
            cons_oop, adtype, x, ones(eltype(x), length(x)))
        function cons_jvp!(J, θ, v)
            pushforward!(cons_oop, J, adtype, θ, v, extras_pushforward)
        end
    elseif cons_jvp == true
        cons_jvp! = (J, θ, v) -> f.cons_jvp(J, θ, v, p)
    else
        cons_jvp! = nothing
    end

    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec
    if cons !== nothing && cons_h == true && f.cons_h === nothing
        fncs = [(x) -> cons_oop(x)[i] for i in 1:num_cons]
        extras_cons_hess = prepare_hessian.(fncs, Ref(soadtype), Ref(x))

        function cons_h!(H, θ)
            for i in 1:num_cons
                hessian!(fncs[i], H[i], soadtype, θ, extras_cons_hess[i])
            end
        end
    elseif cons !== nothing && cons_h == true
        cons_h! = (res, θ) -> f.cons_h(res, θ, p)
    else
        cons_h! = nothing
    end

    lag_hess_prototype = f.lag_hess_prototype

    if f.lag_h === nothing && cons !== nothing && lag_h == true
        lag_extras = prepare_hessian(
            lagrangian, soadtype, vcat(x, [one(eltype(x))], ones(eltype(x), num_cons)))
        lag_hess_prototype = zeros(Bool, length(x), length(x))

        function lag_h!(H::AbstractMatrix, θ, σ, λ)
            if σ == zero(eltype(θ))
                cons_h(H, θ)
                H *= λ
            else
                H .= @view(hessian(lagrangian, soadtype, vcat(θ, [σ], λ), lag_extras)[
                    1:length(θ), 1:length(θ)])
            end
        end

        function lag_h!(h, θ, σ, λ)
            H = eltype(θ).(lag_hess_prototype)
            hessian!(x -> lagrangian(x, σ, λ), H, soadtype, θ, lag_extras)
            k = 0
            rows, cols, _ = findnz(H)
            for (i, j) in zip(rows, cols)
                if i <= j
                    k += 1
                    h[k] = H[i, j]
                end
            end
        end
    elseif cons !== nothing && lag_h == true
        lag_h! = (res, θ, σ, μ) -> f.lag_h(res, θ, σ, μ, p)
    else
        lag_h! = nothing
    end

    return OptimizationFunction{true}(f.f, adtype;
        grad = grad, fg = fg!, hess = hess, hv = hv!, fgh = fgh!,
        cons = cons, cons_j = cons_j!, cons_h = cons_h!,
        cons_vjp = cons_vjp!, cons_jvp = cons_jvp!,
        hess_prototype = hess_sparsity,
        hess_colorvec = hess_colors,
        cons_jac_prototype = cons_jac_prototype,
        cons_jac_colorvec = cons_jac_colorvec,
        cons_hess_prototype = conshess_sparsity,
        cons_hess_colorvec = conshess_colors,
        lag_h = lag_h!,
        lag_hess_prototype = lag_hess_prototype,
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr)
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AutoZygote, num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false)
    x = cache.u0
    p = cache.p

    return OptimizationBase.instantiate_function(
        f, x, adtype, p, num_cons; g, h, hv, fg, fgh, cons_j, cons_vjp, cons_jvp, cons_h)
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, x, adtype::ADTypes.AutoSparse{<:AutoZygote},
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false)
    function _f(θ)
        return f.f(θ, p)[1]
    end

    adtype, soadtype = OptimizationBase.generate_sparse_adtype(adtype)

    if g == true && f.grad === nothing
        extras_grad = prepare_gradient(_f, adtype.dense_ad, x)
        function grad(res, θ)
            gradient!(_f, res, adtype.dense_ad, θ, extras_grad)
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function grad(res, θ, p)
                global p = p
                gradient!(_f, res, adtype.dense_ad, θ)
            end
        end
    elseif g == true
        grad = (G, θ) -> f.grad(G, θ, p)
        if p !== SciMLBase.NullParameters() && p !== nothing
            grad = (G, θ, p) -> f.grad(G, θ, p)
        end
    else
        grad = nothing
    end

    if fg == true && f.fg !== nothing
        if g == false
            extras_grad = prepare_gradient(_f, adtype.dense_ad, x)
        end
        function fg!(res, θ)
            (y, _) = value_and_gradient!(_f, res, adtype.dense_ad, θ, extras_grad)
            return y
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function fg!(res, θ, p)
                global p = p
                (y, _) = value_and_gradient!(_f, res, adtype.dense_ad, θ)
                return y
            end
        end
    elseif fg == true
        fg! = (G, θ) -> f.fg(G, θ, p)
        if p !== SciMLBase.NullParameters() && p !== nothing
            fg! = (G, θ, p) -> f.fg(G, θ, p)
        end
    else
        fg! = nothing
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if f.hess === nothing
        extras_hess = prepare_hessian(_f, soadtype, x) #placeholder logic, can be made much better
        function hess(res, θ)
            hessian!(_f, res, soadtype, θ, extras_hess)
        end
        hess_sparsity = extras_hess.coloring_result.S
        hess_colors = extras_hess.coloring_result.color
    elseif h == true
        hess = (H, θ) -> f.hess(H, θ, p)
    else
        hess = nothing
    end

    if fgh == true && f.fgh !== nothing
        function fgh!(G, H, θ)
            (y, _, _) = value_derivative_and_second_derivative!(_f, G, H, θ, extras_hess)
            return y
        end
    elseif fgh == true
        fgh! = (G, H, θ) -> f.fgh(G, H, θ, p)
    else
        fgh! = nothing
    end

    if hv == true && f.hv !== nothing
        extras_hvp = prepare_hvp(_f, soadtype.dense_ad, x, zeros(eltype(x), size(x)))
        function hv!(H, θ, v)
            hvp!(_f, H, soadtype.dense_ad, θ, v, extras_hvp)
        end
    elseif hv == true
        hv! = (H, θ, v) -> f.hv(H, θ, v, p)
    else
        hv! = nothing
    end

    if f.cons === nothing
        cons = nothing
    else
        function cons(res, θ)
            f.cons(res, θ, p)
        end

        function cons_oop(x)
            _res = Zygote.Buffer(x, num_cons)
            f.cons(_res, x, p)
            return copy(_res)
        end

        function lagrangian(augvars)
            θ = augvars[1:length(x)]
            σ = augvars[length(x) + 1]
            λ = augvars[(length(x) + 2):end]
            return σ * _f(θ) + dot(λ, cons(θ))
        end
    end

    cons_jac_prototype = f.cons_jac_prototype
    cons_jac_colorvec = f.cons_jac_colorvec
    if cons !== nothing && cons_j == true && f.cons_j === nothing
        extras_jac = prepare_jacobian(cons_oop, adtype, x)
        function cons_j!(J, θ)
            jacobian!(cons_oop, J, adtype, θ, extras_jac)
            if size(J, 1) == 1
                J = vec(J)
            end
        end
        cons_jac_prototype = extras_jac.coloring_result.S
        cons_jac_colorvec = extras_jac.coloring_result.color
    elseif cons !== nothing && cons_j == true
        cons_j! = (J, θ) -> f.cons_j(J, θ, p)
    else
        cons_j! = nothing
    end

    if f.cons_vjp === nothing && cons_vjp == true && cons !== nothing
        extras_pullback = prepare_pullback(cons_oop, adtype, x)
        function cons_vjp!(J, θ, v)
            pullback!(cons_oop, J, adtype.dense_ad, θ, v, extras_pullback)
        end
    elseif cons_vjp == true
        cons_vjp! = (J, θ, v) -> f.cons_vjp(J, θ, v, p)
    else
        cons_vjp! = nothing
    end

    if f.cons_jvp === nothing && cons_jvp == true && cons !== nothing
        extras_pushforward = prepare_pushforward(cons_oop, adtype, x)
        function cons_jvp!(J, θ, v)
            pushforward!(cons_oop, J, adtype.dense_ad, θ, v, extras_pushforward)
        end
    elseif cons_jvp == true
        cons_jvp! = (J, θ, v) -> f.cons_jvp(J, θ, v, p)
    else
        cons_jvp! = nothing
    end

    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec
    if cons !== nothing && f.cons_h === nothing && cons_h == true
        fncs = [@closure (x) -> cons_oop(x)[i] for i in 1:num_cons]
        extras_cons_hess = Vector(undef, length(fncs))
        for ind in 1:num_cons
            extras_cons_hess[ind] = prepare_hessian(fncs[ind], soadtype, x)
        end
        colores = getfield.(extras_cons_hess, :coloring_result)
        conshess_sparsity = getfield.(colores, :S)
        conshess_colors = getfield.(colores, :color)
        function cons_h!(H, θ)
            for i in 1:num_cons
                hessian!(fncs[i], H[i], soadtype, θ, extras_cons_hess[i])
            end
        end
    elseif cons_h == true
        cons_h! = (res, θ) -> f.cons_h(res, θ, p)
    else
        cons_h! = nothing
    end

    lag_hess_prototype = f.lag_hess_prototype
    if cons !== nothing && cons_h == true && f.lag_h === nothing
        lag_extras = prepare_hessian(
            lagrangian, soadtype, vcat(x, [one(eltype(x))], ones(eltype(x), num_cons)))
        lag_hess_prototype = lag_extras.coloring_result.S[1:length(θ), 1:length(θ)]
        lag_hess_colors = lag_extras.coloring_result.color

        function lag_h!(H::AbstractMatrix, θ, σ, λ)
            if σ == zero(eltype(θ))
                cons_h(H, θ)
                H *= λ
            else
                H .= hessian(lagrangian, soadtype, vcat(θ, [σ], λ), lag_extras)[
                    1:length(θ), 1:length(θ)]
            end
        end

        function lag_h!(h, θ, σ, λ)
            H = hessian(lagrangian, soadtype, vcat(θ, [σ], λ), lag_extras)[
                1:length(θ), 1:length(θ)]
            k = 0
            rows, cols, _ = findnz(H)
            for (i, j) in zip(rows, cols)
                if i <= j
                    k += 1
                    h[k] = H[i, j]
                end
            end
        end
    elseif cons !== nothing && cons_h == true
        lag_h! = (res, θ, σ, μ) -> f.lag_h(res, θ, σ, μ, p)
    else
        lag_h! = nothing
    end
    return OptimizationFunction{true}(f.f, adtype;
        grad = grad, fg = fg!, hess = hess, hv = hv!, fgh = fgh!,
        cons = cons, cons_j = cons_j!, cons_h = cons_h!,
        hess_prototype = hess_sparsity,
        hess_colorvec = hess_colors,
        cons_jac_prototype = cons_jac_prototype,
        cons_jac_colorvec = cons_jac_colorvec,
        cons_hess_prototype = conshess_sparsity,
        cons_hess_colorvec = conshess_colors,
        lag_h = lag_h!,
        lag_hess_prototype = lag_hess_prototype,
        lag_hess_colorvec = lag_hess_colors,
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr)
end

function OptimizationBase.instantiate_function(
        f::OptimizationFunction{true}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AutoSparse{<:AutoZygote}, num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false)
    x = cache.u0
    p = cache.p

    return OptimizationBase.instantiate_function(
        f, x, adtype, p, num_cons; g, h, hv, fg, fgh, cons_j, cons_vjp, cons_jvp, cons_h)
end

end
