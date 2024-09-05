using OptimizationBase
import OptimizationBase.ArrayInterface
import OptimizationBase.SciMLBase: OptimizationFunction
import OptimizationBase.LinearAlgebra: I
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

function generate_adtype(adtype)
    if !(adtype isa SciMLBase.NoAD) && ADTypes.mode(adtype) isa ADTypes.ForwardMode
        soadtype = DifferentiationInterface.SecondOrder(adtype, AutoReverseDiff()) #make zygote?
    elseif !(adtype isa SciMLBase.NoAD) && ADTypes.mode(adtype) isa ADTypes.ReverseMode
        soadtype = DifferentiationInterface.SecondOrder(AutoForwardDiff(), adtype)
    else
        soadtype = adtype
    end
    return adtype, soadtype
end

function instantiate_function(
        f::OptimizationFunction{true}, x, adtype::ADTypes.AbstractADType,
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false)
    global _p = p
    function _f(θ)
        return f(θ, _p)[1]
    end

    adtype, soadtype = generate_adtype(adtype)

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
        if p !== SciMLBase.NullParameters()
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
        if p !== SciMLBase.NullParameters() && p !== nothing
            function hess(res, θ, p)
                global _p = p
                hessian!(_f, res, soadtype, θ)
            end
        end
    elseif h == true
        hess = (H, θ) -> f.hess(H, θ, p)
        if p !== SciMLBase.NullParameters() && p !== nothing
            hess = (H, θ, p) -> f.hess(H, θ, p)
        end
    else
        hess = nothing
    end

    if fgh == true && f.fgh !== nothing
        function fgh!(G, H, θ)
            (y, _, _) = value_derivative_and_second_derivative!(
                _f, G, H, soadtype, θ, extras_hess)
            return y
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function fgh!(G, H, θ, p)
                global _p = p
                (y, _, _) = value_derivative_and_second_derivative!(
                    _f, G, H, soadtype, θ)
                return y
            end
        end
    elseif fgh == true
        fgh! = (G, H, θ) -> f.fgh(G, H, θ, p)
        if p !== SciMLBase.NullParameters() && p !== nothing
            fgh! = (G, H, θ, p) -> f.fgh(G, H, θ, p)
        end
    else
        fgh! = nothing
    end

    if hv == true && f.hv === nothing
        extras_hvp = prepare_hvp(_f, soadtype, x, zeros(eltype(x), size(x)))
        function hv!(H, θ, v)
            hvp!(_f, H, soadtype, θ, v, extras_hvp)
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function hv!(H, θ, v, p)
                global _p = p
                hvp!(_f, H, soadtype, θ, v)
            end
        end
    elseif hv == true
        hv! = (H, θ, v) -> f.hv(H, θ, v, p)
        if p !== SciMLBase.NullParameters() && p !== nothing
            hv! = (H, θ, v, p) -> f.hv(H, θ, v, p)
        end
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
            _res = zeros(eltype(x), num_cons)
            cons(_res, x)
            return _res
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
    elseif cons_j == true && cons !== nothing
        cons_j! = (J, θ) -> f.cons_j(J, θ, p)
    else
        cons_j! = nothing
    end

    if f.cons_vjp === nothing && cons_vjp == true && cons !== nothing
        extras_pullback = prepare_pullback(cons_oop, adtype, x, ones(eltype(x), num_cons))
        function cons_vjp!(J, θ, v)
            pullback!(cons_oop, J, adtype, θ, v, extras_pullback)
        end
    elseif cons_vjp == true && cons !== nothing
        cons_vjp! = (J, θ, v) -> f.cons_vjp(J, θ, v, p)
    else
        cons_vjp! = nothing
    end

    if f.cons_jvp === nothing && cons_jvp == true && cons !== nothing
        extras_pushforward = prepare_pushforward(
            cons_oop, adtype, x, ones(eltype(x), length(x)))
        function cons_jvp!(J, θ, v)
            pushforward!(cons_oop, J, adtype, θ, v, extras_pushforward)
        end
    elseif cons_jvp == true && cons !== nothing
        cons_jvp! = (J, θ, v) -> f.cons_jvp(J, θ, v, p)
    else
        cons_jvp! = nothing
    end

    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec
    if cons !== nothing && f.cons_h === nothing && cons_h == true
        fncs = [(x) -> cons_oop(x)[i] for i in 1:num_cons]
        extras_cons_hess = prepare_hessian.(fncs, Ref(soadtype), Ref(x))

        function cons_h!(H, θ)
            for i in 1:num_cons
                hessian!(fncs[i], H[i], soadtype, θ, extras_cons_hess[i])
            end
        end
    elseif cons_h == true && cons !== nothing
        cons_h! = (res, θ) -> f.cons_h(res, θ, p)
    else
        cons_h! = nothing
    end

    lag_hess_prototype = f.lag_hess_prototype

    if cons !== nothing && lag_h == true && f.lag_h === nothing
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

        function lag_h!(h::AbstractVector, θ, σ, λ)
            H = hessian(lagrangian, soadtype, vcat(θ, [σ], λ), lag_extras)
            k = 0
            for i in 1:length(θ)
                for j in 1:i
                    k += 1
                    h[k] = H[i, j]
                end
            end
        end

        if p !== SciMLBase.NullParameters() && p !== nothing
            function lag_h!(H::AbstractMatrix, θ, σ, λ, p)
                if σ == zero(eltype(θ))
                    cons_h(H, θ)
                    H *= λ
                else
                    global _p = p
                    H .= @view(hessian(lagrangian, soadtype, vcat(θ, [σ], λ), lag_extras)[
                        1:length(θ), 1:length(θ)])
                end
            end
    
            function lag_h!(h::AbstractVector, θ, σ, λ, p)
                global _p = p
                H = hessian(lagrangian, soadtype, vcat(θ, [σ], λ), lag_extras)
                k = 0
                for i in 1:length(θ)
                    for j in 1:i
                        k += 1
                        h[k] = H[i, j]
                    end
                end
            end
        end
    elseif lag_h == true && cons !== nothing
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

function instantiate_function(
        f::OptimizationFunction{true}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AbstractADType, num_cons = 0;
        kwargs...)
    x = cache.u0
    p = cache.p

    return instantiate_function(f, x, adtype, p, num_cons; kwargs...)
end

function instantiate_function(
        f::OptimizationFunction{false}, x, adtype::ADTypes.AbstractADType,
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false)
    global _p = p
    function _f(θ)
        return f(θ, _p)[1]
    end

    adtype, soadtype = generate_adtype(adtype)

    if g == true && f.grad === nothing
        extras_grad = prepare_gradient(_f, adtype, x)
        function grad(θ)
            gradient(_f, adtype, θ, extras_grad)
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function grad(θ, p)
                global _p = p
                gradient(_f, adtype, θ)
            end
        end
    elseif g == true
        grad = (θ) -> f.grad(θ, p)
        if p !== SciMLBase.NullParameters() && p !== nothing
            grad = (θ, p) -> f.grad(θ, p)
        end
    else
        grad = nothing
    end

    if fg == true && f.fg === nothing
        if g == false
            extras_grad = prepare_gradient(_f, adtype, x)
        end
        function fg!(θ)
            (y, res) = value_and_gradient(_f, adtype, θ, extras_grad)
            return y, res
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function fg!(θ, p)
                global _p = p
                (y, res) = value_and_gradient(_f, adtype, θ)
                return y, res
            end
        end
    elseif fg == true
        fg! = (θ) -> f.fg(θ, p)
        if p !== SciMLBase.NullParameters() && p !== nothing
            fg! = (θ, p) -> f.fg(θ, p)
        end
    else
        fg! = nothing
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if h == true && f.hess === nothing
        extras_hess = prepare_hessian(_f, soadtype, x)
        function hess(θ)
            hessian(_f, soadtype, θ, extras_hess)
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function hess(θ, p)
                global _p = p
                hessian(_f, soadtype, θ)
            end
        end
    elseif h == true
        hess = (θ) -> f.hess(θ, p)
        if p !== SciMLBase.NullParameters() && p !== nothing
            hess = (θ, p) -> f.hess(θ, p)
        end
    else
        hess = nothing
    end

    if fgh == true && f.fgh !== nothing
        function fgh!(θ)
            (y, G, H) = value_derivative_and_second_derivative(_f, adtype, θ, extras_hess)
            return y, G, H
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function fgh!(θ, p)
                global _p = p
                (y, G, H) = value_derivative_and_second_derivative(_f, adtype, θ)
                return y, G, H
            end
        end
    elseif fgh == true
        fgh! = (θ) -> f.fgh(θ, p)
        if p !== SciMLBase.NullParameters() && p !== nothing
            fgh! = (θ, p) -> f.fgh(θ, p)
        end
    else
        fgh! = nothing
    end

    if hv == true && f.hv === nothing
        extras_hvp = prepare_hvp(_f, soadtype, x, zeros(eltype(x), size(x)))
        function hv!(θ, v)
            hvp(_f, soadtype, θ, v, extras_hvp)
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function hv!(θ, v, p)
                global _p = p
                hvp(_f, soadtype, θ, v, extras_hvp)
            end
        end
    elseif hv == true
        hv! = (θ, v) -> f.hv(θ, v, p)
        if p !== SciMLBase.NullParameters() && p !== nothing
            hv! = (θ, v, p) -> f.hv(θ, v, p)
        end
    else
        hv! = nothing
    end

    if f.cons === nothing
        cons = nothing
    else
        function cons(θ)
            return f.cons(θ, p)
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
        extras_jac = prepare_jacobian(cons, adtype, x)
        function cons_j!(θ)
            J = jacobian(cons, adtype, θ, extras_jac)
            if size(J, 1) == 1
                J = vec(J)
            end
            return J
        end
    elseif cons_j == true && cons !== nothing
        cons_j! = (θ) -> f.cons_j(θ, p)
    else
        cons_j! = nothing
    end

    if f.cons_vjp === nothing && cons_vjp == true && cons !== nothing
        extras_pullback = prepare_pullback(cons, adtype, x, ones(eltype(x), num_cons))
        function cons_vjp!(θ, v)
            return pullback(cons, adtype, θ, v, extras_pullback)
        end
    elseif cons_vjp == true && cons !== nothing
        cons_vjp! = (θ, v) -> f.cons_vjp(θ, v, p)
    else
        cons_vjp! = nothing
    end

    if f.cons_jvp === nothing && cons_jvp == true && cons !== nothing
        extras_pushforward = prepare_pushforward(
            cons, adtype, x, ones(eltype(x), length(x)))
        function cons_jvp!(θ, v)
            return pushforward(cons, adtype, θ, v, extras_pushforward)
        end
    elseif cons_jvp == true && cons !== nothing
        cons_jvp! = (θ, v) -> f.cons_jvp(θ, v, p)
    else
        cons_jvp! = nothing
    end

    conshess_sparsity = f.cons_hess_prototype
    conshess_colors = f.cons_hess_colorvec
    if cons !== nothing && cons_h == true && f.cons_h === nothing
        fncs = [(x) -> cons(x)[i] for i in 1:num_cons]
        extras_cons_hess = prepare_hessian.(fncs, Ref(soadtype), Ref(x))

        function cons_h!(θ)
            H = map(1:num_cons) do i
                hessian(fncs[i], soadtype, θ, extras_cons_hess[i])
            end
            return H
        end
    elseif cons_h == true && cons !== nothing
        cons_h! = (θ) -> f.cons_h(θ, p)
    else
        cons_h! = nothing
    end

    lag_hess_prototype = f.lag_hess_prototype

    if cons !== nothing && lag_h == true && f.lag_h === nothing
        lag_extras = prepare_hessian(
            lagrangian, soadtype, vcat(x, [one(eltype(x))], ones(eltype(x), num_cons)))
        lag_hess_prototype = zeros(Bool, length(x), length(x))

        function lag_h!(θ, σ, λ)
            if σ == zero(eltype(θ))
                return λ .* cons_h(θ)
            else
                return hessian(lagrangian, soadtype, vcat(θ, [σ], λ), lag_extras)[
                    1:length(θ), 1:length(θ)]
            end
        end

        if p !== SciMLBase.NullParameters() && p !== nothing
            function lag_h!(θ, σ, λ, p)
                if σ == zero(eltype(θ))
                    return λ .* cons_h(θ)
                else
                    global _p = p
                    return hessian(lagrangian, soadtype, vcat(θ, [σ], λ), lag_extras)[
                        1:length(θ), 1:length(θ)]
                end
            end
        end
    elseif lag_h == true && cons !== nothing
        lag_h! = (θ, σ, λ) -> f.lag_h(θ, σ, λ, p)
    else
        lag_h! = nothing
    end

    return OptimizationFunction{false}(f.f, adtype;
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

function instantiate_function(
        f::OptimizationFunction{false}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AbstractADType, num_cons = 0; kwargs...)
    x = cache.u0
    p = cache.p

    return instantiate_function(f, x, adtype, p, num_cons; kwargs...)
end
