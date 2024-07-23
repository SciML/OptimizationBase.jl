module OptimizationZygoteExt

import OptimizationBase
import OptimizationBase.SciMLBase: OptimizationFunction
import OptimizationBase.ADTypes: AutoZygote
using Zygote, Zygote.ForwardDiff

function OptimizationBase.instantiate_function(f::OptimizationFunction{true}, x,
        adtype::AutoZygote, p,
        num_cons = 0)
    function _f(θ, args...)
        return f(θ, p, args...)[1]
    end

    if f.grad === nothing
        function grad(res, θ, args...)
            val = Zygote.gradient(x -> _f(x, args...), θ)[1]
            if val === nothing
                res .= zero(eltype(θ))
            else
                res .= val
            end
        end
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    if f.hess === nothing
        function hess(res, θ, args...)
            res .= ForwardDiff.jacobian(θ) do θ
                Zygote.gradient(x -> _f(x, args...), θ)[1]
            end
        end
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        function hv(H, θ, v, args...)
            _θ = ForwardDiff.Dual.(θ, v)
            res = similar(_θ)
            grad(res, _θ, args...)
            H .= getindex.(ForwardDiff.partials.(res), 1)
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        function cons(res, θ, args...)
            f.cons(res, θ, p, args...)
        end

        function cons_oop(x, args...)
            _res = Zygote.Buffer(x, num_cons)
            cons(_res, x, args...)
            copy(_res)
        end
    end

    if cons !== nothing && f.cons_j === nothing
        function cons_j(J, θ)
            J .= first(Zygote.jacobian(cons_oop, θ))
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    if cons !== nothing && f.cons_h === nothing
        fncs = [(x) -> cons_oop(x)[i] for i in 1:num_cons]
        cons_h = function (res, θ)
            for i in 1:num_cons
                res[i] .= ForwardDiff.jacobian(θ) do θ
                    Zygote.gradient(fncs[i], θ)[1]
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
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        lag_h = lag_h,
        lag_hess_prototype = f.lag_hess_prototype,
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr)
end

function OptimizationBase.instantiate_function(f::OptimizationFunction{true},
        cache::OptimizationBase.ReInitCache,
        adtype::AutoZygote, num_cons = 0)
    x = cache.u0
    p = cache.p

    return instantiate_function(f, x, adtype, p, num_cons)
end

function OptimizationBase.instantiate_function(f::OptimizationFunction{false}, x,
        adtype::AutoZygote, p,
        num_cons = 0)
    _f = (θ, args...) -> f(θ, p, args...)[1]
    if f.grad === nothing
        grad = function (θ, args...)
            val = Zygote.gradient(x -> _f(x, args...), θ)[1]
            if val === nothing
                return zero(eltype(θ))
            else
                return val
            end
        end
    else
        grad = (θ, args...) -> f.grad(θ, p, args...)
    end

    if f.hess === nothing
        hess = function (θ, args...)
            return ForwardDiff.jacobian(θ) do θ
                return Zygote.gradient(x -> _f(x, args...), θ)[1]
            end
        end
    else
        hess = (θ, args...) -> f.hess(θ, p, args...)
    end

    if f.hv === nothing
        hv = function (H, θ, v, args...)
            _θ = ForwardDiff.Dual.(θ, v)
            res = grad(_θ, args...)
            return getindex.(ForwardDiff.partials.(res), 1)
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

    if cons !== nothing && f.cons_j === nothing
        cons_j = function (θ)
            if num_cons > 1
                return first(Zygote.jacobian(cons_oop, θ))
            else
                return first(Zygote.jacobian(cons_oop, θ))[1, :]
            end
        end
    else
        cons_j = (θ) -> f.cons_j(θ, p)
    end

    if cons !== nothing && f.cons_h === nothing
        fncs = [(x) -> cons_oop(x)[i] for i in 1:num_cons]
        cons_h = function (θ)
            return map(1:num_cons) do i
                Zygote.hessian(fncs[i], θ)
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
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype,
        lag_h = lag_h,
        lag_hess_prototype = f.lag_hess_prototype,
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr)
end

function OptimizationBase.instantiate_function(f::OptimizationFunction{false},
        cache::OptimizationBase.ReInitCache,
        adtype::AutoZygote, num_cons = 0)
    p = cache.p
    x = cache.u0

    return instantiate_function(f, x, adtype, p, num_cons)
end

end
