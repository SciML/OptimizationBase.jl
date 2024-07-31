module OptimizationEnzymeExt

import OptimizationBase, OptimizationBase.ArrayInterface
import OptimizationBase.SciMLBase: OptimizationFunction
import OptimizationBase.SciMLBase
import OptimizationBase.LinearAlgebra: I
import OptimizationBase.ADTypes: AutoEnzyme
using Enzyme
using Core: Vararg

@inline function firstapply(f::F, θ, p) where {F}
    res = f(θ, p)
    if isa(res, AbstractFloat)
        res
    else
        first(res)
    end
end

function inner_grad(θ, bθ, f, p)
    Enzyme.autodiff_deferred(Enzyme.Reverse,
        Const(firstapply),
        Active,
        Const(f),
        Enzyme.Duplicated(θ, bθ),
        Const(p)
    ),
    return nothing
end

function inner_grad_primal(θ, bθ, f, p)
    Enzyme.autodiff_deferred(Enzyme.ReverseWithPrimal,
        Const(firstapply),
        Active,
        Const(f),
        Enzyme.Duplicated(θ, bθ),
        Const(p)
    )[2]
end

function hv_f2_alloc(x, f, p)
    dx = Enzyme.make_zero(x)
    Enzyme.autodiff_deferred(Enzyme.Reverse,
        firstapply,
        Active,
        f,
        Enzyme.Duplicated(x, dx),
        Const(p)
    )
    return dx
end

function inner_cons(x, fcons::Function, p::Union{SciMLBase.NullParameters, Nothing},
        num_cons::Int, i::Int)
    res = zeros(eltype(x), num_cons)
    fcons(res, x, p)
    return res[i]
end

function cons_f2(x, dx, fcons, p, num_cons, i)
    Enzyme.autodiff_deferred(Enzyme.Reverse, inner_cons, Active, Enzyme.Duplicated(x, dx),
        Const(fcons), Const(p), Const(num_cons), Const(i))
    return nothing
end

function inner_cons_oop(
        x::Vector{T}, fcons::Function, p::Union{SciMLBase.NullParameters, Nothing},
        i::Int) where {T}
    return fcons(x, p)[i]
end

function cons_f2_oop(x, dx, fcons, p, i)
    Enzyme.autodiff_deferred(
        Enzyme.Reverse, inner_cons_oop, Active, Enzyme.Duplicated(x, dx),
        Const(fcons), Const(p), Const(i))
    return nothing
end

function OptimizationBase.instantiate_function(f::OptimizationFunction{true}, x,
        adtype::AutoEnzyme, p,
        num_cons = 0)
    if f.grad === nothing
        function grad(res, θ)
            Enzyme.make_zero!(res)
            Enzyme.autodiff(Enzyme.Reverse,
                Const(firstapply),
                Active,
                Const(f.f),
                Enzyme.Duplicated(θ, res),
                Const(p)
            )
        end
    else
        grad = (G, θ) -> f.grad(G, θ, p)
    end

    function fg!(res, θ)
        Enzyme.make_zero!(res)
        y = Enzyme.autodiff(Enzyme.ReverseWithPrimal,
            Const(firstapply),
            Active,
            Const(f.f),
            Enzyme.Duplicated(θ, res),
            Const(p)
        )[2]
        return y
    end

    if f.hess === nothing
        vdθ = Tuple((Array(r) for r in eachrow(I(length(x)) * one(eltype(x)))))
        bθ = zeros(eltype(x), length(x))

        if f.hess_prototype === nothing
            vdbθ = Tuple(zeros(eltype(x), length(x)) for i in eachindex(x))
        else
            #useless right now, looks like there is no way to tell Enzyme the sparsity pattern?
            vdbθ = Tuple((copy(r) for r in eachrow(f.hess_prototype)))
        end

        function hess(res, θ)
            Enzyme.make_zero!.(vdθ)
            Enzyme.make_zero!(bθ)
            Enzyme.make_zero!.(vdbθ)

            Enzyme.autodiff(Enzyme.Forward,
                inner_grad,
                Enzyme.BatchDuplicated(θ, vdθ),
                Enzyme.BatchDuplicatedNoNeed(bθ, vdbθ),
                Const(f.f),
                Const(p)
            )

            for i in eachindex(θ)
                res[i, :] .= vdbθ[i]
            end
        end
    else
        hess = (H, θ) -> f.hess(H, θ, p)
    end

    function fgh!(G, H, θ)
         
    end

    if f.hv === nothing
        function hv(H, θ, v)
            H .= Enzyme.autodiff(
                Enzyme.Forward, hv_f2_alloc, DuplicatedNoNeed, Duplicated(θ, v),
                Const(_f), Const(f.f), Const(p)
            )[1]
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> f.cons(res, θ, p)
    end

    if cons !== nothing && f.cons_j === nothing
        seeds = Tuple((Array(r) for r in eachrow(I(length(x)) * one(eltype(x)))))
        Jaccache = Tuple(zeros(eltype(x), num_cons) for i in 1:length(x))
        y = zeros(eltype(x), num_cons)
        function cons_j(J, θ)
            for i in 1:length(θ)
                Enzyme.make_zero!(Jaccache[i])
            end
            Enzyme.make_zero!(y)
            Enzyme.autodiff(Enzyme.Forward, f.cons, BatchDuplicated(y, Jaccache),
                BatchDuplicated(θ, seeds), Const(p))
            for i in 1:length(θ)
                if J isa Vector
                    J[i] = Jaccache[i][1]
                else
                    copyto!(@view(J[:, i]), Jaccache[i])
                end
            end
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    if cons !== nothing && f.cons_vjp === nothing
        function cons_vjp(res, θ, v)
            
        end
    else
        cons_vjp = (θ, σ) -> f.cons_vjp(θ, σ, p)
    end

    if cons !== nothing && f.cons_h === nothing
        function cons_h(res, θ)
            vdθ = Tuple((Array(r) for r in eachrow(I(length(θ)) * one(eltype(θ)))))
            bθ = zeros(eltype(θ), length(θ))
            vdbθ = Tuple(zeros(eltype(θ), length(θ)) for i in eachindex(θ))
            for i in 1:num_cons
                bθ .= zero(eltype(bθ))
                for el in vdbθ
                    Enzyme.make_zero!(el)
                end
                Enzyme.autodiff(Enzyme.Forward,
                    cons_f2,
                    Enzyme.BatchDuplicated(θ, vdθ),
                    Enzyme.BatchDuplicated(bθ, vdbθ),
                    Const(f.cons),
                    Const(p),
                    Const(num_cons),
                    Const(i))

                for j in eachindex(θ)
                    res[i][j, :] .= vdbθ[j]
                end
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, p)
    end

    if f.lag_h === nothing
        lag_h = nothing # Consider implementing this
    else
        lag_h = (θ, σ, μ) -> f.lag_h(θ, σ, μ, p)
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
        adtype::AutoEnzyme,
        num_cons = 0)
    p = cache.p
    x = cache.u0

    return OptimizationBase.instantiate_function(f, x, adtype, p, num_cons)
end

function OptimizationBase.instantiate_function(f::OptimizationFunction{false}, x,
        adtype::AutoEnzyme, p,
        num_cons = 0)
    if f.grad === nothing
        res = zeros(eltype(x), size(x))
        function grad(θ)
            Enzyme.make_zero!(res)
            Enzyme.autodiff(Enzyme.Reverse,
                Const(firstapply),
                Active,
                Const(f.f),
                Enzyme.Duplicated(θ, res),
                Const(p)
            )
            return res
        end
    else
        grad = (θ) -> f.grad(θ, p)
    end

    if f.hess === nothing
        function hess(θ)
            vdθ = Tuple((Array(r) for r in eachrow(I(length(θ)) * one(eltype(θ)))))

            bθ = zeros(eltype(θ), length(θ))
            vdbθ = Tuple(zeros(eltype(θ), length(θ)) for i in eachindex(θ))

            Enzyme.autodiff(Enzyme.Forward,
                inner_grad,
                Enzyme.BatchDuplicated(θ, vdθ),
                Enzyme.BatchDuplicated(bθ, vdbθ),
                Const(f.f),
                Const(p)
            )

            return reduce(
                vcat, [reshape(vdbθ[i], (1, length(vdbθ[i]))) for i in eachindex(θ)])
        end
    else
        hess = (θ) -> f.hess(θ, p)
    end

    if f.hv === nothing
        function hv(θ, v)
            Enzyme.autodiff(
                Enzyme.Forward, hv_f2_alloc, DuplicatedNoNeed, Duplicated(θ, v),
                Const(_f), Const(f.f), Const(p)
            )[1]
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons_oop = (θ) -> f.cons(θ, p)
    end

    if f.cons !== nothing && f.cons_j === nothing
        seeds = Tuple((Array(r) for r in eachrow(I(length(x)) * one(eltype(x)))))
        function cons_j(θ)
            J = Enzyme.autodiff(
                Enzyme.Forward, f.cons, BatchDuplicated(θ, seeds), Const(p))[1]
            if num_cons == 1
                return reduce(vcat, J)
            else
                return reduce(hcat, J)
            end
        end
    else
        cons_j = (θ) -> f.cons_j(θ, p)
    end

    if f.cons !== nothing && f.cons_h === nothing
        function cons_h(θ)
            vdθ = Tuple((Array(r) for r in eachrow(I(length(θ)) * one(eltype(θ)))))
            bθ = zeros(eltype(θ), length(θ))
            vdbθ = Tuple(zeros(eltype(θ), length(θ)) for i in eachindex(θ))
            res = [zeros(eltype(x), length(θ), length(θ)) for i in 1:num_cons]
            for i in 1:num_cons
                Enzyme.make_zero!(bθ)
                for el in vdbθ
                    Enzyme.make_zero!(el)
                end
                Enzyme.autodiff(Enzyme.Forward,
                    cons_f2_oop,
                    Enzyme.BatchDuplicated(θ, vdθ),
                    Enzyme.BatchDuplicated(bθ, vdbθ),
                    Const(f.cons),
                    Const(p),
                    Const(i))
                for j in eachindex(θ)
                    res[i][j, :] = vdbθ[j]
                end
            end
            return res
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
        cons = cons_oop, cons_j = cons_j, cons_h = cons_h,
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
        adtype::AutoEnzyme,
        num_cons = 0)
    p = cache.p
    x = cache.u0

    return OptimizationBase.instantiate_function(f, x, adtype, p, num_cons)
end

end
