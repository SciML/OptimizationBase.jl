module OptimizationEnzymeExt

import OptimizationBase, OptimizationBase.ArrayInterface
import OptimizationBase.SciMLBase: OptimizationFunction
import OptimizationBase.LinearAlgebra: I
import OptimizationBase.ADTypes: AutoEnzyme
isdefined(Base, :get_extension) ? (using Enzyme) : (using ..Enzyme)

@inline function firstapply(f::F, θ, p, args...) where {F}
    res = f(θ, p, args...)
    if isa(res, AbstractFloat)
        res
    else
        first(res)
    end
end

function OptimizationBase.instantiate_function(f::OptimizationFunction{true}, x,
        adtype::AutoEnzyme, p,
        num_cons = 0)
    if f.grad === nothing
        grad = let
            function (res, θ, args...)
                res .= zero(eltype(res))
                Enzyme.autodiff(Enzyme.Reverse,
                    Const(firstapply),
                    Active,
                    Const(f.f),
                    Enzyme.Duplicated(θ, res),
                    Const(p),
                    Const.(args)...)
            end
        end
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    if f.hess === nothing
        function g(θ, bθ, f, p, args...)
            Enzyme.autodiff_deferred(Enzyme.Reverse,
                Const(firstapply),
                Active,
                Const(f),
                Enzyme.Duplicated(θ, bθ),
                Const(p),
                Const.(args)...),
            return nothing
        end
        function hess(res, θ, args...)
            vdθ = Tuple((Array(r) for r in eachrow(I(length(θ)) * 1.0)))

            bθ = zeros(length(θ))
            vdbθ = Tuple(zeros(length(θ)) for i in eachindex(θ))

            Enzyme.autodiff(Enzyme.Forward,
                g,
                Enzyme.BatchDuplicated(θ, vdθ),
                Enzyme.BatchDuplicated(bθ, vdbθ),
                Const(f.f),
                Const(p),
                args...)

            for i in eachindex(θ)
                res[i, :] .= vdbθ[i]
            end
        end
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        function f2(x, f, p, args...)
            dx = zeros(length(x))
            Enzyme.autodiff_deferred(Enzyme.Reverse,
                firstapply,
                Active,
                f,
                Enzyme.Duplicated(x, dx),
                Const(p),
                Const.(args)...)
            return dx
        end
        hv = function (H, θ, v, args...)
            H .= Enzyme.autodiff(Enzyme.Forward, f2, DuplicatedNoNeed, Duplicated(θ, v),
                Const(_f), Const(f.f), Const(p),
                Const.(args)...)[1]
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> (f.cons(res, θ, p); return nothing)
        cons_oop = (x) -> (_res = zeros(eltype(x), num_cons); cons(_res, x); _res)
    end

    if cons !== nothing && f.cons_j === nothing
        cons_j = function (J, θ)
            if J isa Vector
                J .= Enzyme.jacobian(Enzyme.Forward, cons_oop, θ)[1, :]
            else
                J .= Enzyme.jacobian(Enzyme.Forward, cons_oop, θ)
            end
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    if cons !== nothing && f.cons_h === nothing
        fncs = map(1:num_cons) do i
            function (x)
                res = zeros(eltype(x), num_cons)
                f.cons(res, x, p)
                return res[i]
            end
        end

        function f2(x, dx, fnc)
            Enzyme.autodiff_deferred(Enzyme.Reverse, fnc, Enzyme.Duplicated(x, dx))
            return nothing
        end

        cons_h = function (res, θ)
            vdθ = Tuple((Array(r) for r in eachrow(I(length(θ)) * 1.0)))
            bθ = zeros(length(θ))
            vdbθ = Tuple(zeros(length(θ)) for i in eachindex(θ))
            for i in 1:num_cons
                bθ .= zero(eltype(bθ))
                for el in vdbθ
                    el .= zeros(length(θ))
                end
                Enzyme.autodiff(Enzyme.Forward,
                    f2,
                    Enzyme.BatchDuplicated(θ, vdθ),
                    Enzyme.BatchDuplicated(bθ, vdbθ),
                    Const(fncs[i]))

                for j in eachindex(θ)
                    res[i][j, :] .= vdbθ[j]
                end
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, p)
    end

    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype)
end

function OptimizationBase.instantiate_function(f::OptimizationFunction{true},
        cache::OptimizationBase.ReInitCache,
        adtype::AutoEnzyme,
        num_cons = 0)
    p = cache.p

    if f.grad === nothing
        function grad(res, θ, args...)
            res .= zero(eltype(res))
            Enzyme.autodiff(Enzyme.Reverse,
                Const(firstapply),
                Active,
                Const(f.f),
                Enzyme.Duplicated(θ, res),
                Const(p),
                Const.(args)...)
        end
    else
        grad = (G, θ, args...) -> f.grad(G, θ, p, args...)
    end

    if f.hess === nothing
        function g(θ, bθ, f, p, args...)
            Enzyme.autodiff_deferred(Enzyme.Reverse, Const(firstapply), Active, Const(f),
                Enzyme.Duplicated(θ, bθ),
                Const(p),
                Const.(args)...)
            return nothing
        end
        function hess(res, θ, args...)
            vdθ = Tuple((Array(r) for r in eachrow(I(length(θ)) * 1.0)))

            bθ = zeros(length(θ))
            vdbθ = Tuple(zeros(length(θ)) for i in eachindex(θ))

            Enzyme.autodiff(Enzyme.Forward,
                g,
                Enzyme.BatchDuplicated(θ, vdθ),
                Enzyme.BatchDuplicated(bθ, vdbθ),
                Const(f.f),
                Const(p),
                Const.(args)...)

            for i in eachindex(θ)
                res[i, :] .= vdbθ[i]
            end
        end
    else
        hess = (H, θ, args...) -> f.hess(H, θ, p, args...)
    end

    if f.hv === nothing
        function f2(x, f, p, args...)
            dx = zeros(length(x))
            Enzyme.autodiff_deferred(Enzyme.Reverse, firstapply, Active,
                f,
                Enzyme.Duplicated(x, dx),
                Const(p),
                Const.(args)...)
            return dx
        end
        hv = function (H, θ, v, args...)
            H .= Enzyme.autodiff(Enzyme.Forward, f2, DuplicatedNoNeed, Duplicated(θ, v),
                Const(f.f), Const(p),
                Const.(args)...)[1]
        end
    else
        hv = f.hv
    end

    if f.cons === nothing
        cons = nothing
    else
        cons = (res, θ) -> (f.cons(res, θ, p); return nothing)
        cons_oop = (x) -> (_res = zeros(eltype(x), num_cons); cons(_res, x); _res)
    end

    if cons !== nothing && f.cons_j === nothing
        cons_j = function (J, θ)
            if J isa Vector
                J .= Enzyme.jacobian(Enzyme.Forward, cons_oop, θ)[1, :]
            else
                J .= Enzyme.jacobian(Enzyme.Forward, cons_oop, θ)
            end
        end
    else
        cons_j = (J, θ) -> f.cons_j(J, θ, p)
    end

    if cons !== nothing && f.cons_h === nothing
        fncs = map(1:num_cons) do i
            function (x)
                res = zeros(eltype(x), num_cons)
                f.cons(res, x, p)
                return res[i]
            end
        end
        function f2(fnc, x)
            dx = zeros(length(x))
            Enzyme.autodiff_deferred(Enzyme.Reverse, fnc, Enzyme.Duplicated(x, dx))
            dx
        end
        cons_h = function (res, θ)
            for i in 1:num_cons
                res[i] .= Enzyme.jacobian(Enzyme.Forward, x -> f2(fncs[i], x), θ)
            end
        end
    else
        cons_h = (res, θ) -> f.cons_h(res, θ, p)
    end

    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype)
end

function OptimizationBase.instantiate_function(f::OptimizationFunction{false}, x,
        adtype::AutoEnzyme, p,
        num_cons = 0)
    if f.grad === nothing
        res = zeros(eltype(x), size(x))
        grad = let res = res
            function (θ, args...)
                res .= zero(eltype(res))
                Enzyme.autodiff(Enzyme.Reverse,
                    Const(firstapply),
                    Active,
                    Const(f.f),
                    Enzyme.Duplicated(θ, res),
                    Const(p),
                    Const.(args)...)
                return res
            end
        end
    else
        grad = (θ, args...) -> f.grad(θ, p, args...)
    end

    if f.hess === nothing
        function g(θ, bθ, f, p, args...)
            Enzyme.autodiff_deferred(Enzyme.Reverse,
                Const(firstapply),
                Active,
                Const(f),
                Enzyme.Duplicated(θ, bθ),
                Const(p),
                Const.(args)...),
            return nothing
        end
        function hess(θ, args...)
            vdθ = Tuple((Array(r) for r in eachrow(I(length(θ)) * 1.0)))

            bθ = zeros(length(θ))
            vdbθ = Tuple(zeros(length(θ)) for i in eachindex(θ))

            Enzyme.autodiff(Enzyme.Forward,
                g,
                Enzyme.BatchDuplicated(θ, vdθ),
                Enzyme.BatchDuplicated(bθ, vdbθ),
                Const(f.f),
                Const(p),
                Const.(args)...)

            reduce(vcat, [reshape(vdbθ[i], (1, length(vdbθ[i]))) for i in eachindex(θ)])
        end
    else
        hess = (θ, args...) -> f.hess(θ, p, args...)
    end

    if f.hv === nothing
        dx = zeros(length(x))
        function f2(x, f, p, args...)
            dx .= zero(eltype(dx))
            Enzyme.autodiff_deferred(Enzyme.Reverse,
                firstapply,
                Active,
                f,
                Enzyme.Duplicated(x, dx),
                Const(p),
                Const.(args)...)
            return dx
        end
        hv = function (θ, v, args...)
            Enzyme.autodiff(Enzyme.Forward, f2, DuplicatedNoNeed, Duplicated(θ, v),
                Const(_f), Const(f.f), Const(p),
                Const.(args)...)[1]
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
        cons_j = function (θ)
            J = Enzyme.jacobian(Enzyme.Forward, cons_oop, θ)
            if size(J, 1) == 1
                return J[1, :]
            end
            return J
        end
    else
        cons_j = (θ) -> f.cons_j(θ, p)
    end

    if f.cons !== nothing && f.cons_h === nothing
        fncs = map(1:num_cons) do i
            function (x)
                return f.cons(x, p)[i]
            end
        end

        function f2(x, dx, fnc)
            Enzyme.autodiff_deferred(Enzyme.Reverse, fnc, Enzyme.Duplicated(x, dx))
            return nothing
        end

        cons_h = function (θ)
            vdθ = Tuple((Array(r) for r in eachrow(I(length(θ)) * 1.0)))
            bθ = zeros(length(θ))
            vdbθ = Tuple(zeros(length(θ)) for i in eachindex(θ))
            res = Vector{VecOrMat}(undef, num_cons)
            for i in 1:num_cons
                bθ .= zero(eltype(bθ))
                for el in vdbθ
                    el .= zeros(length(θ))
                end
                Enzyme.autodiff(Enzyme.Forward,
                    f2,
                    Enzyme.BatchDuplicated(θ, vdθ),
                    Enzyme.BatchDuplicated(bθ, vdbθ),
                    Const(fncs[i]))

                res[i] = reduce(vcat,
                    [reshape(vdbθ[i], (1, length(vdbθ[i]))) for i in eachindex(θ)])
            end
            return res
        end
    else
        cons_h = (θ) -> f.cons_h(θ, p)
    end

    return OptimizationFunction{false}(f.f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons_oop, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype)
end

function OptimizationBase.instantiate_function(f::OptimizationFunction{false},
        cache::OptimizationBase.ReInitCache,
        adtype::AutoEnzyme,
        num_cons = 0)
    p = cache.p

    if f.grad === nothing
        res = zeros(eltype(x), size(x))
        grad = let res = res
            function (θ, args...)
                res .= zero(eltype(res))
                Enzyme.autodiff(Enzyme.Reverse,
                    Const(firstapply),
                    Active,
                    Const(f.f),
                    Enzyme.Duplicated(θ, res),
                    Const(p),
                    Const.(args)...)
                return res
            end
        end
    else
        grad = (θ, args...) -> f.grad(θ, p, args...)
    end

    if f.hess === nothing
        function g(θ, bθ, f, p, args...)
            Enzyme.autodiff_deferred(Enzyme.Reverse,
                Const(firstapply),
                Active,
                Const(f),
                Enzyme.Duplicated(θ, bθ),
                Const(p),
                Const.(args)...),
            return nothing
        end
        function hess(θ, args...)
            vdθ = Tuple((Array(r) for r in eachrow(I(length(θ)) * 1.0)))

            bθ = zeros(length(θ))
            vdbθ = Tuple(zeros(length(θ)) for i in eachindex(θ))

            Enzyme.autodiff(Enzyme.Forward,
                g,
                Enzyme.BatchDuplicated(θ, vdθ),
                Enzyme.BatchDuplicated(bθ, vdbθ),
                Const(f.f),
                Const(p),
                Const.(args)...)

            reduce(vcat, [reshape(vdbθ[i], (1, length(vdbθ[i]))) for i in eachindex(θ)])
        end
    else
        hess = (θ, args...) -> f.hess(θ, p, args...)
    end

    if f.hv === nothing
        dx = zeros(length(x))
        function f2(x, f, p, args...)
            dx .= zero(eltype(dx))
            Enzyme.autodiff_deferred(Enzyme.Reverse,
                firstapply,
                Active,
                f,
                Enzyme.Duplicated(x, dx),
                Const(p),
                Const.(args)...)
            return dx
        end
        hv = function (θ, v, args...)
            Enzyme.autodiff(Enzyme.Forward, f2, DuplicatedNoNeed, Duplicated(θ, v),
                Const(_f), Const(f.f), Const(p),
                Const.(args)...)[1]
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
        cons_j = function (θ)
            J = Enzyme.jacobian(Enzyme.Forward, cons_oop, θ)
            if size(J, 1) == 1
                return J[1, :]
            end
            return J
        end
    else
        cons_j = (θ) -> f.cons_j(θ, p)
    end

    if f.cons !== nothing && f.cons_h === nothing
        fncs = map(1:num_cons) do i
            function (x)
                return f.cons(x, p)[i]
            end
        end

        function f2(x, dx, fnc)
            Enzyme.autodiff_deferred(Enzyme.Reverse, fnc, Enzyme.Duplicated(x, dx))
            return nothing
        end

        cons_h = function (θ)
            vdθ = Tuple((Array(r) for r in eachrow(I(length(θ)) * 1.0)))
            bθ = zeros(length(θ))
            vdbθ = Tuple(zeros(length(θ)) for i in eachindex(θ))
            res = Vector{VecOrMat}(undef, num_cons)
            for i in 1:num_cons
                bθ .= zero(eltype(bθ))
                for el in vdbθ
                    el .= zeros(length(θ))
                end
                Enzyme.autodiff(Enzyme.Forward,
                    f2,
                    Enzyme.BatchDuplicated(θ, vdθ),
                    Enzyme.BatchDuplicated(bθ, vdbθ),
                    Const(fncs[i]))

                res[i] = reduce(vcat,
                    [reshape(vdbθ[i], (1, length(vdbθ[i]))) for i in eachindex(θ)])
            end
            return res
        end
    else
        cons_h = (θ) -> f.cons_h(θ, p)
    end

    return OptimizationFunction{false}(f.f, adtype; grad = grad, hess = hess, hv = hv,
        cons = cons_oop, cons_j = cons_j, cons_h = cons_h,
        hess_prototype = f.hess_prototype,
        cons_jac_prototype = f.cons_jac_prototype,
        cons_hess_prototype = f.cons_hess_prototype)
end

end
