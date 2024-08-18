using OptimizationBase
import OptimizationBase.ArrayInterface
import OptimizationBase.SciMLBase: OptimizationFunction
import OptimizationBase.LinearAlgebra: I
import DifferentiationInterface
import DifferentiationInterface: prepare_gradient, prepare_hessian, prepare_hvp,
                                 prepare_jacobian,
                                 gradient!, hessian!, hvp!, jacobian!, gradient, hessian,
                                 hvp, jacobian
using ADTypes
using SparseConnectivityTracer, SparseMatrixColorings

function generate_sparse_adtype(adtype)
    if adtype.sparsity_detector isa ADTypes.NoSparsityDetector &&
       adtype.coloring_algorithm isa ADTypes.NoColoringAlgorithm
        adtype = AutoSparse(adtype.dense_ad; sparsity_detector = TracerSparsityDetector(),
            coloring_algorithm = GreedyColoringAlgorithm())
        if adtype.dense_ad isa ADTypes.AutoFiniteDiff
            soadtype = AutoSparse(
                DifferentiationInterface.SecondOrder(adtype.dense_ad, adtype.dense_ad),
                sparsity_detector = TracerSparsityDetector(),
                coloring_algorithm = GreedyColoringAlgorithm())
        elseif !(adtype.dense_ad isa SciMLBase.NoAD) &&
               ADTypes.mode(adtype.dense_ad) isa ADTypes.ForwardMode
            soadtype = AutoSparse(
                DifferentiationInterface.SecondOrder(adtype.dense_ad, AutoReverseDiff()),
                sparsity_detector = TracerSparsityDetector(),
                coloring_algorithm = GreedyColoringAlgorithm()) #make zygote?
        elseif !(adtype isa SciMLBase.NoAD) &&
               ADTypes.mode(adtype.dense_ad) isa ADTypes.ReverseMode
            soadtype = AutoSparse(
                DifferentiationInterface.SecondOrder(AutoForwardDiff(), adtype.dense_ad),
                sparsity_detector = TracerSparsityDetector(),
                coloring_algorithm = GreedyColoringAlgorithm())
        end
    elseif adtype.sparsity_detector isa ADTypes.NoSparsityDetector &&
           !(adtype.coloring_algorithm isa ADTypes.NoColoringAlgorithm)
        adtype = AutoSparse(adtype.dense_ad; sparsity_detector = TracerSparsityDetector(),
            coloring_algorithm = adtype.coloring_algorithm)
        if adtype.dense_ad isa ADTypes.AutoFiniteDiff
            soadtype = AutoSparse(
                DifferentiationInterface.SecondOrder(adtype.dense_ad, adtype.dense_ad),
                sparsity_detector = TracerSparsityDetector(),
                coloring_algorithm = adtype.coloring_algorithm)
        elseif !(adtype.dense_ad isa SciMLBase.NoAD) &&
               ADTypes.mode(adtype.dense_ad) isa ADTypes.ForwardMode
            soadtype = AutoSparse(
                DifferentiationInterface.SecondOrder(adtype.dense_ad, AutoReverseDiff()),
                sparsity_detector = TracerSparsityDetector(),
                coloring_algorithm = adtype.coloring_algorithm)
        elseif !(adtype isa SciMLBase.NoAD) &&
               ADTypes.mode(adtype.dense_ad) isa ADTypes.ReverseMode
            soadtype = AutoSparse(
                DifferentiationInterface.SecondOrder(AutoForwardDiff(), adtype.dense_ad),
                sparsity_detector = TracerSparsityDetector(),
                coloring_algorithm = adtype.coloring_algorithm)
        end
    elseif !(adtype.sparsity_detector isa ADTypes.NoSparsityDetector) &&
           adtype.coloring_algorithm isa ADTypes.NoColoringAlgorithm
        adtype = AutoSparse(adtype.dense_ad; sparsity_detector = adtype.sparsity_detector,
            coloring_algorithm = GreedyColoringAlgorithm())
        if adtype.dense_ad isa ADTypes.AutoFiniteDiff
            soadtype = AutoSparse(
                DifferentiationInterface.SecondOrder(adtype.dense_ad, adtype.dense_ad),
                sparsity_detector = adtype.sparsity_detector,
                coloring_algorithm = GreedyColoringAlgorithm())
        elseif !(adtype.dense_ad isa SciMLBase.NoAD) &&
               ADTypes.mode(adtype.dense_ad) isa ADTypes.ForwardMode
            soadtype = AutoSparse(
                DifferentiationInterface.SecondOrder(adtype.dense_ad, AutoReverseDiff()),
                sparsity_detector = adtype.sparsity_detector,
                coloring_algorithm = GreedyColoringAlgorithm())
        elseif !(adtype isa SciMLBase.NoAD) &&
               ADTypes.mode(adtype.dense_ad) isa ADTypes.ReverseMode
            soadtype = AutoSparse(
                DifferentiationInterface.SecondOrder(AutoForwardDiff(), adtype.dense_ad),
                sparsity_detector = adtype.sparsity_detector,
                coloring_algorithm = GreedyColoringAlgorithm())
        end
    else
        if adtype.dense_ad isa ADTypes.AutoFiniteDiff
            soadtype = AutoSparse(
                DifferentiationInterface.SecondOrder(adtype.dense_ad, adtype.dense_ad),
                sparsity_detector = adtype.sparsity_detector,
                coloring_algorithm = adtype.coloring_algorithm)
        elseif !(adtype.dense_ad isa SciMLBase.NoAD) &&
               ADTypes.mode(adtype.dense_ad) isa ADTypes.ForwardMode
            soadtype = AutoSparse(
                DifferentiationInterface.SecondOrder(adtype.dense_ad, AutoReverseDiff()),
                sparsity_detector = adtype.sparsity_detector,
                coloring_algorithm = adtype.coloring_algorithm)
        elseif !(adtype isa SciMLBase.NoAD) &&
               ADTypes.mode(adtype.dense_ad) isa ADTypes.ReverseMode
            soadtype = AutoSparse(
                DifferentiationInterface.SecondOrder(AutoForwardDiff(), adtype.dense_ad),
                sparsity_detector = adtype.sparsity_detector,
                coloring_algorithm = adtype.coloring_algorithm)
        end
    end
    return adtype, soadtype
end

function instantiate_function(
        f::OptimizationFunction{true}, x, adtype::ADTypes.AutoSparse{<:AbstractADType},
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false)
    function _f(θ)
        return f.f(θ, p)[1]
    end

    adtype, soadtype = generate_sparse_adtype(adtype)

    if g == true && f.grad === nothing
        extras_grad = prepare_gradient(_f, adtype.dense_ad, x)
        function grad(res, θ)
            gradient!(_f, res, adtype.dense_ad, θ, extras_grad)
        end
    elseif g == true
        grad = (G, θ) -> f.grad(G, θ, p)
    else
        grad = nothing
    end

    if fg == true && f.fg !== nothing
        function fg!(res, θ)
            (y, _) = value_and_gradient!(_f, res, adtype.dense_ad, θ, extras_grad)
            return y
        end
    elseif fg == true
        fg! = (G, θ) -> f.fg(G, θ, p)
    else
        fg! = nothing
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if f.hess === nothing && h == true
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

    if hv == true && f.hv === nothing
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

        function cons_oop(x, p = p)
            _res = zeros(eltype(x), num_cons)
            f.cons(_res, x, p)
            return _res
        end

        function lagrangian(x, σ = one(eltype(x)), λ = ones(eltype(x), num_cons))
            return σ * _f(x) + dot(λ, cons_oop(x))
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
    elseif cons_j === true && cons !== nothing
        cons_j! = (J, θ) -> f.cons_j(J, θ, p)
    else
        cons_j! = nothing
    end

    if f.cons_vjp === nothing && cons_vjp == true
        extras_pullback = prepare_pullback(cons_oop, adtype, x)
        function cons_vjp!(J, θ, v)
            pullback!(cons_oop, J, adtype.dense_ad, θ, v, extras_pullback)
        end
    elseif cons_vjp === true && cons !== nothing
        cons_vjp! = (J, θ, v) -> f.cons_vjp(J, θ, v, p)
    else
        cons_vjp! = nothing
    end

    if f.cons_jvp === nothing && cons_jvp == true
        extras_pushforward = prepare_pushforward(cons_oop, adtype, x)
        function cons_jvp!(J, θ, v)
            pushforward!(cons_oop, J, adtype.dense_ad, θ, v, extras_pushforward)
        end
    elseif cons_jvp === true && cons !== nothing
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
    elseif cons_h == true && cons !== nothing
        cons_h! = (res, θ) -> f.cons_h(res, θ, p)
    else
        cons_h! = nothing
    end

    lag_hess_prototype = f.lag_hess_prototype
    lag_hess_colors = f.lag_hess_colorvec
    if cons !== nothing && lag_h == true && f.lag_h === nothing
        lag_extras = prepare_hessian(lagrangian, soadtype, x)
        lag_hess_prototype = lag_extras.coloring_result.S
        lag_hess_colors = lag_extras.coloring_result.color

        function lag_h!(H::AbstractMatrix, θ, σ, λ)
            if σ == zero(eltype(θ))
                cons_h(H, θ)
                H *= λ
            else
                hessian!(x -> lagrangian(x, σ, λ), H, soadtype, θ, lag_extras)
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
    elseif lag_h == true
        lag_h! = (H, θ, σ, λ) -> f.lag_h(H, θ, σ, λ, p)
    else
        lag_h! = nothing
    end
    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv!,
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
        lag_hess_colorvec = lag_hess_colors,
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr)
end

function instantiate_function(
        f::OptimizationFunction{true}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AutoSparse{<:AbstractADType}, num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false)
    x = cache.u0
    p = cache.p

    return instantiate_function(f, x, adtype, p, num_cons; g = g, h = h, hv = hv, fg = fg,
        fgh = fgh, cons_j = cons_j, cons_vjp = cons_vjp, cons_jvp = cons_jvp, cons_h = cons_h,
        lag_h = lag_h)
end

function instantiate_function(
        f::OptimizationFunction{false}, x, adtype::ADTypes.AutoSparse{<:AbstractADType},
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false)
    function _f(θ)
        return f(θ, p)[1]
    end

    adtype, soadtype = generate_sparse_adtype(adtype)

    if g == true && f.grad === nothing
        extras_grad = prepare_gradient(_f, adtype.dense_ad, x)
        function grad(θ)
            gradient(_f, adtype.dense_ad, θ, extras_grad)
        end
    elseif g == true
        grad = (θ) -> f.grad(θ, p)
    else
        grad = nothing
    end

    if fg == true && f.fg !== nothing
        function fg!(θ)
            (y, G) = value_and_gradient(_f, adtype.dense_ad, θ, extras_grad)
            return y, G
        end
    elseif fg == true
        fg! = (θ) -> f.fg(θ, p)
    else
        fg! = nothing
    end

    if fgh == true && f.fgh !== nothing
        function fgh!(θ)
            (y, G, H) = value_derivative_and_second_derivative(_f, soadtype, θ, extras_hess)
            return y, G, H
        end
    elseif fgh == true
        fgh! = (θ) -> f.fgh(θ, p)
    else
        fgh! = nothing
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if h == true && f.hess === nothing
        extras_hess = prepare_hessian(_f, soadtype, x) #placeholder logic, can be made much better
        function hess(θ)
            hessian(_f, soadtype, θ, extras_hess)
        end
        hess_sparsity = extras_hess.coloring_result.S
        hess_colors = extras_hess.coloring_result.color
    elseif h == true
        hess = (θ) -> f.hess(θ, p)
    else
        hess = nothing
    end

    if hv == true && f.hv === nothing
        extras_hvp = prepare_hvp(_f, soadtype.dense_ad, x, zeros(eltype(x), size(x)))
        function hv!(θ, v)
            hvp(_f, soadtype.dense_ad, θ, v, extras_hvp)
        end
    elseif hv == true
        hv! = (θ, v) -> f.hv(θ, v, p)
    else
        hv! = nothing
    end

    if f.cons === nothing
        cons = nothing
    else
        function cons(θ)
            f.cons(θ, p)
        end

        function lagrangian(x, σ = one(eltype(x)), λ = ones(eltype(x), num_cons))
            return σ * _f(x) + dot(λ, cons(x))
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
        cons_jac_prototype = extras_jac.coloring_result.S
        cons_jac_colorvec = extras_jac.coloring_result.color
    elseif cons_j === true && cons !== nothing
        cons_j! = (θ) -> f.cons_j(θ, p)
    else
        cons_j! = nothing
    end

    if f.cons_vjp === nothing && cons_vjp == true
        extras_pullback = prepare_pullback(cons, adtype, x)
        function cons_vjp!(θ, v)
            pullback(cons, adtype, θ, v, extras_pullback)
        end
    elseif cons_vjp === true && cons !== nothing
        cons_vjp! = (θ, v) -> f.cons_vjp(θ, v, p)
    else
        cons_vjp! = nothing
    end

    if f.cons_jvp === nothing && cons_jvp == true
        extras_pushforward = prepare_pushforward(cons, adtype, x)
        function cons_jvp!(θ, v)
            pushforward(cons, adtype, θ, v, extras_pushforward)
        end
    elseif cons_jvp === true && cons !== nothing
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
        colores = getfield.(extras_cons_hess, :coloring_result)
        conshess_sparsity = getfield.(colores, :S)
        conshess_colors = getfield.(colores, :color)
    elseif cons_h == true && cons !== nothing
        cons_h! = (res, θ) -> f.cons_h(res, θ, p)
    else
        cons_h! = nothing
    end

    lag_hess_prototype = f.lag_hess_prototype
    lag_hess_colors = f.lag_hess_colorvec
    if cons !== nothing && lag_h == true && f.lag_h === nothing
        lag_extras = prepare_hessian(lagrangian, soadtype, x)
        function lag_h!(θ, σ, λ)
            if σ == zero(eltype(θ))
                return λ * cons_h!(θ)
            else
                hess = hessian(x -> lagrangian(x, σ, λ), soadtype, θ, lag_extras)
                return hess
            end
        end
        lag_hess_prototype = lag_extras.coloring_result.S
        lag_hess_colors = lag_extras.coloring_result.color
    elseif lag_h == true && cons !== nothing
        lag_h! = (θ, σ, μ) -> f.lag_h(θ, σ, μ, p)
    else
        lag_h! = nothing
    end
    return OptimizationFunction{true}(f.f, adtype; grad = grad, hess = hess, hv = hv!,
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
        lag_hess_colorvec = lag_hess_colors,
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr)
end

function instantiate_function(
        f::OptimizationFunction{false}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AutoSparse{<:AbstractADType}, num_cons = 0)
    x = cache.u0
    p = cache.p

    return instantiate_function(f, x, adtype, p, num_cons)
end
