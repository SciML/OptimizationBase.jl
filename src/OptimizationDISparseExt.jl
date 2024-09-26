using OptimizationBase
import OptimizationBase.ArrayInterface
import OptimizationBase.SciMLBase: OptimizationFunction
import OptimizationBase.LinearAlgebra: I
import DifferentiationInterface
import DifferentiationInterface: prepare_gradient, prepare_hessian, prepare_hvp,
                                 prepare_jacobian, value_and_gradient!,
                                 value_derivative_and_second_derivative!,
                                 value_and_gradient, value_derivative_and_second_derivative,
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
    global _p = p
    function _f(θ)
        return f.f(θ, _p)[1]
    end

    adtype, soadtype = generate_sparse_adtype(adtype)

    if g == true && f.grad === nothing
        extras_grad = prepare_gradient(_f, adtype.dense_ad, x)
        function grad(res, θ)
            gradient!(_f, res, extras_grad, adtype.dense_ad, θ)
        end
        if p !== SciMLBase.NullParameters()
            function grad(res, θ, p)
                global _p = p
                gradient!(_f, res, extras_grad, adtype.dense_ad, θ)
            end
        end
    elseif g == true
        grad = (G, θ, p = p) -> f.grad(G, θ, p)
    else
        grad = nothing
    end

    if fg == true && f.fg === nothing
        if g == false
            extras_grad = prepare_gradient(_f, adtype.dense_ad, x)
        end
        function fg!(res, θ)
            (y, _) = value_and_gradient!(_f, res, extras_grad, adtype.dense_ad, θ)
            return y
        end
        if p !== SciMLBase.NullParameters()
            function fg!(res, θ, p)
                global _p = p
                (y, _) = value_and_gradient!(_f, res, extras_grad, adtype.dense_ad, θ)
                return y
            end
        end
    elseif fg == true
        fg! = (G, θ, p = p) -> f.fg(G, θ, p)
    else
        fg! = nothing
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if f.hess === nothing && h == true
        extras_hess = prepare_hessian(_f, soadtype, x) #placeholder logic, can be made much better
        function hess(res, θ)
            hessian!(_f, res, extras_hess, soadtype, θ)
        end
        hess_sparsity = extras_hess.coloring_result.S
        hess_colors = extras_hess.coloring_result.color

        if p !== SciMLBase.NullParameters() && p !== nothing
            function hess(res, θ, p)
                global _p = p
                hessian!(_f, res, extras_hess, soadtype, θ)
            end
        end
    elseif h == true
        hess = (H, θ, p = p) -> f.hess(H, θ, p)
    else
        hess = nothing
    end

    if fgh == true && f.fgh === nothing
        function fgh!(G, H, θ)
            (y, _, _) = value_derivative_and_second_derivative!(
                _f, G, H, extras_hess, soadtype.dense_ad, θ)  # TODO: adtype was missing?
            return y
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function fgh!(G, H, θ, p)
                global _p = p
                (y, _, _) = value_derivative_and_second_derivative!(
                    _f, G, H, extras_hess, soadtype.dense_ad, θ)  # TODO: adtype was missing?
                return y
            end
        end
    elseif fgh == true
        fgh! = (G, H, θ, p = p) -> f.fgh(G, H, θ, p)
    else
        fgh! = nothing
    end

    if hv == true && f.hv === nothing
        extras_hvp = prepare_hvp(_f, soadtype.dense_ad, x, zeros(eltype(x), size(x)))
        function hv!(H, θ, v)
            hvp!(_f, H, extras_hvp, soadtype.dense_ad, θ, v)
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function hv!(H, θ, v, p)
                global _p = p
                hvp!(_f, H, extras_hvp, soadtype.dense_ad, θ, v)
            end
        end
    elseif hv == true
        hv! = (H, θ, v, p = p) -> f.hv(H, θ, v, p)
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
            jacobian!(cons_oop, J, extras_jac, adtype, θ)
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

    if f.cons_vjp === nothing && cons_vjp == true && cons !== nothing
        extras_pullback = prepare_pullback(
            cons_oop, adtype.dense_ad, x, ones(eltype(x), num_cons))
        function cons_vjp!(J, θ, v)
            pullback!(cons_oop, J, extras_pullback, adtype.dense_ad, θ, v)
        end
    elseif cons_vjp === true && cons !== nothing
        cons_vjp! = (J, θ, v) -> f.cons_vjp(J, θ, v, p)
    else
        cons_vjp! = nothing
    end

    if f.cons_jvp === nothing && cons_jvp == true && cons !== nothing
        extras_pushforward = prepare_pushforward(
            cons_oop, adtype.dense_ad, x, ones(eltype(x), length(x)))
        function cons_jvp!(J, θ, v)
            pushforward!(cons_oop, J, extras_pushforward, adtype.dense_ad, θ, v)
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
                hessian!(fncs[i], H[i], extras_cons_hess[i], soadtype, θ)
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
        lag_extras = prepare_hessian(
            lagrangian, soadtype, vcat(x, [one(eltype(x))], ones(eltype(x), num_cons)))
        lag_hess_prototype = lag_extras.coloring_result.S[1:length(x), 1:length(x)]
        lag_hess_colors = lag_extras.coloring_result.color

        function lag_h!(H::AbstractMatrix, θ, σ, λ)
            if σ == zero(eltype(θ))
                cons_h(H, θ)
                H *= λ
            else
                H .= hessian(lagrangian, lag_extras, soadtype, vcat(θ, [σ], λ))[
                    1:length(θ), 1:length(θ)]
            end
        end

        function lag_h!(h, θ, σ, λ)
            H = hessian(lagrangian, lag_extras, soadtype, vcat(θ, [σ], λ))[
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

        if p !== SciMLBase.NullParameters() && p !== nothing
            function lag_h!(H::AbstractMatrix, θ, σ, λ, p)
                if σ == zero(eltype(θ))
                    cons_h(H, θ)
                    H *= λ
                else
                    global _p = p
                    H .= hessian(lagrangian, lag_extras, soadtype, vcat(θ, [σ], λ))[
                        1:length(θ), 1:length(θ)]
                end
            end

            function lag_h!(h, θ, σ, λ, p)
                global _p = p
                H = hessian(lagrangian, lag_extras, soadtype, vcat(θ, [σ], λ))[
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
        end
    elseif lag_h == true
        lag_h! = (H, θ, σ, λ, p = p) -> f.lag_h(H, θ, σ, λ, p)
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
        lag_hess_colorvec = lag_hess_colors,
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr)
end

function instantiate_function(
        f::OptimizationFunction{true}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AutoSparse{<:AbstractADType}, num_cons = 0; kwargs...)
    x = cache.u0
    p = cache.p

    return instantiate_function(f, x, adtype, p, num_cons; kwargs...)
end

function instantiate_function(
        f::OptimizationFunction{false}, x, adtype::ADTypes.AutoSparse{<:AbstractADType},
        p = SciMLBase.NullParameters(), num_cons = 0;
        g = false, h = false, hv = false, fg = false, fgh = false,
        cons_j = false, cons_vjp = false, cons_jvp = false, cons_h = false,
        lag_h = false)
    global _p = p
    function _f(θ)
        return f(θ, _p)[1]
    end

    adtype, soadtype = generate_sparse_adtype(adtype)

    if g == true && f.grad === nothing
        extras_grad = prepare_gradient(_f, adtype.dense_ad, x)
        function grad(θ)
            gradient(_f, extras_grad, adtype.dense_ad, θ)
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function grad(θ, p)
                global _p = p
                gradient(_f, extras_grad, adtype.dense_ad, θ)
            end
        end
    elseif g == true
        grad = (θ, p = p) -> f.grad(θ, p)
    else
        grad = nothing
    end

    if fg == true && f.fg === nothing
        if g == false
            extras_grad = prepare_gradient(_f, adtype.dense_ad, x)
        end
        function fg!(θ)
            (y, G) = value_and_gradient(_f, extras_grad, adtype.dense_ad, θ)
            return y, G
        end
        if p !== SciMLBase.NullParameters() && p !== nothing
            function fg!(θ, p)
                global _p = p
                (y, G) = value_and_gradient(_f, extras_grad, adtype.dense_ad, θ)
                return y, G
            end
        end
    elseif fg == true
        fg! = (θ, p = p) -> f.fg(θ, p)
    else
        fg! = nothing
    end

    if fgh == true && f.fgh === nothing
        function fgh!(θ)
            (y, G, H) = value_derivative_and_second_derivative(_f, extras_hess, soadtype, θ)
            return y, G, H
        end

        if p !== SciMLBase.NullParameters() && p !== nothing
            function fgh!(θ, p)
                global _p = p
                (y, G, H) = value_derivative_and_second_derivative(
                    _f, extras_hess, soadtype, θ)
                return y, G, H
            end
        end
    elseif fgh == true
        fgh! = (θ, p = p) -> f.fgh(θ, p)
    else
        fgh! = nothing
    end

    hess_sparsity = f.hess_prototype
    hess_colors = f.hess_colorvec
    if h == true && f.hess === nothing
        extras_hess = prepare_hessian(_f, soadtype, x) #placeholder logic, can be made much better
        function hess(θ)
            hessian(_f, extras_hess, soadtype, θ)
        end
        hess_sparsity = extras_hess.coloring_result.S
        hess_colors = extras_hess.coloring_result.color

        if p !== SciMLBase.NullParameters() && p !== nothing
            function hess(θ, p)
                global _p = p
                hessian(_f, extras_hess, soadtype, θ)
            end
        end
    elseif h == true
        hess = (θ, p = p) -> f.hess(θ, p)
    else
        hess = nothing
    end

    if hv == true && f.hv === nothing
        extras_hvp = prepare_hvp(_f, soadtype.dense_ad, x, zeros(eltype(x), size(x)))
        function hv!(θ, v)
            hvp(_f, extras_hvp, soadtype.dense_ad, θ, v)
        end

        if p !== SciMLBase.NullParameters() && p !== nothing
            function hv!(θ, v, p)
                global _p = p
                hvp(_f, extras_hvp, soadtype.dense_ad, θ, v)
            end
        end
    elseif hv == true
        hv! = (θ, v, p = p) -> f.hv(θ, v, p)
    else
        hv! = nothing
    end

    if f.cons === nothing
        cons = nothing
    else
        function cons(θ)
            f.cons(θ, p)
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
            J = jacobian(cons, extras_jac, adtype, θ)
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

    if f.cons_vjp === nothing && cons_vjp == true && cons !== nothing
        extras_pullback = prepare_pullback(
            cons, adtype.dense_ad, x, ones(eltype(x), num_cons))
        function cons_vjp!(θ, v)
            pullback(cons, extras_pullback, adtype.dense_ad, θ, v)
        end
    elseif cons_vjp === true && cons !== nothing
        cons_vjp! = (θ, v) -> f.cons_vjp(θ, v, p)
    else
        cons_vjp! = nothing
    end

    if f.cons_jvp === nothing && cons_jvp == true && cons !== nothing
        extras_pushforward = prepare_pushforward(
            cons, adtype.dense_ad, x, ones(eltype(x), length(x)))
        function cons_jvp!(θ, v)
            pushforward(cons, extras_pushforward, adtype.dense_ad, θ, v)
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
                hessian(fncs[i], extras_cons_hess[i], soadtype, θ)
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
        lag_extras = prepare_hessian(
            lagrangian, soadtype, vcat(x, [one(eltype(x))], ones(eltype(x), num_cons)))
        function lag_h!(θ, σ, λ)
            if σ == zero(eltype(θ))
                return λ .* cons_h!(θ)
            else
                hess = hessian(lagrangian, lag_extras, soadtype, vcat(θ, [σ], λ))[
                    1:length(θ), 1:length(θ)]
                return hess
            end
        end
        lag_hess_prototype = lag_extras.coloring_result.S[1:length(θ), 1:length(θ)]
        lag_hess_colors = lag_extras.coloring_result.color

        if p !== SciMLBase.NullParameters() && p !== nothing
            function lag_h!(θ, σ, λ, p)
                if σ == zero(eltype(θ))
                    return λ .* cons_h!(θ)
                else
                    global _p = p
                    hess = hessian(lagrangian, lag_extras, vcat(θ, [σ], λ))[
                        1:length(θ), 1:length(θ)]
                    return hess
                end
            end
        end
    elseif lag_h == true && cons !== nothing
        lag_h! = (θ, σ, μ, p = p) -> f.lag_h(θ, σ, μ, p)
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
        lag_hess_colorvec = lag_hess_colors,
        sys = f.sys,
        expr = f.expr,
        cons_expr = f.cons_expr)
end

function instantiate_function(
        f::OptimizationFunction{false}, cache::OptimizationBase.ReInitCache,
        adtype::ADTypes.AutoSparse{<:AbstractADType}, num_cons = 0; kwargs...)
    x = cache.u0
    p = cache.p

    return instantiate_function(f, x, adtype, p, num_cons; kwargs...)
end
