using OptimizationBase, Test, DifferentiationInterface, SparseArrays, Symbolics
using ForwardDiff, Zygote, ReverseDiff, FiniteDiff, Tracker
using ModelingToolkit, Enzyme, Random

x0 = zeros(2)
rosenbrock(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
l1 = rosenbrock(x0)

function g!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    G[2] = 200.0 * (x[2] - x[1]^2)
end

function h!(H, x)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
end

G1 = Array{Float64}(undef, 2)
G2 = Array{Float64}(undef, 2)
H1 = Array{Float64}(undef, 2, 2)
H2 = Array{Float64}(undef, 2, 2)

g!(G1, x0)
h!(H1, x0)

cons = (res, x, p) -> (res .= [x[1]^2 + x[2]^2]; return nothing)
optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoModelingToolkit(), cons = cons)
optprob = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoModelingToolkit(),
    nothing, 1)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2
res = Array{Float64}(undef, 1)
optprob.cons(res, x0)
@test res == [0.0]
J = Array{Float64}(undef, 2)
optprob.cons_j(J, [5.0, 3.0])
@test J == [10.0, 6.0]
H3 = [Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 == [[2.0 0.0; 0.0 2.0]]

function con2_c(res, x, p)
    res .= [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    return nothing
end
optf = OptimizationFunction(rosenbrock,
    OptimizationBase.AutoModelingToolkit(),
    cons = con2_c)
optprob = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoModelingToolkit(),
    nothing, 2)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2
res = Array{Float64}(undef, 2)
optprob.cons(res, x0)
@test res == [0.0, 0.0]
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

G2 = Array{Float64}(undef, 2)
H2 = Array{Float64}(undef, 2, 2)

optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoEnzyme(), cons = cons)
optprob = OptimizationBase.instantiate_function(
    optf, x0, OptimizationBase.AutoEnzyme(),
    nothing, 1)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2
res = Array{Float64}(undef, 1)
optprob.cons(res, x0)
@test res == [0.0]
J = Array{Float64}(undef, 2)
optprob.cons_j(J, [5.0, 3.0])
@test J == [10.0, 6.0]
H3 = [Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 == [[2.0 0.0; 0.0 2.0]]
G2 = Array{Float64}(undef, 2)
H2 = Array{Float64}(undef, 2, 2)
optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoEnzyme(), cons = con2_c)
optprob = OptimizationBase.instantiate_function(
    optf, x0, OptimizationBase.AutoEnzyme(),
    nothing, 2)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2
res = Array{Float64}(undef, 2)
optprob.cons(res, x0)
@test res == [0.0, 0.0]
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

G2 = Array{Float64}(undef, 2)
H2 = Array{Float64}(undef, 2, 2)

optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoReverseDiff(), cons = con2_c)
optprob = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoReverseDiff(),
    nothing, 2)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2
res = Array{Float64}(undef, 2)
optprob.cons(res, x0)
@test res == [0.0, 0.0]
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
H3 == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

G2 = Array{Float64}(undef, 2)
H2 = Array{Float64}(undef, 2, 2)

optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoReverseDiff(), cons = con2_c)
optprob = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoReverseDiff(compile = true),
    nothing, 2)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2
res = Array{Float64}(undef, 2)
optprob.cons(res, x0)
@test res == [0.0, 0.0]
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
H3 == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

G2 = Array{Float64}(undef, 2)
H2 = Array{Float64}(undef, 2, 2)

optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoZygote(), cons = con2_c)
optprob = OptimizationBase.instantiate_function(optf, x0, OptimizationBase.AutoZygote(),
    nothing, 2)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2
res = Array{Float64}(undef, 2)
optprob.cons(res, x0)
@test res == [0.0, 0.0]
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
H3 == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoModelingToolkit(true, true),
    cons = con2_c)
optprob = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoModelingToolkit(true, true),
    nothing, 2)
using SparseArrays
sH = sparse([1, 1, 2, 2], [1, 2, 1, 2], zeros(4))
@test findnz(sH)[1:2] == findnz(optprob.hess_prototype)[1:2]
optprob.hess(sH, x0)
@test sH == H2
res = Array{Float64}(undef, 2)
optprob.cons(res, x0)
@test res == [0.0, 0.0]
sJ = sparse([1, 1, 2, 2], [1, 2, 1, 2], zeros(4))
@test findnz(sJ)[1:2] == findnz(optprob.cons_jac_prototype)[1:2]
optprob.cons_j(sJ, [5.0, 3.0])
@test all(isapprox(sJ, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
sH3 = [sparse([1, 2], [1, 2], zeros(2)), sparse([1, 1, 2], [1, 2, 1], zeros(3))]
@test getindex.(findnz.(sH3), Ref([1, 2])) ==
      getindex.(findnz.(optprob.cons_hess_prototype), Ref([1, 2]))
optprob.cons_h(sH3, x0)
@test Array.(sH3) == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff())
optprob = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoForwardDiff(),
    nothing)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 ≈ H2

optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoZygote())
optprob = OptimizationBase.instantiate_function(optf,
    x0,
    OptimizationBase.AutoZygote(),
    nothing)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2

optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoReverseDiff())
optprob = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoReverseDiff(),
    nothing)
optprob.grad(G2, x0)
@test G1 == G2
optprob.hess(H2, x0)
@test H1 == H2

optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoTracker())
optprob = OptimizationBase.instantiate_function(optf,
    x0,
    OptimizationBase.AutoTracker(),
    nothing)
optprob.grad(G2, x0)
@test G1 == G2
@test_broken optprob.hess(H2, x0)

prob = OptimizationProblem(optf, x0)

optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoFiniteDiff())
optprob = OptimizationBase.instantiate_function(
    optf, x0, OptimizationBase.AutoFiniteDiff(),
    nothing)
optprob.grad(G2, x0)
@test G1≈G2 rtol=1e-6
optprob.hess(H2, x0)
@test H1≈H2 rtol=1e-6

# Test new constraints
cons = (res, x, p) -> (res .= [x[1]^2 + x[2]^2])
optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoFiniteDiff(), cons = cons)
optprob = OptimizationBase.instantiate_function(
    optf, x0, OptimizationBase.AutoFiniteDiff(),
    nothing, 1)
optprob.grad(G2, x0)
@test G1≈G2 rtol=1e-6
optprob.hess(H2, x0)
@test H1≈H2 rtol=1e-6
res = Array{Float64}(undef, 1)
optprob.cons(res, x0)
@test res == [0.0]
optprob.cons(res, [1.0, 4.0])
@test res == [17.0]
J = zeros(1, 2)
optprob.cons_j(J, [5.0, 3.0])
@test J ≈ [10.0 6.0]
H3 = [Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 ≈ [[2.0 0.0; 0.0 2.0]]

H4 = Array{Float64}(undef, 2, 2)
μ = randn(1)
σ = rand()
# optprob.lag_h(H4, x0, σ, μ)
# @test H4≈σ * H1 + μ[1] * H3[1] rtol=1e-6

cons_jac_proto = Float64.(sparse([1 1])) # Things break if you only use [1 1]; see FiniteDiff.jl
cons_jac_colors = 1:2
optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoFiniteDiff(), cons = cons,
    cons_jac_prototype = cons_jac_proto,
    cons_jac_colorvec = cons_jac_colors)
optprob = OptimizationBase.instantiate_function(
    optf, x0, OptimizationBase.AutoFiniteDiff(),
    nothing, 1)
@test optprob.cons_jac_prototype == sparse([1.0 1.0]) # make sure it's still using it
@test optprob.cons_jac_colorvec == 1:2
J = zeros(1, 2)
optprob.cons_j(J, [5.0, 3.0])
@test J ≈ [10.0 6.0]

function con2_c(res, x, p)
    res .= [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
end
optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoFiniteDiff(), cons = con2_c)
optprob = OptimizationBase.instantiate_function(
    optf, x0, OptimizationBase.AutoFiniteDiff(),
    nothing, 2)
optprob.grad(G2, x0)
@test G1≈G2 rtol=1e-6
optprob.hess(H2, x0)
@test H1≈H2 rtol=1e-6
res = Array{Float64}(undef, 2)
optprob.cons(res, x0)
@test res == [0.0, 0.0]
optprob.cons(res, [1.0, 2.0])
@test res ≈ [5.0, 0.682941969615793]
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

cons_jac_proto = Float64.(sparse([1 1; 1 1]))
cons_jac_colors = 1:2
optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoFiniteDiff(), cons = con2_c,
    cons_jac_prototype = cons_jac_proto,
    cons_jac_colorvec = cons_jac_colors)
optprob = OptimizationBase.instantiate_function(
    optf, x0, OptimizationBase.AutoFiniteDiff(),
    nothing, 2)
@test optprob.cons_jac_prototype == sparse([1.0 1.0; 1.0 1.0]) # make sure it's still using it
@test optprob.cons_jac_colorvec == 1:2
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test all(isapprox(J, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
H2 = Array{Float64}(undef, 2, 2)
optprob.hess(H2, [5.0, 3.0])
@test all(isapprox(H2, [28802.0 -2000.0; -2000.0 200.0]; rtol = 1e-3))

cons_j = (J, θ, p) -> optprob.cons_j(J, θ)
hess = (H, θ, p) -> optprob.hess(H, θ)
sH = sparse([1, 1, 2, 2], [1, 2, 1, 2], zeros(4))
sJ = sparse([1, 1, 2, 2], [1, 2, 1, 2], zeros(4))
optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoForwardDiff(), hess = hess,
    hess_prototype = copy(sH), cons = con2_c, cons_j = cons_j,
    cons_jac_prototype = copy(sJ))
optprob1 = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoForwardDiff(),
    nothing, 2)
@test optprob1.hess_prototype == sparse([0.0 0.0; 0.0 0.0]) # make sure it's still using it
optprob1.hess(sH, [5.0, 3.0])
@test all(isapprox(sH, [28802.0 -2000.0; -2000.0 200.0]; rtol = 1e-3))
@test optprob1.cons_jac_prototype == sparse([0.0 0.0; 0.0 0.0]) # make sure it's still using it
optprob1.cons_j(sJ, [5.0, 3.0])
@test all(isapprox(sJ, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))

grad = (G, θ, p) -> optprob.grad(G, θ)
hess = (H, θ, p) -> optprob.hess(H, θ)
cons_j = (J, θ, p) -> optprob.cons_j(J, θ)
cons_h = (res, θ, p) -> optprob.cons_h(res, θ)
sH = sparse([1, 1, 2, 2], [1, 2, 1, 2], zeros(4))
sJ = sparse([1, 1, 2, 2], [1, 2, 1, 2], zeros(4))
sH3 = [sparse([1, 2], [1, 2], zeros(2)), sparse([1, 1, 2], [1, 2, 1], zeros(3))]
optf = OptimizationFunction(rosenbrock, SciMLBase.NoAD(), grad = grad, hess = hess,
    cons = con2_c, cons_j = cons_j, cons_h = cons_h,
    hess_prototype = sH, cons_jac_prototype = sJ,
    cons_hess_prototype = sH3)
optprob2 = OptimizationBase.instantiate_function(optf, x0, SciMLBase.NoAD(), nothing, 2)
optprob2.hess(sH, [5.0, 3.0])
@test all(isapprox(sH, [28802.0 -2000.0; -2000.0 200.0]; rtol = 1e-3))
optprob2.cons_j(sJ, [5.0, 3.0])
@test all(isapprox(sJ, [10.0 6.0; -0.149013 -0.958924]; rtol = 1e-3))
optprob2.cons_h(sH3, [5.0, 3.0])
@test Array.(sH3)≈[
    [2.0 0.0; 0.0 2.0],
    [2.8767727327346804 0.2836621681849162; 0.2836621681849162 -6.622738308376736e-9]
] rtol=1e-4

optf = OptimizationFunction(rosenbrock,
    OptimizationBase.AutoSparseFiniteDiff(),
    cons = con2_c)
optprob = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoSparseFiniteDiff(),
    nothing, 2)
G2 = Array{Float64}(undef, 2)
optprob.grad(G2, x0)
@test G1≈G2 rtol=1e-4
H2 = Array{Float64}(undef, 2, 2)
optprob.hess(H2, x0)
@test H1≈H2 rtol=1e-4
res = Array{Float64}(undef, 2)
optprob.cons(res, x0)
@test res≈[0.0, 0.0] atol=1e-4
optprob.cons(res, [1.0, 2.0])
@test res ≈ [5.0, 0.682941969615793]
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test J≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-3
H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoSparseFiniteDiff())
optprob = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoSparseFiniteDiff(),
    nothing)
optprob.grad(G2, x0)
@test G1≈G2 rtol=1e-6
optprob.hess(H2, x0)
@test H1≈H2 rtol=1e-4

optf = OptimizationFunction(rosenbrock,
    OptimizationBase.AutoSparseForwardDiff(),
    cons = con2_c)
optprob = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoSparseForwardDiff(),
    nothing, 2)
G2 = Array{Float64}(undef, 2)
optprob.grad(G2, x0)
@test G1≈G2 rtol=1e-4
H2 = Array{Float64}(undef, 2, 2)
optprob.hess(H2, x0)
@test H1≈H2 rtol=1e-4
res = Array{Float64}(undef, 2)
optprob.cons(res, x0)
@test res≈[0.0, 0.0] atol=1e-4
optprob.cons(res, [1.0, 2.0])
@test res ≈ [5.0, 0.682941969615793]
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test J≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-3
H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoSparseForwardDiff())
optprob = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoSparseForwardDiff(),
    nothing)
optprob.grad(G2, x0)
@test G1≈G2 rtol=1e-6
optprob.hess(H2, x0)
@test H1≈H2 rtol=1e-6

optf = OptimizationFunction(rosenbrock,
    OptimizationBase.AutoSparseReverseDiff(),
    cons = con2_c)
optprob = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoSparseReverseDiff(true),
    nothing, 2)
G2 = Array{Float64}(undef, 2)
optprob.grad(G2, x0)
@test G1≈G2 rtol=1e-4
H2 = Array{Float64}(undef, 2, 2)
optprob.hess(H2, x0)
@test H1≈H2 rtol=1e-4
res = Array{Float64}(undef, 2)
optprob.cons(res, x0)
@test res≈[0.0, 0.0] atol=1e-4
optprob.cons(res, [1.0, 2.0])
@test res ≈ [5.0, 0.682941969615793]
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test J≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-3
H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

optf = OptimizationFunction(rosenbrock,
    OptimizationBase.AutoSparseReverseDiff(),
    cons = con2_c)
optprob = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoSparseReverseDiff(),
    nothing, 2)
G2 = Array{Float64}(undef, 2)
optprob.grad(G2, x0)
@test G1≈G2 rtol=1e-4
H2 = Array{Float64}(undef, 2, 2)
optprob.hess(H2, x0)
@test H1≈H2 rtol=1e-4
res = Array{Float64}(undef, 2)
optprob.cons(res, x0)
@test res≈[0.0, 0.0] atol=1e-4
optprob.cons(res, [1.0, 2.0])
@test res ≈ [5.0, 0.682941969615793]
J = Array{Float64}(undef, 2, 2)
optprob.cons_j(J, [5.0, 3.0])
@test J≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-3
H3 = [Array{Float64}(undef, 2, 2), Array{Float64}(undef, 2, 2)]
optprob.cons_h(H3, x0)
@test H3 ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

optf = OptimizationFunction(rosenbrock, OptimizationBase.AutoSparseReverseDiff())
optprob = OptimizationBase.instantiate_function(optf, x0,
    OptimizationBase.AutoSparseReverseDiff(),
    nothing)
optprob.grad(G2, x0)
@test G1≈G2 rtol=1e-6
optprob.hess(H2, x0)
@test H1≈H2 rtol=1e-6

@testset "OOP" begin
    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoEnzyme(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoEnzyme(),
        nothing, 1)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoEnzyme(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoEnzyme(),
        nothing, 2)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0])≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoFiniteDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoFiniteDiff(),
        nothing, 1)

    @test optprob.grad(x0)≈G1 rtol=1e-6
    @test optprob.hess(x0)≈H1 rtol=1e-6

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0])≈[10.0, 6.0] rtol=1e-6

    @test optprob.cons_h(x0) ≈ [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoFiniteDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoFiniteDiff(),
        nothing, 2)

    @test optprob.grad(x0)≈G1 rtol=1e-6
    @test optprob.hess(x0)≈H1 rtol=1e-6
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0])≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test optprob.cons_h(x0) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoForwardDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoForwardDiff(),
        nothing, 1)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoForwardDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoForwardDiff(),
        nothing, 2)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0])≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoReverseDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoReverseDiff(),
        nothing, 1)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoReverseDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoReverseDiff(),
        nothing, 2)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0])≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoReverseDiff(true),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoReverseDiff(true),
        nothing, 1)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoReverseDiff(true),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoReverseDiff(true),
        nothing, 2)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0])≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoSparseForwardDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseForwardDiff(),
        nothing, 1)

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoSparseForwardDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseForwardDiff(),
        nothing, 2)

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test Array(optprob.cons_j([5.0, 3.0]))≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test Array.(optprob.cons_h(x0)) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoSparseFiniteDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseFiniteDiff(),
        nothing, 1)

    @test optprob.grad(x0)≈G1 rtol=1e-4
    @test Array(optprob.hess(x0)) ≈ H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) ≈ [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoSparseFiniteDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseForwardDiff(),
        nothing, 2)

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test Array(optprob.cons_j([5.0, 3.0]))≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test Array.(optprob.cons_h(x0)) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoSparseReverseDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseReverseDiff(),
        nothing, 1)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1

    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoSparseReverseDiff(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseReverseDiff(),
        nothing, 2)

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test Array(optprob.cons_j([5.0, 3.0]))≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test Array.(optprob.cons_h(x0)) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoSparseReverseDiff(true),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseReverseDiff(true),
        nothing, 1)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoSparseReverseDiff(true),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(optf, x0,
        OptimizationBase.AutoSparseReverseDiff(true),
        nothing, 2)

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test Array(optprob.cons_j([5.0, 3.0]))≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test Array.(optprob.cons_h(x0)) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoZygote(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoZygote(),
        nothing, 1)

    @test optprob.grad(x0) == G1
    @test optprob.hess(x0) == H1
    @test optprob.cons(x0) == [0.0]

    @test optprob.cons_j([5.0, 3.0]) == [10.0, 6.0]

    @test optprob.cons_h(x0) == [[2.0 0.0; 0.0 2.0]]

    cons = (x, p) -> [x[1]^2 + x[2]^2, x[2] * sin(x[1]) - x[1]]
    optf = OptimizationFunction{false}(rosenbrock,
        OptimizationBase.AutoZygote(),
        cons = cons)
    optprob = OptimizationBase.instantiate_function(
        optf, x0, OptimizationBase.AutoZygote(),
        nothing, 2)

    @test optprob.grad(x0) == G1
    @test Array(optprob.hess(x0)) ≈ H1
    @test optprob.cons(x0) == [0.0, 0.0]
    @test optprob.cons_j([5.0, 3.0])≈[10.0 6.0; -0.149013 -0.958924] rtol=1e-6
    @test Array.(optprob.cons_h(x0)) ≈ [[2.0 0.0; 0.0 2.0], [-0.0 1.0; 1.0 0.0]]
end
