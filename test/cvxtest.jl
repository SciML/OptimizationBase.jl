using Optimization, OptimizationBase, ForwardDiff, SymbolicAnalysis

function f(x, p = nothing)
    return exp(x[1]) + x[1]^2
end

optf = OptimizationFunction(f, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, [0.4])

@time sol = solve(prob, Optimization.LBFGS(), maxiters = 1000)

x0 = zeros(2)
rosenbrock(x, p = nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
l1 = rosenbrock(x0)

optf = OptimizationFunction(rosenbrock, AutoEnzyme())
prob = OptimizationProblem(optf, x0)
@time res = solve(prob, Optimization.LBFGS(), maxiters = 100)

function con2_c(res, x, p)
    res .= [x[1]^2 + x[2]^2, (x[2] * sin(x[1]) + x[1])-5]
end

optf = OptimizationFunction(rosenbrock, AutoZygote(), cons = con2_c)
prob = OptimizationProblem(optf, x0, lcons = [1.0, -Inf], ucons = [1.0, 0.0], lb = [-1.0, -1.0], ub = [1.0, 1.0])
@time res = solve(prob, Optimization.LBFGS(), maxiters = 100)

m = 100
σ = 0.005
q = Matrix{Float64}(I) .+ 2.0

M = SymmetricPositiveDefinite(5)
@variables X[1:5, 1:5]
data2 = [exp(M, q, σ * rand(M; vector_at=q)) for i in 1:m];

f(x, p = nothing) = sum(SymbolicAnalysis.distance(M, data2[i], x)^2 for i in 1:5)
optf = OptimizationFunction(f, Optimization.AutoZygote(); expr = prob.f.expr, sys = prob.f.sys)
prob = OptimizationProblem(optf, data2[1]; manifold = M)

opt = OptimizationManopt.GradientDescentOptimizer()
@time sol = solve(prob, opt, maxiters = 1000)