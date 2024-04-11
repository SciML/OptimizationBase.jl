using Optimization, OptimizationBase, ForwardDiff

function f(x, p = nothing)
    return exp(x[1]) + x[1]^2
end

optf = OptimizationFunction(f, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, [0.4])

sol = solve(prob, Optimization.LBFGS(), maxiters = 1000)