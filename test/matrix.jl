using OptimizationBase, Optimization, Test, LinearAlgebra, OptimizationOptimJL, OptimizationOptimJL.Optim, Zygote

# 1. Matrix Factorization
function matrix_factorization_objective(x, A)
    U, V = x
    return norm(A - U * V')
end

A_mf = rand(4, 4)  # Original matrix
U_mf = rand(4, 2)  # Factor matrix U
V_mf = rand(4, 2)  # Factor matrix V

optf = OptimizationFunction{false}(matrix_factorization_objective, AutoZygote())
prob = OptimizationProblem(optf, [U_mf, V_mf], A_mf)
optf = OptimizationBase.instantiate_function(optf, [U_mf, V_mf], optf.adtype, A_mf, g = true)
G = optf.grad([U_mf, V_mf])

# 2. Principal Component Analysis (PCA)
function pca_objective(X, A)
    return -tr(X' * A * X)  # Minimize the negative of the trace for maximization
end

function orthogonality_constraint(X, A)
    return norm(X' * X - I)
end

A_pca = rand(4, 4)  # Covariance matrix (can be symmetric positive definite)
A_pca = A_pca * A_pca'
X_pca = rand(4, 2)  # Matrix to hold principal components
optf = OptimizationFunction{false}(pca_objective, AutoZygote(), cons = orthogonality_constraint)
optf = OptimizationBase.instantiate_function(optf, X_pca, optf.adtype, A_pca, g = true, cons_j = true)
G = optf.grad(X_pca)
