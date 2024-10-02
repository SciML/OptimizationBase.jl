function symify_cache(f::OptimizationFunction, prob)
    obj_expr = f.expr
    cons_expr = f.cons_expr === nothing ? nothing : getfield.(f.cons_expr, Ref(:lhs))

    return obj_expr, cons_expr
end
