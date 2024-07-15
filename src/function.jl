
function symbolify(e::Expr)
    if !(e.args[1] isa Symbol)
        e.args[1] = Symbol(e.args[1])
    end
    symbolify.(e.args)
    return e
end

function symbolify(e)
    return e
end

function rep_pars_vals!(e::Expr, p)
    rep_pars_vals!.(e.args, Ref(p))
    replace!(e.args, p...)
end

function rep_pars_vals!(e, p) end

"""
    instantiate_function(f, x, ::AbstractADType, p, num_cons = 0)::OptimizationFunction

This function is used internally by Optimization.jl to construct
the necessary extra functions (gradients, Hessians, etc.) before
optimization. Each of the ADType dispatches use the supplied automatic
differentiation type in order to specify how the construction process
occurs.

If no ADType is given, then the default `NoAD` dispatch simply
defines closures on any supplied gradient function to enclose the
parameters to match the interfaces for the specific optimization
libraries (i.e. (G,x)->f.grad(G,x,p)). If a function is not given
and the `NoAD` dispatch is used, or if the AD dispatch is currently
not capable of defining said derivative, then the constructed
`OptimizationFunction` will simply use `nothing` to specify and undefined
function.

The return of `instantiate_function` is an `OptimizationFunction` which
is then used in the optimization process. If an optimizer requires a
function that is not defined, an error is thrown.

For more information on the use of automatic differentiation, see the
documentation of the `AbstractADType` types.
"""

function instantiate_function(f::OptimizationFunction, x, adtype::ADTypes.AbstractADType,
        p, num_cons = 0)
    adtypestr = string(adtype)
    _strtind = findfirst('.', adtypestr)
    strtind = isnothing(_strtind) ? 5 : _strtind + 5
    open_nrmlbrkt_ind = findfirst('(', adtypestr)
    open_squigllybrkt_ind = findfirst('{', adtypestr)
    open_brkt_ind = isnothing(open_squigllybrkt_ind) ? open_nrmlbrkt_ind :
                    min(open_nrmlbrkt_ind, open_squigllybrkt_ind)
    adpkg = adtypestr[strtind:(open_brkt_ind - 1)]
    throw(ArgumentError("The passed automatic differentiation backend choice is not available. Please load the corresponding AD package $adpkg."))
end
