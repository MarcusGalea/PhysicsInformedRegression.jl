using Pkg
Pkg.activate("docs/")
using Documenter, PhysicsInformedRegression
makedocs(
    sitename = "PhysicsInformedRegression.jl",
    modules = [PhysicsInformedRegression],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)