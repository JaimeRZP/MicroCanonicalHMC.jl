# make.jl
using Documenter, GaussianProcess

makedocs(sitename = "MCHMC.jl",
         modules = [GaussianProcess],
         pages = ["Home" => "index.md",
                  "API" => "api.md"])
deploydocs(
    repo = "github.com/JaimeRZP/MCHCM.jl"
)