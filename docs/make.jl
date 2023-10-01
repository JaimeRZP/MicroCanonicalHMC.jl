# make.jl
using Documenter, MicroCanonicalHMC

makedocs(sitename = "MicroCanonicalHMC.jl",
         modules = [MicroCanonicalHMC],
         pages = ["Home" => "index.md",
                  "API" => "api.md"])
         
deploydocs(repo = "github.com/JaimeRZP/MicroCanonicalHMC.jl")