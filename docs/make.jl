using Documenter
using LeniaDynamics

makedocs(
    sitename = "LeniaDynamics.jl",
    modules = [LeniaDynamics],
    pages = [
        "Home"        => "index.md",
        "Tutorial"    => "tutorial.md",
        "Performance" => "performance.md",
        "API"         => "api.md",
    ],
)

# If you host this package on GitHub and want a docs website via GitHub Pages,
# uncomment and set `repo` to your repository URL.
# deploydocs(repo = "github.com/USER/REPO.git")
