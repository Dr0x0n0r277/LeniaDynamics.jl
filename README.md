# LeniaDynamics.jl

<<<<<<< HEAD
Julia balíček pro simulaci **Lenia** (kontinuální varianta “Game of Life”) se zaměřením na:
- **výkon** (FFT konvoluce vs. naivní konvoluce),
- **reprodukovatelné experimenty** (oddělené `scripts/` prostředí s commitnutým `Manifest.toml`),
- **interaktivní vizualizaci** přes **GLMakie** (GUI ve `scripts/run_gui.jl`),
- volitelně **CUDA** akceleraci (package extension).

---

## Co umí kód (high-level)

### Core simulátor (balíček)
- **2 backendy konvoluce**
  - `:fft` — rychlá periodická konvoluce přes FFT (doporučeno pro větší N).
  - `:naive` — přímé sčítání (užitečné pro validaci / malé radius), s **Threads**.
- **Integrátory** (výběr v `step!`/`run!`)
  - `Euler()`, `RK2()`, `RK4()` (viz `src/integrators.jl`).
- **Parametrizace Lenia**
  - `KernelSpec` (radius, ringy, váhy…)
  - `LeniaParams` (kernel + growth funkce + `μ`, `σ`, `dt`, …)
- **Inicializace a běh**
  - `initialize_state(N; init=:spot|..., seed=..., backend=:fft|:naive)`
  - `step!(st, p; integrator=...)`, `run!(st, p, steps; integrator=...)`
- **Volitelná CUDA akcelerace**
  - detekce přes `has_cuda()`
  - převod stavu na GPU: `to_device(st, :cuda)` (typicky `Float32`)

### GUI (GLMakie, skript)
`./scripts/run_gui.jl` nabízí interaktivní UI (aktuální rozsah funkcí je popsaný i přímo v hlavičce skriptu), typicky:
- živý náhled simulace,
- ovládání parametrů a integrátoru,
- “Advanced” okno: View (colormap/gamma/auto-contrast), History graf,
- I/O: snapshoty / recording (frames do `scripts/recordings/`),
- Config: export / load parametrů a scénářů (TOML),
- paint do pole A (LMB přidává, RMB maže), hover readout.

---

## Struktura repozitáře

```
.
├── src/                 # source code
├── test/                # testy pro Pkg.test()
├── scripts/             # spustitelné ukázky/bench/GUI 
│   ├── Project.toml
│   ├── Manifest.toml   
│   ├── run_example.jl
│   ├── bench.jl
│   └── run_gui.jl
├── docs/                # Documenter.jl
└── README.md
```

---

## Instalace

### Varianta A: instalace z Git URL (pro uživatele)
V Julia REPL:

```julia
import Pkg
Pkg.add(url="https://github.com/Dr0x0n0r277/LeniaDynamics.jl.git")
```

Pak:

```julia
using LeniaDynamics
```

> Nahraď `USER/REPO` reálným repozitářem.

### Varianta B: lokální vývoj (`Pkg.develop`)
Pokud máš repo naklonované lokálně:

```julia
import Pkg
Pkg.develop(path="C:/cesta/k/LeniaDynamics.jl")
```

---

## Rychlý start (example)

### PowerShell (Windows)
Z kořene repa:

```powershell
# 1) připrav scripts prostředí (poprvé / po pullu)
julia --project=scripts -e "import Pkg; Pkg.instantiate(); Pkg.precompile()"

# 2) spusť ukázku
julia --project=scripts scripts/run_example.jl
```

### Julia REPL
Spusť Julii v kořeni repa a:

```julia
import Pkg
Pkg.activate("scripts")
Pkg.instantiate()
Pkg.precompile()

include("scripts/run_example.jl")
```

---

## Benchmarky (výkon)

Benchmark script porovnává backendy a používá `BenchmarkTools`.

### PowerShell (Windows)
```powershell
julia --project=scripts -e "import Pkg; Pkg.instantiate(); Pkg.precompile()"
julia --project=scripts scripts/bench.jl
```

#### Parametry přes env proměnné (PowerShell)
`bench.jl` podporuje např. `LENIA_N` a `LENIA_STEPS`:

```powershell
$env:LENIA_N = "512"
$env:LENIA_STEPS = "2"
julia --project=scripts scripts/bench.jl
```

Zrušení proměnné:
```powershell
Remove-Item Env:LENIA_N
Remove-Item Env:LENIA_STEPS
```

### Julia REPL
```julia
import Pkg
Pkg.activate("scripts")
Pkg.instantiate()
Pkg.precompile()

include("scripts/bench.jl")
```

#### Častý problém: `FFTW not found`
`bench.jl` dělá `using FFTW`, takže `FFTW` musí být v `scripts/` env:

```julia
import Pkg
Pkg.activate("scripts")
Pkg.add("FFTW")
Pkg.instantiate()
```

---

## GUI (GLMakie)

GUI je záměrně ve **scripts** prostředí (Makie stack je “těžký” a nechceme ho v core balíčku).

### PowerShell (Windows)
```powershell
# 1) připrav scripts env
julia --project=scripts -e "import Pkg; Pkg.instantiate(); Pkg.precompile()"

# 2) spusť GUI
julia --project=scripts scripts/run_gui.jl
```

### Julia REPL
```julia
import Pkg
Pkg.activate("scripts")
Pkg.instantiate()
Pkg.precompile()

include("scripts/run_gui.jl")
```

#### Pokud spadne na chybějící GLMakie
Doinstaluj ho do scripts env:

```julia
import Pkg
Pkg.activate("scripts")
Pkg.add("GLMakie")
Pkg.instantiate()
```

---

## Nastavení vláken (Threads) — doporučeno pro výkon

### PowerShell (Windows)
```powershell
$env:JULIA_NUM_THREADS = "8"
julia --project=scripts scripts/bench.jl
```

Nebo použij přímo přepínač:
```powershell
julia -t auto --project=scripts scripts/bench.jl
```

### Julia REPL
V běžícím REPL počet threadů nezměníš; musíš spustit Julii s `-t` nebo `JULIA_NUM_THREADS`.

---

## CUDA (volitelné)

Balíček obsahuje extension `ext/LeniaDynamicsCUDAExt.jl`. Pokud máš GPU a chceš to zapnout:

### PowerShell / terminál
```powershell
julia --project=scripts -e "import Pkg; Pkg.add(\"CUDA\"); Pkg.instantiate()"
```

### Julia REPL (scripts env)
```julia
import Pkg
Pkg.activate("scripts")
Pkg.add("CUDA")
Pkg.instantiate()
```

Pak v kódu:
```julia
using LeniaDynamics
has_cuda() && @info "CUDA available"

st = initialize_state(256; init=:spot, seed=42, backend=:fft)
st_gpu = to_device(st, :cuda)   # typicky Float32
```

> Pozn.: aktuální CUDA cesta je zaměřená primárně na `Float32` (viz extension).

---

## Minimal API příklad (core balíček)
=======
A reusable, performance-oriented **Lenia (continuous Game of Life)** simulator in **Julia**, focused on:

- fast evolution via **FFT-based convolution** (CPU / FFTW),
- a threaded **naive** backend for validation and tiny kernels,
- optional **CUDA** acceleration (GPU) if `CUDA.jl` is installed,
- reproducible scripts, benchmarks, tests, and basic docs.

## Quick start (package API)
>>>>>>> aba54a83b4fd1c07653f884c4d06ffe32d094890

```julia
using LeniaDynamics

<<<<<<< HEAD
spec = KernelSpec(radius=13,
                  rings=Float32[0.45, 0.75],
                  ring_widths=Float32[0.15, 0.12],
                  ring_weights=Float32[1.0, 0.7])
=======
spec = KernelSpec(
    radius=13,
    rings=Float32[0.45, 0.75],
    ring_widths=Float32[0.15, 0.12],
    ring_weights=Float32[1.0, 0.7],
)
>>>>>>> aba54a83b4fd1c07653f884c4d06ffe32d094890

p = LeniaParams(kernel=spec, growth=gaussian_growth, μ=0.15f0, σ=0.015f0, dt=0.10f0)

st = initialize_state(256; init=:spot, seed=42, backend=:fft)
<<<<<<< HEAD
run!(st, p, 50; integrator=RK2())
```

---

## Testy

### PowerShell
```powershell
julia --project=. -e "import Pkg; Pkg.instantiate(); Pkg.test()"
```

### Julia REPL
```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
```

---

## Dokumentace (Documenter.jl)

### PowerShell
```powershell
julia --project=docs -e "import Pkg; Pkg.instantiate()"
julia --project=docs docs/make.jl
```

### Julia REPL
```julia
import Pkg
Pkg.activate("docs")
Pkg.instantiate()

include("docs/make.jl")
```

---

## Troubleshooting

### 1) GLMakie / OpenGL problémy
- Aktualizuj GPU ovladače.
- Na některých strojích může pomoci přepnout renderer / použít WGL/ANGLE (Makie/GPU stack je citlivý na konfiguraci).

### 2) Reprodukovatelnost `scripts/`
Ujisti se, že `scripts/Manifest.toml` neobsahuje absolutní Windows cestu pro `LeniaDynamics`.
Správně má být u balíčku v manifestu:

```toml
path = ".."
```

Pokud je tam absolutní cesta, smaž `scripts/Manifest.toml` a regeneruj ho přes `Pkg.develop(path="..")`.

---

## License
Viz `LICENSE`.
=======
run!(st, p, 200; integrator=RK2())
# result is in st.A
```

## Installation / running tests

From the package root:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.test()"
```

For local development (REPL):

```julia
using Pkg
Pkg.develop(path="/path/to/LeniaDynamics")
Pkg.instantiate()
```

If you host the repository online, you can install via URL:

```julia
using Pkg
Pkg.add(url="https://github.com/USER/REPO.git")
```

## Scripts (reproducible environment)

The `scripts/` folder contains a separate Julia environment (committed `Project.toml` + `Manifest.toml`).

- Example run + microbenchmarks:

```bash
julia --project=scripts scripts/run_example.jl
```

- Backend benchmark sweep:

```bash
julia --project=scripts scripts/bench.jl
```

- GUI (GLMakie):

```bash
julia --project=scripts scripts/run_gui.jl
```

## CUDA (optional)

CUDA support is provided via a package extension. To enable it, add CUDA to **your environment**:

```julia
import Pkg
Pkg.add("CUDA")
```

Then use `backend=:cuda` (or `to_device(st, :cuda)`).

## Project structure

- `src/` — package code
- `test/` — runnable tests (`Pkg.test()`)
- `docs/` — Documenter.jl site (`julia --project=docs docs/make.jl`)
- `scripts/` — reproducible examples, GUI, benchmarks
- `.github/workflows/ci.yml` — GitHub Actions CI

## License

MIT
>>>>>>> aba54a83b4fd1c07653f884c4d06ffe32d094890
