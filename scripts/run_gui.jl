# scripts/run_gui.jl
import Pkg

# -----------------------------------------------------------------------------
# LeniaDynamics GUI (GLMakie)
#
# Umístění:
#   Tento soubor patří do:  LeniaDynamics_relXXX/scripts/run_gui.jl
# Spuštění:
#   julia --project=./scripts ./scripts/run_gui.jl
# nebo ve VSCode REPL:
#   include("scripts/run_gui.jl")
#
# Funkce:
# - Help (samostatné okno) + tlačítko Zpět (nemění velikost hlavního okna)
# - Advance (samostatné okno): View (colormap+gamma+auto-contrast), velký History graf,
#   I/O (snapshot/record), Config (export/load params/scenario)
# - View colormap přepínání funguje i na Makie verzích, kde menu.selection vrací String
# - Malování do pole A (Paint): toggle v Advanced. LMB přidává, RMB maže.
# - Hover readout: ukazuje A[y,x]
# - Bez FPS počítání
# -----------------------------------------------------------------------------

Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))
Pkg.instantiate()

using GLMakie
using Statistics
using Dates
using TOML
using Printf
using LeniaDynamics

# top-level constants (musí být mimo funkci)
const ADV_HIST_LEN = 600

# -----------------------------
# Helpers
# -----------------------------
@inline function integrator_from_symbol(sym::Symbol)
    sym === :Euler && return Euler()
    sym === :RK2   && return RK2()
    sym === :RK4   && return RK4()
    return RK2()
end

@inline function growth_from_symbol(sym::Symbol)
    sym === :bump && return bump_growth
    return gaussian_growth
end

# Menu helper: labels + values (kompatibilní napříč Makie verzemi)
function menu_labeled(fig; labels::Vector{String}, values::Vector)
    @assert length(labels) == length(values)
    m = Menu(fig; options = labels)
    return m, values
end

function _menu_options(menu::Menu)
    # v některých verzích je to Observable, jinde plain Vector
    opts = getproperty(menu, :options)
    return opts isa Observable ? opts[] : opts
end

@inline function selected_value(menu::Menu, values)
    sel = menu.selection[]
    if sel isa Integer
        return values[sel]
    elseif sel isa AbstractString
        opts = _menu_options(menu)
        idx = findfirst(==(sel), opts)
        idx === nothing && error("Menu selection not found in options: $(sel)")
        return values[idx]
    else
        error("Unsupported Menu selection type: $(typeof(sel))")
    end
end

function set_menu_to_value!(menu::Menu, values, target)
    idx = findfirst(==(target), values)
    idx === nothing && return nothing

    sel = menu.selection[]
    if sel isa Integer
        menu.selection[] = idx
    else
        opts = _menu_options(menu)
        menu.selection[] = opts[idx]  # nastav label
    end
    return nothing
end

@inline clamp01(x::Float32) = x < 0f0 ? 0f0 : (x > 1f0 ? 1f0 : x)

# -----------------------------
# Help text
# -----------------------------
const HELP_TEXT = """
Nápověda k parametrům (LeniaDynamics)

dt
  Časový krok integrace. Menší dt = stabilnější, ale pomalejší.

μ
  Střed růstové funkce (cílová hodnota).

σ
  Šířka růstové funkce. Menší σ = „ostřejší“ pravidla.

Growth
  gaussian – hladká gaussovská růstová funkce
  bump     – kompaktnější bump růst

Integrator
  Euler / RK2 / RK4 – numerická integrace (RK4 bývá stabilnější, ale pomalejší).

Kernel
  radius – poloměr kernelu (dosah okolí)
  rings  – počet prstenců (1–3)
  r1..r3 – pozice prstenců (0..1)
  w1..w3 – šířky prstenců
  β1..β3 – váhy prstenců

Init
  noise / spot / sprinkle / primordia – způsob inicializace

seed
  Semínko RNG pro opakovatelnost.

noise amp
  Amplituda šumu pro init/noise.

Paint (Advanced)
  Zapnutí malování do pole A:
    LMB = přidává buňky
    RMB = maže buňky
  Brush r = poloměr štětce v pixelech mřížky
  Brush s = intenzita přidání/ubrání (0..1)
"""

# -----------------------------
# GUI runner
# -----------------------------
function run_gui(; N::Int=256)
    GLMakie.activate!()
    script_dir = @__DIR__

    # Simulation objects
    st_ref    = Ref{LeniaState{Float32, Matrix{Float32}}}()
    p_ref     = Ref{LeniaParams}()
    integ_ref = Ref{Integrator}()

    # keep screens alive (Help/Advanced)
    help_screen_ref = Ref{Any}(nothing)
    adv_screen_ref  = Ref{Any}(nothing)

    # guard for applying scenario (avoid feedback loops)
    ui_suspend = Ref(false)

    # -----------------------------
    # Observables
    # -----------------------------
    running = Observable(false)

    # Advanced state
    steps_per_tick    = Observable(1)
    view_autocontrast = Observable(false)
    view_gamma        = Observable(1.0)
    colormap_sym      = Observable(:viridis)

    # Paint tool (toggle in Advanced)
    paint_enabled  = Observable(false)     # default OFF
    painting       = Observable(false)     # držím tlačítko myši
    paint_is_erase = Ref(false)            # RMB = erase
    brush_radius   = Observable(6)         # px
    brush_strength = Observable(0.25f0)    # 0..1

    # Percentiles pro autocontrast
    p05_obs = Observable(0.0f0)
    p95_obs = Observable(1.0f0)

    # Recording
    recording        = Observable(false)
    record_frames    = Observable(120)
    record_remaining = Ref(0)
    record_dir       = Ref{String}("")
    record_idx       = Ref(0)

    # History (mean/max)
    mean_x_obs = Observable(Float32[])
    mean_y_obs = Observable(Float32[])
    max_x_obs  = Observable(Float32[])
    max_y_obs  = Observable(Float32[])
    hist_t    = Float32[]
    hist_mean = Float32[]
    hist_max  = Float32[]
    t0 = time()

    # UI state
    preset_sym = Observable(:default)
    backend    = Observable(:fft)
    init_sym   = Observable(:spot)
    seed       = Observable(1)

    dt    = Observable(0.10f0)
    mu    = Observable(0.15f0)
    sigma = Observable(0.015f0)

    growth_kind     = Observable(:gaussian)
    integrator_kind = Observable(:RK2)

    radius     = Observable(13)
    ring_count = Observable(2)

    r1 = Observable(0.45f0); w1 = Observable(0.15f0); b1 = Observable(1.0f0)
    r2 = Observable(0.75f0); w2 = Observable(0.12f0); b2 = Observable(0.7f0)
    r3 = Observable(0.85f0); w3 = Observable(0.10f0); b3 = Observable(0.0f0)

    noise_amp = Observable(0.05f0)

    # Dirty flags
    params_dirty = Observable(true)
    integ_dirty  = Observable(true)

    onany(dt, mu, sigma, growth_kind, radius, ring_count,
          r1, w1, b1, r2, w2, b2, r3, w3, b3) do args...
        params_dirty[] = true
    end
    on(integrator_kind) do _
        integ_dirty[] = true
    end

    # -----------------------------
    # Kernel + rebuild
    # -----------------------------
    function current_kernel()
        cnt = ring_count[]
        if cnt == 1
            rings       = Float32[r1[]]
            widths      = Float32[w1[]]
            ringweights = Float32[b1[]]
        elseif cnt == 2
            rings       = Float32[r1[], r2[]]
            widths      = Float32[w1[], w2[]]
            ringweights = Float32[b1[], b2[]]
        else
            rings       = Float32[r1[], r2[], r3[]]
            widths      = Float32[w1[], w2[], w3[]]
            ringweights = Float32[b1[], b2[], b3[]]
        end
        return KernelSpec(radius = radius[], rings = rings, ring_widths = widths, ring_weights = ringweights)
    end

    function rebuild_params_if_needed!()
        if params_dirty[]
            p_ref[] = LeniaParams(
                kernel   = current_kernel(),
                growth   = growth_from_symbol(growth_kind[]),
                μ        = mu[],
                σ        = sigma[],
                dt       = dt[],
                feedback = nothing
            )
            params_dirty[] = false
            empty!(st_ref[].cache)
        end
        return nothing
    end

    function rebuild_integrator_if_needed!()
        if integ_dirty[]
            integ_ref[] = integrator_from_symbol(integrator_kind[])
            integ_dirty[] = false
        end
        return nothing
    end

    function reset_state!()
        running[] = false
        st = initialize_state(
            N;
            init      = init_sym[],
            seed      = seed[],
            backend   = backend[],
            noise_amp = noise_amp[]
        )
        st_ref[] = st
        params_dirty[] = true
        integ_dirty[]  = true
        rebuild_params_if_needed!()
        rebuild_integrator_if_needed!()
        return st
    end

    function apply_preset!(name::Symbol)
        running[] = false
        st, p, integ = make_preset(name; N = N, seed = seed[], backend = backend[])
        st_ref[] = st
        p_ref[] = p
        integ_ref[] = integ

        # sync GUI from preset
        dt[]    = p.dt
        mu[]    = p.μ
        sigma[] = p.σ
        growth_kind[] = (p.growth === bump_growth) ? :bump : :gaussian
        integrator_kind[] = integ isa Euler ? :Euler : (integ isa RK4 ? :RK4 : :RK2)

        radius[]     = p.kernel.radius
        ring_count[] = length(p.kernel.rings)

        r1[] = p.kernel.rings[1]; w1[] = p.kernel.ring_widths[1]; b1[] = p.kernel.ring_weights[1]
        if ring_count[] >= 2
            r2[] = p.kernel.rings[2]; w2[] = p.kernel.ring_widths[2]; b2[] = p.kernel.ring_weights[2]
        else
            r2[] = 0.75f0; w2[] = 0.12f0; b2[] = 0f0
        end
        if ring_count[] >= 3
            r3[] = p.kernel.rings[3]; w3[] = p.kernel.ring_widths[3]; b3[] = p.kernel.ring_weights[3]
        else
            r3[] = 0.85f0; w3[] = 0.10f0; b3[] = 0f0
        end

        params_dirty[] = false
        integ_dirty[]  = false
        empty!(st.cache)
        return st
    end

    # initial state
    apply_preset!(preset_sym[])

    # --------------------------------
    # Main figure / Layout
    # --------------------------------
    fig = Figure(size = (1600, 820), fontsize = 14)

    ax = Axis(fig[1, 1], title = "LeniaDynamics GUI", aspect = DataAspect())
    hidespines!(ax); hidedecorations!(ax)

    # Když je Paint zapnutý, vypni zoom/pan/rectanglezoom (a po vypnutí je vrať)
    function set_axis_navigation!(ax::Axis, enabled::Bool)
        for name in (:scrollzoom, :dragpan, :rectanglezoom)
            try
                enabled ? Makie.activate_interaction!(ax, name) :
                          Makie.deactivate_interaction!(ax, name)
            catch
            end
        end
        return nothing
    end
    set_axis_navigation!(ax, true)
    on(paint_enabled) do onoff
        set_axis_navigation!(ax, !onoff)
    end

    # view buffer
    Aview_ref = Ref(similar(st_ref[].A))
    Aview_obs = Observable(Aview_ref[])

    function ensure_view_buffer!()
        A = st_ref[].A
        if size(Aview_ref[]) != size(A)
            Aview_ref[] = similar(A)
            Aview_obs[] = Aview_ref[]
        end
        return nothing
    end

    function update_percentiles!(A::AbstractMatrix{<:Real})
        v = vec(Float32.(A))
        p05_obs[] = Float32(quantile(v, 0.05))
        p95_obs[] = Float32(quantile(v, 0.95))
        return nothing
    end

    function update_view!()
        ensure_view_buffer!()
        A = st_ref[].A
        V = Aview_ref[]

        g = Float64(view_gamma[])
        invg = (g <= 0.0) ? 1.0 : (1.0 / g)

        if view_autocontrast[]
            lo = Float32(p05_obs[])
            hi = Float32(p95_obs[])
            if !(isfinite(lo) && isfinite(hi)) || hi <= lo + 1f-6
                lo = 0f0; hi = 1f0
            end
            inv = 1f0 / (hi - lo)

            @inbounds for j in axes(A, 2), i in axes(A, 1)
                x = (A[i, j] - lo) * inv
                x = x < 0f0 ? 0f0 : (x > 1f0 ? 1f0 : x)
                if g != 1.0
                    x = Float32((Float64(x))^invg)
                end
                V[i, j] = x
            end
        else
            @inbounds for j in axes(A, 2), i in axes(A, 1)
                x = A[i, j]
                x = x < 0f0 ? 0f0 : (x > 1f0 ? 1f0 : x)
                if g != 1.0
                    x = Float32((Float64(x))^invg)
                end
                V[i, j] = x
            end
        end

        notify(Aview_obs)
        return nothing
    end

    # Image plot with initial colormap value
    H0 = size(st_ref[].A, 1)
    W0 = size(st_ref[].A, 2)
    imgplt = image!(ax, (1, W0), (1, H0), Aview_obs; interpolate=false, colormap=colormap_sym[])

    # Ensure colormap changes actually apply (Makie-version safe)
    on(colormap_sym) do cm
        try
            imgplt.colormap[] = cm
        catch
            try
                imgplt.attributes[:colormap][] = cm
            catch
            end
        end
    end

    # Right controls
    ctrl = fig[1, 2] = GridLayout(tellheight = false)
    rowgap!(ctrl, 10)
    colgap!(ctrl, 12)

    # Fix column width after second column exists
    colsize!(fig.layout, 2, Fixed(520))
    colgap!(fig.layout, 20)

    btnw = 150
    btnh = 36

    start_btn = Button(fig, label="Start/Stop",         width=btnw, height=btnh)
    step_btn  = Button(fig, label="Step",          width=btnw, height=btnh)
    reset_btn = Button(fig, label="Reset",         width=btnw, height=btnh)

    clear_btn = Button(fig, label="Clear",         width=btnw, height=btnh)
    calib_btn = Button(fig, label="AutoCalibrate", width=btnw, height=btnh)
    help_btn  = Button(fig, label="Help",          width=btnw, height=btnh)

    ctrl[1, 1] = start_btn
    ctrl[1, 2] = step_btn
    ctrl[1, 3] = reset_btn

    ctrl[2, 1] = clear_btn
    ctrl[2, 2] = calib_btn
    ctrl[2, 3] = help_btn

    stat_label  = Label(fig, "mean(A): -   max(A): -")
    hover_label = Label(fig, "hover: -")
    ctrl[3, 1:3] = GridLayout()
    ctrl[3, 1:3][1, 1] = stat_label
    ctrl[3, 1:3][2, 1] = hover_label

    # Menus
    preset_labels = ["Preset: default", "Preset: primordia", "Preset: sustain", "Preset: noisy"]
    preset_values = Symbol[:default, :primordia, :sustain, :noisy]
    preset_menu, preset_vals = menu_labeled(fig; labels = preset_labels, values = preset_values)

    backend_labels = ["Backend: FFT (fast)", "Backend: naive (slow)"]
    backend_values = Symbol[:fft, :naive]
    backend_menu, backend_vals = menu_labeled(fig; labels = backend_labels, values = backend_values)

    init_labels = ["Init: noise", "Init: spot", "Init: sprinkle", "Init: primordia"]
    init_values = Symbol[:noise, :spot, :sprinkle, :primordia]
    init_menu, init_vals = menu_labeled(fig; labels = init_labels, values = init_values)

    growth_labels = ["Growth: gaussian", "Growth: bump"]
    growth_values = Symbol[:gaussian, :bump]
    growth_menu, growth_vals = menu_labeled(fig; labels = growth_labels, values = growth_values)

    integ_labels = ["Integrator: Euler", "Integrator: RK2", "Integrator: RK4"]
    integ_values = Symbol[:Euler, :RK2, :RK4]
    integ_menu, integ_vals = menu_labeled(fig; labels = integ_labels, values = integ_values)

    ctrl[4, 1] = preset_menu
    ctrl[4, 2] = backend_menu
    ctrl[4, 3] = init_menu
    ctrl[5, 1] = growth_menu
    ctrl[5, 2] = integ_menu

    advance_btn = Button(fig, label="Advance", width=btnw, height=btnh)
    ctrl[5, 3] = advance_btn

    # Sliders
    sg = SliderGrid(fig,
        (label = "dt",        range = 0.005:0.001:0.25, startvalue = Float64(dt[])),
        (label = "μ",         range = 0.0:0.001:1.0,    startvalue = Float64(mu[])),
        (label = "σ",         range = 0.001:0.001:0.25, startvalue = Float64(sigma[])),
        (label = "radius",    range = 3:1:80,           startvalue = radius[]),
        (label = "rings",     range = 1:1:3,            startvalue = ring_count[]),
        (label = "r1",        range = 0.0:0.005:1.0,    startvalue = Float64(r1[])),
        (label = "w1",        range = 0.01:0.005:0.50,  startvalue = Float64(w1[])),
        (label = "β1",        range = 0.0:0.01:2.0,     startvalue = Float64(b1[])),
        (label = "r2",        range = 0.0:0.005:1.0,    startvalue = Float64(r2[])),
        (label = "w2",        range = 0.01:0.005:0.50,  startvalue = Float64(w2[])),
        (label = "β2",        range = 0.0:0.01:2.0,     startvalue = Float64(b2[])),
        (label = "r3",        range = 0.0:0.005:1.0,    startvalue = Float64(r3[])),
        (label = "w3",        range = 0.01:0.005:0.50,  startvalue = Float64(w3[])),
        (label = "β3",        range = 0.0:0.01:2.0,     startvalue = Float64(b3[])),
        (label = "seed",      range = 1:1:999,          startvalue = seed[]),
        (label = "noise amp", range = 0.0:0.005:0.30,   startvalue = Float64(noise_amp[]))
    )
    ctrl[6, 1:3] = sg

    # -----------------------------
    # Paint impl
    # -----------------------------
    function apply_brush!(A::AbstractMatrix{Float32}, cx::Int, cy::Int; erase::Bool, r::Int, strength::Float32)
        H, W = size(A, 1), size(A, 2)
        r2 = r*r
        sgn = erase ? -1f0 : 1f0
        @inbounds for dy in -r:r
            yy = cy + dy
            (1 <= yy <= H) || continue
            for dx in -r:r
                xx = cx + dx
                (1 <= xx <= W) || continue
                (dx*dx + dy*dy <= r2) || continue
                A[yy, xx] = clamp01(A[yy, xx] + sgn * strength)
            end
        end
        return nothing
    end

    function cursor_index()
        p = mouseposition(ax)  # data coords v ose
        x = Float64(p[1]); y = Float64(p[2])
        if !(isfinite(x) && isfinite(y))
            return nothing
        end

        # KLÍČ: swap x<->y (tím se zruší zrcadlení přes y=x)
        x, y = y, x
        A = st_ref[].A
        H, W = size(A, 1), size(A, 2)

        ix = clamp(Int(round(x)), 1, W)  # col
        iy = clamp(Int(round(y)), 1, H)  # row

        return CartesianIndex(iy, ix)
    end

    function paint_at_cursor!(erase::Bool)
        idx = cursor_index()
        idx === nothing && return

        iy, ix = Tuple(idx)  # (row, col)
        A = st_ref[].A

        r = max(1, Int(brush_radius[]))
        s = Float32(brush_strength[])

        apply_brush!(A, ix, iy; erase = erase, r = r, strength = s)  # cx=col, cy=row
        empty!(st_ref[].cache)
        return nothing
    end

    # -----------------------------
    # Help window
    # -----------------------------
    function open_help!()
        # zavři předchozí okno nápovědy, pokud existuje
        if help_screen_ref[] !== nothing
            try close(help_screen_ref[]) catch end
            help_screen_ref[] = nothing
        end

    # -----------------------------
    # Text nápovědy ve 2 sloupcích
    # -----------------------------
    help_left = """
    HLAVNÍ OKNO (Main GUI)

    Ovládání simulace
    • Start/Stop  – přepíná běh simulace (running)
    • Step        – provede 1 integrační krok
    • Reset       – znovu inicializuje stav podle Init/seed/noise amp
    • Clear       – vynuluje pole A (A .= 0)
    • AutoCalibrate – zavolá LeniaDynamics.auto_calibrate!(; target = μ)
    • Help        – otevře tuto nápovědu
    • Advance     – otevře pokročilé funkce (samostatné okno)

    Menu (výběry)
    • Preset      – přednastavení (default/primordia/sustain/noisy)
    • Backend     – :fft (rychlé) / :naive (pomalejší) – projeví se po Reset / Preset
    • Init        – inicializace (noise/spot/sprinkle/primordia)
    • Growth      – gaussian / bump
    • Integrator  – Euler / RK2 / RK4

    Slidery (parametry)
    • dt          – časový krok integrace
    • μ           – střed růstové funkce
    • σ           – šířka růstové funkce
    • radius      – poloměr kernelu
    • rings       – počet prstenců (1–3)
    • r1..r3      – pozice prstenců (0..1)
    • w1..w3      – šířky prstenců
    • β1..β3      – váhy prstenců
    • seed        – RNG seed
    • noise amp   – amplituda šumu pro init/noise

    Info vpravo
    • mean(A), max(A) – základní statistiky pole A
    • hover: x,y,A     – hodnota A[y,x] pod kurzorem

    Myš nad polem
    • Hover jen čte hodnotu
    • Paint (zapíná se v Advanced):
        LMB přidává, RMB maže
        tažením myši kreslí průběžně
        když je Paint ON, vypíná se zoom/pan (aby kreslení fungovalo spolehlivě)

    Spuštění
    • julia --project=./scripts ./scripts/run_gui.jl
    • nebo ve VSCode: include("scripts/run_gui.jl")
    """

        help_right = """
    ADVANCED OKNO

    View (render)
    • Colormap      – volba mapy barev (viridis, magma, plasma, inferno, …)
    • gamma         – gamma korekce zobrazení (jen render)
    • Auto-contrast – normalizace podle p05..p95 (percentily pole A)
    • p05..p95      – zobrazené percentily (diagnostika)
    • steps/tick    – kolik sim kroků proběhne za jeden render tick
    • Paint         – zap/vyp malování do pole A
    • brush r       – poloměr štětce (pixely mřížky)
    • brush s       – síla zásahu (0..1)

    History (mean/max)
    • Graf historie mean(A) a max(A) v čase (posledních ~600 vzorků)

    I/O
    • Snapshot – uloží PNG snímek hlavního okna
            do scripts/snapshots/snapshot_YYYYMMDD_HHMMSS.png
    • Record   – spustí ukládání PNG snímků během běhu (running = true)
            do scripts/recordings/rec_YYYYMMDD_HHMMSS/frame_00001.png …
    • Stop     – zastaví nahrávání
    • frames   – počet snímků pro Record

    Config
    • Export params     – uloží parametry sliderů do scripts/advanced_params.toml
    • Load params       – načte advanced_params.toml a aplikuje do GUI
    • Export scenario   – uloží “stav GUI” (params + UI + view) do advanced_scenario.toml
    • Load scenario (apply) – načte a aplikuje bez resetu stavu
    • Load scenario (reset) – načte a potom resetuje stav

    Poznámky
    • Změny v render (colormap/gamma/autocontrast) neovlivňují simulaci, jen zobrazení.
    • Když změníš kernel/growth/integrator, projeví se při dalším kroku simulace.
    """

        # -----------------------------
        # Okno Help
        # -----------------------------
        help_fig = Figure(size = (1100, 900), fontsize = 14)
        gl = help_fig[1, 1] = GridLayout()
        rowgap!(gl, 12)
        colgap!(gl, 12)

        header = gl[1, 1] = GridLayout()
        colgap!(header, 10)

        title = Label(help_fig, "Help / Nápověda", fontsize = 22)
        back_btn = Button(help_fig, label = "Zpět", width = 100, height = 34)

        header[1, 1] = title
        header[1, 2] = back_btn

        body = gl[2, 1] = GridLayout()
        colgap!(body, 30)

        body[1, 1] = Label(help_fig, help_left;  tellwidth = false, halign = :left, justification = :left)
        body[1, 2] = Label(help_fig, help_right; tellwidth = false, halign = :left, justification = :left)

        # Nastav šířky až po vytvoření sloupců (vyhne se chybě "invalid column").
        try
            colsize!(body, 1, Fixed(520))
            colsize!(body, 2, Fixed(520))
        catch
        end

        screen = try
            GLMakie.Screen(; size = (1100, 900))
        catch
            GLMakie.Screen(; resolution = (1100, 900))
        end

        display(screen, help_fig)
        help_screen_ref[] = screen

        on(back_btn.clicks) do _
            try close(screen) catch end
            help_screen_ref[] = nothing
        end

        return nothing
    end

    # -----------------------------
    # History
    # -----------------------------
    function push_history!(A::AbstractMatrix{<:Real})
        t = Float32(time() - t0)
        mA = Float32(mean(A))
        xA = Float32(maximum(A))

        push!(hist_t, t)
        push!(hist_mean, mA)
        push!(hist_max, xA)

        if length(hist_t) > ADV_HIST_LEN
            deleteat!(hist_t, 1)
            deleteat!(hist_mean, 1)
            deleteat!(hist_max, 1)
        end

        mean_x_obs[] = copy(hist_t)
        mean_y_obs[] = copy(hist_mean)
        max_x_obs[]  = copy(hist_t)
        max_y_obs[]  = copy(hist_max)
        return nothing
    end

    # -----------------------------
    # Advanced window
    # -----------------------------
    params_path   = joinpath(script_dir, "advanced_params.toml")
    scenario_path = joinpath(script_dir, "advanced_scenario.toml")

    function collect_params_dict()
        return Dict(
            "dt" => Float64(dt[]),
            "mu" => Float64(mu[]),
            "sigma" => Float64(sigma[]),
            "radius" => Int(radius[]),
            "rings" => Int(ring_count[]),
            "r1" => Float64(r1[]), "w1" => Float64(w1[]), "b1" => Float64(b1[]),
            "r2" => Float64(r2[]), "w2" => Float64(w2[]), "b2" => Float64(b2[]),
            "r3" => Float64(r3[]), "w3" => Float64(w3[]), "b3" => Float64(b3[])
        )
    end

    function collect_scenario_dict()
        return Dict(
            "params" => collect_params_dict(),
            "ui" => Dict(
                "preset" => String(preset_sym[]),
                "backend" => String(backend[]),
                "init" => String(init_sym[]),
                "growth_kind" => String(growth_kind[]),
                "integrator_kind" => String(integrator_kind[]),
                "seed" => Int(seed[]),
                "noise_amp" => Float64(noise_amp[]),
                "steps_per_tick" => Int(steps_per_tick[])
            ),
            "view" => Dict(
                "autocontrast" => Bool(view_autocontrast[]),
                "gamma" => Float64(view_gamma[]),
                "colormap" => String(colormap_sym[]),
                "paint_enabled" => Bool(paint_enabled[]),
                "brush_radius" => Int(brush_radius[]),
                "brush_strength" => Float64(brush_strength[])
            )
        )
    end

    function apply_params_dict!(d::Dict)
        getf(key, default) = haskey(d, key) ? d[key] : default

        sg.sliders[1].value[]  = Float64(getf("dt", Float64(dt[])))
        sg.sliders[2].value[]  = Float64(getf("mu", Float64(mu[])))
        sg.sliders[3].value[]  = Float64(getf("sigma", Float64(sigma[])))
        sg.sliders[4].value[]  = Int(getf("radius", Int(radius[])))
        sg.sliders[5].value[]  = Int(getf("rings", Int(ring_count[])))

        sg.sliders[6].value[]  = Float64(getf("r1", Float64(r1[])))
        sg.sliders[7].value[]  = Float64(getf("w1", Float64(w1[])))
        sg.sliders[8].value[]  = Float64(getf("b1", Float64(b1[])))

        sg.sliders[9].value[]  = Float64(getf("r2", Float64(r2[])))
        sg.sliders[10].value[] = Float64(getf("w2", Float64(w2[])))
        sg.sliders[11].value[] = Float64(getf("b2", Float64(b2[])))

        sg.sliders[12].value[] = Float64(getf("r3", Float64(r3[])))
        sg.sliders[13].value[] = Float64(getf("w3", Float64(w3[])))
        sg.sliders[14].value[] = Float64(getf("b3", Float64(b3[])))

        params_dirty[] = true
        rebuild_params_if_needed!()
        return nothing
    end

    function apply_scenario_dict!(d::Dict; do_reset::Bool=false)
        ui_suspend[] = true
        try
            if haskey(d, "params") && d["params"] isa Dict
                apply_params_dict!(d["params"])
            end

            if haskey(d, "ui") && d["ui"] isa Dict
                u = d["ui"]

                if haskey(u, "seed");       sg.sliders[15].value[] = Int(u["seed"]) end
                if haskey(u, "noise_amp");  sg.sliders[16].value[] = Float64(u["noise_amp"]) end
                if haskey(u, "steps_per_tick"); steps_per_tick[] = Int(u["steps_per_tick"]) end

                if haskey(u, "backend")
                    backend[] = Symbol(u["backend"])
                    set_menu_to_value!(backend_menu, backend_vals, backend[])
                end
                if haskey(u, "init")
                    init_sym[] = Symbol(u["init"])
                    set_menu_to_value!(init_menu, init_vals, init_sym[])
                end
                if haskey(u, "growth_kind")
                    growth_kind[] = Symbol(u["growth_kind"])
                    set_menu_to_value!(growth_menu, growth_vals, growth_kind[])
                end
                if haskey(u, "integrator_kind")
                    integrator_kind[] = Symbol(u["integrator_kind"])
                    set_menu_to_value!(integ_menu, integ_vals, integrator_kind[])
                end
                if haskey(u, "preset")
                    preset_sym[] = Symbol(u["preset"])
                    set_menu_to_value!(preset_menu, preset_vals, preset_sym[])
                end

                integ_dirty[] = true
                rebuild_integrator_if_needed!()
            end

            if haskey(d, "view") && d["view"] isa Dict
                v = d["view"]
                if haskey(v, "autocontrast"); view_autocontrast[] = Bool(v["autocontrast"]) end
                if haskey(v, "gamma");        view_gamma[] = Float64(v["gamma"]) end
                if haskey(v, "colormap");     colormap_sym[] = Symbol(v["colormap"]) end

                if haskey(v, "paint_enabled"); paint_enabled[] = Bool(v["paint_enabled"]) end
                if haskey(v, "brush_radius");  brush_radius[] = Int(v["brush_radius"]) end
                if haskey(v, "brush_strength"); brush_strength[] = Float32(v["brush_strength"]) end
            end
        finally
            ui_suspend[] = false
        end

        if do_reset
            reset_state!()
        else
            rebuild_params_if_needed!()
            rebuild_integrator_if_needed!()
        end

        update_percentiles!(st_ref[].A)
        push_history!(st_ref[].A)
        update_view!()
        return nothing
    end

    function open_advanced!()
        if adv_screen_ref[] !== nothing
            try close(adv_screen_ref[]) catch end
            adv_screen_ref[] = nothing
        end

        adv_fig = Figure(size = (1200, 800), fontsize = 14)
        gl = adv_fig[1, 1] = GridLayout()
        rowgap!(gl, 12)
        colgap!(gl, 12)

        # header
        header = gl[1, 1] = GridLayout()
        colgap!(header, 10)
        title = Label(adv_fig, "Advanced", fontsize = 22)
        back_btn = Button(adv_fig, label = "Zpět", width = 100, height = 34)
        header[1, 1] = title
        header[1, 2] = back_btn

        status_lbl = Label(adv_fig, " ")

        # (1) View
        view_box = gl[2, 1] = GridLayout()
        rowgap!(view_box, 8)
        colgap!(view_box, 10)

        view_box[1, 1] = Label(adv_fig, "View (render)", fontsize = 16)

        cmap_labels = ["viridis", "grays", "magma", "plasma", "inferno", "cividis"]
        cmap_values = Symbol[:viridis, :grays, :magma, :plasma, :inferno, :cividis]
        cmap_menu, cmap_vals = menu_labeled(adv_fig; labels = cmap_labels, values = cmap_values)

        ac_btn   = Button(adv_fig, label = view_autocontrast[] ? "Auto-contrast: ON" : "Auto-contrast: OFF",
                          width=180, height=34)
        paint_btn = Button(adv_fig, label = paint_enabled[] ? "Paint: ON" : "Paint: OFF",
                           width=180, height=34)

        # NEW: reset zobrazení (vrátí zoom/pan)
        reset_view_btn = Button(adv_fig, label="Reset zobrazení", width=180, height=34)

        ac_info  = Label(adv_fig, "p05..p95: -")

        gamma_sg = SliderGrid(adv_fig,
            (label="gamma", range = 0.2:0.05:3.0, startvalue = view_gamma[])
        )
        speed_sg = SliderGrid(adv_fig,
            (label="steps/tick", range = 1:1:20, startvalue = steps_per_tick[])
        )
        brush_sg = SliderGrid(adv_fig,
            (label="brush r", range = 1:1:30, startvalue = brush_radius[]),
            (label="brush s", range = 0.01:0.01:1.0, startvalue = Float64(brush_strength[]))
        )

        view_box[2, 1] = cmap_menu
        view_box[2, 2] = ac_btn
        view_box[2, 3] = paint_btn
        view_box[2, 4] = reset_view_btn  # NEW
        view_box[3, 1:4] = gamma_sg
        view_box[4, 1:4] = speed_sg
        view_box[5, 1:4] = brush_sg
        view_box[6, 1:4] = ac_info

        # (2) History (větší)
        his_box = gl[3:6, 1] = GridLayout()
        rowgap!(his_box, 8)
        his_box[1, 1] = Label(adv_fig, "History (mean/max)", fontsize=16)
        his_ax = Axis(his_box[2, 1]; xlabel="t [s]", ylabel="value")
        lines!(his_ax, mean_x_obs, mean_y_obs)
        lines!(his_ax, max_x_obs,  max_y_obs)

        # (3) I/O
        io_box = gl[7, 1] = GridLayout()
        rowgap!(io_box, 8)
        colgap!(io_box, 10)

        io_box[1, 1] = Label(adv_fig, "I/O", fontsize=16)
        snapshot_btn = Button(adv_fig, label="Snapshot", width=150, height=36)
        rec_btn      = Button(adv_fig, label="Record",   width=150, height=36)
        stoprec_btn  = Button(adv_fig, label="Stop",     width=150, height=36)
        rec_sg = SliderGrid(adv_fig,
            (label="frames", range = 10:10:600, startvalue = record_frames[])
        )

        io_box[2, 1] = snapshot_btn
        io_box[2, 2] = rec_btn
        io_box[2, 3] = stoprec_btn
        io_box[3, 1:3] = rec_sg

        # (4) Config
        cfg_box = gl[8, 1] = GridLayout()
        rowgap!(cfg_box, 8)
        colgap!(cfg_box, 10)

        cfg_box[1, 1] = Label(adv_fig, "Config", fontsize=16)
        export_params_btn = Button(adv_fig, label="Export params", width=160, height=36)
        export_hist_btn   = Button(adv_fig, label="Export history", width=160, height=36) # NEW
        load_params_btn   = Button(adv_fig, label="Load params",   width=160, height=36)
        export_sc_btn     = Button(adv_fig, label="Export scenario", width=160, height=36)
        load_sc_apply_btn = Button(adv_fig, label="Load scenario (apply)", width=220, height=36)
        load_sc_reset_btn = Button(adv_fig, label="Load scenario (reset)", width=220, height=36)

        cfg_box[2, 1] = export_params_btn
        cfg_box[2, 2] = export_hist_btn     # NEW (napravo od export params)
        cfg_box[2, 3] = load_params_btn

        cfg_box[3, 1] = export_sc_btn
        cfg_box[3, 2] = load_sc_apply_btn
        cfg_box[3, 3] = load_sc_reset_btn

        gl[9, 1] = status_lbl

        # Wiring: View
        on(cmap_menu.selection) do _
            colormap_sym[] = selected_value(cmap_menu, cmap_vals)
        end
        set_menu_to_value!(cmap_menu, cmap_vals, colormap_sym[])

        on(ac_btn.clicks) do _
            view_autocontrast[] = !view_autocontrast[]
            ac_btn.label[] = view_autocontrast[] ? "Auto-contrast: ON" : "Auto-contrast: OFF"
            update_view!()
        end

        on(paint_btn.clicks) do _
            paint_enabled[] = !paint_enabled[]
            painting[] = false
            paint_btn.label[] = paint_enabled[] ? "Paint: ON" : "Paint: OFF"
        end

        # NEW: Reset zobrazení (hlavní pole + history graf)
        on(reset_view_btn.clicks) do _
            try
                reset_limits!(ax)
            catch
                try autolimits!(ax) catch end
            end
            try
                reset_limits!(his_ax)
            catch
                try autolimits!(his_ax) catch end
            end
            status_lbl.text[] = "Zobrazení resetováno."
        end

        on(gamma_sg.sliders[1].value) do v
            view_gamma[] = Float64(v)
            update_view!()
        end

        on(speed_sg.sliders[1].value) do v
            steps_per_tick[] = Int(round(v))
        end

        on(brush_sg.sliders[1].value) do v
            brush_radius[] = Int(round(v))
        end
        on(brush_sg.sliders[2].value) do v
            brush_strength[] = Float32(v)
        end

        on(rec_sg.sliders[1].value) do v
            record_frames[] = Int(round(v))
        end

        onany(p05_obs, p95_obs) do args...
            ac_info.text[] = "p05..p95: $(round(Float64(p05_obs[]),digits=4)) .. $(round(Float64(p95_obs[]),digits=4))"
        end
        ac_info.text[] = "p05..p95: $(round(Float64(p05_obs[]),digits=4)) .. $(round(Float64(p95_obs[]),digits=4))"

        # Wiring: I/O
        on(snapshot_btn.clicks) do _
            try
                snap_dir = joinpath(script_dir, "snapshots")
                isdir(snap_dir) || mkpath(snap_dir)
                ts = Dates.format(now(), "yyyymmdd_HHMMSS")
                path = joinpath(snap_dir, "snapshot_$ts.png")
                save(path, fig)
                status_lbl.text[] = "Saved snapshot: $(path)"
            catch err
                status_lbl.text[] = "Snapshot failed (viz REPL)"
                @error "Snapshot failed" exception=(err, catch_backtrace())
            end
        end

        on(rec_btn.clicks) do _
            try
                rec_root = joinpath(script_dir, "recordings")
                isdir(rec_root) || mkpath(rec_root)
                ts = Dates.format(now(), "yyyymmdd_HHMMSS")
                d = joinpath(rec_root, "rec_$ts")
                mkpath(d)
                record_dir[] = d
                record_idx[] = 0
                record_remaining[] = max(1, Int(record_frames[]))
                recording[] = true
                status_lbl.text[] = "Recording $(record_remaining[]) frames to: $(d)"
            catch err
                status_lbl.text[] = "Record start failed (viz REPL)"
                @error "Record start failed" exception=(err, catch_backtrace())
            end
        end

        on(stoprec_btn.clicks) do _
            recording[] = false
            record_remaining[] = 0
            status_lbl.text[] = "Recording stopped."
        end

        # Wiring: Config
        on(export_params_btn.clicks) do _
            try
                d = collect_params_dict()
                open(params_path, "w") do io
                    TOML.print(io, d)
                end
                status_lbl.text[] = "Exported params: $(params_path)"
            catch err
                status_lbl.text[] = "Export params failed (viz REPL)"
                @error "Export params failed" exception=(err, catch_backtrace())
            end
        end

        # NEW: Export history -> CSV (t, mean, max)
        on(export_hist_btn.clicks) do _
            try
                hist_dir = joinpath(script_dir, "history")
                isdir(hist_dir) || mkpath(hist_dir)
                ts = Dates.format(now(), "yyyymmdd_HHMMSS")
                path = joinpath(hist_dir, "history_$ts.csv")

                open(path, "w") do io
                    println(io, "t,mean,max")
                    @inbounds for i in eachindex(hist_t)
                        println(io, "$(hist_t[i]),$(hist_mean[i]),$(hist_max[i])")
                    end
                end
                status_lbl.text[] = "Exported history: $(path)"
            catch err
                status_lbl.text[] = "Export history failed (viz REPL)"
                @error "Export history failed" exception=(err, catch_backtrace())
            end
        end

        on(load_params_btn.clicks) do _
            try
                if !isfile(params_path)
                    status_lbl.text[] = "No file: $(params_path)  (klikni nejdřív Export params)"
                    return
                end
                d = TOML.parsefile(params_path)
                apply_params_dict!(d)
                rebuild_params_if_needed!()
                update_percentiles!(st_ref[].A)
                push_history!(st_ref[].A)
                update_view!()
                status_lbl.text[] = "Loaded params: $(params_path)"
            catch err
                status_lbl.text[] = "Load params failed (viz REPL)"
                @error "Load params failed" exception=(err, catch_backtrace())
            end
        end

        on(export_sc_btn.clicks) do _
            try
                d = collect_scenario_dict()
                open(scenario_path, "w") do io
                    TOML.print(io, d)
                end
                status_lbl.text[] = "Exported scenario: $(scenario_path)"
            catch err
                status_lbl.text[] = "Export scenario failed (viz REPL)"
                @error "Export scenario failed" exception=(err, catch_backtrace())
            end
        end

        on(load_sc_apply_btn.clicks) do _
            try
                if !isfile(scenario_path)
                    status_lbl.text[] = "No file: $(scenario_path)  (klikni nejdřív Export scenario)"
                    return
                end
                d = TOML.parsefile(scenario_path)
                apply_scenario_dict!(d; do_reset=false)
                status_lbl.text[] = "Loaded scenario (apply): $(scenario_path)"
            catch err
                status_lbl.text[] = "Load scenario failed (viz REPL)"
                @error "Load scenario failed" exception=(err, catch_backtrace())
            end
        end

        on(load_sc_reset_btn.clicks) do _
            try
                if !isfile(scenario_path)
                    status_lbl.text[] = "No file: $(scenario_path)  (klikni nejdřív Export scenario)"
                    return
                end
                d = TOML.parsefile(scenario_path)
                apply_scenario_dict!(d; do_reset=true)
                status_lbl.text[] = "Loaded scenario (reset): $(scenario_path)"
            catch err
                status_lbl.text[] = "Load scenario(reset) failed (viz REPL)"
                @error "Load scenario(reset) failed" exception=(err, catch_backtrace())
            end
        end

        # show window
        screen = try
            GLMakie.Screen(; size = (1200, 800))
        catch
            GLMakie.Screen(; resolution = (1200, 800))
        end
        display(screen, adv_fig)
        adv_screen_ref[] = screen

        on(back_btn.clicks) do _
            try close(screen) catch end
            adv_screen_ref[] = nothing
        end

        return nothing
    end

    # -----------------------------
    # Wiring: menus (suspend-safe)
    # -----------------------------
    on(preset_menu.selection) do _
        ui_suspend[] && return
        v = selected_value(preset_menu, preset_vals)
        preset_sym[] = v
        apply_preset!(v)
        update_percentiles!(st_ref[].A)
        push_history!(st_ref[].A)
        update_view!()
    end

    on(backend_menu.selection) do _
        ui_suspend[] && return
        backend[] = selected_value(backend_menu, backend_vals)
        running[] = false
        start_btn.label[] = "Start/Stop"
    end

    on(init_menu.selection) do _
        ui_suspend[] && return
        init_sym[] = selected_value(init_menu, init_vals)
    end

    on(growth_menu.selection) do _
        ui_suspend[] && return
        growth_kind[] = selected_value(growth_menu, growth_vals)
    end

    on(integ_menu.selection) do _
        ui_suspend[] && return
        integrator_kind[] = selected_value(integ_menu, integ_vals)
    end

    # -----------------------------
    # Wiring: sliders
    # -----------------------------
    on(sg.sliders[1].value)  do v; dt[]    = Float32(v) end
    on(sg.sliders[2].value)  do v; mu[]    = Float32(v) end
    on(sg.sliders[3].value)  do v; sigma[] = Float32(v) end
    on(sg.sliders[4].value)  do v; radius[]     = Int(round(v)) end
    on(sg.sliders[5].value)  do v; ring_count[] = Int(round(v)) end
    on(sg.sliders[6].value)  do v; r1[] = Float32(v) end
    on(sg.sliders[7].value)  do v; w1[] = Float32(v) end
    on(sg.sliders[8].value)  do v; b1[] = Float32(v) end
    on(sg.sliders[9].value)  do v; r2[] = Float32(v) end
    on(sg.sliders[10].value) do v; w2[] = Float32(v) end
    on(sg.sliders[11].value) do v; b2[] = Float32(v) end
    on(sg.sliders[12].value) do v; r3[] = Float32(v) end
    on(sg.sliders[13].value) do v; w3[] = Float32(v) end
    on(sg.sliders[14].value) do v; b3[] = Float32(v) end
    on(sg.sliders[15].value) do v; seed[] = Int(round(v)) end
    on(sg.sliders[16].value) do v; noise_amp[] = Float32(v) end

    # -----------------------------
    # Buttons
    # -----------------------------
    on(start_btn.clicks) do _
        running[] = !running[]
        start_btn.label[] = "Start/Stop"
    end

    on(step_btn.clicks) do _
        running[] = false
        start_btn.label[] = "Start/Stop"
        rebuild_params_if_needed!()
        rebuild_integrator_if_needed!()
        step!(st_ref[], p_ref[]; integrator = integ_ref[])
        update_percentiles!(st_ref[].A)
        push_history!(st_ref[].A)
        update_view!()
    end

    on(reset_btn.clicks) do _
        reset_state!()
        update_percentiles!(st_ref[].A)
        push_history!(st_ref[].A)
        update_view!()
    end

    on(clear_btn.clicks) do _
        running[] = false
        start_btn.label[] = "Start/Stop"
        st_ref[].A .= 0f0
        empty!(st_ref[].cache)
        update_percentiles!(st_ref[].A)
        push_history!(st_ref[].A)
        update_view!()
    end

    on(calib_btn.clicks) do _
        running[] = false
        start_btn.label[] = "Start/Stop"
        rebuild_params_if_needed!()
        auto_calibrate!(st_ref[], p_ref[]; target = p_ref[].μ)
        empty!(st_ref[].cache)
        update_percentiles!(st_ref[].A)
        push_history!(st_ref[].A)
        update_view!()
    end

    on(help_btn.clicks) do _
        try open_help!() catch err
            @error "Help se nepodařilo otevřít" exception=(err, catch_backtrace())
        end
    end

    on(advance_btn.clicks) do _
        try open_advanced!() catch err
            @error "Advanced se nepodařilo otevřít" exception=(err, catch_backtrace())
        end
    end

    # -----------------------------
    # Mouse: painting + hover
    # -----------------------------
    on(events(fig.scene).mousebutton) do ev
        paint_enabled[] || return

        if ev.action == Makie.Mouse.press
            if ev.button == Makie.Mouse.left || ev.button == Makie.Mouse.right
                running[] = false
                start_btn.label[] = "Start/Stop"

                paint_is_erase[] = (ev.button == Makie.Mouse.right)
                painting[] = true

                paint_at_cursor!(paint_is_erase[])
                update_percentiles!(st_ref[].A)
                push_history!(st_ref[].A)
                update_view!()
            end
        elseif ev.action == Makie.Mouse.release
            painting[] = false
        end
    end

    on(events(fig.scene).mouseposition) do _
        # Hover readout
        try
            idx = cursor_index()
            if idx === nothing
                hover_label.text[] = "hover: -"
            else
                iy, ix = Tuple(idx)
                v = st_ref[].A[iy, ix]
                hover_label.text[] = "hover: x=$(ix)  y=$(iy)   A=$(round(Float64(v), digits=6))"
            end
        catch
            hover_label.text[] = "hover: -"
        end

        # Paint while dragging
        if paint_enabled[] && painting[]
            paint_at_cursor!(paint_is_erase[])
            update_percentiles!(st_ref[].A)
            push_history!(st_ref[].A)
            update_view!()
        end
    end

    # -----------------------------
    # Tick loop (version-safe)
    # -----------------------------
    last_stat_t = Ref(time())

    evs = events(fig.scene)
    tick_obs =
        hasproperty(evs, :tick)       ? getproperty(evs, :tick) :
        hasproperty(evs, :frame_tick) ? getproperty(evs, :frame_tick) :
        error("This Makie version has no tick/frame_tick event; update Makie/GLMakie.")

    on(tick_obs) do _
        if running[]
            rebuild_params_if_needed!()
            rebuild_integrator_if_needed!()

            nsteps = max(1, Int(steps_per_tick[]))
            for _ in 1:nsteps
                step!(st_ref[], p_ref[]; integrator = integ_ref[])
            end

            update_view!()
        end

        # recording: ukládej jen když running + recording
        if recording[] && running[]
            try
                if record_remaining[] > 0
                    record_idx[] += 1
                    fname = @sprintf("frame_%05d.png", record_idx[])
                    path = joinpath(record_dir[], fname)
                    save(path, fig)
                    record_remaining[] -= 1
                end
                if record_remaining[] <= 0
                    recording[] = false
                end
            catch err
                recording[] = false
                @error "Recording failed" exception=(err, catch_backtrace())
            end
        end

        t = time()
        if t - last_stat_t[] >= 0.5
            A = st_ref[].A
            stat_label.text[] = "mean(A): $(round(mean(A); digits=4))   max(A): $(round(maximum(A); digits=4))"

            update_percentiles!(A)
            push_history!(A)

            if view_autocontrast[]
                update_view!()
            end

            last_stat_t[] = t
        end
    end

    # initial
    update_percentiles!(st_ref[].A)
    push_history!(st_ref[].A)
    update_view!()

    display(fig)
    return fig
end

run_gui()
