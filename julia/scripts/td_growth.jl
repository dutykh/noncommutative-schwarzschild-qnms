# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Late times.  When V_h = V(r_h) < 0 the operator -d^2/dr*^2 + V has essential spectrum
# [V_h, infinity); since V >= V_h on the whole line (checked below) there is no eigenvalue under
# V_h, and the Green function has its rightmost singularity at the branch point w = i mu,
# mu = sqrt(-V_h), where k = sqrt(w^2 - V_h) vanishes.  Generically the Green function is analytic
# in k there, and its part odd in k gives Psi ~ A exp(mu t) t^(-3/2) at fixed r*.  This script
# measures the late rate on the null grid (d = 0.1, 0.05) and by the method of lines (h = 0.2,
# 0.1), by least squares on log|Psi| = c + lam t - p log t over the second half of the run, with
# the local rate d log|Psi|/dt at the end; it records the onset of growth at each observer (the
# minimum of the envelope of |Psi|) and the profile Psi(T, r*) at the final time; for V_h = 0
# it measures the local power index of the tail instead.
include(joinpath(@__DIR__, "..", "TimeDomain.jl"))
using .NCQNM, .TimeDomain, Printf

const DATA = joinpath(@__DIR__, "..", "..", "data", "timedomain")

# (model, l, m, Theta, final time, save the series)
allcases = [("swc", 1, 1, 0.5, 3000.0, true), ("swc", 1, 1, 1.0, 2000.0, true),
            ("swc", 2, 2, 0.5, 2000.0, false), ("swc", 2, 2, 1.0, 1200.0, false),
            ("swc", 2, 1, 0.5, 2000.0, false), ("swc", 2, 1, 1.0, 1200.0, false),
            ("swc", 1, 1, 0.25, 6000.0, false),
            ("ctz", 1, 1, 0.5, 3000.0, true), ("ctz", 1, 1, 1.0, 2000.0, true),
            ("jkp_rth", 1, 1, 0.5, 2000.0, true), ("jkp_rth", 1, 1, 0.25, 2000.0, false),
            ("jkp_rth", 1, 1, 0.1, 2000.0, true), ("jkp_rth", 1, 1, 0.05, 2000.0, false),
            ("jkp_rth", 2, 2, 0.5, 2000.0, false), ("jkp_rth", 2, 2, 0.25, 2000.0, false),
            ("jkp_rth", 2, 2, 0.1, 2000.0, false), ("jkp_rth", 2, 2, 0.05, 2000.0, false),
            ("jkp_rth", 2, 1, 0.5, 2000.0, false), ("jkp_rth", 2, 1, 0.25, 2000.0, false),
            ("jkp_rth", 2, 1, 0.1, 2000.0, false), ("jkp_rth", 2, 1, 0.05, 2000.0, false),
            ("jkp_tth", 1, 1, 0.5, 2000.0, true)]
cases = isempty(ARGS) ? allcases : [c for c in allcases if c[1] in ARGS]
const OBS = [10.0, 50.0, -100.0, -300.0]

"""
Onset of growth: the envelope max|Psi| over [t - 10, t + 10] after the arrival of the signal
(first value above 1e-10, the data having amplitude 1); the onset is the last time at which the
envelope decreases, after which it only grows (the arrival itself when it never decreases).
"""
function onset(t, y)
    n = length(t)
    dt = t[2] - t[1]
    K = max(1, round(Int, 10 / dt))
    ay = abs.(y)
    env = [maximum(ay[max(1, i - K):min(n, i + K)]) for i in 1:n]
    i0 = findfirst(env .> 1e-10)
    i0 === nothing && return NaN, NaN
    i = findlast(j -> env[j] < env[j-1], i0+1:n-K)
    i = i === nothing ? i0 : i0 + i
    return t[i], env[i]
end

function analyse(t, y, Tend, mu)
    sel = findall(t .>= Tend / 2)
    tt, yy = t[sel], y[sel]
    f2 = growth_fit(tt, yy)
    f3 = growth_fit(tt, yy; withpow=true)
    tl, lr = local_rate(t, y, max(1, round(Int, 20 / (t[2] - t[1]))))
    ton, von = onset(t, y)
    return (lam2=f2.lam, lam3=f3.lam, p=f3.p, loc=lr[end], ton=ton, von=von)
end

rows = String[]
tailrows = String[]
bsrows = String[]
for (name, l, m, Th, Tend, save) in cases
    mod = model(name, l, m, Th)
    T = tortoise(mod)
    Vh = potential(mod, T.uh)
    mu = sqrt(max(0.0, -Vh))
    # V >= V_h on [-600, 600] and the lowest Dirichlet eigenvalues on [-L, 400] (no bound state:
    # the lowest one stays above V_h and tends to it like (pi/L)^2)
    xg = collect(-600.0:0.05:600.0)
    Vg = potential_at(mod, T, xg)
    Vmin = minimum(Vg)
    ev = Float64[]
    for L in (400.0, 800.0)
        xb = collect(-L:0.1:400.0)
        push!(ev, bound_states(potential_at(mod, T, xb), 0.1, 1)[1])
    end
    push!(bsrows, @sprintf("%s,%d,%d,%.3f,%.10e,%.10e,%.10e,%.10e", name, l, m, Th, Vh, Vmin, ev...))
    @printf("%s (%d,%d) Theta=%.2f  V_h=%.6e  mu=%.6f  min V on [-600,600] = %.6e  lowest Dirichlet eigenvalue (L=400, 800): %.6e, %.6e\n",
            name, l, m, Th, Vh, mu, Vmin, ev...)
    series = nothing
    for d in (0.1, 0.05)
        r = gpp_evolve(xs -> potential_at(mod, T, xs), d, Tend + 300, Tend + 50; vc=20.0, sigma=3.0, obs=OBS)
        for (k, xo) in enumerate(OBS)
            t, y = r.t[k], r.psi[k]
            if mu > 0
                a = analyse(t, y, Tend, mu)
                push!(rows, @sprintf("%s,%d,%d,%.3f,GPP,%.3f,%.1f,%.1f,%.8f,%.8f,%.8f,%.4f,%.8f,%.2f,%.3e", name, l, m, Th,
                                     d, xo, Tend, mu, a.lam2, a.lam3, a.p, a.loc, a.ton, a.von))
                @printf("   GPP d=%.2f r*=%5.0f: lam(exp)=%.6f lam(exp*t^-p)=%.6f p=%.3f local=%.6f (mu=%.6f, lam-mu=%+.1e) onset t=%.1f |Psi|=%.1e\n",
                        d, xo, a.lam2, a.lam3, a.p, a.loc, mu, a.lam3 - mu, a.ton, a.von)
            else
                ps = Float64[]
                for tc in (250.0, 500.0, 1000.0, 1500.0, 1900.0)
                    tc * 1.05 > t[end] && (push!(ps, NaN); continue)
                    i1 = searchsortedfirst(t, tc / 1.05); i2 = searchsortedfirst(t, tc * 1.05)
                    push!(ps, -(log(abs(y[i2])) - log(abs(y[i1]))) / (log(t[i2]) - log(t[i1])))
                end
                push!(tailrows, @sprintf("%s,%d,%d,%.3f,GPP,%.3f,%.1f,%s", name, l, m, Th, d, xo,
                                         join([@sprintf("%.4f", p) for p in ps], ",")))
                @printf("   GPP d=%.2f r*=%5.0f: tail index p(t) at t=250,500,1000,1500,1900: %s\n", d, xo,
                        join([@sprintf("%.3f", p) for p in ps], " "))
            end
        end
        if d == 0.1 && save
            series = r
        end
    end
    # method of lines on an interval whose ends cannot signal to any observer before Tend
    prof = nothing
    for h in (0.2, 0.1)
        a = floor((40.0 - 300.0 - Tend) / 2 - 20.0)
        b = ceil((Tend + 1.6 + 50.0) / 2 + 20.0)
        xs = collect(a:h:b)
        V = potential_at(mod, T, xs)
        dt = h / 2
        r = mol_evolve(V, a, h, dt, Tend; x0=40.0, sigma=2.0, ingoing=true, obs=OBS, every=round(Int, 0.2 / dt))
        for (k, xo) in enumerate(OBS)
            t, y = r.t, r.psi[k]
            if mu > 0
                an = analyse(t, y, Tend, mu)
                push!(rows, @sprintf("%s,%d,%d,%.3f,MOL,%.3f,%.1f,%.1f,%.8f,%.8f,%.8f,%.4f,%.8f,%.2f,%.3e", name, l, m, Th,
                                     h, xo, Tend, mu, an.lam2, an.lam3, an.p, an.loc, an.ton, an.von))
                @printf("   MOL h=%.2f r*=%5.0f: lam(exp)=%.6f lam(exp*t^-p)=%.6f p=%.3f local=%.6f (lam-mu=%+.1e) onset t=%.1f |Psi|=%.1e\n",
                        h, xo, an.lam2, an.lam3, an.p, an.loc, an.lam3 - mu, an.ton, an.von)
            else
                ps = Float64[]
                for tc in (250.0, 500.0, 1000.0, 1500.0, 1900.0)
                    tc * 1.05 > t[end] && (push!(ps, NaN); continue)
                    i1 = searchsortedfirst(t, tc / 1.05); i2 = searchsortedfirst(t, tc * 1.05)
                    push!(ps, -(log(abs(y[i2])) - log(abs(y[i1]))) / (log(t[i2]) - log(t[i1])))
                end
                push!(tailrows, @sprintf("%s,%d,%d,%.3f,MOL,%.3f,%.1f,%s", name, l, m, Th, h, xo,
                                         join([@sprintf("%.4f", p) for p in ps], ",")))
                @printf("   MOL h=%.2f r*=%5.0f: tail index p(t) at t=250,500,1000,1500,1900: %s\n", h, xo,
                        join([@sprintf("%.3f", p) for p in ps], " "))
            end
        end
        if h == 0.1
            prof = (r.x, r.P)
        end
    end
    if mu > 0 && prof !== nothing
        x, P = prof
        lp = log10.(abs.(P) .+ 1e-300)
        imax = argmax(lp)
        # width of the region where |Psi| is within a factor e of its maximum
        near = findall(lp .> lp[imax] - 1 / log(10))
        @printf("   profile at t=%.0f: max |Psi| at r* = %.1f (log10 = %.2f); |Psi| > max/e on [%.1f, %.1f]; log10|Psi| at r* = -600, -300, -100, -20, 0, 10, 50: %s\n",
                Tend, x[imax], lp[imax], x[first(near)], x[last(near)],
                join([@sprintf("%.2f", lp[argmin(abs.(x .- xx))]) for xx in (-600.0, -300.0, -100.0, -20.0, 0.0, 10.0, 50.0)], " "))
        push!(rows, @sprintf("%s,%d,%d,%.3f,MOL-profile,0.100,%.1f,%.1f,%.8f,NaN,NaN,NaN,NaN,%.1f,%.1f", name, l, m, Th,
                             x[imax], Tend, mu, x[first(near)], x[last(near)]))
    end
    if save && series !== nothing
        tt = collect(0.0:2.0:Tend)
        cols = Vector{Float64}[]
        for k in eachindex(OBS)
            c = fill(NaN, length(tt))
            for (ti, yi) in zip(series.t[k], series.psi[k])
                j = round(Int, ti / 2) + 1
                abs(ti - 2(j - 1)) < 1e-6 && 1 <= j <= length(tt) && (c[j] = yi)
            end
            push!(cols, c)
        end
        tag = @sprintf("%s_l%dm%d_Th%.2f", name, l, m, Th)
        write_columns(joinpath(DATA, "series_$(tag).csv"), ["t", "psi_10", "psi_50", "psi_m100", "psi_m300"], tt, cols...;
                      comment="$name (l,m)=($l,$m) Theta=$Th, null grid d = 0.1, Gaussian (sigma 3) on u = 0 at v = 20; Psi at r* = 10, 50, -100, -300",
                      digits=9)
    end
end
tag = isempty(ARGS) ? "all" : join(ARGS, "_")
open(joinpath(DATA, "growth_$(tag).csv"), "w") do io
    println(io, "# late-time growth: fits of log|Psi| over [T/2, T]; lam_exp from c + lam t, lam_pow and p from c + lam t - p log t; local = dlog|Psi|/dt at T;")
    println(io, "# onset = time of the minimum of the envelope max|Psi| over +-10; rows MOL-profile: rstar_obs = position of max|Psi(T, r*)|, onset columns = interval where |Psi| > max/e")
    println(io, "model,l,m,Theta,method,res,rstar_obs,T,mu,lam_exp,lam_pow,p,local,t_onset,psi_onset")
    foreach(r -> println(io, r), rows)
end
if !isempty(tailrows)
    open(joinpath(DATA, "tails_$(tag).csv"), "w") do io
        println(io, "# local power index p(t) = -dlog|Psi|/dlog t (centred over t/1.05 .. 1.05 t)")
        println(io, "model,l,m,Theta,method,res,rstar_obs,p_250,p_500,p_1000,p_1500,p_1900")
        foreach(r -> println(io, r), tailrows)
    end
end
open(joinpath(DATA, "boundstates_$(tag).csv"), "w") do io
    println(io, "# V_h, minimum of V on [-600, 600], lowest Dirichlet eigenvalue of -d2 + V on [-L, 400] (second differences, h = 0.1) for L = 400 and 800")
    println(io, "model,l,m,Theta,V_h,Vmin,ev_L400,ev_L800")
    foreach(r -> println(io, r), bsrows)
end
