# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Validation of the time-domain method on the Schwarzschild black hole (scalar field, l = 1, 2).
# Quasinormal frequencies: method of lines (h = 0.2, 0.1; dt = h/4) and null grid (d = 0.2, 0.1,
# 0.05, 0.025, with Richardson extrapolation in d^2), fundamental by sliding matrix-pencil
# windows, first overtone after removal of the fitted fundamental by the annihilating filter;
# reference values by Leaver's continued fraction.  Late times: the local power index
# p(t) = -d log|Psi| / d log t, expected to tend to 2l + 3 at fixed r*.
include(joinpath(@__DIR__, "..", "TimeDomain.jl"))
include(joinpath(@__DIR__, "..", "leaver.jl"))
using .NCQNM, .TimeDomain, .Leaver, Printf

const DATA = joinpath(@__DIR__, "..", "..", "data", "timedomain")
setprecision(BigFloat, 128)
ref = Dict{Tuple{Int,Int},ComplexF64}()
for (l, n, g) in ((1, 0, 0.2929 - 0.0977im), (1, 1, 0.2645 - 0.3063im), (2, 0, 0.4836 - 0.0968im),
                  (2, 1, 0.4639 - 0.2956im))
    ref[(l, n)] = ComplexF64(schwarzschild_qnm(l, n, Complex{BigFloat}(g); J=40_000))
end

rows = String[]
function record!(l, n, method, res, xo, q)
    e = abs(q.w - ref[(l, n)])
    push!(rows, @sprintf("%d,%d,%s,%.3f,%.1f,%.12f,%.12f,%.3e,%.3e,%.1f,%d,%.1f", l, n, method, res, xo,
                         real(q.w), imag(q.w), e, q.spread, q.W, q.M, q.t0))
    @printf("  l=%d n=%d %-4s res=%.3f r*=%4.0f  w = %.12f %+.12fi  err %.2e  spread %.1e  (W=%g, M=%d, t0=%g)\n",
            l, n, method, res, xo, real(q.w), imag(q.w), e, q.spread, q.W, q.M, q.t0)
end

function overtone(t, y, wf, guide, t0s)
    ds = 0.2
    stride = max(1, round(Int, ds / (t[2] - t[1])))
    tt = t[1:stride:end]; yy = y[1:stride:end]
    yf = annihilate(yy, exp(-im * wf * (tt[2] - tt[1])))
    res = qnm_scan(tt[1:end-2], yf, guide; t0s=t0s, Ws=(20.0, 30.0, 40.0), Ms=(4, 6, 8),
                   ds=tt[2] - tt[1], K=9)
    return res[1]
end

println("Reference (Leaver): ", join([@sprintf("l=%d n=%d %.12f%+.12fi", k..., real(v), imag(v)) for (k, v) in sort(collect(ref), by=first)], "; "))
gppw = Dict{Tuple{Int,Float64,Float64},ComplexF64}()
for l in (1, 2)
    mod = schwarzschild(l)
    T = tortoise(mod)
    # method of lines: ingoing Gaussian (sigma = 2) from r* = 40; the ringdown reaches r* = xo
    # at about t = 36.8 + xo
    for h in (0.2, 0.1)
        a, b = -300.0, 400.0
        xs = collect(a:h:b)
        V = potential_at(mod, T, xs)
        dt = h / 4
        r = mol_evolve(V, a, h, dt, 300.0; x0=40.0, sigma=2.0, ingoing=true, obs=[10.0, 50.0],
                       every=round(Int, 0.1 / dt))
        for (k, xo) in enumerate((10.0, 50.0))
            tr = 36.8 + xo
            q = qnm_scan(r.t, r.psi[k], ref[(l, 0)]; t0s=tr .+ (0.0:1.0:80.0), Ws=(40.0, 60.0, 80.0),
                         Ms=(6, 8), ds=0.25, K=11)[1]
            record!(l, 0, "MOL", h, xo, q)
            q1 = overtone(r.t, r.psi[k], q.w, ref[(l, 1)], tr .+ (-6.0:0.5:28.0))
            record!(l, 1, "MOL", h, xo, q1)
        end
    end
    # null grid: Gaussian (sigma = 3) on u = 0 centred at v = 20; ringdown at about t = 16.8 + xo
    for d in (0.2, 0.1, 0.05, 0.025)
        r = gpp_evolve(xs -> potential_at(mod, T, xs), d, 300.0, 360.0; vc=20.0, sigma=3.0, obs=[10.0, 50.0])
        for (k, xo) in enumerate((10.0, 50.0))
            tr = 16.8 + xo
            q = qnm_scan(r.t[k], r.psi[k], ref[(l, 0)]; t0s=tr .+ (0.0:1.0:80.0), Ws=(40.0, 60.0, 80.0),
                         Ms=(6, 8), ds=0.2, K=11)[1]
            record!(l, 0, "GPP", d, xo, q)
            gppw[(l, d, xo)] = q.w
            if d <= 0.05
                q1 = overtone(r.t[k], r.psi[k], q.w, ref[(l, 1)], tr .+ (-6.0:0.5:28.0))
                record!(l, 1, "GPP", d, xo, q1)
            end
        end
    end
    # Richardson extrapolation of the null-grid fundamental in d^2, and the observed order
    for xo in (10.0, 50.0)
        w = [gppw[(l, d, xo)] for d in (0.2, 0.1, 0.05, 0.025)]
        for k in 1:3
            wr = (4w[k+1] - w[k]) / 3
            push!(rows, @sprintf("%d,0,GPP-Richardson,%.3f,%.1f,%.12f,%.12f,%.3e,NaN,NaN,0,NaN", l,
                                 [0.2, 0.1, 0.05][k] / 2, xo, real(wr), imag(wr), abs(wr - ref[(l, 0)])))
            @printf("  l=%d GPP Richardson (d=%.3f,%.3f) r*=%g: w = %.12f %+.12fi  err %.2e\n", l,
                    [0.2, 0.1, 0.05][k], [0.1, 0.05, 0.025][k], xo, real(wr), imag(wr), abs(wr - ref[(l, 0)]))
        end
        p1 = log2(abs(w[1] - w[2]) / abs(w[2] - w[3])); p2 = log2(abs(w[2] - w[3]) / abs(w[3] - w[4]))
        @printf("  l=%d GPP observed order r*=%g: %.2f, %.2f\n", l, xo, p1, p2)
    end
end
open(joinpath(DATA, "qnm_schwarzschild.csv"), "w") do io
    println(io, "# Schwarzschild scalar quasinormal frequencies from the time domain (M w), error against Leaver's continued fraction;")
    println(io, "# res = h (method of lines) or d (null grid); spread = max deviation over the 9 to 11 most stable consecutive windows")
    println(io, "l,n,method,res,rstar_obs,Re_w,Im_w,error,spread,W,M,t0")
    foreach(r -> println(io, r), rows)
end

# Late times: power-law tails on the null grid (d = 0.1 and 0.05) and by the method of lines
println("\nLate-time tails")
tailrows = String[]
for l in (1, 2)
    mod = schwarzschild(l)
    T = tortoise(mod)
    Tend = 2000.0
    obs = [10.0, 50.0, -100.0, -300.0]
    series = Dict{Float64,Any}()
    for d in (0.1, 0.05)
        r = gpp_evolve(xs -> potential_at(mod, T, xs), d, Tend + 300, Tend + 50; vc=20.0, sigma=3.0, obs=obs)
        for (k, xo) in enumerate(obs)
            t, y = r.t[k], r.psi[k]
            ps = Float64[]
            for tc in (250.0, 500.0, 1000.0, 1500.0, 1900.0)
                i1 = searchsortedfirst(t, tc / 1.05); i2 = searchsortedfirst(t, min(tc * 1.05, t[end]))
                push!(ps, -(log(abs(y[i2])) - log(abs(y[i1]))) / (log(t[i2]) - log(t[i1])))
            end
            push!(tailrows, @sprintf("%d,GPP,%.3f,%.1f,%s", l, d, xo, join([@sprintf("%.4f", p) for p in ps], ",")))
            @printf("  l=%d GPP d=%.2f r*=%5.0f  p(t) at t=250,500,1000,1500,1900: %s\n", l, d, xo,
                    join([@sprintf("%.3f", p) for p in ps], " "))
            if d == 0.1
                sel = [i for i in eachindex(t) if abs(t[i] / 2 - round(t[i] / 2)) < 1e-6]
                series[xo] = (t[sel], y[sel])
            end
        end
    end
    tt = collect(0.0:2.0:Tend)
    cols = Vector{Float64}[]
    for xo in obs
        ts, ys = series[xo]
        c = fill(NaN, length(tt))
        for (ti, yi) in zip(ts, ys)
            k = round(Int, ti / 2) + 1
            1 <= k <= length(tt) && (c[k] = yi)
        end
        push!(cols, c)
    end
    write_columns(joinpath(DATA, "series_schwarzschild_l$(l).csv"), ["t", "psi_10", "psi_50", "psi_m100", "psi_m300"],
                  tt, cols...; comment="Schwarzschild l=$l, null grid d = 0.1, Gaussian (sigma 3) on u = 0 at v = 20; Psi at r* = 10, 50, -100, -300",
                  digits=9)
    # method of lines, h = 0.2, dt = 0.1, on [-1150, 1050] (no boundary signal reaches any observer before t = 2000)
    a, b, h = -1150.0, 1050.0, 0.2
    xs = collect(a:h:b)
    V = potential_at(mod, T, xs)
    r = mol_evolve(V, a, h, h / 2, Tend; x0=40.0, sigma=2.0, ingoing=true, obs=obs, every=10)
    for (k, xo) in enumerate(obs)
        t, y = r.t, r.psi[k]
        ps = Float64[]
        for tc in (250.0, 500.0, 1000.0, 1500.0, 1900.0)
            i1 = searchsortedfirst(t, tc / 1.05); i2 = searchsortedfirst(t, tc * 1.05)
            push!(ps, -(log(abs(y[i2])) - log(abs(y[i1]))) / (log(t[i2]) - log(t[i1])))
        end
        push!(tailrows, @sprintf("%d,MOL,%.3f,%.1f,%s", l, h, xo, join([@sprintf("%.4f", p) for p in ps], ",")))
        @printf("  l=%d MOL h=%.2f r*=%5.0f  p(t) at t=250,500,1000,1500,1900: %s\n", l, h, xo,
                join([@sprintf("%.3f", p) for p in ps], " "))
    end
end
open(joinpath(DATA, "tails_schwarzschild.csv"), "w") do io
    println(io, "# local power index p(t) = -dlog|Psi|/dlog t (centred over t/1.05 .. 1.05 t); expected limit 2l+3 at fixed r*")
    println(io, "l,method,res,rstar_obs,p_250,p_500,p_1000,p_1500,p_1900")
    foreach(r -> println(io, r), tailrows)
end
