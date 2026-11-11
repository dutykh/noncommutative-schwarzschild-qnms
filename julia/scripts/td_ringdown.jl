# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Ringdown frequencies of the deformed models "swc", "ctz" and "jkp_tth" from the time domain.
# Method of lines (h = 0.2 and 0.1, dt = h/4; ingoing Gaussian of width 2 from r* = 40) and null
# grid (d = 0.1, 0.05, 0.025, Gaussian of width 3 on u = 0 at v = 20, Richardson extrapolation in
# d^2); observers at r* = 10 and 50; fundamental tone from sliding matrix-pencil windows (the
# extra terms of orders M = 8, 10 absorb the non-oscillating branch-cut part of the signal, which
# grows when V_h < 0); comparison with the collocation values where they exist.
include(joinpath(@__DIR__, "..", "TimeDomain.jl"))
using .NCQNM, .TimeDomain, Printf

const DATA = joinpath(@__DIR__, "..", "..", "data", "timedomain")
const SCHW = Dict(1 => 0.292936133267283 - 0.097659988913578im, 2 => 0.483643872210713 - 0.096758775978288im)

# (model, l, m, Theta, reference or NaN)
const NOREF = ComplexF64(NaN, NaN)
cases = [("swc", 1, 1, 0.1, NOREF), ("swc", 1, 1, 0.25, 0.2929159992 - 0.0980117193im),
         ("swc", 1, 1, 0.5, 0.2928268754 - 0.0990516347im), ("swc", 1, 1, 1.0, NOREF),
         ("swc", 2, 2, 0.5, 0.4835721421 - 0.0981167760im), ("swc", 2, 2, 1.0, NOREF),
         ("swc", 2, 1, 0.5, 0.4825060395 - 0.0983904184im), ("swc", 2, 1, 1.0, NOREF),
         ("ctz", 1, 1, 0.1, 0.292980685 - 0.097721078im), ("ctz", 1, 1, 0.25, 0.293211373 - 0.098041021im),
         ("ctz", 1, 1, 0.5, 0.293994252 - 0.099172782im), ("ctz", 1, 1, 1.0, NOREF),
         ("ctz", 2, 2, 0.5, 0.485334084 - 0.098236978im),
         ("jkp_tth", 1, 1, 0.5, 0.292867760 - 0.097586697im)]
if !isempty(ARGS)
    cases = [c for c in cases if c[1] in ARGS]
end

rows = String[]
summary = String[]
for (name, l, m, Th, wref) in cases
    mod = model(name, l, m, Th)
    T = tortoise(mod)
    Vh = potential(mod, T.uh)
    guide = isnan(real(wref)) ? SCHW[l] : wref
    @printf("%s (l,m)=(%d,%d) Theta=%.2f  V_h=%.6e  reference %s\n", name, l, m, Th, Vh,
            isnan(real(wref)) ? "none" : @sprintf("%.10f%+.10fi", real(wref), imag(wref)))
    est = Dict{Tuple{String,Float64,Float64},ComplexF64}()
    sprd = Dict{Tuple{String,Float64,Float64},Float64}()
    function rec(method, res, xo, q)
        e = isnan(real(wref)) ? NaN : abs(q.w - wref)
        est[(method, res, xo)] = q.w; sprd[(method, res, xo)] = q.spread
        push!(rows, @sprintf("%s,%d,%d,%.3f,%s,%.4f,%.1f,%.12f,%.12f,%.3e,%.3e,%.1f,%d,%.1f", name, l, m, Th,
                             method, res, xo, real(q.w), imag(q.w), e, q.spread, q.W, q.M, q.t0))
        @printf("   %-4s res=%.3f r*=%3.0f  w = %.11f %+.11fi  spread %.1e  err %.1e  (W=%g M=%d t0=%g)\n",
                method, res, xo, real(q.w), imag(q.w), q.spread, e, q.W, q.M, q.t0)
    end
    for h in (0.2, 0.1)
        a, b = -300.0, 400.0
        xs = collect(a:h:b)
        V = potential_at(mod, T, xs)
        dt = h / 4
        r = mol_evolve(V, a, h, dt, 300.0; x0=40.0, sigma=2.0, ingoing=true, obs=[10.0, 50.0],
                       every=round(Int, 0.1 / dt))
        for (k, xo) in enumerate((10.0, 50.0))
            tr = 36.8 + xo
            q = qnm_scan(r.t, r.psi[k], guide; t0s=tr .+ (0.0:1.0:80.0), Ws=(40.0, 60.0, 80.0),
                         Ms=(6, 8, 10), ds=0.25, K=11)[1]
            rec("MOL", h, xo, q)
        end
    end
    for d in (0.1, 0.05, 0.025)
        r = gpp_evolve(xs -> potential_at(mod, T, xs), d, 300.0, 360.0; vc=20.0, sigma=3.0, obs=[10.0, 50.0])
        for (k, xo) in enumerate((10.0, 50.0))
            tr = 16.8 + xo
            q = qnm_scan(r.t[k], r.psi[k], guide; t0s=tr .+ (0.0:1.0:80.0), Ws=(40.0, 60.0, 80.0),
                         Ms=(6, 8, 10), ds=0.2, K=11)[1]
            rec("GPP", d, xo, q)
        end
    end
    for xo in (10.0, 50.0), (d1, d2) in ((0.1, 0.05), (0.05, 0.025))
        wr = (4est[("GPP", d2, xo)] - est[("GPP", d1, xo)]) / 3
        e = isnan(real(wref)) ? NaN : abs(wr - wref)
        push!(rows, @sprintf("%s,%d,%d,%.3f,GPP-Richardson,%.4f,%.1f,%.12f,%.12f,%.3e,NaN,NaN,0,NaN", name, l, m,
                             Th, d2 / 2, xo, real(wr), imag(wr), e))
        est[("GPPR", d2, xo)] = wr
        @printf("   GPP Richardson (%.3f,%.3f) r*=%3.0f  w = %.11f %+.11fi  err %.1e\n", d1, d2, xo, real(wr), imag(wr), e)
    end
    # retained value: method of lines h = 0.1 at r* = 10; error bar from the other estimates
    w = est[("MOL", 0.1, 10.0)]
    others = [est[("MOL", 0.1, 50.0)], est[("MOL", 0.2, 10.0)], est[("GPPR", 0.025, 10.0)], est[("GPPR", 0.025, 50.0)]]
    eb = maximum(abs.(others .- w))
    e = isnan(real(wref)) ? NaN : abs(w - wref)
    push!(summary, @sprintf("%s,%d,%d,%.3f,%.12f,%.12f,%.2e,%.2e,%s,%s,%.3e", name, l, m, Th, real(w), imag(w),
                            eb, sprd[("MOL", 0.1, 10.0)],
                            isnan(real(wref)) ? "NaN" : @sprintf("%.10f", real(wref)),
                            isnan(real(wref)) ? "NaN" : @sprintf("%.10f", imag(wref)), e))
    @printf("   RETAINED w = %.11f %+.11fi   method spread %.1e   |w - ref| = %.1e\n", real(w), imag(w), eb, e)
end
tag = isempty(ARGS) ? "all" : join(ARGS, "_")
open(joinpath(DATA, "qnm_ringdown_$(tag).csv"), "w") do io
    println(io, "# fundamental tone M w from the time domain; res = h (MOL) or d (GPP); spread = max deviation over the 11 most stable consecutive windows")
    println(io, "model,l,m,Theta,method,res,rstar_obs,Re_w,Im_w,err_vs_ref,spread,W,M,t0")
    foreach(r -> println(io, r), rows)
end
open(joinpath(DATA, "qnm_ringdown_summary_$(tag).csv"), "w") do io
    println(io, "# retained tone: method of lines h = 0.1 at r* = 10; method_spread = max distance to (MOL r*=50, MOL h=0.2, GPP Richardson r*=10 and 50)")
    println(io, "model,l,m,Theta,Re_w,Im_w,method_spread,window_spread,Re_ref,Im_ref,err_vs_ref")
    foreach(r -> println(io, r), summary)
end
