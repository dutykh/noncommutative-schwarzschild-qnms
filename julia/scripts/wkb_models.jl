# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# WKB frequencies of the scalar field for the second-order metrics "swc", "jkp_rth", "jkp_tth" and
# "ctz" (NCQNM.jl) at Theta^2 = 0.0025, 0.01, 0.0625, 0.25 (Theta = 0.05, 0.1, 0.25, 0.5), M = 1:
# third order (Iyer-Will), sixth order (Konoplya) with the Pade approximant P_{3/3}, and 13th order
# with P_{6/6}, P_{6/7}, P_{7/6} (Matyjasek-Opala).  The spread of the three highest Pade values
# estimates the accuracy of the WKB answer itself; where accurate frequencies are known they are
# compared.  The WKB answer depends only on the Taylor coefficients of V(r*) at the top of the
# barrier, so the quotient (w(Theta) - w(0))/Theta^2 measures how regularly the barrier alone moves
# the frequency.
# Output: data/wkb/models_wkb.csv
include(joinpath(@__DIR__, "..", "NCQNM.jl"))
include(joinpath(@__DIR__, "..", "wkb.jl"))
using .NCQNM, .WKB, Polynomials, Printf

setprecision(BigFloat, 256)
const T = BigFloat
const OUT = joinpath(@__DIR__, "..", "..", "data", "wkb")
mkpath(OUT)
const THS = ["0.05", "0.1", "0.25", "0.5"]
cases = [("swc", 1, 1), ("swc", 2, 2), ("swc", 2, 1), ("jkp_rth", 1, 1), ("jkp_rth", 2, 2),
         ("jkp_tth", 1, 1), ("ctz", 1, 1), ("ctz", 2, 2)]
# accurate frequencies by collocation, typed in here, keyed by (model, l, m, n, Theta)
accurate = Dict(("swc", 1, 1, 0, "0.5") => 0.2928268754 - 0.0990516347im,
                ("swc", 1, 1, 0, "0.25") => 0.2929159992 - 0.0980117193im,
                ("swc", 1, 1, 0, "0.1") => 0.2929332547 - 0.0977164376im,
                ("swc", 1, 1, 0, "0.05") => 0.2929354261 - 0.0976741072im,
                ("swc", 2, 2, 0, "0.5") => 0.4835721421 - 0.0981167760im,
                ("swc", 2, 1, 0, "0.5") => 0.4825060395 - 0.0983904184im,
                ("ctz", 1, 1, 0, "0.1") => 0.292980685 - 0.097721078im,
                ("ctz", 1, 1, 0, "0.25") => 0.293211373 - 0.098041021im,
                ("ctz", 1, 1, 0, "0.5") => 0.293994252 - 0.099172782im,
                ("ctz", 1, 1, 1, "0.1") => 0.264525617 - 0.306442989im,
                ("ctz", 1, 1, 1, "0.5") => 0.266089779 - 0.310857250im,
                ("ctz", 2, 2, 0, "0.5") => 0.485334084 - 0.098236978im)
schw = Dict((1, 0) => 0.292936133267283 - 0.097659988913578im, (1, 1) => 0.264448650604833 - 0.306257391559047im,
            (2, 0) => 0.483643872210713 - 0.096758775978288im, (2, 1) => 0.463850579019766 - 0.295603936987963im)
fmt(w) = @sprintf("%.8f%+.8fi", Float64(real(w)), Float64(imag(w)))
rows = String[]
w0 = Dict{Tuple{Int,Int},Complex{T}}()
for l in (1, 2), n in (0, 1)
    mod = schwarzschild(l; T=T)
    uh, _ = horizon(mod)
    w0[(l, n)] = wkb_omegas(mod.F, mod.Vn, mod.Vd, uh, n; N=13).pade[(6, 7)]
end
for (name, l, m) in cases, n in (0, 1)
    println("== $name (l, m) = ($l, $m), n = $n;  Schwarzschild P6/7 ", fmt(w0[(l, n)]))
    for th in THS
        Th = parse(T, th)
        mod = model(name, l, m, Th; T=T)
        uh, _ = horizon(mod)
        r = wkb_omegas(mod.F, mod.Vn, mod.Vd, uh, n; N=13)
        hi = [r.pade[(6, 6)], r.pade[(6, 7)], r.pade[(7, 6)]]
        spread = maximum(abs(a - b) for a in hi, b in hi)
        best = r.pade[(6, 7)]
        slope = (best - w0[(l, n)]) / Th^2
        Vh = potential(mod, uh)
        @printf("  Th2=%-7.4f r_peak=%.6f V_peak=%.7f V_h=%+.5f | WKB3 %s  WKB6 %s  P33 %s  P67 %s (spread %.0e) | (w-w0)/Th2 %s\n",
                Float64(Th^2), Float64(1 / r.u0), Float64(r.v[1]), Float64(Vh), fmt(r.plain[3]), fmt(r.plain[6]),
                fmt(r.pade[(3, 3)]), fmt(best), Float64(spread), fmt(slope))
        key = (name, l, m, n, th)
        if haskey(accurate, key)
            ex = accurate[key]
            @printf("      accurate %s : errors WKB3 %.1e  WKB6 %.1e  P33 %.1e  P66 %.1e  P67 %.1e  P76 %.1e ; shift from Schwarzschild %.1e\n",
                    fmt(ex), abs(r.plain[3] - ex), abs(r.plain[6] - ex), abs(r.pade[(3, 3)] - ex), abs(hi[1] - ex),
                    abs(hi[2] - ex), abs(hi[3] - ex), abs(ex - schw[(l, n)]))
            push!(rows, @sprintf("%s,%d,%d,%d,%.6f,accurate,%.10f,%.10f", name, l, m, n, Float64(Th^2), real(ex), imag(ex)))
        end
        for (lab, w) in (("wkb3", r.plain[3]), ("wkb6", r.plain[6]), ("wkb13", r.plain[13]), ("pade33", r.pade[(3, 3)]),
                         ("pade66", hi[1]), ("pade67", hi[2]), ("pade76", hi[3]))
            push!(rows, @sprintf("%s,%d,%d,%d,%.6f,%s,%.10f,%.10f", name, l, m, n, Float64(Th^2), lab,
                                 Float64(real(w)), Float64(imag(w))))
        end
        push!(rows, @sprintf("%s,%d,%d,%d,%.6f,peak,%.10f,%.10f", name, l, m, n, Float64(Th^2), Float64(1 / r.u0), Float64(r.v[1])))
    end
end
open(joinpath(OUT, "models_wkb.csv"), "w") do io
    println(io, "# WKB quasinormal frequencies (M = 1) of the scalar field, models of NCQNM.jl: wkb3, wkb6, wkb13")
    println(io, "# (plain formula of that order), pade33 (sixth order, P3/3), pade66, pade67, pade76 (12th and 13th")
    println(io, "# orders); accurate = collocation values typed in the script; peak = (r at the top of")
    println(io, "# the barrier, V there) stored in the (re, im) columns.")
    println(io, "model,l,m,n,Theta2,approximation,re,im")
    foreach(r -> println(io, r), rows)
end
println("written ", joinpath(OUT, "models_wkb.csv"))
