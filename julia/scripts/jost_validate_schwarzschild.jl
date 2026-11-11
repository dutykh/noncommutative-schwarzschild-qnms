# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Validation of the Jost function in the Schwarzschild limit: the zeros of J for l = 1, 2 and
# overtones n = 0, 1, 2 against Leaver's continued fraction.  The continued fraction is evaluated
# at depth 10^5 and 2 10^5 (its tail error decays like exp(-c sqrt(depth))), and each zero of J
# is computed with two settings that differ in precision, matching radius, matching direction,
# rotation radius, Taylor order and starting point of the horizon series.
# Output: data/jost/schwarzschild_validation.csv
include(joinpath(@__DIR__, "..", "NCQNM.jl"))
include(joinpath(@__DIR__, "..", "leaver.jl"))
include(joinpath(@__DIR__, "..", "jost.jl"))
using .NCQNM, .Leaver, .NCJost, Printf

const OUT = joinpath(@__DIR__, "..", "..", "data", "jost")
mkpath(OUT)

digs(x, ref) = x == ref ? 99.0 : -log10(Float64(abs(x - ref) / abs(ref)))
fmt(x::BigFloat, d=40) = @sprintf("%.*f", d, x)

guesses = Dict(1 => [0.2929 - 0.0977im, 0.2645 - 0.3063im, 0.2295 - 0.5401im],
               2 => [0.4836 - 0.0968im, 0.4639 - 0.2956im, 0.4305 - 0.5086im])

rows = String[]
push!(rows, "l,n,re_jost,im_jost,re_leaver,im_leaver,digits_vs_leaver,digits_two_settings,leaver_depth_digits")
for l in (1, 2), n in 0:2
    setprecision(BigFloat, 256) do
        g = Complex{BigFloat}(guesses[l][n+1])
        wL1 = schwarzschild_qnm(l, n, g; J=100_000)
        wL2 = schwarzschild_qnm(l, n, g; J=200_000)
        P = jost_problem(schwarzschild(l; T=BigFloat), horizon(schwarzschild(l; T=BigFloat))...)
        wa, ha, ia = qnm_secant(P, g; digits=60, rtol=BigFloat(10)^(-60))
        # second setting: more bits, other path and matching data
        wb = setprecision(BigFloat, 320) do
            Pb = jost_problem(schwarzschild(l; T=BigFloat), horizon(schwarzschild(l; T=BigFloat))...)
            th = NCJost.recessive_theta(g) - 0.25
            wb, hb, ib = qnm_secant(Pb, Complex{BigFloat}(g) * (1 + BigFloat(10)^(-6));
                                    digits=80, rtol=BigFloat(10)^(-75), theta=th, rho0=0.2,
                                    narc=16, N=190, x1frac=0.3, tol=BigFloat(10)^(-80))
            wb
        end
        dL = digs(wa, wL2)
        d2 = digs(wa, wb)
        dd = digs(wL1, wL2)
        @printf("l=%d n=%d  Jost  %s %s i\n", l, n, fmt(real(wa), 45), fmt(imag(wa), 45))
        @printf("           Leaver %s %s i\n", fmt(real(wL2), 45), fmt(imag(wL2), 45))
        @printf("           digits: Jost vs Leaver %.1f, two Jost settings %.1f, Leaver depths %.1f; R=%.1f theta=%.3f\n",
                dL, d2, dd, ia.R, ia.theta)
        push!(rows, join([l, n, fmt(real(wa)), fmt(imag(wa)), fmt(real(wL2)), fmt(imag(wL2)),
                          @sprintf("%.1f", dL), @sprintf("%.1f", d2), @sprintf("%.1f", dd)], ","))
        flush(stdout)
    end
end
open(joinpath(OUT, "schwarzschild_validation.csv"), "w") do io
    foreach(r -> println(io, r), rows)
end
println("written ", joinpath(OUT, "schwarzschild_validation.csv"))
