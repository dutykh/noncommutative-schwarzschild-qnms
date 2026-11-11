# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Quasinormal frequencies of the deformed Schwarzschild models as zeros of the Jost function.
#
# Run from the root of the repository as  julia --project=julia julia/scripts/jost_production.jl
# followed by the model and l, m:
#   jkp_rth l m                  every zero in the box B at Theta = 1/2, 1/4, 1/10, 1/20, with the
#                                Theta = 1/2 fundamental and first overtone followed continuously
#                                down to Theta = 1/20
#   swc, ctz, swc_tth or jkp_tth, then l m
#                                n = 0, 1 continued from Schwarzschild (Theta = 0) through
#                                Theta = 1/100, 1/10, 1/4, 1/2, 1, with the labels confirmed by
#                                the winding number of J
#
# Every zero is computed twice (for jkp_rth the three least damped of each box): 256 bits with
# matching accuracy 10^-60, and 320 bits with accuracy 10^-80, another matching direction,
# rotation radius, Taylor order and starting point of the horizon series; the number of digits
# on which the two agree is recorded.
# Output: data/jost/<model>_l<l>m<m>.csv (and <model>_l<l>m<m>_tracks.csv for jkp_rth).
include(joinpath(@__DIR__, "..", "NCQNM.jl"))
include(joinpath(@__DIR__, "..", "jost.jl"))
using .NCQNM, .NCJost, Printf

const OUT = joinpath(@__DIR__, "..", "..", "data", "jost")
mkpath(OUT)
const BOX = (0.02, 1.0, -0.6, -0.004)

name = ARGS[1]
l = parse(Int, ARGS[2])
m = parse(Int, ARGS[3])

problem(Th, bits) = setprecision(BigFloat, bits) do
    mod = Th == 0 ? schwarzschild(l; T=BigFloat) : model(name, l, m, Th; T=BigFloat)
    jost_problem(mod, horizon(mod)...)
end

digs(x, ref) = x == ref ? 99.0 : -log10(Float64(abs(x - ref) / abs(ref)))
fmt(x::BigFloat, d=40) = @sprintf("%.*f", d, x)

"""
Zero polished with the two settings; returns the 256-bit root and the digits of agreement
(NaN when `second` is false and only the first setting is run).
"""
function polish(Th, w; second::Bool=true)
    wa, ia = setprecision(BigFloat, 256) do
        P = problem(Th, 256)
        wa, _, ia = qnm_secant(P, Complex{BigFloat}(w); digits=60, rtol=BigFloat(10)^(-60))
        wa, ia
    end
    second || return wa, NaN, ia
    wb = setprecision(BigFloat, 320) do
        P = problem(Th, 320)
        th0 = NCJost.recessive_theta(ComplexF64(wa))
        th = th0 > 1.2 ? th0 - 0.25 : th0 + 0.25
        wb, _, _ = qnm_secant(P, Complex{BigFloat}(wa) * (1 + BigFloat(10)^(-8)); digits=80,
                              rtol=BigFloat(10)^(-75), theta=th, rho0=0.2, narc=16, N=190,
                              x1frac=0.3, tol=BigFloat(10)^(-80))
        wb
    end
    return wa, digs(wa, wb), ia
end

"""Root followed in Theta along `ths` from w0 (secant from the linear extrapolation)."""
function track(ths, w0)
    out = Complex{BigFloat}[]
    setprecision(BigFloat, 128) do
        for Th in ths
            P = problem(Th, 128)
            g = length(out) >= 2 ? 2out[end] - out[end-1] : (isempty(out) ? Complex{BigFloat}(w0) : out[end])
            w, _, inf = qnm_secant(P, g; digits=30, rtol=BigFloat(10)^(-26))
            inf.passdiff < 1e-22 || @warn "tracking not converged" Th w
            push!(out, w)
        end
    end
    return out
end

rows = ["model,l,m,Theta,label,re,im,digits_two_settings,R,theta,box_winding,count_confirmed"]
row(Th, label, w, d, ia, W, ok) = join([name, l, m, @sprintf("%.4f", Float64(Th)), label,
                                        fmt(real(w)), fmt(imag(w)), @sprintf("%.1f", d),
                                        @sprintf("%.1f", ia.R), @sprintf("%.4f", ia.theta),
                                        W, ok], ",")

if name == "jkp_rth"
    ths_prod = [1 // 2, 1 // 4, 1 // 10, 1 // 20]
    # all zeros in the box at each production Theta
    mapped = Dict{Rational{Int},Vector{Complex{BigFloat}}}()
    for Th in ths_prod
        t = @elapsed (zs, W, ok) = setprecision(BigFloat, 128) do
            grid_zeros(problem(Th, 128), BOX; digits=28)
        end
        sort!(zs, by=z -> -imag(z))
        mapped[Th] = zs
        @printf("%s (%d,%d) Theta=%.3f: %d zeros found, winding number %d, confirmed %s (%.0f s)\n",
                name, l, m, Float64(Th), length(zs), W, ok, t)
        for (i, z) in enumerate(zs)
            wa, d, ia = polish(Th, z; second=(i <= 3))
            @printf("   z%-2d %s %s i   digits %.1f\n", i - 1, fmt(real(wa), 32), fmt(imag(wa), 32), d)
            push!(rows, row(Th, "z$(i-1)", wa, d, ia, W, ok))
            zs[i] = wa
        end
        flush(stdout)
    end
    # the Theta = 1/2 fundamental and first overtone followed down to Theta = 1/20
    ths_track = [k // 100 for k in 50:-1:5]
    trows = ["model,l,m,Theta,branch,re,im"]
    for (bname, w0) in (("n0", mapped[1//2][1]), ("n1", mapped[1//2][2]))
        tr = track(ths_track, w0)
        for (Th, w) in zip(ths_track, tr)
            push!(trows, join([name, l, m, @sprintf("%.2f", Float64(Th)), bname,
                               fmt(real(w), 25), fmt(imag(w), 25)], ","))
            if Th in ths_prod
                dist, j = findmin([abs(w - z) for z in mapped[Th]])
                @printf("   branch %s at Theta=%.2f is z%d (distance %.1e)\n", bname, Float64(Th), j - 1, Float64(dist))
            end
        end
    end
    open(joinpath(OUT, "$(name)_l$(l)m$(m)_tracks.csv"), "w") do io
        foreach(r -> println(io, r), trows)
    end
else
    ths_prod = [1 // 100, 1 // 10, 1 // 4, 1 // 2, 1 // 1]
    ths_track = vcat([0 // 1, 1 // 100, 1 // 20], [k // 20 for k in 2:20])
    sch = Dict(1 => [0.2929361332672827 - 0.0976599889135782im, 0.2644486506048325 - 0.3062573915590471im],
               2 => [0.4836438722107130 - 0.0967587759782879im, 0.4638505790197656 - 0.2956039369879625im],
               3 => [0.6753662325366205 - 0.0964996277340096im, 0.6606714979559625 - 0.2922847851384119im])
    tracks = [track(ths_track, sch[l][n+1]) for n in 0:1]
    for Th in ths_prod
        i = findfirst(==(Th), ths_track)
        w0, w1 = tracks[1][i], tracks[2][i]
        # labels: the winding number of J around the strip down to below the first overtone
        b = (BOX[1], BOX[2], Float64(imag(w1)) - 0.08, BOX[4])
        W, _ = setprecision(BigFloat, 128) do
            NCJost.box_winding(problem(Th, 128), b; digits=28)
        end
        ok = W == 2
        @printf("%s (%d,%d) Theta=%.2f: winding number of the strip down to Im w = %.3f is %d\n",
                name, l, m, Float64(Th), b[3], W)
        for (n, w) in enumerate((w0, w1))
            wa, d, ia = polish(Th, w)
            @printf("   n=%d %s %s i   digits %.1f\n", n - 1, fmt(real(wa), 32), fmt(imag(wa), 32), d)
            push!(rows, row(Th, "n$(n-1)", wa, d, ia, W, ok))
        end
        flush(stdout)
    end
end
open(joinpath(OUT, "$(name)_l$(l)m$(m).csv"), "w") do io
    foreach(r -> println(io, r), rows)
end
println("written ", joinpath(OUT, "$(name)_l$(l)m$(m).csv"))
