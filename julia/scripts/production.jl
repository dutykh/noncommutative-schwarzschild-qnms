# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Tones of the second-order metrics by spectral collocation: the Joukowski quartic pencil in double
# precision locates the tones that converge between two resolutions; each is then refined by
# Newton's method in 256-bit arithmetic at N = 80 and N = 100, and the agreement of the two gives
# the number of correct digits.  Also the first-order coefficient omega_2 of
# omega(Theta) = omega_S + Theta^2 omega_2 + O(Theta^4), by Richardson extrapolation of
# (omega - omega_S)/Theta^2 at Theta^2 = h, 2h, 4h.
include(joinpath(@__DIR__, "..", "NCQNM.jl"))
using .NCQNM, LinearAlgebra, Printf

setprecision(BigFloat, 256)
const OUT = joinpath(@__DIR__, "..", "..", "data", "production")
mkpath(OUT)

function candidates(name, l, m, Th; nmax = 3)
    mod = model(name, l, m, Th)
    s1 = solve_joukowski(mod, 60); s2 = solve_joukowski(mod, 80)
    pick(s) = s.w[s.phys .& (real.(s.w) .> 1e-4) .& (imag.(s.w) .< 0) .& (imag.(s.w) .> -1.2) .& (abs.(s.w) .< 3)]
    w1, w2 = pick(s1), pick(s2)
    c = [z for z in w2 if minimum(abs.(w1 .- z); init = Inf) < 1e-5]
    sort!(c, by = z -> -imag(z))
    return c[1:min(nmax, length(c))]
end

function refine(name, l, m, Th, w0; Ns = (80, 100))
    vals = Complex{BigFloat}[]
    for N in Ns
        mod = model(name, l, m, BigFloat(Th); T = BigFloat)
        d = discretize(mod, N)
        w, _, _ = newton_refine(d, w0)
        push!(vals, w)
    end
    digits = -log10(Float64(abs(vals[2] - vals[1]) / abs(vals[2])))
    return vals[end], digits
end


open(joinpath(OUT, "tones.csv"), "w") do io
    println(io, "model,l,m,Theta,n,re,im,digits")
    for name in ("swc", "ctz", "swc_tth"), l in 1:3, m in (name == "swc_tth" ? (0:l) : (1:l)),
        Th in (0.1, 0.25, 0.5, 1.0)
        ws = candidates(name, l, m, Th)
        for (n, w0) in enumerate(ws)
            w, dg = refine(name, l, m, Th, w0)
            println(io, name, ",", l, ",", m, ",", Th, ",", n - 1, ",",
                    string(round(real(w), digits = 30)), ",", string(round(imag(w), digits = 30)), ",",
                    @sprintf("%.1f", dg))
            flush(io)
        end
        @printf("%-8s l=%d m=%d Theta=%.2f: %d tones\n", name, l, m, Th, length(ws))
    end
end

# first-order coefficients
open(joinpath(OUT, "first_order.csv"), "w") do io
    println(io, "model,l,m,n,re_omega2,im_omega2,re_omegaS,im_omegaS")
    h = big"1e-4"
    for name in ("swc", "ctz", "swc_tth"), l in 1:3, m in (name == "swc_tth" ? (0:l) : (1:l))
        ws = candidates(name, l, m, 0.1; nmax = 2)
        for (n, w0) in enumerate(ws)
            wS, _ = refine(name, l, m, 0.0, w0)
            q = Complex{BigFloat}[]
            for s in (1, 2, 4)
                w, _ = refine(name, l, m, sqrt(s * h), w0)
                push!(q, (w - wS) / (s * h))
            end
            # Richardson for q(x) = w2 + w4 x + w6 x^2 at x = h, 2h, 4h
            w2 = (8q[1] - 6q[2] + q[3]) / 3
            println(io, name, ",", l, ",", m, ",", n - 1, ",", string(round(real(w2), digits = 20)), ",",
                    string(round(imag(w2), digits = 20)), ",", string(round(real(wS), digits = 25)), ",",
                    string(round(imag(wS), digits = 25)))
            flush(io)
        end
        @printf("first order %-8s l=%d m=%d done\n", name, l, m)
    end
end
