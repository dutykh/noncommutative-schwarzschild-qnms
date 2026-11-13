# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# The guide of the time-domain fits only names the mode.  The evolutions of td_ringdown.jl (method of
# lines, h = 0.1, observer r* = 10, Theta = 0.5) are fitted by sliding matrix-pencil windows in which
# the fundamental is the fitted frequency closest to a guide.  Here the guide is the zero of the Jost
# function, the Schwarzschild tone, or the Schwarzschild tone displaced by 0.01 (1 - i); and, without
# any guide, the fundamental is taken in each window as the least damped frequency with Re w > 0.1.
#
# Run from the root of the repository as  julia --project=julia julia/scripts/td_blind.jl
# Output: data/timedomain/blind_fits.txt (the printout).
include(joinpath(@__DIR__, "..", "TimeDomain.jl"))
using .NCQNM, .TimeDomain, Printf

const OUT = joinpath(@__DIR__, "..", "..", "data", "timedomain", "blind_fits.txt")
const SCHW = Dict(1 => 0.292936133267283 - 0.097659988913578im, 2 => 0.483643872210713 - 0.096758775978288im)
const JOST = Dict(("swc", 1, 1) => 0.2928268753542784 - 0.0990516347063569im,
                  ("swc", 2, 1) => 0.4825060394987666 - 0.0983904183571474im,
                  ("swc", 2, 2) => 0.4835721421282132 - 0.0981167759756296im)

open(OUT, "w") do io
    out(s) = (print(s); print(io, s))
    out("# Fundamental tones at Theta = 0.5 from the method of lines (h = 0.1, r* = 10), fitted with\n")
    out("# three guides and with none; err = |w - Jost zero|, spread over the 11 most stable windows.\n")
    for (name, l, m) in sort(collect(keys(JOST)))
        wref = JOST[(name, l, m)]
        mod = model(name, l, m, 0.5)
        Tt = tortoise(mod)
        h = 0.1; a, b = -300.0, 400.0
        V = potential_at(mod, Tt, collect(a:h:b))
        dt = h / 4
        r = mol_evolve(V, a, h, dt, 300.0; x0=40.0, sigma=2.0, ingoing=true, obs=[10.0], every=round(Int, 0.1 / dt))
        t, y = r.t, r.psi[1]
        tr = 46.8
        for (lab, guide) in (("Jost zero", wref), ("Schwarzschild", SCHW[l]), ("Schwarzschild + 0.01(1-i)", SCHW[l] + 0.01 - 0.01im))
            q = qnm_scan(t, y, ComplexF64(guide); t0s=tr .+ (0.0:1.0:80.0), Ws=(40.0, 60.0, 80.0), Ms=(6, 8, 10), ds=0.25, K=11)[1]
            out(@sprintf("%s (%d,%d) guide %-26s w = %.12f %+.12f i  err %.2e  spread %.1e  (W = %g, M = %d)\n",
                         name, l, m, lab, real(q.w), imag(q.w), abs(q.w - wref), q.spread, q.W, q.M))
        end
        best = (Inf, 0.0im, 0.0, 0)
        for W in (40.0, 60.0, 80.0), M in (6, 8, 10)
            t0s = [t0 for t0 in tr .+ (0.0:1.0:80.0) if t0 + W <= t[end]]
            sc = pencil_scan(t, y, t0s, W, M; ds=0.25)
            z = ComplexF64[(c = filter(w -> real(w) > 0.1, ws); isempty(c) ? ComplexF64(NaN, NaN) : c[1]) for ws in sc]
            good = .!isnan.(real.(z))
            sum(good) < 11 && continue
            zm, s, _ = TimeDomain.plateau(z[good], 11)
            s < best[1] && (best = (s, zm, W, M))
        end
        out(@sprintf("%s (%d,%d) no guide                   w = %.12f %+.12f i  err %.2e  spread %.1e  (W = %g, M = %d)\n",
                     name, l, m, real(best[2]), imag(best[2]), abs(best[2] - wref), best[1], best[3], best[4]))
    end
end
