# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Tones n = 0, 1 of the metric of Chaichian, Tureanu and Zet ("ctz") and of the correct metric
# ("swc") at the values Theta^2 = 0.2, 0.4, 0.6, 0.8, 1 of the published tables (units M = 1):
# the quartic pencil in double precision locates them, and Newton's method in 256-bit arithmetic
# refines them at N = 80 and N = 100; the agreement of the two gives the digits.
# Output: data/production/published_points.csv.
include(joinpath(@__DIR__, "..", "NCQNM.jl"))
using .NCQNM, LinearAlgebra, Printf

setprecision(BigFloat, 256)
const OUT = joinpath(@__DIR__, "..", "..", "data", "production")
schw = Dict(1 => [0.2929361332672827 - 0.0976599889135782im, 0.2644486506048325 - 0.3062573915590471im],
            2 => [0.4836438722107130 - 0.0967587759782879im, 0.4638505790197656 - 0.2956039369879625im])
open(joinpath(OUT, "published_points.csv"), "w") do io
    println(io, "model,l,m,n,Theta2,re,im,digits")
    for name in ("ctz", "swc"), (l, m) in ((1, 1), (2, 1), (2, 2)), n in (0, 1)
        z = schw[l][n+1]
        for T2 in (0.2, 0.4, 0.6, 0.8, 1.0)
            Th = sqrt(big(T2))
            s = solve_joukowski(model(name, l, m, Float64(Th)), 80)
            ws = s.w[s.phys]
            z = ws[argmin(abs.(ws .- z))]
            vals = Complex{BigFloat}[]
            for N in (80, 100)
                d = discretize(model(name, l, m, Th; T = BigFloat), N)
                w, _, _ = newton_refine(d, Complex{BigFloat}(z))
                push!(vals, w)
            end
            dg = -log10(Float64(abs(vals[2] - vals[1])))
            @printf(io, "%s,%d,%d,%d,%.1f,%s,%s,%.1f\n", name, l, m, n, T2, string(round(real(vals[2]), digits = 20)),
                    string(round(imag(vals[2]), digits = 20)), dg)
            flush(io)
        end
    end
end
println("done")
