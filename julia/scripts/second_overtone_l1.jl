# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# The second overtone l = m = 1 of the correct metric and of the metric of Chaichian, Tureanu and
# Zet (units M = 1), which the double-precision pencil does not resolve at N = 60 and 80: Newton's
# method in 256-bit arithmetic from the Schwarzschild value, at N = 80, 100 and 120.  Output:
# data/production/second_overtone_l1.csv.
include(joinpath(@__DIR__, "..", "NCQNM.jl"))
using .NCQNM, LinearAlgebra, Printf

setprecision(BigFloat, 256)
const OUT = joinpath(@__DIR__, "..", "..", "data", "production")
open(joinpath(OUT, "second_overtone_l1.csv"), "w") do io
    println(io, "model,l,m,Theta,n,N,re,im")
    for name in ("swc", "ctz"), Th in ("0.1", "0.25", "0.5", "1.0")
        w0 = Complex{BigFloat}(big"0.2295393349313016718477" - big"0.5401334250191072134701" * im)
        for N in (80, 100, 120)
            mod = model(name, 1, 1, parse(BigFloat, Th); T = BigFloat)
            d = discretize(mod, N)
            w, _, it = newton_refine(d, w0)
            @printf(io, "%s,1,1,%s,2,%d,%s,%s\n", name, Th, N, string(round(real(w), digits = 25)),
                    string(round(imag(w), digits = 25)))
            @printf("%s Theta=%s N=%d: %.16f %+.16fi (%d iterations)\n", name, Th, N, Float64(real(w)),
                    Float64(imag(w)), it)
            flush(io)
        end
    end
end
