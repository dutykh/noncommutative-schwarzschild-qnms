# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Fundamental tones of the metric of Chaichian, Tureanu and Zet ("ctz") and of the correct metric
# ("swc") for Theta^2 from 0 to 1 (units M = 1), by the quartic Joukowski pencil in double
# precision at N = 80, each checked against N = 60; the curves of the figure that compares them
# with the published values.  Output: data/figures/fundamentals_vs_theta2.csv.
include(joinpath(@__DIR__, "..", "NCQNM.jl"))
using .NCQNM, LinearAlgebra, Printf

const OUT = joinpath(@__DIR__, "..", "..", "data", "figures")
guess = Dict((1, 1) => 0.2929 - 0.0977im, (2, 1) => 0.4836 - 0.0968im, (2, 2) => 0.4836 - 0.0968im)
open(joinpath(OUT, "fundamentals_vs_theta2.csv"), "w") do io
    println(io, "model,l,m,Theta2,re,im,diff_N60")
    for name in ("ctz", "swc"), (l, m) in ((1, 1), (2, 1), (2, 2))
        z = guess[(l, m)]
        for T2 in 0.0:0.025:1.0
            mod = T2 == 0 ? model(name, l, m, 1e-8) : model(name, l, m, sqrt(T2))
            s80 = solve_joukowski(mod, 80); s60 = solve_joukowski(mod, 60)
            w80 = s80.w[s80.phys]; w60 = s60.w[s60.phys]
            z = w80[argmin(abs.(w80 .- z))]
            d = minimum(abs.(w60 .- z))
            @printf(io, "%s,%d,%d,%.4f,%.12f,%.12f,%.1e\n", name, l, m, T2, real(z), imag(z), d)
        end
    end
end
println("done")
