# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Validation in the Schwarzschild limit: Leaver's continued fraction (reference) against the
# collocation pencil, which reduces to a quadratic when V_h = 0 (k = w on the physical sheet).
include(joinpath(@__DIR__, "..", "NCQNM.jl"))
include(joinpath(@__DIR__, "..", "leaver.jl"))
using .NCQNM, .Leaver, LinearAlgebra, Printf

setprecision(BigFloat, 256)
guesses = Dict(1 => [0.2929 - 0.0977im, 0.2645 - 0.3063im, 0.2295 - 0.5401im, 0.2033 - 0.7883im],
               2 => [0.4836 - 0.0968im, 0.4639 - 0.2956im, 0.4305 - 0.5086im, 0.3939 - 0.7329im])
ref = Dict{Int,Vector{Complex{BigFloat}}}()
for l in (1, 2)
    ref[l] = Complex{BigFloat}[]
    for (i, g) in enumerate(guesses[l])
        w = schwarzschild_qnm(l, i - 1, Complex{BigFloat}(g); J=40_000)
        w2 = schwarzschild_qnm(l, i - 1, Complex{BigFloat}(g); J=80_000)
        push!(ref[l], w2)
        @printf("Leaver l=%d n=%d  Mw = %.15f %+.15fi   (depth change %.1e)\n", l, i - 1,
                Float64(real(w2)), Float64(imag(w2)), Float64(abs(w2 - w)))
    end
end

# Collocation in double precision: quadratic pencil M1 + w (Mw + Mk) + w^2 (Mw2 + Mkw)
function quadratic_modes(l, N)
    mod = schwarzschild(l; T=Float64)
    d = discretize(mod, N)
    P0 = d.M[:one]; P1 = d.M[:w] + d.M[:k]; P2 = d.M[:w2] + d.M[:kw]
    A, B = linearize([P0, P1, P2])
    w = eigvals(A, B)
    return w[isfinite.(w)]
end
for l in (1, 2)
    for N in (40, 60, 80)
        w = quadratic_modes(l, N)
        errs = [minimum(abs.(w .- ComplexF64(r))) for r in ref[l]]
        @printf("collocation l=%d N=%3d: distance to the Leaver tones n=0..3: %s\n", l, N,
                join([@sprintf("%.1e", e) for e in errs], "  "))
    end
end
