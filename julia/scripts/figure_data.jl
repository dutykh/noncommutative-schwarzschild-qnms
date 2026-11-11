# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Data of two figures of the article.
#  1. The quartic Joukowski pencil of the correct metric, l = m = 1, Theta = M: all its eigenvalues
#     Lambda at N = 60 and N = 80, the corresponding (w, k), the sheet (|Lambda| > sqrt|V_h|), and
#     the distance from each eigenvalue at N = 80 to the nearest one at N = 60.
#  2. Spectral convergence for l = m = 1, Theta = 0.5 M, overtones n = 0, 1: the error of the
#     quartic pencil in double precision and of Newton's method in 256-bit arithmetic, as a
#     function of the number N of Chebyshev polynomials, against the zeros of the Jost function.
include(joinpath(@__DIR__, "..", "NCQNM.jl"))
using .NCQNM, LinearAlgebra, Printf

setprecision(BigFloat, 256)
const OUT = joinpath(@__DIR__, "..", "..", "data", "figures")
mkpath(OUT)

# 1. the two planes
mod = model("swc", 1, 1, 1.0)
s60 = solve_joukowski(mod, 60)
s80 = solve_joukowski(mod, 80)
open(joinpath(OUT, "joukowski_swc_l1m1_Th1.csv"), "w") do io
    println(io, "# quartic pencil of the correct metric, l = m = 1, Theta = 1 (M = 1), N = 80; V_h = ", s80.Vh)
    println(io, "re_Lambda,im_Lambda,re_w,im_w,physical,dist_N60")
    for j in eachindex(s80.Lam)
        d = minimum(abs.(s60.w .- s80.w[j]))
        @printf(io, "%.12e,%.12e,%.12e,%.12e,%d,%.3e\n", real(s80.Lam[j]), imag(s80.Lam[j]),
                real(s80.w[j]), imag(s80.w[j]), s80.phys[j], d)
    end
end

# 2. convergence against the Jost zeros (swc, l = m = 1, Theta = 0.5)
ref = [big"0.2928268753542784082064307" - big"0.0990516347063569079474662" * im,
       big"0.2648416262722882036269" - big"0.3105621631114951518984" * im]
modb = model("swc", 1, 1, big"0.5"; T = BigFloat)
open(joinpath(OUT, "convergence_swc_l1m1_Th0p5.csv"), "w") do io
    println(io, "N,n,err_double_qz,err_newton_256bit")
    for N in 16:4:100
        s = solve_joukowski(mod.name == "swc" ? model("swc", 1, 1, 0.5) : mod, N)
        for (n, z) in enumerate(ref)
            ws = s.w[s.phys]
            zd = ws[argmin(abs.(ws .- ComplexF64(z)))]
            ed = abs(zd - ComplexF64(z))
            d = discretize(modb, N)
            wb, _, _ = newton_refine(d, Complex{BigFloat}(zd))
            eb = Float64(abs(wb - z))
            @printf(io, "%d,%d,%.4e,%.4e\n", N, n - 1, ed, eb)
            @printf("N = %3d  n = %d  double %.2e  256-bit %.2e\n", N, n - 1, ed, eb)
        end
    end
end
