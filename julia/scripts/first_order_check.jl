# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# A second Richardson triplet for the first-order coefficients omega_2.  The coefficients of
# data/production/first_order.csv extrapolate the difference quotients (omega - omega_S)/Theta^2 at
# Theta^2 = h, 2h, 4h with h = 1e-4, which presupposes a regular expansion in Theta^2.  This script
# repeats the extrapolation with the triplet h/2, h, 2h and compares the two, for the fundamentals
# with (l, m) = (1, 1), (2, 1), (2, 2), (3, 1), (3, 3) and the first overtone of (1, 1) of the correct
# metric, and for the tones with l <= 2 among them of the metric of Chaichian, Tureanu and Zet;
# Newton's method in 256-bit arithmetic at N = 80 and N = 100, linear map.
#
# Run from the root of the repository as  julia --project=julia julia/scripts/first_order_check.jl
# Output: data/production/first_order_check.csv.
include(joinpath(@__DIR__, "..", "NCQNM.jl"))
using .NCQNM, Printf

setprecision(BigFloat, 256)
const T = BigFloat
const OUT = joinpath(@__DIR__, "..", "..", "data", "production")
tone(mod, N, w0) = newton_refine(discretize(mod, N), w0)[1]

const WS = Dict((1, 0) => 0.292936133267283 - 0.097659988913578im, (1, 1) => 0.264448650604834 - 0.306257391559067im,
                (2, 0) => 0.483643872210713 - 0.096758775978288im, (3, 0) => 0.675366232536621 - 0.096499627734010im)
const GUESS = Dict((1, 1, 0) => -2.81e-4 - 5.648e-3im, (1, 1, 1) => 2.442e-3 - 1.7457e-2im,
                   (2, 1, 0) => -4.4776e-3 - 6.6604e-3im, (2, 2, 0) => -2.029e-4 - 5.4835e-3im,
                   (3, 1, 0) => -1.21732e-2 - 6.8342e-3im, (3, 3, 0) => -2.461e-4 - 5.3416e-3im)

open(joinpath(OUT, "first_order_check.csv"), "w") do io
    println(io, "model,l,m,n,N,re_omega2_half,im_omega2_half,re_omega2,im_omega2,abs_difference")
    h = T(1) / 10000
    for name in ("swc", "ctz"), (l, m, n) in sort(collect(keys(GUESS)))
        name == "ctz" && l == 3 && continue
        for N in (80, 100)
            wS = tone(model(name, l, m, T(0); T=T), N, Complex{T}(WS[(l, n)]))
            q = Dict{Int,Complex{T}}()
            for s in (1, 2, 4, 8)                     # Theta^2 = s h/2
                Th2 = s * h / 2
                w = tone(model(name, l, m, sqrt(Th2); T=T), N, wS + Th2 * Complex{T}(GUESS[(l, m, n)]))
                q[s] = (w - wS) / Th2
            end
            # q(x) = w2 + w4 x + w6 x^2 at x = a, 2a, 4a gives w2 = (8 q(a) - 6 q(2a) + q(4a))/3
            w2half = (8q[1] - 6q[2] + q[4]) / 3      # a = h/2
            w2 = (8q[2] - 6q[4] + q[8]) / 3          # a = h, as in production.jl
            println(io, name, ",", l, ",", m, ",", n, ",", N, ",", string(real(w2half)), ",", string(imag(w2half)), ",",
                    string(real(w2)), ",", string(imag(w2)), ",", @sprintf("%.3e", Float64(abs(w2half - w2))))
            flush(io)
            @printf("%s (%d,%d) n=%d N=%d  omega_2 = %.10e %+.10e i   difference %.2e\n", name, l, m, n, N,
                    Float64(real(w2)), Float64(imag(w2)), Float64(abs(w2half - w2)))
        end
    end
end
