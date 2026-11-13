# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# The completions of the axis for |m| = 1.  Near the axis the second-order metric leaves the angular
# eigenvalue of the modes with |m| = 1 undetermined at first order: the completions of the model of
# symbolic/axis_model.py add to the potential Theta^2 beta kappa L (2l+1) f/(8 r^4), with beta = +1
# (Dirichlet) or -1 (Neumann).  Since omega_2 depends linearly on the potential, their effect on
# omega_2 is beta kappa L (2l+1)/8 times
#     R = d omega / d e  for the Schwarzschild potential plus e f/r^4  (units M = 1),
# computed here by central differences (e = +-1e-15) in 256-bit arithmetic at N = 80 and N = 100,
# for l = 1, 2, 3 and n = 0, 1.  For (l, m, n) = (1, 1, 0) of the correct metric the predicted
# omega_2 of the two completions is checked by a direct computation with the modified potential
# (Richardson extrapolation at Theta^2 = 1e-4, 2e-4, 4e-4, N = 100).
#
# Run from the root of the repository as  julia --project=julia julia/scripts/axis_test.jl
# Output: data/limits/axis_response.csv and data/limits/axis_direct.csv.
include(joinpath(@__DIR__, "..", "NCQNM.jl"))
using .NCQNM, Polynomials, Printf

setprecision(BigFloat, 256)
const T = BigFloat
const OUT = joinpath(@__DIR__, "..", "..", "data", "limits")
mkpath(OUT)
B(s) = parse(BigFloat, s)

"""The model with the extra potential e (u^4 - 2 u^5) = e f/r^4 added."""
function add_f_r4(mod::NCQNM.Model{S}, e) where {S}
    u = Polynomial([zero(S), one(S)])
    return NCQNM.Model{S}(mod.name * "+fr4", mod.l, mod.m, mod.Theta, mod.F,
                          mod.Vn + S(e) * (u^4 - 2u^5) * mod.Vd, mod.Vd)
end
tone(mod, N, w0) = newton_refine(discretize(mod, N), w0)[1]

# starting values: Schwarzschild tones (Leaver's continued fraction)
const WS = Dict((1, 0) => 0.292936133267283 - 0.097659988913578im, (1, 1) => 0.264448650604834 - 0.306257391559067im,
                (2, 0) => 0.483643872210713 - 0.096758775978288im, (2, 1) => 0.463850579019766 - 0.295603936987963im,
                (3, 0) => 0.675366232536621 - 0.096499627734010im, (3, 1) => 0.660671497955963 - 0.292284785138412im)

open(joinpath(OUT, "axis_response.csv"), "w") do io
    println(io, "l,n,N,re_wS,im_wS,re_R,im_R")
    for (l, n) in sort(collect(keys(WS)))
        for N in (80, 100)
            s = schwarzschild(l; T=T)
            w0 = tone(s, N, Complex{T}(WS[(l, n)]))
            e = T(1e-15)
            R = (tone(add_f_r4(s, e), N, w0) - tone(add_f_r4(s, -e), N, w0)) / (2e)
            println(io, l, ",", n, ",", N, ",", string(real(w0)), ",", string(imag(w0)), ",",
                    string(real(R)), ",", string(imag(R)))
            flush(io)
            @printf("l=%d n=%d N=%d  R = %.10f %+.10f i\n", l, n, N, Float64(real(R)), Float64(imag(R)))
        end
    end
end

# direct check for (1, 1, 0) of the correct metric: kappa L (2l+1)/8 = (5/16)(2)(3)/8 = 15/64
open(joinpath(OUT, "axis_direct.csv"), "w") do io
    println(io, "beta,re_omega2,im_omega2")
    h = T(1) / 10000
    wS0 = Complex(B("0.2929361332672827086223011"), -B("0.0976599889135782221553126"))
    for beta in (1, -1)
        q = Complex{T}[]
        for s in (1, 2, 4)
            Th2 = s * h
            mm = add_f_r4(model("swc", 1, 1, sqrt(Th2); T=T), beta * Th2 * T(15) / 64)
            w = tone(mm, 100, wS0 + Th2 * Complex{T}(-2.8e-4, -5.65e-3))
            push!(q, (w - wS0) / Th2)
        end
        w2 = (8q[1] - 6q[2] + q[3]) / 3
        println(io, beta, ",", string(real(w2)), ",", string(imag(w2)))
        @printf("beta=%+d: omega_2 = %.6e %+.6e i\n", beta, Float64(real(w2)), Float64(imag(w2)))
    end
end
