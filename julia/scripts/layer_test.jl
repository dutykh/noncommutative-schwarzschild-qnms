# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# A change of the potential confined to the horizon layer.  The potential of the correct metric
# ("swc"), l = m = 1, receives the extra term (units M = 1)
#     Delta V = c Theta^4 (2/r)^3 / ((r - 2) + Theta^2) = 8 c Theta^4 u^4 / (1 - (2 - Theta^2) u),
# of fourth order in Theta at every fixed r > 2 but of second order in the layer r - 2 = O(Theta^2),
# where it reaches (32 c/33) Theta^2 at the horizon; at infinity it decays like 1/r^4.  For the
# fundamental (n = 0) and the first overtone (n = 1), and Theta = 0.2, 0.1, 0.05, 0.02, 0.01,
# 0.005, the tone is computed with c = 0, +1/256 and -1/256 by Newton's method in 256-bit
# arithmetic at the degrees N = 100 and N = 120, with the collocation map that resolves the layer
# (eps = distance from the horizon to the pole of the extra term, at r = 2 - Theta^2).  The tone
# with c != 0 is reached from the tone with c = 0 by continuation in c, in eight steps.
#
# Run from the root of the repository, either for every case in turn,
#     julia --project=julia julia/scripts/layer_test.jl
# or for one case, so that the cases can run side by side,
#     julia --project=julia julia/scripts/layer_test.jl 1 0.005
# Output: data/limits/layer_n<n>_Th<Theta>.csv, one file per case, read by figures/fig_limits.py.
include(joinpath(@__DIR__, "..", "NCQNM.jl"))
using .NCQNM, Polynomials, Printf

setprecision(BigFloat, 256)
const T = BigFloat
const OUT = joinpath(@__DIR__, "..", "..", "data", "limits")
mkpath(OUT)
B(s) = parse(BigFloat, s)

"""The model with the extra potential 8 c Theta^4 u^4 / (1 - (2 - Theta^2) u) added."""
function add_layer_term(mod::NCQNM.Model{S}, c) where {S}
    Th = mod.Theta
    u = Polynomial([zero(S), one(S)])
    q = 1 - (2 - Th^2) * u
    return NCQNM.Model{S}(mod.name * "+layer", mod.l, mod.m, Th, mod.F,
                          mod.Vn * q + 8 * S(c) * Th^4 * u^4 * mod.Vd, mod.Vd * q)
end

# Schwarzschild tones and first-order coefficients of the correct metric (data/production/first_order.csv)
const WS = Dict(0 => Complex(B("0.2929361332672827086223011"), -B("0.0976599889135782221553126")),
                1 => Complex(B("0.2644486506048138630631387"), -B("0.3062573915590669111535644")))
const W2 = Dict(0 => Complex(B("-0.000281210569696518"), -B("0.005648147701350282")),
                1 => Complex(B("0.002442297338829383"), -B("0.017456496713672230")))

function run_case(n::Int, Th::Float64)
    Thb = T(Th)
    m0 = model("swc", 1, 1, Thb; T=T)
    uh = horizon(m0)[1]
    ep = 1 / (2 - Thb^2) - uh
    path = joinpath(OUT, "layer_n$(n)_Th$(Th).csv")
    open(path, "w") do io
        println(io, "l,m,n,Theta,c,N,re,im")
        for N in (100, 120)
            d0 = discretize(m0, N; eps=ep)
            base, _, _ = newton_refine(d0, WS[n] + Thb^2 * W2[n])
            println(io, "1,1,", n, ",", Th, ",0,", N, ",", string(real(base)), ",", string(imag(base)))
            flush(io)
            for c in (T(1) / 256, -T(1) / 256)
                w = base
                for s in 1:8
                    d = discretize(add_layer_term(m0, c * s / 8), N; eps=ep)
                    w, _, _ = newton_refine(d, w)
                end
                println(io, "1,1,", n, ",", Th, ",", Float64(c), ",", N, ",", string(real(w)), ",", string(imag(w)))
                flush(io)
                @printf("n=%d Theta=%.3f N=%d c=%+.5f  delta w = %.6e %+.6e i\n", n, Th, N, Float64(c),
                        Float64(real(w - base)), Float64(imag(w - base)))
            end
        end
    end
end

if length(ARGS) == 2
    run_case(parse(Int, ARGS[1]), parse(Float64, ARGS[2]))
else
    for n in (0, 1), Th in (0.2, 0.1, 0.05, 0.02, 0.01, 0.005)
        run_case(n, Th)
    end
end
