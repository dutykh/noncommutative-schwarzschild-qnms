# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# WKB frequencies of the scalar field on the metric of Chaichian, Tureanu and Zet ("ctz"), compared
# with the third-order WKB tables of Heidari et al., Eur. Phys. J. C 84 (2024) 566 (Tables 1-3),
# with the Pade value of Zhao et al., Nucl. Phys. B 1004 (2024) 116545 (Table II, Theta = 0), and
# with accurate frequencies (collocation of NCQNM.jl).  M = 1, Theta^2 in units of M^2.
#
# The radial equation is Psi'' + (w^2 - V) Psi = 0 in the tortoise coordinate, with
#     dr/dr* = F = f (1 + Theta^2 (A0 - B0)/2),     f = 1 - 2/r,
#     V = f (l(l+1)/r^2 + 2/r^3) + Theta^2 [ W00 + a0 Wa + c0 Wc + s1 Ws1 + s2 Ws2 ]   (NCQNM.jl),
# where W00 = (f/r) f'(A0 - B0) + (1/(4 r^2)) D(r^2 D(A0 - B0 + C0 + D0)) - D(D(A0 - B0))/4 and
# Ws2 = (1/(4 r^2)) D(r^2 D(r^-2)), with D = f d/dr the zeroth-order tortoise derivative.  The
# variants tested against the published numbers are
#   cct        the potential above (that of NCQNM.jl, the reference);
#   tort1      dr/dr* = f (1 + Theta^2 (A0 - B0)), the tortoise relation as printed by Heidari et
#              al. (their eq. (2.19), without the factor 1/2), potential unchanged;
#   tort0      dr/dr* = f, the Theta^2 term of the tortoise relation dropped;
#   Vprint     the potential as printed in their eq. (2.23), with (A0 + B0 - C0 + D0) and (A0 + B0)
#              in place of (A0 - B0 + C0 + D0) and (A0 - B0) in the two derivative terms;
#   Vprint1    Vprint together with tort1;
#   Dfull      the derivatives D inside the Theta^2 part of V taken with the full F (factor 1/2),
#              which differs from cct at order Theta^4 only;
#   Dfull1     the full F without the factor 1/2 used everywhere (in V and in the WKB derivatives);
#   sqrtF      dr/dr* = f sqrt((1 + Theta^2 A0)/(1 + Theta^2 B0)), the unexpanded ratio of the
#              metric coefficients, potential as in cct.
# Outputs: data/wkb/ctz_wkb.csv (all values), data/wkb/ctz_heidari_comparison.csv.
include(joinpath(@__DIR__, "..", "NCQNM.jl"))
include(joinpath(@__DIR__, "..", "wkb.jl"))
using .NCQNM, .WKB, Polynomials, LinearAlgebra, Printf

setprecision(BigFloat, 256)
BLAS.set_num_threads(2)
const T = BigFloat
const OUT = joinpath(@__DIR__, "..", "..", "data", "wkb")
mkpath(OUT)

# ---------------------------------------------------------------------------------------------
# Exact rational functions of u = 1/r (M = 1), for the deformation functions of the metric.
# ---------------------------------------------------------------------------------------------
const Q = Rational{BigInt}
const RP = Polynomial{Q}
struct RF
    p::RP
    q::RP
end
Base.:+(a::RF, b::RF) = RF(a.p * b.q + b.p * a.q, a.q * b.q)
Base.:-(a::RF, b::RF) = RF(a.p * b.q - b.p * a.q, a.q * b.q)
Base.:*(a::RF, b::RF) = RF(a.p * b.p, a.q * b.q)
Base.:*(c::Number, a::RF) = RF(Q(c) * a.p, a.q)
lowest(a::RF) = (g = gcd(a.p, a.q); RF(div(a.p, g), div(a.q, g)))
const uu = RP([0, 1])
const ff = 1 - 2uu
"""Derivative d/dr* = Fp d/dr = -u^2 Fp d/du of a rational function, Fp a polynomial in u."""
Dr(a::RF, Fp::RP) = lowest(RF(-uu^2 * Fp * (derivative(a.p) * a.q - a.p * derivative(a.q)), a.q^2))
rf(p::RP) = RF(p, RP([1]))
# Chaichian-Tureanu-Zet deformations, alpha = 2M = 2, written in u
const den = 4 * (1 - 2uu)
const A0 = RF(uu^3 * (4 - 11uu), den)
const B0 = RF(-uu^3 * (2 - 3uu), den)
const C0 = RF(uu^2 * (1 - 17uu + 34uu^2), 4den)
const D0 = RF(-uu^3 * (1 - uu), den)
const r2 = RF(RP([1]), uu^2)
const ir2 = rf(uu^2)
W00of(Fp, X, Y) = rf(2uu^3 * ff) * (A0 - B0) + (1 // 4) * (ir2 * Dr(r2 * Dr(X, Fp), Fp)) - (1 // 4) * Dr(Dr(Y, Fp), Fp)
Ws2of(Fp) = (1 // 4) * (ir2 * Dr(r2 * Dr(ir2, Fp), Fp))
const W00 = lowest(W00of(ff, A0 - B0 + C0 + D0, A0 - B0))
const Ws2 = lowest(Ws2of(ff))
const F2 = lowest((1 // 2) * (rf(ff) * (A0 - B0)))
let P = NCQNM.MODEL_PIECES["ctz"]
    @assert W00.q == RP([1]) && W00.p == RP(P["W00"][1]) "W00 differs from model_pieces.jl"
    @assert F2.q == RP([1]) && F2.p == RP(P["F2"][1]) "F2 differs from model_pieces.jl"
    @assert Ws2.q == RP([1]) && Ws2.p == RP(P["Ws2"][1]) "Ws2 differs from model_pieces.jl"
end
const dWprint = lowest(W00of(ff, A0 + B0 - C0 + D0, A0 + B0) - W00)
tb(p::RP) = Polynomial(T[T(numerator(c)) / T(denominator(c)) for c in coeffs(p)])
println("ctz: W00, Ws2 and F2 rebuilt from the metric agree exactly with model_pieces.jl")
println("printed potential minus ours: Theta^2 x (", dWprint.p, ") / (", dWprint.q, ")")

# ---------------------------------------------------------------------------------------------
# A variant as (series map of u^2 F, series map of V, Vn, Vd, u_h); th2 is an exact rational.
# ---------------------------------------------------------------------------------------------
polyG(F) = (us, K) -> series_compose(Polynomial([zero(T), zero(T), one(T)]) * F, us, K)
ratV(Vn, Vd) = (us, K) -> series_div(series_compose(Vn, us, K), series_compose(Vd, us, K), K)
addrf(Vn, Vd, c, d::RF) = (Vn * tb(d.q) + c * tb(d.p) * Vd, Vd * tb(d.q))
function variant(l, m, th2::Q, kind::Symbol)
    Th2 = T(numerator(th2)) / T(denominator(th2))
    mod = model("ctz", l, m, sqrt(Th2); T=T)
    uh, _ = horizon(mod)
    F, Vn, Vd = mod.F, mod.Vn, mod.Vd
    F2T = tb(F2.p)
    am = abs(m)
    s2 = (2l + 1 - 2am) // (32am)                  # kax16 = 1 for ctz
    if kind == :cct
    elseif kind == :tort1
        F = F + Th2 * F2T
    elseif kind == :tort0
        F = F - Th2 * F2T
    elseif kind == :Vprint || kind == :Vprint1
        Vn, Vd = addrf(Vn, Vd, Th2, dWprint)
        kind == :Vprint1 && (F = F + Th2 * F2T)
    elseif kind == :Dfull || kind == :Dfull1
        Fp = kind == :Dfull ? ff + th2 * F2.p : ff + 2th2 * F2.p
        dV = lowest((W00of(Fp, A0 - B0 + C0 + D0, A0 - B0) - W00) + s2 * (Ws2of(Fp) - Ws2))
        Vn, Vd = addrf(Vn, Vd, Th2, dV)
        kind == :Dfull1 && (F = F + Th2 * F2T)
    elseif kind == :sqrtF
        A, B = A0, B0
        G = (us, K) -> begin
            a = Th2 .* series_div(series_compose(tb(A.p), us, K), series_compose(tb(A.q), us, K), K)
            b = Th2 .* series_div(series_compose(tb(B.p), us, K), series_compose(tb(B.q), us, K), K)
            a[1] += 1
            b[1] += 1
            fu = series_compose(Polynomial(T[0, 0, 1, -2]), us, K)
            series_mul(fu, series_sqrt(series_div(a, b, K), K), K)
        end
        return G, ratV(Vn, Vd), Vn, Vd, uh
    else
        error("unknown variant $kind")
    end
    return polyG(F), ratV(Vn, Vd), Vn, Vd, uh
end
function wkb_variant(l, m, n, th2::Q, kind::Symbol; N=13)
    G, Vs, Vn, Vd, uh = variant(l, m, th2, kind)
    u0, _ = peak_u(Vn, Vd, uh)
    v, _ = rstar_taylor(G, Vs, u0, 2N)
    a, _ = wkb_poly(v, n, N)
    plain = [sqrt(sum(a[1:k+1])) for k in 1:N]
    pade = Dict((nt, N - nt) => sqrt(pade1(a, nt, N - nt)) for nt in 0:N)
    for k in 2:N-1, nt in 0:k
        pade[(nt, k - nt)] = sqrt(pade1(a, nt, k - nt))
    end
    return (; plain, pade)
end

# ---------------------------------------------------------------------------------------------
# Published third-order values (Heidari et al., Tables 1, 2, 3) and accurate frequencies.
# ---------------------------------------------------------------------------------------------
const TH2 = Q[0, 1//5, 2//5, 3//5, 4//5, 1]
heid = Dict(
    (1, 1, 0) => [0.29111 - 0.09800im, 0.29232 - 0.09940im, 0.29346 - 0.10075im, 0.29450 - 0.10199im, 0.29532 - 0.10272im, 0.29641 - 0.10434im],
    (1, 1, 1) => [0.26221 - 0.30743im, 0.26513 - 0.31105im, 0.26754 - 0.31458im, 0.26927 - 0.31782im, 0.26855 - 0.31928im, 0.27144 - 0.32402im],
    (2, 1, 0) => [0.48321 - 0.096805im, 0.48443 - 0.098256im, 0.48559 - 0.099694im, 0.48670 - 0.101110im, 0.48775 - 0.102470im, 0.48877 - 0.103800im],
    (2, 1, 1) => [0.46319 - 0.29581im, 0.46522 - 0.30000im, 0.46699 - 0.30424im, 0.46849 - 0.30848im, 0.46963 - 0.31249im, 0.47051 - 0.31640im],
    (2, 1, 2) => [0.43166 - 0.50343im, 0.43509 - 0.50993im, 0.43795 - 0.51669im, 0.44023 - 0.52356im, 0.44141 - 0.53004im, 0.44188 - 0.53637im],
    (2, 2, 0) => [0.48321 - 0.096805im, 0.48465 - 0.098013im, 0.48606 - 0.099226im, 0.48741 - 0.100390im, 0.48873 - 0.101530im, 0.49002 - 0.102680im],
    (2, 2, 1) => [0.46319 - 0.29581im, 0.46526 - 0.29937im, 0.46714 - 0.30300im, 0.46869 - 0.30642im, 0.47001 - 0.30976im, 0.47121 - 0.31316im],
    (2, 2, 2) => [0.43166 - 0.50343im, 0.43490 - 0.50910im, 0.43776 - 0.51499im, 0.43956 - 0.52044im, 0.44072 - 0.52579im, 0.44164 - 0.53133im])
const KEYS = sort(collect(keys(heid)))
schw = Dict((1, 0) => 0.292936133267283 - 0.097659988913578im, (1, 1) => 0.264448650604833 - 0.306257391559047im,
            (2, 0) => 0.483643872210713 - 0.096758775978288im, (2, 1) => 0.463850579019766 - 0.295603936987963im,
            (2, 2) => 0.430544054376576 - 0.508558402154275im)
"""Accurate frequency by collocation (N = 60 and 70), the eigenvalue nearest to the guess."""
function accurate(l, m, n, th2::Q, guess)
    th2 == 0 && return schw[(l, n)], 0.0
    mod = model("ctz", l, m, sqrt(Float64(th2)))
    w = ComplexF64[]
    for N in (60, 70)
        s = solve_joukowski(mod, N)
        c = s.w[s.phys]
        push!(w, c[argmin(abs.(c .- guess))])
    end
    return w[2], abs(w[2] - w[1])
end

# ---------------------------------------------------------------------------------------------
# 1. Which variant reproduces the published third-order numbers?
# ---------------------------------------------------------------------------------------------
const KINDS = (:cct, :tort1, :tort0, :Vprint, :Vprint1, :Dfull, :Dfull1, :sqrtF)
allrows = String[]
cmprows = String[]
dev = Dict(k => zeros(length(TH2)) for k in KINDS)
best = Dict{Tuple{Int,Int,Int,Q},Any}()
exact = Dict{Tuple{Int,Int,Int,Q},Tuple{ComplexF64,Float64}}()
for key in KEYS, (i, th2) in enumerate(TH2)
    l, m, n = key
    for kind in KINDS
        r = wkb_variant(l, m, n, th2, kind; N=kind == :cct ? 13 : 3)
        w3 = ComplexF64(r.plain[3])
        dev[kind][i] = max(dev[kind][i], abs(real(w3) - real(heid[key][i])), abs(imag(w3) - imag(heid[key][i])))
        push!(allrows, @sprintf("%s,%d,%d,%d,%.4f,wkb3,%.10f,%.10f", kind, l, m, n, Float64(th2), real(w3), imag(w3)))
        if kind == :cct
            best[(l, m, n, th2)] = r
            for (lab, w) in (("wkb6", r.plain[6]), ("wkb13", r.plain[13]), ("pade33", r.pade[(3, 3)]),
                             ("pade66", r.pade[(6, 6)]), ("pade67", r.pade[(6, 7)]), ("pade76", r.pade[(7, 6)]))
                push!(allrows, @sprintf("cct,%d,%d,%d,%.4f,%s,%.10f,%.10f", l, m, n, Float64(th2), lab,
                                        Float64(real(w)), Float64(imag(w))))
            end
            exact[(l, m, n, th2)] = accurate(l, m, n, th2, ComplexF64(r.pade[(6, 7)]))
        end
    end
end
println("\nLargest deviation (Re or Im) of the third-order WKB value of each variant from the")
println("published Tables 1-3 of Heidari et al., over the eight (l, m, n), at each Theta^2:")
@printf("%-8s", "variant")
foreach(th2 -> @printf("  Th2=%-6.1f", Float64(th2)), TH2)
println()
for kind in KINDS
    @printf("%-8s", kind)
    foreach(d -> @printf("  %-10.1e", d), dev[kind])
    println()
end

# ---------------------------------------------------------------------------------------------
# 2. Accuracy of the WKB values against the accurate frequencies, and size of the shifts.
# ---------------------------------------------------------------------------------------------
fmt(w) = @sprintf("%.6f%+.6fi", real(w), imag(w))
println("\nctz model: published WKB3, our WKB3, sixth order with Pade P3/3, 13th order with P6/7,")
println("accurate value (collocation), errors |WKB - accurate| and the shift |w(Theta) - w(0)|")
for key in KEYS
    l, m, n = key
    println("(l, m, n) = $key")
    for (i, th2) in enumerate(TH2)
        r = best[(l, m, n, th2)]
        ex, acc = exact[(l, m, n, th2)]
        w3 = ComplexF64(r.plain[3]); p33 = ComplexF64(r.pade[(3, 3)]); p67 = ComplexF64(r.pade[(6, 7)])
        sh = abs(ex - schw[(l, n)])
        @printf("  Th2=%.1f  pub %s  WKB3 %s  P33 %s  P67 %s  exact %s (+-%.0e)  err: pub %.1e WKB3 %.1e P33 %.1e P67 %.1e  shift %.1e\n",
                Float64(th2), fmt(heid[key][i]), fmt(w3), fmt(p33), fmt(p67), fmt(ex), acc,
                abs(heid[key][i] - ex), abs(w3 - ex), abs(p33 - ex), abs(p67 - ex), sh)
        rD = wkb_variant(l, m, n, th2, :Dfull; N=3)
        push!(cmprows, @sprintf("%d,%d,%d,%.1f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.10f,%.10f,%.1e",
                                l, m, n, Float64(th2), real(heid[key][i]), imag(heid[key][i]),
                                real(w3), imag(w3), Float64(real(rD.plain[3])), Float64(imag(rD.plain[3])),
                                real(p33), imag(p33), real(p67), imag(p67), real(ex), imag(ex), acc))
        push!(allrows, @sprintf("accurate,%d,%d,%d,%.4f,collocation,%.12f,%.12f", l, m, n, Float64(th2), real(ex), imag(ex)))
    end
end

# Reference points: accurate tones by collocation, typed in here (Theta = 0.1, 0.25, 0.5)
println("\nReference points (accurate values typed in), ctz:")
refs = [((1, 1, 0), 0.1, 0.292980685 - 0.097721078im), ((1, 1, 1), 0.1, 0.264525617 - 0.306442989im),
        ((1, 1, 0), 0.25, 0.293211373 - 0.098041021im), ((1, 1, 0), 0.5, 0.293994252 - 0.099172782im),
        ((1, 1, 1), 0.5, 0.266089779 - 0.310857250im), ((2, 2, 0), 0.5, 0.485334084 - 0.098236978im),
        ((2, 1, 0), 0.5, 0.485050502 - 0.098558200im)]
for (key, Th, ref) in refs
    l, m, n = key
    th2 = rationalize(BigInt, Th)^2
    r = wkb_variant(l, m, n, th2, :cct; N=13)
    w3 = ComplexF64(r.plain[3]); w6 = ComplexF64(r.plain[6]); p33 = ComplexF64(r.pade[(3, 3)])
    p66 = ComplexF64(r.pade[(6, 6)]); p67 = ComplexF64(r.pade[(6, 7)])
    @printf("  %s Theta=%.2f  exact %s | WKB3 %.1e  WKB6 %.1e  P33 %.1e  P66 %.1e  P67 %.1e | shift %.1e\n",
            key, Th, fmt(ref), abs(w3 - ref), abs(w6 - ref), abs(p33 - ref), abs(p66 - ref), abs(p67 - ref),
            abs(ref - schw[(l, n)]))
    for (lab, w) in (("wkb3", w3), ("wkb6", w6), ("pade33", p33), ("pade66", p66), ("pade67", p67))
        push!(allrows, @sprintf("cct,%d,%d,%d,%.4f,%s,%.10f,%.10f", l, m, n, Th^2, lab, real(w), imag(w)))
    end
    push!(allrows, @sprintf("accurate,%d,%d,%d,%.4f,supplied,%.12f,%.12f", l, m, n, Th^2, real(ref), imag(ref)))
end

# Zhao et al., Table II, Theta = 0 (Schwarzschild, l = 1, n = 0): "0.306 - 0.101i" with P66
let r = best[(1, 1, 0, Q(0))], z = 0.306 - 0.101im
    d = [(abs(ComplexF64(w) - z), k) for (k, w) in r.pade]
    dm, km = minimum(d)
    @printf("\nZhao et al., Theta = 0: published %s; P6/6 gives %s (difference %.1e);\n", fmt(z),
            fmt(ComplexF64(r.pade[(6, 6)])), abs(ComplexF64(r.pade[(6, 6)]) - z))
    @printf("closest of all Pade approximants of orders up to 13: P%d/%d, still %.1e away\n", km[1], km[2], dm)
end

open(joinpath(OUT, "ctz_wkb.csv"), "w") do io
    println(io, "# ctz scalar quasinormal frequencies, M = 1: WKB values for the variants described in")
    println(io, "# julia/scripts/wkb_ctz_heidari.jl (wkb3 for every variant; wkb6, wkb13 and Pade values for cct)")
    println(io, "# and accurate values (collocation N = 70, or supplied: typed in the script).")
    println(io, "variant,l,m,n,Theta2,approximation,re,im")
    foreach(r -> println(io, r), allrows)
end
open(joinpath(OUT, "ctz_heidari_comparison.csv"), "w") do io
    println(io, "# Heidari et al. (2024) Tables 1-3 against: wkb3 of the cct model, wkb3 of the Dfull variant,")
    println(io, "# sixth order with Pade P3/3 and 13th order with P6/7 (cct), accurate value (collocation, N = 70,")
    println(io, "# with |w(70) - w(60)| as acc).  M = 1.")
    println(io, "l,m,n,Theta2,pub_re,pub_im,wkb3_re,wkb3_im,wkb3Dfull_re,wkb3Dfull_im,p33_re,p33_im,p67_re,p67_im,exact_re,exact_im,acc")
    foreach(r -> println(io, r), cmprows)
end
println("\nwritten ", joinpath(OUT, "ctz_wkb.csv"), " and ctz_heidari_comparison.csv")
