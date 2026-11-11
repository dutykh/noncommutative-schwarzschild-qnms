# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Validation of the higher-order WKB formula (wkb.jl) on three test problems.
#  1. Poeschl-Teller barrier V = V0 / cosh^2(b x), whose quasinormal frequencies are
#     w = b ( sqrt(V0/b^2 - 1/4) - i (n + 1/2) ); the WKB series is then the Taylor series of the
#     exact w^2 in the order parameter and converges.
#  2. The explicit third-order formula of Iyer and Will against the recursion.
#  3. Schwarzschild scalar field, V = (1 - 2/r)(l(l+1)/r^2 + 2/r^3), M = 1: the published values of
#     Konoplya, Zhidenko and Zinhailo, Class. Quantum Grav. 36 (2019) 155002, Table III (l = 1,
#     orders 2 to 12, units 2M = 1) and of Matyjasek and Opala, Phys. Rev. D 96 (2017) 024011,
#     Table II (sixth order, Pade P_{6/6} and P_{6/7}, M = 1), and the accurate frequencies of
#     Leaver's continued fraction (leaver.jl).
# Output: data/wkb/validation.csv
include(joinpath(@__DIR__, "..", "NCQNM.jl"))
include(joinpath(@__DIR__, "..", "leaver.jl"))
include(joinpath(@__DIR__, "..", "wkb.jl"))
using .NCQNM, .Leaver, .WKB, Polynomials, Printf

setprecision(BigFloat, 256)
const T = BigFloat
const OUT = joinpath(@__DIR__, "..", "..", "data", "wkb")
mkpath(OUT)
rows = String[]
row(test, l, n, q, w, ref) = push!(rows, @sprintf("%s,%d,%d,%s,%.12f,%.12f,%.12f,%.12f,%.3e", test, l, n, q,
                                                    Float64(real(w)), Float64(imag(w)), real(ref), imag(ref),
                                                    Float64(abs(ComplexF64(w) - ref))))
fmt(w) = @sprintf("%.9f %+.9fi", Float64(real(w)), Float64(imag(w)))

# 1. Poeschl-Teller, V0 = 1, b = 1/2
println("1. Poeschl-Teller barrier V = 1/cosh^2(x/2): error of the WKB formula of order N")
let V0 = T(1), b = T(1) / 2, K = 26
    ch = zeros(T, K + 1)
    for k in 0:2:K
        ch[k+1] = b^k / factorial(big(k))
    end
    v = series_div(vcat(V0, zeros(T, K)), series_mul(ch, ch, K), K)
    for n in 0:2
        a, E = wkb_poly(v, n, 13)
        ex = b * (sqrt(V0 / b^2 - T(1) / 4) - im * (n + T(1) / 2))
        errs = [abs(sqrt(sum(a[1:N+1])) - ex) for N in 1:13]
        @printf("   n=%d  N=1: %.1e  N=3: %.1e  N=6: %.1e  N=9: %.1e  N=13: %.1e   (odd E_m: %.0e)\n", n,
                Float64(errs[1]), Float64(errs[3]), Float64(errs[6]), Float64(errs[9]), Float64(errs[13]),
                Float64(maximum(abs.(E[2:2:end]))))
        for N in 1:13
            row("poeschl_teller", -1, n, "order$N", sqrt(sum(a[1:N+1])), ComplexF64(ex))
        end
    end
end

# 3. Schwarzschild: accurate values by Leaver's continued fraction
guesses = Dict(0 => [0.1105 - 0.1049im, 0.0861 - 0.3481im],
               1 => [0.2929 - 0.0977im, 0.2645 - 0.3063im, 0.2295 - 0.5401im, 0.2033 - 0.7883im],
               2 => [0.4836 - 0.0968im, 0.4639 - 0.2956im, 0.4305 - 0.5086im, 0.3939 - 0.7329im])
leaver = Dict{Tuple{Int,Int},ComplexF64}()
for l in 0:2, (i, g) in enumerate(guesses[l])
    leaver[(l, i - 1)] = schwarzschild_qnm(l, i - 1, ComplexF64(g); J=200_000)
end

# Konoplya-Zhidenko-Zinhailo Table III, l = 1, orders k = 2..12, units 2M = 1
kzz = Dict(0 => [0.589020 - 0.215341im, 0.582228 - 0.196003im, 0.585907 - 0.194772im, 0.586124 - 0.195422im,
                 0.585819 - 0.195523im, 0.585746 - 0.195305im, 0.585862 - 0.195267im, 0.585867 - 0.195284im,
                 0.585942 - 0.195259im, 0.586003 - 0.195441im, 0.585669 - 0.195552im],
           1 => [0.576605 - 0.659933im, 0.524424 - 0.614865im, 0.527253 - 0.611565im, 0.528949 - 0.613028im,
                 0.528942 - 0.613036im, 0.528634 - 0.612771im, 0.528827 - 0.612547im, 0.528642 - 0.612387im,
                 0.529018 - 0.611952im, 0.529876 - 0.612694im, 0.528618 - 0.614152im],
           2 => [0.570510 - 1.11164im, 0.447086 - 1.05363im, 0.434729 - 1.08358im, 0.458264 - 1.09324im,
                 0.462028 - 1.08433im, 0.463192 - 1.08483im, 0.466558 - 1.07700im, 0.456354 - 1.07262im,
                 0.453724 - 1.07884im, 0.459693 - 1.08136im, 0.457483 - 1.08658im],
           3 => [NaN + 0im, 0.347404 - 1.49726im, 0.320285 - 1.62404im, 0.428444 - 1.64878im,
                 0.444189 - 1.59034im, 0.461173 - 1.59516im, 0.494969 - 1.48625im, 0.292972 - 1.43170im,
                 0.254679 - 1.64697im, 0.379148 - 1.67075im, 0.360239 - 1.75845im])
# Matyjasek-Opala Table II (M = 1): Pade P_{6/6}, P_{6/7}, sixth-order WKB
mo = Dict((0, 0) => (0.111328786 - 0.104885261im, 0.110346431 - 0.104924037im, 0.110467018 - 0.100816251im),
          (0, 1) => (0.0869872660 - 0.347994694im, 0.0874777581 - 0.347875503im, 0.0890291001 - 0.344528856im),
          (1, 0) => (0.292936154 - 0.0976600269im, 0.292936143 - 0.0976599978im, 0.292909644 - 0.0977616179im),
          (1, 1) => (0.264443075 - 0.306258587im, 0.264446919 - 0.306257973im, 0.264471051 - 0.306518241im),
          (1, 2) => (0.229227121 - 0.540073930im, 0.229216213 - 0.540091386im, 0.231014233 - 0.542165408im),
          (1, 3) => (0.202925543 - 0.788885062im, 0.202747941 - 0.788724995im, 0.222094287 - 0.795168920im),
          (2, 0) => (0.483643872 - 0.0967587762im, 0.483643872 - 0.0967587759im, 0.483641882 - 0.0967660891im),
          (2, 1) => (0.463850543 - 0.295603953im, 0.463850565 - 0.295603954im, 0.463846744 - 0.295626915im),
          (2, 2) => (0.430543707 - 0.508556964im, 0.430544040 - 0.508556573im, 0.430385788 - 0.508699895im),
          (2, 3) => (0.393853836 - 0.738081384im, 0.393855027 - 0.738098967im, 0.393206798 - 0.739885188im))

println("\n2-3. Schwarzschild scalar field (M = 1)")
maxkzz = 0.0
maxmo = 0.0
maxiw = 0.0
for l in 0:2, n in 0:3
    haskey(mo, (l, n)) || continue
    mod = schwarzschild(l; T=T)
    uh, _ = horizon(mod)
    r = wkb_omegas(mod.F, mod.Vn, mod.Vd, uh, n; N=13)
    iw, _, _ = iyer_will3(r.v, n)
    global maxiw = max(maxiw, Float64(abs(sqrt(iw) - r.plain[3])))
    ex = haskey(leaver, (l, n)) ? leaver[(l, n)] : NaN + 0im
    p66, p67, w6 = mo[(l, n)]
    global maxmo = max(maxmo, abs(ComplexF64(r.pade[(6, 6)]) - p66), abs(ComplexF64(r.pade[(6, 7)]) - p67),
                       abs(ComplexF64(r.plain[6]) - w6))
    row("schwarzschild_MO", l, n, "pade66", r.pade[(6, 6)], p66)
    row("schwarzschild_MO", l, n, "pade67", r.pade[(6, 7)], p67)
    row("schwarzschild_MO", l, n, "order6", r.plain[6], w6)
    if l == 1
        for k in 2:12
            isnan(real(kzz[n][k-1])) && continue
            dev = abs(2 * ComplexF64(r.plain[k]) - kzz[n][k-1])
            global maxkzz = max(maxkzz, dev)
            row("schwarzschild_KZZ_2M=1", l, n, "order$k", 2 * r.plain[k], kzz[n][k-1])
        end
    end
    if !isnan(real(ex))
        for k in 1:13
            row("schwarzschild_leaver", l, n, "order$k", r.plain[k], ex)
        end
        for (nt, mt) in ((2, 1), (1, 2), (3, 3), (4, 2), (2, 4), (6, 6), (6, 7), (7, 6))
            row("schwarzschild_leaver", l, n, "pade$(nt)$(mt)", r.pade[(nt, mt)], ex)
        end
    end
    @printf("   l=%d n=%d  exact %s\n", l, n, isnan(real(ex)) ? "(not computed)" : fmt(ex))
    for (lab, w) in (("WKB3", r.plain[3]), ("WKB6", r.plain[6]), ("P3/3", r.pade[(3, 3)]),
                     ("P6/6", r.pade[(6, 6)]), ("P6/7", r.pade[(6, 7)]))
        @printf("        %-5s %s   error %.1e\n", lab, fmt(w), isnan(real(ex)) ? NaN : Float64(abs(w - ex)))
    end
end
@printf("\n   explicit Iyer-Will third order vs recursion: max difference %.1e\n", maxiw)
@printf("   KZZ Table III (orders 2..12, 6 decimals): max deviation %.1e\n", maxkzz)
@printf("   Matyjasek-Opala Table II (WKB6, P66, P67, 9-10 decimals): max deviation %.1e\n", maxmo)

open(joinpath(OUT, "validation.csv"), "w") do io
    println(io, "# WKB validation: test, l (-1 for Poeschl-Teller), overtone n, approximation, value (Re, Im),")
    println(io, "# reference (Re, Im) and absolute difference.  Units M = 1 except the KZZ rows (2M = 1).")
    println(io, "test,l,n,approximation,re,im,ref_re,ref_im,absdiff")
    foreach(r -> println(io, r), rows)
end
println("written ", joinpath(OUT, "validation.csv"))
