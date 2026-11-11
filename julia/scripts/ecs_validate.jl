# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Exterior complex scaling in the tortoise coordinate, checked against known frequencies.
# Schwarzschild l = 1, 2 (n = 0, 1, 2) against Leaver's continued fraction, with the dependence on
# the angle, the resolution, the real segment and the truncation; then the deformed models "ctz"
# (1,1), (2,2) at Theta = 0.1, 0.5, 1 and "jkp_tth" (1,1) at Theta = 0.5 against the reference
# values of the collocation method.
#
# Output: data/scaling/validation.csv and data/scaling/schwarzschild_stability.csv.
include(joinpath(@__DIR__, "..", "ECS.jl"))
include(joinpath(@__DIR__, "..", "leaver.jl"))
using .ECS, .Leaver, Printf, LinearAlgebra

setprecision(BigFloat, 256)
const OUT = joinpath(@__DIR__, "..", "..", "data", "scaling")
mkpath(OUT)

digits_of(w, r) = -log10(abs(w - r) / abs(r))

# ---------------------------------------------------------------------------------------------
# Schwarzschild references by Leaver's continued fraction
# ---------------------------------------------------------------------------------------------
guesses = Dict(1 => [0.2929 - 0.0977im, 0.2645 - 0.3063im, 0.2295 - 0.5401im],
               2 => [0.4836 - 0.0968im, 0.4639 - 0.2956im, 0.4305 - 0.5086im])
leaver = Dict{Int,Vector{ComplexF64}}()
for l in (1, 2)
    leaver[l] = ComplexF64[]
    for (i, g) in enumerate(guesses[l])
        w1 = schwarzschild_qnm(l, i - 1, Complex{BigFloat}(g); J=40_000)
        w2 = schwarzschild_qnm(l, i - 1, Complex{BigFloat}(g); J=80_000)
        push!(leaver[l], ComplexF64(w2))
        @printf("Leaver l=%d n=%d  Mw = %.15f %+.15fi  (depth change %.1e)\n", l, i - 1,
                Float64(real(w2)), Float64(imag(w2)), Float64(abs(w2 - w1)))
    end
end

rows = String[]
push!(rows, "model,l,m,Theta,n,Re_w,Im_w,spread,ref_Re,ref_Im,abs_err,digits,ref_source")

nearest(ws, r) = argmin(abs.(ws .- r))

for l in (1, 2)
    S = setup("schwarzschild", l)
    c = certify(S; th=1.35, T=240.0, N=36)
    println("Schwarzschild l=$l: certified tones (least damped first)")
    for (n, r) in enumerate(leaver[l])
        i = nearest(c.w, r)
        w = c.w[i]
        @printf("  n=%d  w = %.15f %+.15fi  spread %.1e  |w - Leaver| = %.1e  (%.1f digits)\n",
                n - 1, real(w), imag(w), c.spread[i], abs(w - r), digits_of(w, r))
        push!(rows, @sprintf("schwarzschild,%d,0,0,%d,%.15f,%.15f,%.2e,%.15f,%.15f,%.2e,%.2f,Leaver",
                             l, n - 1, real(w), imag(w), c.spread[i], real(r), imag(r),
                             abs(w - r), digits_of(w, r)))
    end
end

# ---------------------------------------------------------------------------------------------
# Stability of the Schwarzschild l = 1 tones: angle, resolution, segment, truncation
# ---------------------------------------------------------------------------------------------
stab = String["scan,param,l,n,Re_w,Im_w,abs_err,revealed"]
S1 = setup("schwarzschild", 1)
function record!(scan, param, C)
    lam = spectrum(S1, C)
    w = omega_of.(lam)
    for (n, r) in enumerate(leaver[1])
        i = nearest(w, r)
        rev = angle(r) > -min(C.thm, C.thp)
        push!(stab, @sprintf("%s,%.4f,1,%d,%.15f,%.15f,%.3e,%d", scan, param, n - 1,
                             real(w[i]), imag(w[i]), abs(w[i] - r), rev))
    end
end
for th in 0.70:0.05:1.45
    record!("angle", th, ecs_contour(aL=5.0, aR=5.0, thm=th, thp=th, TL=240.0, TR=240.0, N=36))
end
for N in 12:4:44
    record!("degree", N, ecs_contour(aL=5.0, aR=5.0, thm=1.35, thp=1.35, TL=240.0, TR=240.0, N=N))
end
for a in (2.5, 4.0, 5.0, 8.0, 12.0, 16.0, 20.0)
    record!("segment", a, ecs_contour(aL=a, aR=a, thm=1.35, thp=1.35, TL=240.0, TR=240.0, N=36))
end
for T in (40.0, 60.0, 80.0, 100.0, 140.0, 180.0, 220.0, 260.0)
    record!("truncation", T, ecs_contour(aL=5.0, aR=5.0, thm=1.35, thp=1.35, TL=T, TR=T, N=36))
end
# different angles at the two ends
for (tm, tp) in ((0.9, 1.4), (1.4, 0.9), (1.2, 1.45), (1.45, 1.2))
    record!(@sprintf("angles_%.2f_%.2f", tm, tp), tm, ecs_contour(aL=5.0, aR=5.0, thm=tm, thp=tp,
                                                               TL=240.0, TR=240.0, N=36))
end
open(joinpath(OUT, "schwarzschild_stability.csv"), "w") do io
    foreach(r -> println(io, r), stab)
end
println("Stability of the l = 1 tones (error against Leaver):")
for scan in ("angle", "degree", "segment", "truncation")
    println("  $scan:")
    for r in filter(x -> startswith(x, scan * ","), stab)
        f = split(r, ",")
        f[4] == "0" && print(@sprintf("    %8s", f[2]))
        print(@sprintf("  n=%s %.1e%s", f[4], parse(Float64, f[7]), f[8] == "1" ? "" : "(hidden)"))
        f[4] == "2" && println()
    end
end

# ---------------------------------------------------------------------------------------------
# The deformed models against the collocation references
# ---------------------------------------------------------------------------------------------
refs = [("ctz", 1, 1, 1 // 10, [0.292980685 - 0.097721078im, 0.264525617 - 0.306442989im]),
        ("ctz", 1, 1, 1 // 2, [0.293994252 - 0.099172782im, 0.266089779 - 0.310857250im]),
        ("ctz", 1, 1, 1 // 1, ComplexF64[]),
        ("ctz", 2, 2, 1 // 10, ComplexF64[]),
        ("ctz", 2, 2, 1 // 2, [0.485334084 - 0.098236978im]),
        ("ctz", 2, 2, 1 // 1, ComplexF64[]),
        ("jkp_tth", 1, 1, 1 // 2, [0.292867760 - 0.097586697im])]
for (name, l, m, Th, rr) in refs
    S = setup(name, l, m, Th)
    c = certify(S; th=1.35, T=240.0, N=36, wmax=1.0)
    @printf("%s (%d,%d) Theta = %s  V_h = %.8f\n", name, l, m, string(Th), Float64(S.Vh))
    for (n, w) in enumerate(c.w[1:min(3, end)])
        j = isempty(rr) ? 0 : argmin(abs.(rr .- w))
        if j > 0 && abs(rr[j] - w) < 1e-4
            r = rr[j]
            @printf("  n=%d  w = %.12f %+.12fi  spread %.1e  ref diff %.1e (%.1f digits)\n", n - 1,
                    real(w), imag(w), c.spread[n], abs(w - r), digits_of(w, r))
            push!(rows, @sprintf("%s,%d,%d,%s,%d,%.15f,%.15f,%.2e,%.9f,%.9f,%.2e,%.2f,collocation",
                                 name, l, m, string(Float64(Th)), n - 1, real(w), imag(w),
                                 c.spread[n], real(r), imag(r), abs(w - r), digits_of(w, r)))
        else
            @printf("  n=%d  w = %.12f %+.12fi  spread %.1e\n", n - 1, real(w), imag(w), c.spread[n])
            push!(rows, @sprintf("%s,%d,%d,%s,%d,%.15f,%.15f,%.2e,,,,,", name, l, m,
                                 string(Float64(Th)), n - 1, real(w), imag(w), c.spread[n]))
        end
    end
end
open(joinpath(OUT, "validation.csv"), "w") do io
    foreach(r -> println(io, r), rows)
end
println("written ", joinpath(OUT, "validation.csv"), " and schwarzschild_stability.csv")
