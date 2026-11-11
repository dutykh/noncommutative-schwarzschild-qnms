# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# The model "jkp_rth" as a case study.  Its potential has a simple pole at r = 2M, inside the
# horizon, whose images in the tortoise plane lie at z = X + i pi A_h (2n + 1), and on the real
# axis it is a plateau V_h ~ -1/32 for z < X, a step near z = X, then the barrier near z = 0;
# X ~ 4 ln(Theta) - 6.5 moves away as Theta decreases.
#
# 1. The horizon-side ray must leave the real axis to the left of the poles: when the wedge it
#    sweeps contains a pole, the eigenvalues jump to another sheet, independent of the angle.
# 2. The certified frequencies along a logarithmic grid 0.01 ⩽ Theta ⩽ 0.5, the continuation
#    of the fundamental tone, and the least damped frequency.
# 3. The new family between the step and the barrier: consecutive members obey a quantisation
#    w_{n+1} - w_n ~ pi / Lambda with a complex length Lambda, compared with the distance from the
#    poles to the singular point r = 0 of the barrier.
#
# Output: data/scaling/jkp_rth_pole_crossing.csv, jkp_rth_tracks.csv, jkp_rth_family.csv.
include(joinpath(@__DIR__, "..", "ECS.jl"))
using .ECS, Printf, LinearAlgebra

setprecision(BigFloat, 256)
const OUT = joinpath(@__DIR__, "..", "..", "data", "scaling")
mkpath(OUT)

# ---------------------------------------------------------------------------------------------
# 1. Sweeping across a pole
# ---------------------------------------------------------------------------------------------
S = setup("jkp_rth", 1, 1, 1 // 2)
X = pole_abscissa(S)
Yp = abs(Float64(imag(pole_images(S; nmax=0)[1][2])))   # height of the first pole above the axis
wref = certify(S; th=1.3).w[1]
rows = ["aL,thm,wedge_contains_pole,Re_w,Im_w,dist_to_fundamental"]
println("Sweeping across the pole at z = $(round(X, digits=3)) -/+ $(round(Yp, digits=3)) i:")
for aL in (4.0, 6.0), thm in 0.30:0.05:1.40
    lam = spectrum(S, ecs_contour(aL=aL, aR=5.0, thm=thm, thp=1.0, TL=200.0, TR=200.0, N=28))
    w = omega_of.(lam)
    i = argmin(abs.(w .- wref))
    inside = (-X - aL) * tan(thm) > Yp
    push!(rows, @sprintf("%.1f,%.2f,%d,%.12f,%.12f,%.3e", aL, thm, inside, real(w[i]), imag(w[i]),
                         abs(w[i] - wref)))
    @printf("  aL = %.0f  thm = %.2f  pole swept: %-5s  w = %.10f %+.10fi   |w - w0| = %.1e\n", aL,
            thm, inside, real(w[i]), imag(w[i]), abs(w[i] - wref))
end
open(io -> foreach(r -> println(io, r), rows), joinpath(OUT, "jkp_rth_pole_crossing.csv"), "w")

# ---------------------------------------------------------------------------------------------
# 2. Following the frequencies as Theta decreases
# ---------------------------------------------------------------------------------------------
grid(n) = [round(Int, 1e6 * 0.5 * 0.02^(k / (n - 1))) // 1_000_000 for k in 0:n-1]
trows = ["l,m,Theta,X_pole,Re_w,Im_w,spread,fundamental,least_damped"]
frows = ["l,m,Theta,X_pole,members,Re_Lambda,Im_Lambda,Re_Lambda_pred,Im_Lambda_pred"]
schw = Dict(1 => 0.292936133267283 - 0.097659988913578im, 2 => 0.483643872210713 - 0.096758775978288im)
for (l, m, n) in ((1, 1, 41), (2, 2, 21), (2, 1, 21))
    wf = nothing
    println("jkp_rth ($l,$m): the fundamental tone and the least damped frequency as Theta decreases")
    for Th in grid(n)
        St = setup("jkp_rth", l, m, Th)
        Xt = pole_abscissa(St)
        c = certify(St; th=1.3, T=220.0, N=28, wmax=l == 1 ? 1.0 : 1.3)
        isempty(c.w) && continue
        if wf === nothing
            wf = c.w[argmin(abs.(c.w .- schw[l]))]
        else
            j = argmin(abs.(c.w .- wf))
            abs(c.w[j] - wf) > 0.05 && @printf("   (jump of %.3f in the continuation)\n", abs(c.w[j] - wf))
            wf = c.w[j]
        end
        jl = argmax(imag.(c.w))
        for (k, w) in enumerate(c.w)
            push!(trows, @sprintf("%d,%d,%.6f,%.6f,%.14f,%.14f,%.2e,%d,%d", l, m, Float64(Th), Xt,
                                  real(w), imag(w), c.spread[k], w == wf, k == jl))
        end
        @printf("   Theta = %.4f  X = %7.3f  fundamental %.8f %+.8fi (|w - w_Schw| = %.2e)  least damped %.8f %+.8fi\n",
                Float64(Th), Xt, real(wf), imag(wf), abs(wf - schw[l]), real(c.w[jl]), imag(c.w[jl]))
        # 3. the family: certified frequencies other than the fundamental with 0.3 < Re w < 0.85,
        #    consecutive members in order of Re w
        fam = sort(filter(w -> w != wf && 0.3 < real(w) < 0.85, c.w), by=real)
        if length(fam) >= 3
            d = diff(fam)
            Lam = pi / (sum(d) / length(d))
            # A wave running towards the horizon is reflected at the step with an amplitude
            # ~ exp(-2 i w z1), z1 the lower image of the pole (the integral of V' exp(-2 i w z)
            # closes in the lower half plane); a wave running outwards is reflected by the barrier
            # with ~ exp(2 i w z2), z2 the upper image of r = 0.  The round trip
            # exp(2 i w (z2 - z1)) = const gives w_{n+1} - w_n = pi / (z2 - z1).
            zpole = ComplexF64(pole_images(St; nmax=1)[1][2])          # n = -1: lower image
            zr0 = ComplexF64(zofr(St, big"1e-40"; n=0))                 # r -> 0 above the horizon
            pred = zr0 - zpole
            push!(frows, @sprintf("%d,%d,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f", l, m, Float64(Th), Xt,
                                  length(fam), real(Lam), imag(Lam), real(pred), imag(pred)))
        end
    end
end
open(io -> foreach(r -> println(io, r), trows), joinpath(OUT, "jkp_rth_tracks.csv"), "w")
open(io -> foreach(r -> println(io, r), frows), joinpath(OUT, "jkp_rth_family.csv"), "w")
println("written jkp_rth_pole_crossing.csv, jkp_rth_tracks.csv, jkp_rth_family.csv in ", OUT)
