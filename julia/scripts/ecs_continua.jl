# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# The two continua of the scaled operator and how they move with the angles.  The essential
# spectrum is the union of the ray lambda = e^{-2 i th_p} tau from the threshold 0 (far side) and
# the ray lambda = V_h + e^{-2 i th_m} tau from the threshold V_h (horizon side), tau ⩾ 0; on the
# truncated contour each becomes a string of eigenvalues along its ray.  The quasinormal
# frequencies stay where they are and are uncovered once the rays have passed them.  Every
# eigenvalue with |lambda| < 0.6 is classified as a certified frequency or by its nearest ray, with
# its distance to that ray.
#
# Output: data/scaling/continua.csv.
include(joinpath(@__DIR__, "..", "ECS.jl"))
using .ECS, Printf, LinearAlgebra

setprecision(BigFloat, 256)
const OUT = joinpath(@__DIR__, "..", "..", "data", "scaling")
mkpath(OUT)

"""Distance from lambda to the ray c + e^{i phi} [0, oo)."""
function ray_distance(lam, c, phi)
    v = lam - c
    tau = max(0.0, real(v * cis(-phi)))
    return abs(v - cis(phi) * tau)
end

angles = [(0.3, 0.3), (0.6, 0.6), (0.9, 0.9), (1.2, 1.2), (0.5, 1.1), (1.1, 0.5)]
cases = [("schwarzschild", 1, 0, 0 // 1), ("swc", 1, 1, 1 // 2), ("jkp_rth", 1, 1, 1 // 2)]
rows = ["model,l,m,Theta,Vh,thm,thp,Re_lambda,Im_lambda,Re_w,Im_w,class,dist_ray"]
for (name, l, m, Th) in cases
    S = setup(name, l, m, Th)
    Vh = Float64(S.Vh)
    cert = certify(S; th=1.35, T=240.0, N=36, wmax=1.0).lambda
    @printf("%s (%d,%d) Theta = %s   V_h = %.6f   thresholds 0 and V_h\n", name, l, m, string(Th), Vh)
    for (thm, thp) in angles
        C = ecs_contour(aL=safe_aL(S), aR=5.0, thm=thm, thp=thp, TL=240.0, TR=240.0, N=36)
        lam = spectrum(S, C)
        nq, df_max, dh_max, nf, nh = 0, 0.0, 0.0, 0, 0
        for x in lam
            abs(x) < 0.6 || continue
            df = ray_distance(x, 0.0, -2thp)
            dh = ray_distance(x, Vh, -2thm)
            if any(abs.(cert .- x) .< 1e-8)
                cls, d = "qnm", min(df, dh)
                nq += 1
            elseif df <= dh
                cls, d = "far", df
                abs(x) < 0.3 && (df_max = max(df_max, df); nf += 1)
            else
                cls, d = "horizon", dh
                abs(x) < 0.3 && (dh_max = max(dh_max, dh); nh += 1)
            end
            w = omega_of(x)
            push!(rows, @sprintf("%s,%d,%d,%.4f,%.10f,%.2f,%.2f,%.12e,%.12e,%.12e,%.12e,%s,%.3e", name, l,
                                 m, Float64(Th), Vh, thm, thp, real(x), imag(x), real(w), imag(w), cls, d))
        end
        @printf("   thm = %.1f thp = %.1f: %2d frequencies uncovered; |lambda| < 0.3: %3d on the far ray (max distance %.1e), %3d on the horizon ray (max distance %.1e)\n",
                thm, thp, nq, nf, df_max, nh, dh_max)
    end
end
open(io -> foreach(r -> println(io, r), rows), joinpath(OUT, "continua.csv"), "w")
println("written continua.csv in ", OUT)
