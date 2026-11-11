# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Quasinormal frequencies of the deformed Schwarzschild models by exterior complex scaling in the
# tortoise coordinate.  For every case, the frequencies independent of the contour (angles,
# resolution, real segment, truncation and, when V has poles, a contour passing above them) inside
# |w| < wmax and inside the sector revealed by the angles, compared with the reference values of
# the collocation method where they exist; and the whole spectrum of the base contour (the two
# rotated continua with the quasinormal frequencies), for the maps.
#
# Output: data/scaling/modes.csv, data/scaling/spectra.csv.
include(joinpath(@__DIR__, "..", "ECS.jl"))
using .ECS, Printf, LinearAlgebra

setprecision(BigFloat, 256)
const OUT = joinpath(@__DIR__, "..", "..", "data", "scaling")
mkpath(OUT)

const TH = 1.35         # rotation angle of both rays for the base contour
const NDEG = 36         # Chebyshev degree of each piece
const TRUNC = 240.0     # length of each ray

# reference values (collocation with the Joukowski map), w in units M = 1
refs = Dict(("swc", 1, 1, 1 // 2) => [0.2928268754 - 0.0990516347im],
            ("swc", 1, 1, 1 // 4) => [0.2929159992 - 0.0980117193im],
            ("swc", 2, 2, 1 // 2) => [0.4835721421 - 0.0981167760im, 0.4640571004 - 0.2997518205im],
            ("swc", 2, 1, 1 // 2) => [0.4825060395 - 0.0983904184im],
            ("ctz", 1, 1, 1 // 10) => [0.292980685 - 0.097721078im, 0.264525617 - 0.306442989im],
            ("ctz", 1, 1, 1 // 2) => [0.293994252 - 0.099172782im, 0.266089779 - 0.310857250im],
            ("ctz", 2, 2, 1 // 2) => [0.485334084 - 0.098236978im],
            ("jkp_tth", 1, 1, 1 // 2) => [0.292867760 - 0.097586697im],
            ("jkp_rth", 1, 1, 1 // 2) => [0.292365174 - 0.096848229im],
            ("jkp_rth", 2, 2, 1 // 2) => [0.485279618 - 0.099957259im],
            ("jkp_rth", 1, 1, 1 // 10) => [0.304331924 - 0.097865003im, 0.198546716 - 0.099826508im],
            ("jkp_rth", 1, 1, 1 // 20) => [0.171798021 - 0.076535121im, 0.298164043 - 0.084756713im])

cases = Tuple{String,Int,Int,Rational{Int}}[]
for (l, m) in ((1, 1), (2, 2), (2, 1)), Th in (1 // 2, 1 // 4, 1 // 10)
    push!(cases, ("swc", l, m, Th))
end
for (l, m) in ((1, 1), (2, 2)), Th in (1 // 10, 1 // 2, 1 // 1)
    push!(cases, ("ctz", l, m, Th))
end
push!(cases, ("jkp_tth", 1, 1, 1 // 2))
for (l, m) in ((1, 1), (2, 2), (2, 1)), Th in (1 // 2, 1 // 4, 1 // 10, 1 // 20)
    push!(cases, ("jkp_rth", l, m, Th))
end

mrows = ["model,l,m,Theta,Vh,X_pole,index,Re_w,Im_w,spread,Re_lambda,Im_lambda,ref_Re,ref_Im,abs_err,digits"]
srows = ["model,l,m,Theta,thm,thp,Re_lambda,Im_lambda,Re_w,Im_w,certified"]
for (name, l, m, Th) in cases
    t0 = time()
    S = setup(name, l, m, Th)
    X = pole_abscissa(S)
    wmax = l == 1 ? 1.0 : 1.3
    c = certify(S; th=TH, T=TRUNC, N=NDEG, wmax=wmax)
    Vh = Float64(S.Vh)
    @printf("%s (%d,%d) Theta = %s   V_h = %.8f%s   %d certified frequencies (%.0f s)\n", name, l, m,
            string(Th), Vh, X === nothing ? "" : @sprintf("   poles at Re z = %.3f", X),
            length(c.w), time() - t0)
    rr = get(refs, (name, l, m, Th), ComplexF64[])
    for (k, w) in enumerate(c.w)
        lam = c.lambda[k]
        j = isempty(rr) ? 0 : argmin(abs.(rr .- w))
        refcols = ",,,"
        if j > 0 && abs(rr[j] - w) < 1e-5
            err = abs(rr[j] - w)
            dg = -log10(err / abs(rr[j]))
            refcols = @sprintf("%.10f,%.10f,%.2e,%.2f", real(rr[j]), imag(rr[j]), err, dg)
            @printf("   %2d  w = %.12f %+.12fi   spread %.1e   reference %.10f %+.10fi   diff %.1e\n",
                    k - 1, real(w), imag(w), c.spread[k], real(rr[j]), imag(rr[j]), err)
        else
            @printf("   %2d  w = %.12f %+.12fi   spread %.1e\n", k - 1, real(w), imag(w), c.spread[k])
        end
        push!(mrows, @sprintf("%s,%d,%d,%.4f,%.10f,%s,%d,%.14f,%.14f,%.2e,%.14f,%.14f,%s", name, l, m,
                              Float64(Th), Vh, X === nothing ? "" : @sprintf("%.6f", X), k - 1,
                              real(w), imag(w), c.spread[k], real(lam), imag(lam), refcols))
    end
    for lam in c.base
        w = omega_of(lam)
        abs(w) < 1.6 || continue
        cert = any(abs.(c.lambda .- lam) .< 1e-14)
        push!(srows, @sprintf("%s,%d,%d,%.4f,%.3f,%.3f,%.12e,%.12e,%.12e,%.12e,%d", name, l, m,
                              Float64(Th), TH, TH, real(lam), imag(lam), real(w), imag(w), cert))
    end
end

# mirror modes: the conjugate contour gives the conjugate eigenvalues, w -> -conj(w)
S = setup("swc", 1, 1, 1 // 2)
C = ecs_contour(aL=5.0, aR=5.0, thm=TH, thp=TH, TL=TRUNC, TR=TRUNC, N=NDEG)
l1 = spectrum(S, C)
l2 = spectrum(S, conj_contour(C))
@printf("mirror check (swc (1,1), Theta = 1/2): max over the spectrum of |lambda(conj contour) - conj(lambda)| = %.1e\n",
        maximum(match_modes(conj.(l1), l2)))

open(io -> foreach(r -> println(io, r), mrows), joinpath(OUT, "modes.csv"), "w")
open(io -> foreach(r -> println(io, r), srows), joinpath(OUT, "spectra.csv"), "w")
println("written modes.csv and spectra.csv in ", OUT)
