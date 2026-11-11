# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Exterior complex scaling in the tortoise coordinate.  The radial equation
#     -psi'' + V(r(z)) psi = lambda psi,     lambda = w^2,     dr/dz = F(r),   z = r*,
# is posed on a contour of the complex z plane which is real on a middle segment and leaves it
# along two rays: z = a_R + e^{i th_p} t at the far side (t ⩾ 0) and z = -a_L - e^{i th_m} t at
# the horizon side.  A quasinormal solution behaves as exp(i w z) at the far side and as
# exp(-i k z), k^2 = lambda - V_h, at the horizon side, and both decay along the rays when
# arg w > -th_p and arg k > -th_m.  The quasinormal frequencies thus become eigenvalues
# lambda = w^2 of an ordinary linear problem, independent of the contour, while the two continua
# of the scaled operator lie on the rays  lambda = e^{-2 i th_p} [0, oo)  and
# lambda = V_h + e^{-2 i th_m} [0, oo).  The contour must not sweep across a singularity of
# V(r(z)) when it is deformed from the real axis.
#
# The tortoise coordinate is used in closed form.  With P(r) = r^d F(1/r) (monic, simple zeros
# r_h, r_1, ..., r_{d-1}) one has 1/F = 1 + sum_j A_j/(r - r_j), A_j = r_j^d / P'(r_j), hence
#     z(r) = r - r0 + A_h [log(r - r_h) - log(r0 - r_h)] + sum_{j != h} A_j [Log(r - r_j) - Log(r0 - r_j)],
# normalised by z(r0) = 0, with A_h = 1/(2 kappa).  In rho = log(r - r_h) the horizon term is
# linear, and r(z) is continued along the contour by Newton's method on z(rho) in BigFloat, so
# that r - r_h keeps its relative precision however close to the horizon the contour goes.
#
# Discretisation: the contour is a chain of straight pieces, each carrying Chebyshev-Lobatto
# collocation; psi and dpsi/dz are continuous at the vertices (the derivative condition replaces
# the collocation equation there, and is eliminated), psi = 0 at the two truncated ends.

module ECS

using LinearAlgebra
using Polynomials
using Printf

include(joinpath(@__DIR__, "NCQNM.jl"))
using .NCQNM: Model, model, schwarzschild, horizon, potential

export Setup, setup, zofrho, zofr, Contour, polygon_contour, ecs_contour, bent_contour,
       conj_contour, Discretization, discretize_ecs, ecs_eigen, spectrum, omega_of, k_of,
       match_modes, stable_modes, revealed, pole_images, pole_abscissa, safe_aL, certify

const CB = Complex{BigFloat}

# ---------------------------------------------------------------------------------------------
# The model and its tortoise coordinate
# ---------------------------------------------------------------------------------------------

struct Setup
    name::String
    l::Int
    m::Int
    Theta::Rational{Int}
    mb::Model{BigFloat}         # exact coefficients carried in BigFloat
    uh::BigFloat
    rh::BigFloat
    Vh::BigFloat
    twokappa::BigFloat          # F'(r_h) = u_h^2 H(u_h)
    Ah::BigFloat                # residue of 1/F at r_h, = 1/(2 kappa)
    rj::Vector{CB}              # the other zeros of P
    Aj::Vector{CB}              # their residues
    r0::BigFloat                # z(r0) = 0
    rho0::BigFloat
end

"""
    setup(name, l, m, Theta; r0=3)

Model `name` ("schwarzschild", "swc", "ctz", "jkp_rth", "swc_tth", "jkp_tth"), with the partial
fractions of 1/F.
Theta is a rational number (exact coefficients); the working precision is that of BigFloat.
"""
function setup(name::String, l::Int, m::Int=0, Theta::Rational{Int}=0 // 1; r0=3)
    mb = name == "schwarzschild" ? schwarzschild(l; T=BigFloat) :
         model(name, l, m, Theta; T=BigFloat)
    uh, H = horizon(mb)
    Fp = mb.F
    for _ in 1:5                              # polish u_h at the working precision
        uh -= Fp(uh) / derivative(Fp)(uh)
    end
    rh = 1 / uh
    Vh = potential(mb, uh)
    twokappa = -uh^2 * derivative(Fp)(uh)     # dF/dr = -u^2 dF/du
    f = coeffs(Fp)
    d = length(f) - 1
    @assert abs(f[1] - 1) < 1e-60 "F(0) = 1 expected"
    P = Polynomial(reverse(f))                # r^d F(1/r)
    dP = derivative(P)
    seeds = ComplexF64.(roots(Polynomial(Float64.(coeffs(P)))))
    rb = CB[]
    for s in seeds
        x = CB(s)
        for _ in 1:200
            dx = P(x) / dP(x)
            x -= dx
            abs(dx) < big"1e-70" * max(1, abs(x)) && break
        end
        push!(rb, x)
    end
    ih = argmin(abs.(rb .- rh))
    @assert abs(rb[ih] - rh) < big"1e-50" "horizon root not found"
    A = [x^d / dP(x) for x in rb]
    Ah = real(A[ih])
    @assert abs(Ah - 1 / twokappa) < big"1e-50" * Ah "residue at the horizon"
    @assert abs(sum(A) + f[2]) < big"1e-50" "sum of residues"
    keep = setdiff(1:length(rb), ih)
    r0b = BigFloat(r0)
    return Setup(name, l, m, Theta, mb, uh, rh, Vh, twokappa, Ah, rb[keep], A[keep], r0b,
                 log(r0b - rh))
end

"""z(rho) and dz/drho, with r = r_h + e^rho (Log of the other factors on its principal branch)."""
function zofrho(S::Setup, rho::CB)
    er = exp(rho)
    r = S.rh + er
    z = r - S.r0 + S.Ah * (rho - S.rho0)
    s = one(CB)
    for (rj, Aj) in zip(S.rj, S.Aj)
        z += Aj * (log(r - rj) - log(S.r0 - rj))
        s += Aj / (r - rj)
    end
    return z, er * s + S.Ah, r
end

"""z at the point r, with log(r - r_h) taken on the branch shifted by 2 pi i n."""
zofr(S::Setup, r; n::Int=0) = zofrho(S, log(CB(r) - S.rh) + 2n * big(pi) * im)[1]

"""
Images in the z plane of the poles of V (zeros of Vd), on the sheets n = -nmax..nmax; for a real
pole below the horizon, n = 0 is the image above the real axis and n = -1 the one below.
"""
function pole_images(S::Setup; nmax::Int=1)
    out = Tuple{CB,CB}[]
    degree(S.mb.Vd) == 0 && return out
    for up in roots(Polynomial(Float64.(coeffs(S.mb.Vd))))
        u = CB(up)
        for _ in 1:100
            u -= S.mb.Vd(u) / derivative(S.mb.Vd)(u)
        end
        rp = 1 / u
        # A real pole below the horizon: log(r_p - r_h) = log(r_h - r_p) + i pi (2n + 1), the image
        # n = 0 above the real z axis and n = -1 below it, with the branch fixed explicitly (the
        # sign of a vanishing imaginary part of r_p must not choose it).
        isreal_pole = abs(imag(rp)) < big"1e-40" * abs(rp) && real(rp) < S.rh
        for n in -nmax:nmax
            if isreal_pole
                rho = log(S.rh - real(rp)) + (2n + 1) * big(pi) * im
                push!(out, (CB(real(rp)), zofrho(S, CB(rho))[1]))
            else
                push!(out, (rp, zofr(S, rp; n=n)))
            end
        end
    end
    return out
end

"""rho at the points zs, continued by Newton's method from (z_start, rho_start)."""
function rho_along(S::Setup, zs::AbstractVector, z_start, rho_start::CB)
    out = Vector{CB}(undef, length(zs))
    rho = rho_start
    zp = CB(z_start)
    for (i, zt) in enumerate(zs)
        ztb = CB(zt)
        nsub = max(1, ceil(Int, Float64(abs(ztb - zp)) / 0.2))
        for js in 1:nsub
            zj = zp + (ztb - zp) * js / nsub
            for it in 1:100
                zr, dz, _ = zofrho(S, rho)
                delta = (zj - zr) / dz
                rho += delta
                abs(delta) < big"1e-60" && break
                it == 100 && error("Newton did not converge at z = $zt")
            end
        end
        zr, _, _ = zofrho(S, rho)
        @assert abs(zr - ztb) < big"1e-50"
        out[i] = rho
        zp = ztb
    end
    return out
end

# ---------------------------------------------------------------------------------------------
# Contours
# ---------------------------------------------------------------------------------------------

struct Contour
    verts::Vector{ComplexF64}   # ends of the straight pieces, from the far horizon side to the far side
    N::Vector{Int}              # Chebyshev degree of each piece
    thm::Float64                # angle of the horizon-side ray
    thp::Float64                # angle of the far-side ray
end

"""
Chain of straight pieces through the corners (which must contain 0), each segment cut into pieces
of length at most hmax carrying Chebyshev degree N.
"""
function polygon_contour(corners::Vector{<:Number}, thm, thp; hmax=12.0, N::Int=28)
    @assert any(==(0), corners) "z = 0 must be a corner"
    verts = ComplexF64[corners[1]]
    for i in 1:length(corners)-1
        a, b = ComplexF64(corners[i]), ComplexF64(corners[i+1])
        n = max(1, ceil(Int, abs(b - a) / hmax - 1e-9))
        for j in 1:n
            push!(verts, j == n ? b : a + (b - a) * j / n)
        end
    end
    return Contour(verts, fill(N, length(verts) - 1), Float64(thm), Float64(thp))
end

"""Exterior complex scaling: real segment [-aL, aR], rays of angles thm (horizon) and thp (far)."""
function ecs_contour(; aL=5.0, aR=5.0, thm=1.0, thp=1.0, TL=200.0, TR=200.0, hmax=12.0, N::Int=28)
    corners = ComplexF64[-aL - cis(thm) * TL, -aL, 0, aR, aR + cis(thp) * TR]
    return polygon_contour(corners, thm, thp; hmax=hmax, N=N)
end

"""Mirror contour (complex conjugate): its eigenvalues are the conjugates, the mirror modes -conj(w)."""
conj_contour(C::Contour) = Contour(conj.(C.verts), C.N, -C.thm, -C.thp)

# ---------------------------------------------------------------------------------------------
# Chebyshev-Lobatto collocation
# ---------------------------------------------------------------------------------------------

"""Nodes y_j = -cos(pi j/N) (increasing) and the first-derivative matrix on [-1, 1]."""
function cheb(N::Int)
    y = [-cospi(j / N) for j in 0:N]
    w = [(-1.0)^j * ((j == 0 || j == N) ? 0.5 : 1.0) for j in 0:N]
    D = zeros(N + 1, N + 1)
    for i in 0:N, j in 0:N
        i == j && continue
        dy = 2 * sinpi((i + j) / (2N)) * sinpi((i - j) / (2N))      # y_i - y_j without cancellation
        D[i+1, j+1] = (w[j+1] / w[i+1]) / dy
    end
    for i in 1:N+1
        D[i, i] = -sum(D[i, j] for j in 1:N+1 if j != i)
    end
    return y, D
end

struct Discretization
    S::Setup
    C::Contour
    z::Vector{ComplexF64}        # all nodes, in contour order
    r::Vector{ComplexF64}
    V::Vector{ComplexF64}
    L::Matrix{ComplexF64}        # reduced operator on the interior nodes
    inner::Vector{Int}           # node indices of the unknowns of L
    verts::Vector{Int}           # node indices of the interior vertices
    back::Matrix{ComplexF64}     # psi at the interior vertices = back * psi_inner
end

function discretize_ecs(S::Setup, C::Contour)
    np = length(C.N)
    offs = zeros(Int, np + 1)
    offs[1] = 1
    for p in 1:np
        offs[p+1] = offs[p] + C.N[p]
    end
    nt = offs[np+1]
    z = zeros(ComplexF64, nt)
    Ds = Vector{Matrix{Float64}}(undef, np)
    cs = zeros(ComplexF64, np)
    for p in 1:np
        y, D = cheb(C.N[p])
        a, b = C.verts[p], C.verts[p+1]
        z[offs[p]:offs[p+1]] .= a .+ (b - a) .* (y .+ 1) ./ 2
        z[offs[p]] = a
        z[offs[p+1]] = b
        Ds[p] = D
        cs[p] = (b - a) / 2
    end
    # r(z) by continuation from the vertex z = 0
    i0 = findfirst(==(0), z)
    @assert i0 !== nothing
    rho = Vector{CB}(undef, nt)
    rho[i0] = CB(S.rho0)
    rho[i0+1:nt] = rho_along(S, z[i0+1:nt], z[i0], rho[i0])
    rho[i0-1:-1:1] = rho_along(S, z[i0-1:-1:1], z[i0], rho[i0])
    rb = S.rh .+ exp.(rho)
    ub = 1 ./ rb
    Vb = [S.mb.Vn(u) / S.mb.Vd(u) for u in ub]
    r = ComplexF64.(rb)
    V = ComplexF64.(Vb)
    # assembly on all nodes
    vtx = [offs[p] for p in 2:np]
    ends = [1, nt]
    inner = setdiff(1:nt, vcat(vtx, ends))
    A = zeros(ComplexF64, nt, nt)          # rows: collocation at inner nodes, matching at vertices
    for p in 1:np
        cols = offs[p]:offs[p+1]
        D = Ds[p]
        D2 = D * D
        for i in 2:C.N[p]
            g = offs[p] + i - 1
            A[g, cols] .= -D2[i, :] ./ cs[p]^2
            A[g, g] += V[g]
        end
    end
    for (q, g) in enumerate(vtx)
        p = q                                  # vertex between pieces p and p+1
        A[g, offs[p]:offs[p+1]] .+= Ds[p][end, :] ./ cs[p]
        A[g, offs[p+1]:offs[p+2]] .-= Ds[p+1][1, :] ./ cs[p+1]
    end
    AII = A[inner, inner]
    AIV = A[inner, vtx]
    CI = A[vtx, inner]
    CV = A[vtx, vtx]
    back = -(CV \ CI)
    L = AII + AIV * back
    return Discretization(S, C, z, r, V, L, inner, vtx, back)
end

"""Eigenvalues (and, on request, eigenfunctions on all nodes) of the scaled operator."""
function ecs_eigen(d::Discretization; vectors::Bool=false)
    if !vectors
        return eigvals(d.L), nothing
    end
    E = eigen(d.L)
    nt = length(d.z)
    Psi = zeros(ComplexF64, nt, length(E.values))
    Psi[d.inner, :] = E.vectors
    Psi[d.verts, :] = d.back * E.vectors
    return E.values, Psi
end

omega_of(lam) = sqrt(complex(lam))

"""k = sqrt(lambda - V_h) on the branch that decays along the horizon-side ray of angle thm."""
function k_of(lam, Vh, thm)
    k = sqrt(complex(lam - Vh))
    return imag(k * cis(thm)) >= 0 ? k : -k
end

"""Is lambda inside the sector revealed by the two rays (with a margin in angle)?"""
function revealed(lam, Vh, thm, thp; margin=0.0)
    w = sqrt(complex(lam))
    k = sqrt(complex(lam - Vh))
    return angle(w) > -thp + margin && angle(k) > -thm + margin && imag(lam) < 0
end

"""Distance from each lambda of `base` to the nearest eigenvalue of `other`."""
match_modes(base, other) = [minimum(abs.(other .- l)) for l in base]

"""
Eigenvalues of `runs[1]` inside the window that move by less than tol (relative to max(1,|lambda|))
in every other run; returns (lambda, spread).
"""
function stable_modes(runs::Vector{<:AbstractVector}; window=l -> true, tol=1e-8)
    base = filter(window, runs[1])
    spread = zeros(length(base))
    for other in runs[2:end]
        spread = max.(spread, match_modes(base, other))
    end
    keep = spread .< tol .* max.(1, abs.(base))
    return base[keep], spread[keep]
end

# ---------------------------------------------------------------------------------------------
# Contours adapted to the model, and the certification of the quasinormal frequencies
# ---------------------------------------------------------------------------------------------

"""Real part of the images of the poles of V in the z plane (they share it), or nothing."""
function pole_abscissa(S::Setup)
    P = pole_images(S; nmax=0)
    isempty(P) && return nothing
    return maximum(Float64(real(z)) for (_, z) in P)
end

"""
Left end of the real segment.  The poles of V lie at Re z = X; the horizon-side ray must leave
the real axis to the left of them, so that the wedge it sweeps contains none of them.
"""
function safe_aL(S::Setup; margin=4.0, amin=5.0)
    X = pole_abscissa(S)
    return X === nothing ? amin : max(amin, -X + margin)
end

"""
Contour passing above the poles: real on [-a1, aR], then a straight descent to the point
(X - margin) - i y (between the real axis and the poles, which lie near Im z = -pi A_h), then the
horizon-side ray; the far side as in `ecs_contour`.
"""
function bent_contour(S::Setup; a1=4.0, aR=5.0, margin=4.0, y=pi, thm=1.0, thp=1.0, TL=200.0,
                      TR=200.0, hmax=12.0, N::Int=28)
    X = pole_abscissa(S)
    X === nothing && error("bent_contour: the model has no poles")
    c1 = (X - margin) - im * y
    corners = ComplexF64[c1 - cis(thm) * TL, c1, -a1, 0, aR, aR + cis(thp) * TR]
    return polygon_contour(corners, thm, thp; hmax=hmax, N=N)
end

spectrum(S::Setup, C::Contour) = ecs_eigen(discretize_ecs(S, C))[1]

"""
Quasinormal frequencies certified by their independence of the contour: a base run (angles th,
segment [-aL, aR], truncation T, degree N) and variants in resolution (N + dN), angles
(th + dth), segment and truncation (aL + 3, aR + 3, T + 40), and, for a model whose potential has
poles, the contour that passes above them.  Kept: eigenvalues inside the sector revealed by the
base angles (minus `margin`), |w| < wmax, moving by less than tol in every variant.  Returns
(lambda, w, spread, base), with base the whole spectrum of the base run.
"""
function certify(S::Setup; th=1.3, dth=(0.12, 0.08), aL=safe_aL(S), aR=5.0, T=220.0, N::Int=28,
                 dN::Int=8, wmax=1.2, margin=0.06, tol=1e-6, hmax=12.0, bent::Bool=true)
    run(; aL=aL, aR=aR, thm=th, thp=th, T=T, N=N) =
        spectrum(S, ecs_contour(aL=aL, aR=aR, thm=thm, thp=thp, TL=T, TR=T, hmax=hmax, N=N))
    base = run()
    runs = [base, run(N=N + dN), run(thm=th + dth[1], thp=th + dth[2]),
            run(aL=aL + 3, aR=aR + 3, T=T + 40)]
    if bent && pole_abscissa(S) !== nothing
        push!(runs, spectrum(S, bent_contour(S; thm=th + 0.05, thp=th + 0.05, TL=T, TR=T,
                                             hmax=hmax, N=N)))
    end
    Vh = Float64(S.Vh)
    win(l) = abs(sqrt(complex(l))) < wmax && revealed(l, Vh, th, th; margin=margin)
    lam, spr = stable_modes(runs; window=win, tol=tol)
    p = sortperm(imag.(omega_of.(lam)), rev=true)
    return (lambda=lam[p], w=omega_of.(lam[p]), spread=spr[p], base=base)
end

end # module
