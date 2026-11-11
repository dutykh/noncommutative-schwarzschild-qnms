# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Quasinormal modes of the noncommutative Schwarzschild black holes of model() below,
# with the master equation of Chen, Chiang and Tsao.  Units M = 1 (alpha = 2M = 2), Theta in
# units of M, mu = M/Theta.  With u = 1/r the radial equation  F (F psi')' + (w^2 - V) psi = 0
# (prime = d/dr) reads  G (G psi_u)_u + (w^2 - V) psi = 0  with G = u^2 F(u).  The horizon is the
# smallest positive zero u_h of F; there F = (u_h - u) H(u) with H > 0 on [0, u_h].
#
# The quasinormal solution is factored as
#     psi = (u_h - u)^{s} u^{-2 i w} exp(i w / u) phi(u),     s = i k / G1,
# with G1 = G'(u_h) < 0 and k^2 = w^2 - V_h, V_h = V(u_h).  Then phi solves
#     u^2 D H^2 phi'' + B phi' + E phi = 0,        D = u_h - u,
# with B and E polynomials in u, linear in the five monomials 1, w, w^2, k, k w.  Everything is
# built with exact polynomial division, so no digits are lost near the two ends.

module NCQNM

using LinearAlgebra
using Polynomials

export Model, model, schwarzschild, potential, horizon, factored_equation, discretize,
       joukowski_pencil, linearize, solve_joukowski, residual, kfun, newton_refine

# ---------------------------------------------------------------------------------------------
# Models: F(u) a polynomial in u = 1/r, V(u) = Vn(u)/Vd(u) a rational function, Vd without zero on [0, u_h].
# ---------------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "model_pieces.jl"))

struct Model{T<:AbstractFloat}
    name::String
    l::Int
    m::Int
    Theta::T
    F::Polynomial{T}      # dr/dr* as a polynomial in u
    Vn::Polynomial{T}     # numerator of the effective potential
    Vd::Polynomial{T}     # denominator of the effective potential
end

topoly(c::Vector{Rational{BigInt}}, ::Type{T}) where {T} =
    Polynomial(T[T(numerator(x)) / T(denominator(x)) for x in c])

"""
    model(name, l, m, Theta; T=Float64)

Deformed Schwarzschild metric `name` ("swc", "ctz", "jkp_rth", "swc_tth", "jkp_tth"; see symbolic/
export_models.py) with the master equation of Chen, Chiang and Tsao.  Units M = 1.
F = 1 - 2u + Theta^2 F2(u),
V = f (L u^2 + 2u^3) + Theta^2 [ W00 + a0 Wa + c0 Wc + kax16 (s1 Ws1 + s2 Ws2) ],
with a0 = |m|(2l+1)/2, c0 = a0 - l(l+1), s1 = (2l+1)(l(l+1) - m^2)/(64|m|),
s2 = (2l+1-2|m|)/(32|m|).  The axis sums exist only for m != 0 when kax16 != 0.
"""
function model(name::String, l::Int, m::Int, Theta; T::Type=Float64)
    P = MODEL_PIECES[name]
    kax16 = P["kax16"]
    @assert l >= 0 && abs(m) <= l
    kax16 != 0 && m == 0 && error("model $name: the axis sums diverge for m = 0")
    Th2 = T(Theta)^2
    u = Polynomial([zero(T), one(T)])
    L = T(l * (l + 1))
    am = abs(m)
    F2n, F2d = P["F2"]
    @assert F2d == [1 // 1]
    F = 1 - 2u + Th2 * topoly(F2n, T)
    a0 = T(am * (2l + 1)) / 2
    c0 = a0 - L
    s1 = am == 0 ? zero(T) : T((2l + 1) * (l * (l + 1) - am^2)) / (64am)
    s2 = am == 0 ? zero(T) : T(2l + 1 - 2am) / (32am)
    coef = Dict("W00" => one(T), "Wa" => a0, "Wc" => c0,
                "Ws1" => T(kax16) * s1, "Ws2" => T(kax16) * s2)
    dens = Polynomial{T}[]
    for k in keys(coef)
        d = topoly(P[k][2], T)
        degree(d) > 0 && !any(x -> x == d, dens) && push!(dens, d)
    end
    Vd = isempty(dens) ? one(u) : prod(dens)
    Vn = (1 - 2u) * (L * u^2 + 2u^3) * Vd
    for (k, cf) in coef
        n = topoly(P[k][1], T)
        d = topoly(P[k][2], T)
        q, rr = divrem(Vd, d)
        @assert maximum(abs, coeffs(rr); init=zero(T)) < 100eps(T)
        Vn += Th2 * cf * n * q
    end
    return Model{T}(name, l, m, T(Theta), F, Vn, Vd)
end

"""Schwarzschild scalar field, the Theta = 0 limit (any l >= 0)."""
function schwarzschild(l::Int; T::Type=Float64)
    u = Polynomial([zero(T), one(T)])
    f = 1 - 2u
    return Model{T}("schwarzschild", l, 0, zero(T), f, f * (T(l * (l + 1)) * u^2 + 2u^3), one(u))
end

potential(mod::Model, u) = mod.Vn(u) / mod.Vd(u)

# ---------------------------------------------------------------------------------------------
# Horizon and the factored equation.
# ---------------------------------------------------------------------------------------------

"""Smallest positive zero u_h of F (Newton from u = 1/2), and H = F/(u_h - u)."""
function horizon(mod::Model{T}) where {T}
    F = mod.F
    dF = derivative(F)
    uh = T(1) / 2
    for _ in 1:200
        du = F(uh) / dF(uh)
        uh -= du
        abs(du) < 10eps(T) * abs(uh) && break
    end
    for _ in 1:3
        uh -= F(uh) / dF(uh)
    end
    # synthetic division of F by (u_h - u)
    q, rem = divrem(F, Polynomial([uh, -one(T)]))
    @assert abs(rem(zero(T))) < sqrt(eps(T)) * maximum(abs, coeffs(F)) "division by (u_h - u) failed"
    return uh, q
end

"""
Coefficient polynomials of the regularized equation  A2 phi'' + B phi' + E phi = 0  (d/du),
returned as a NamedTuple with A2 and the parts of B and E multiplying 1, w, w^2, k, k w,
together with u_h, V_h and G1.
"""
function factored_equation(mod::Model{T}) where {T}
    uh, H = horizon(mod)
    u = Polynomial([zero(T), one(T)])
    D = uh - u
    Vn, Vd = mod.Vn, mod.Vd
    Vh = Vn(uh) / Vd(uh)
    G1 = -uh^2 * H(uh)                       # G'(u_h) < 0
    dH = derivative(H)
    K = 2 * D * H - u * H + u * D * dH       # GG' = u^3 D H K
    # B = -2 s u^2 H^2 - 2 i w D H^2 (2u + 1) + u H K,     s = i k / G1
    # (complex factors kept outside: B = B1 + (i w) Bw + (s) Bs)
    B1 = u * H * K
    Bw = -2 * D * H^2 * (2u + 1)             # multiplies i w
    Bs = -2 * u^2 * H^2                      # multiplies s
    # C = G^2 (S'' + S'^2) + G G' S' + w^2 - V, with S' = -(s/D + 2iw/u + iw/u^2).  Write
    # C = Css s^2 + Csw (i w s) + Cs s + Cww (i w)^2 + Cw (i w) + C0 + w^2 - V  (w^2 = -(i w)^2).
    Css = u^4 * H^2
    Csw = H^2 * (4 * u^3 * D + 2 * u^2 * D)
    Cs = -u^4 * H^2 - u * H * K * u^2
    Cww = H^2 * (4 * u^2 * D^2 + D^2 + 4 * u * D^2)     # from (iw)^2 terms of S'^2
    Cw = 2 * u^2 * D^2 * H^2 + 2 * u * D^2 * H^2 - u * H * K * (2 * u * D + D)
    C0 = -Vn                                 # the equation is multiplied through by Vd
    # w^2 - V: the w^2 = -(iw)^2 term combines with Cww: total (iw)^2 coefficient Cww - 1.
    Cww_tot = Cww - 1
    # s^2 = -k^2 / G1^2 = -(w^2 - V_h)/G1^2 = ((iw)^2 + V_h)/G1^2
    # i w s = i w (i k/G1) = -(w k)/G1 ;  s = i k / G1.
    # Collect C by monomials in (w, k), with complex coefficients:
    #   1   : C0 + Css * V_h/G1^2
    #   w   : i Cw
    #   w^2 : -(Cww_tot) - Css/G1^2         [ (iw)^2 = -w^2 ]
    #   k   : (i/G1) Cs
    #   k w : -(1/G1) Csw
    parts = Dict{Symbol,Polynomial{Complex{T}}}()
    c(p) = Polynomial(Complex{T}.(coeffs(p)))
    parts[:one] = c(C0 + Vd * Css * (Vh / G1^2))
    parts[:w] = im * c(Vd * Cw)
    parts[:w2] = -c(Vd * Cww_tot) - c(Vd * Css) / G1^2
    parts[:k] = (im / G1) * c(Vd * Cs)
    parts[:kw] = -(1 / G1) * c(Vd * Csw)
    # C vanishes to second order at u = 0 and at u = u_h for every (w, k) with k^2 = w^2 - V_h.
    # The functions 1, w, w^2, k, k w are linearly independent on that curve, so each part
    # vanishes there on its own and is divisible by u^2 D; the division is checked below.
    E = Dict{Symbol,Polynomial{Complex{T}}}()
    for (key, p) in parts
        q, rr = divrem(p, c(u^2 * D))
        E[key] = q
        E_rem = maximum(abs, coeffs(rr); init=zero(T))
        if E_rem > sqrt(eps(T)) * max(one(T), maximum(abs, coeffs(p)))
            error("E part $key not divisible by u^2 D (remainder $E_rem)")
        end
    end
    Bp = Dict{Symbol,Polynomial{Complex{T}}}()
    Bp[:one] = c(Vd * B1)
    Bp[:w] = im * c(Vd * Bw)
    Bp[:k] = (im / G1) * c(Vd * Bs)
    A2 = c(Vd * u^2 * D * H^2)
    return (; A2, B=Bp, E, uh, Vh, G1, H)
end

# ---------------------------------------------------------------------------------------------
# Chebyshev collocation on y in [-1, 1], u = u_h (1 + y)/2 (y = -1 infinity, y = 1 horizon).
# ---------------------------------------------------------------------------------------------

"""Values of T_n, T_n', T_n'' (n = 0..N) at the Gauss-Chebyshev points (interior roots)."""
function chebyshev_tables(N::Int, ::Type{T}) where {T}
    y = [cos(T(pi) * (2j + 1) / (2N + 2)) for j in 0:N]
    V0 = zeros(T, N + 1, N + 1); V1 = similar(V0); V2 = similar(V0)
    for (i, yi) in enumerate(y)
        t0, t1 = one(T), yi
        d0, d1 = zero(T), one(T)
        s0, s1 = zero(T), zero(T)
        for n in 0:N
            if n == 0
                V0[i, 1] = t0; V1[i, 1] = d0; V2[i, 1] = s0
            elseif n == 1
                V0[i, 2] = t1; V1[i, 2] = d1; V2[i, 2] = s1
            else
                t2 = 2yi * t1 - t0
                d2 = 2t1 + 2yi * d1 - d0
                s2 = 4d1 + 2yi * s1 - s0
                V0[i, n+1] = t2; V1[i, n+1] = d2; V2[i, n+1] = s2
                t0, t1 = t1, t2; d0, d1 = d1, d2; s0, s1 = s1, s2
            end
        end
    end
    return y, V0, V1, V2
end

"""
    discretize(mod, N; eps=nothing)

Collocation matrices for  T(w, k) = M1 + w Mw + w^2 Mw2 + k Mk + k w Mkw  acting on the
Chebyshev coefficients of phi; also returns V_h, u_h, G1.  With eps = nothing the map is linear,
u = u_h (1 + y)/2.  With eps = epsilon > 0 the near-horizon layer is resolved by the map
    u = u_h - epsilon (exp(sigma) - 1),   sigma = S (1 - y)/2,   S = log(1 + u_h/epsilon),
which sends a singularity of the coefficients at u = u_h + epsilon to sigma = -infinity
(epsilon is the distance from the horizon to the pole of V in the corrected metric).
"""
function discretize(mod::Model{T}, N::Int; eps=nothing) where {T}
    eq = factored_equation(mod)
    y, T0, T1, T2 = chebyshev_tables(N, T)
    uh = eq.uh
    if eps === nothing
        uu = uh .* (1 .+ y) ./ 2
        uy = fill(uh / 2, length(y))
        uyy = zeros(T, length(y))
    else
        ep = T(eps)
        S = log(1 + uh / ep)
        sg = S .* (1 .- y) ./ 2
        uu = uh .- ep .* (exp.(sg) .- 1)
        uy = (ep * S / 2) .* exp.(sg)
        uyy = -(ep * S^2 / 4) .* exp.(sg)
    end
    ev(p) = Complex{T}.(p.(uu))
    Dg(v) = Diagonal(v)
    a2 = ev(eq.A2)
    D2 = Dg(a2 ./ uy .^ 2) * T2 - Dg(a2 .* uyy ./ uy .^ 3) * T1     # A2 d^2/du^2
    D1(b) = Dg(ev(b) ./ uy) * T1                                       # B d/du
    M = Dict{Symbol,Matrix{Complex{T}}}()
    M[:one] = D2 + D1(eq.B[:one]) + Dg(ev(eq.E[:one])) * T0
    M[:w] = D1(eq.B[:w]) + Dg(ev(eq.E[:w])) * T0
    M[:w2] = Dg(ev(eq.E[:w2])) * T0
    M[:k] = D1(eq.B[:k]) + Dg(ev(eq.E[:k])) * T0
    M[:kw] = Dg(ev(eq.E[:kw])) * T0
    # rows scaled to unit size (the eigenvalues of the pencil are unchanged)
    sc = [1 / max(maximum(abs, M[:one][i, :]), maximum(abs, M[:w][i, :]),
                  maximum(abs, M[:w2][i, :]), maximum(abs, M[:k][i, :]), floatmin(T))
          for i in 1:length(y)]
    for key in keys(M)
        M[key] = Dg(Complex{T}.(sc)) * M[key]
    end
    return (; M, Vh=eq.Vh, uh=eq.uh, G1=eq.G1)
end

"""k(w) on the physical sheet: k ~ w at infinity, cut on the segment between +-sqrt(V_h)."""
kfun(w, Vh) = w * sqrt(1 - Vh / w^2)

"""Quartic matrix polynomial in Lambda (Joukowski map w = (L^2 + V_h)/(2L), k = (L^2 - V_h)/(2L))."""
function joukowski_pencil(d)
    M, Vh = d.M, d.Vh
    P4 = M[:w2] + M[:kw]
    P3 = 2 * (M[:w] + M[:k])
    P2 = 4 * M[:one] + 2Vh * M[:w2]
    P1 = 2Vh * (M[:w] - M[:k])
    P0 = Vh^2 * (M[:w2] - M[:kw])
    return [P0, P1, P2, P3, P4]
end

"""First companion linearization  A x = Lambda B x  of sum_i Lambda^i P_i."""
function linearize(P::Vector{<:AbstractMatrix{S}}) where {S}
    d = length(P) - 1
    n = size(P[1], 1)
    A = zeros(S, d * n, d * n)
    B = zeros(S, d * n, d * n)
    for i in 1:d-1
        A[(i-1)*n+1:i*n, i*n+1:(i+1)*n] = I(n)
        B[(i-1)*n+1:i*n, (i-1)*n+1:i*n] = I(n)
    end
    for i in 0:d-1
        A[(d-1)*n+1:d*n, i*n+1:(i+1)*n] = -P[i+1]
    end
    B[(d-1)*n+1:d*n, (d-1)*n+1:d*n] = P[d+1]
    return A, B
end

"""
    solve_joukowski(mod, N; eigensolver)

All eigenvalues Lambda of the quartic pencil, mapped to (w, k); flag = |Lambda| > sqrt(|V_h|)
(the physical sheet, cut on the segment between the branch points).
"""
function solve_joukowski(mod::Model{T}, N::Int; geneig=nothing, eps=nothing) where {T}
    d = discretize(mod, N; eps=eps)
    P = joukowski_pencil(d)
    A, B = linearize(P)
    Lam = geneig === nothing ? eigvals(A, B) : geneig(A, B)
    Lam = Lam[isfinite.(Lam) .& (abs.(Lam) .> 0)]
    Vh = d.Vh
    w = (Lam .^ 2 .+ Vh) ./ (2 .* Lam)
    k = (Lam .^ 2 .- Vh) ./ (2 .* Lam)
    phys = abs.(Lam) .> sqrt(abs(Vh))
    return (; w, k, Lam, phys, Vh, uh=d.uh)
end

"""Smallest singular value of T(w, k(w)) relative to the norms of the coefficient matrices."""
function residual(d, w)
    k = kfun(w, d.Vh)
    Tm = d.M[:one] + w * d.M[:w] + w^2 * d.M[:w2] + k * d.M[:k] + k * w * d.M[:kw]
    nrm = opnorm(d.M[:one], 1) + abs(w) * opnorm(d.M[:w], 1) + abs(w)^2 * opnorm(d.M[:w2], 1) +
          abs(k) * opnorm(d.M[:k], 1) + abs(k * w) * opnorm(d.M[:kw], 1)
    return minimum(svdvals(Tm)) / nrm
end


"""
    newton_refine(d, w0; maxit=60, tol=nothing)

Newton's method for  T(w) a = 0,  e1' a = 1  (the first Chebyshev coefficient normalised), on the
physical sheet, k(w) = w sqrt(1 - V_h/w^2).  The Jacobian of (T(w) a, e1' a - 1) with respect to
(a, w) is the bordered matrix [T(w)  T'(w) a; e1'  0], with
T'(w) = Mw + 2 w Mw2 + k'(w) (Mk + w Mkw) + k Mkw and k'(w) = w/k.  The starting vector comes
from three steps of inverse iteration with T(w0).  Works in the arithmetic of d.
"""
function newton_refine(d, w0; maxit::Int=60, tol=nothing)
    M = d.M
    S = eltype(M[:one])
    R = real(S)
    tol === nothing && (tol = 100 * eps(R))
    Vh = d.Vh
    w = S(w0)
    n = size(M[:one], 1)
    Tm(w) = (k = kfun(w, Vh); M[:one] + w * M[:w] + w^2 * M[:w2] + k * (M[:k] + w * M[:kw]))
    # starting vector: inverse iteration with the shifted matrix (three steps)
    a = ones(S, n)
    F = lu(Tm(w))
    for _ in 1:3
        a = F \ a
        a ./= a[1]
    end
    for it in 1:maxit
        k = kfun(w, Vh)
        dk = w / k
        Tw = Tm(w)
        dT = M[:w] + 2w * M[:w2] + dk * (M[:k] + w * M[:kw]) + k * M[:kw]
        J = zeros(S, n + 1, n + 1)
        J[1:n, 1:n] = Tw
        J[1:n, n+1] = dT * a
        J[n+1, 1] = one(S)
        rhs = vcat(Tw * a, a[1] - one(S))
        delta = J \ rhs
        a .-= delta[1:n]
        w -= delta[n+1]
        abs(delta[n+1]) < tol * abs(w) && return w, a, it
    end
    return w, a, maxit
end

end # module
