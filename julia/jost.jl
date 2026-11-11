# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Jost function of the radial equation  G (G psi_u)_u + (w^2 - V) psi = 0,  G = u^2 F,  u = 1/r,
# in arbitrary precision; its zeros on the physical sheet are the quasinormal frequencies.
#
# Horizon.  The point u = u_h is a regular singular point with exponents +-s, s = i k / G1,
# G1 = G'(u_h) < 0 and k = w sqrt(1 - V_h/w^2) (principal root).  The ingoing solution is
#     psi_H = x^s sum_{n >= 0} b_n x^n,     x = u_h - u,  b_0 = 1,
# convergent for |x| < rho_H, the distance from u_h to the nearest other singular point (u = 0,
# another zero of F, or a zero of the denominator Vd of V).
#
# Continuation.  With A = Vd G^2 and B = Vd G G' the equation reads
#     A psi'' + B psi' + (w^2 Vd - Vn) psi = 0,
# with polynomial coefficients; at any regular point u0 the Taylor coefficients of psi follow from
# a linear recurrence, and psi_H is continued by Taylor steps of high order along the path
#     u_h - x1  ->  rho0  (real axis)  ->  rho0 e^{-i theta}  (arc |u| = rho0)  ->  e^{-i theta}/R.
# This path is homotopic, in the plane punctured at the singular points, to the real segment
# followed by a small arc around u = 0, i.e. to the large arc 0 <= arg r <= theta in r = 1/u.
#
# Infinity.  The point u = 0 is an irregular singular point of rank one.  The outgoing solution
#     psi_out = e^{i w/u} u^{-2 i w} phi(u),     phi ~ sum_{n >= 0} a_n u^n,  a_0 = 1,
# has a divergent (Gevrey-one) series, a_n ~ (n-1)!/(2 i w)^n.  For theta = pi/2 - arg(w) one has
# i w/u = -|w| R on the ray, so psi_out is recessive there, by e^{-2|w|R} against the incoming
# solution, and is the unique solution with this behaviour.  The series is truncated at its
# smallest term, whose relative size is about e^{-2|w|R}.
#
# Jost function.  J(w) = G (psi_H psi_out' - psi_H' psi_out) is minus the Wronskian in the tortoise
# coordinate, independent of the matching point, and vanishes exactly when psi_H is outgoing.  The
# truncation error of the series perturbs the data of psi_out by a relative amount e^{-2|w|R}, and
# since the incoming solution dominates on the ray by e^{2|w|R}, its weight in that perturbation is
# e^{-4|w|R}: the zeros of the computed J are exact up to O(e^{-4|w|R}) and the arithmetic.

module NCJost

using Polynomials

export JostProblem, jost_problem, jost, kphys, qnm_secant, horizon_series, outgoing_series,
       jost_contour_winding, box_winding, multistart, zeros_in_box, grid_zeros

# ---------------------------------------------------------------------------------------------
# Polynomial helpers (ascending coefficient vectors)
# ---------------------------------------------------------------------------------------------

"""Coefficients of p(u0 + h) in powers of h (repeated synthetic division)."""
function taylor_shift(c::AbstractVector, u0)
    S = promote_type(eltype(c), typeof(u0))
    a = S.(c)
    n = length(a)
    for i in 1:n-1
        for j in n-1:-1:i
            a[j] += u0 * a[j+1]
        end
    end
    return a
end

"""Coefficients of p(u_h - x) in powers of x."""
function reflect_shift(c::AbstractVector, uh)
    a = taylor_shift(c, uh)
    return [isodd(i - 1) ? -a[i] : a[i] for i in eachindex(a)]
end

coef(p::Polynomial{T}) where {T} = (c = collect(T, coeffs(p)); isempty(c) ? T[zero(T)] : c)

"""Exact division of p by u^2: the two lowest coefficients must vanish to rounding."""
function divide_u2(p::Polynomial{T}, what::String) where {T}
    c = coef(p)
    scale = max(one(T), maximum(abs, c))
    length(c) >= 1 && abs(c[1]) > 1000eps(T) * scale && error("$what: u^0 coefficient $(c[1])")
    length(c) >= 2 && abs(c[2]) > 1000eps(T) * scale && error("$what: u^1 coefficient $(c[2])")
    return length(c) <= 2 ? T[zero(T)] : c[3:end]
end

pad(v::Vector{S}, n::Int) where {S} = length(v) >= n ? v : vcat(v, zeros(S, n - length(v)))

"""k(w) on the physical sheet: k ~ w at infinity, cut on the segment between +-sqrt(V_h)."""
kphys(w, Vh) = w * sqrt(1 - Vh / w^2)

# ---------------------------------------------------------------------------------------------
# The problem
# ---------------------------------------------------------------------------------------------

struct JostProblem{T<:AbstractFloat}
    name::String
    l::Int
    m::Int
    Theta::T
    uh::T
    Vh::T
    G1::T
    G::Vector{T}       # G = u^2 F
    A::Vector{T}       # Vd G^2
    B::Vector{T}       # Vd G G'
    Vd::Vector{T}
    Vn::Vector{T}
    # horizon, x = u_h - u:  x^2 At psi_xx + x Bt psi_x + (w^2 Vdx - Vnx) psi = 0
    At::Vector{T}
    Bt::Vector{T}
    Vdx::Vector{T}
    Vnx::Vector{T}
    # infinity, psi = e^{i w/u} u^{-2 i w} phi:
    #   Q2 phi'' + (Q1a + i w Q1b) phi' + (Q0a + i w Q0b + w^2 Q0c) phi = 0
    Q2::Vector{T}
    Q1a::Vector{T}
    Q1b::Vector{T}
    Q0a::Vector{T}
    Q0b::Vector{T}
    Q0c::Vector{T}
    sing::Vector{ComplexF64}   # finite singular points of the equation, u_h included
    rhoH::T                    # radius of convergence of the horizon series
end

"""
    jost_problem(mod, uh, H)

Set up the Jost problem of a model (fields name, l, m, Theta, F, Vn, Vd of NCQNM.Model) with
horizon u_h and H = F/(u_h - u) (NCQNM.horizon).
"""
function jost_problem(mod, uh::T, H::Polynomial{T}) where {T<:AbstractFloat}
    u = Polynomial([zero(T), one(T)])
    F, Vn, Vd = mod.F, mod.Vn, mod.Vd
    dF = derivative(F)
    G = u^2 * F
    dG = derivative(G)
    A = Vd * G^2
    B = Vd * G * dG
    Vh = Vn(uh) / Vd(uh)
    G1 = -uh^2 * H(uh)
    # horizon pieces: G = x Ghat(x), Ghat(0) = -G1
    Ghat = Polynomial(reflect_shift(coef(u^2 * H), uh))
    Vdx = Polynomial(reflect_shift(coef(Vd), uh))
    xp = Polynomial([zero(T), one(T)])
    At = Vdx * Ghat^2
    Bt = Vdx * Ghat * (Ghat + xp * derivative(Ghat))
    Vnx = reflect_shift(coef(Vn), uh)
    # infinity pieces (all divided by u^2)
    Q2 = Vd * u^2 * F^2
    Q1a = Vd * u * F * (2F + u * dF)
    Q1b = -2 * Vd * F^2 * (1 + 2u)
    Q0a = divide_u2(-Vn, "Vn")
    Q0b = -Vd * F * (2F + dF + 2u * dF)
    Q0c = divide_u2(Vd * (1 - F^2 * (1 + 2u)^2), "1 - F^2 (1+2u)^2")
    # singular points: zeros of A = Vd u^4 F^2
    zF = roots(Polynomial(Float64.(coef(F))))
    zD = degree(Vd) > 0 ? roots(Polynomial(Float64.(coef(Vd)))) : ComplexF64[]
    sing = ComplexF64[0.0; ComplexF64.(zF); ComplexF64.(zD)]
    others = [z for z in sing if abs(z - Float64(uh)) > 1e-10]
    rhoH = T(minimum(abs.(others .- Float64(uh))))
    # refine rhoH with the exact pole of Vd when it is the nearest point (jkp_rth: u = 1/2)
    if degree(Vd) == 1
        zd = -coef(Vd)[1] / coef(Vd)[2]
        abs(zd - uh) <= rhoH * (1 + 1e-6) && (rhoH = abs(zd - uh))
    end
    sing = ComplexF64[ComplexF64(Float64(uh)); others]
    return JostProblem{T}(mod.name, mod.l, mod.m, T(mod.Theta), uh, Vh, G1, coef(G), coef(A),
                          coef(B), coef(Vd), coef(Vn), coef(At), coef(Bt), coef(Vdx), Vnx,
                          coef(Q2), coef(Q1a), coef(Q1b), Q0a, coef(Q0b), Q0c, sing, rhoH)
end

# ---------------------------------------------------------------------------------------------
# Horizon: Frobenius series
# ---------------------------------------------------------------------------------------------

"""
    horizon_series(P, w, k, x1; tol, nmax)

psi_H(u1) and psi_H'(u1) (d/du) at u1 = u_h - x1, with psi_H = x^s (1 + b_1 x + ...),
s = i k / G1; also the number of terms used.
"""
function horizon_series(P::JostProblem{T}, w::Complex{T}, k::Complex{T}, x1::T;
                        tol::T=T(10)^(-60), nmax::Int=20000) where {T}
    C = Complex{T}
    s = im * k / P.G1
    n0 = max(length(P.At), length(P.Bt), length(P.Vdx), length(P.Vnx))
    At = pad(C.(P.At), n0)
    Bt = pad(C.(P.Bt), n0)
    P0 = w^2 .* pad(C.(P.Vdx), n0) .- pad(C.(P.Vnx), n0)
    b = C[one(C)]
    S = one(C)
    Sd = s
    xs = x1
    quiet = 0
    nused = 0
    for n in 1:nmax
        acc = zero(C)
        for i in 1:min(n, n0 - 1)
            σ = (n - i) + s
            acc += (At[i+1] * σ * (σ - 1) + Bt[i+1] * σ + P0[i+1]) * b[n-i+1]
        end
        σ = n + s
        In = At[1] * σ * (σ - 1) + Bt[1] * σ + P0[1]
        bn = -acc / In
        push!(b, bn)
        term = bn * xs
        S += term
        Sd += (n + s) * term
        xs *= x1
        nused = n
        if abs(term) * (n + abs(s) + 1) < tol * abs(S)
            quiet += 1
            quiet >= 6 && break
        else
            quiet = 0
        end
    end
    xsp = exp(s * log(x1))
    return xsp * S, -xsp * Sd / x1, nused
end

# ---------------------------------------------------------------------------------------------
# Continuation by Taylor steps
# ---------------------------------------------------------------------------------------------

"""Taylor coefficients c_0..c_N of the solution with c_0 = y0, c_1 = y1 at the shift point."""
function taylor_coeffs!(c::Vector{S}, α::Vector{S}, β::Vector{S}, πc::Vector{S},
                        y0::S, y1::S, N::Int) where {S}
    c[1] = y0
    c[2] = y1
    dα, dβ, dπ = length(α) - 1, length(β) - 1, length(πc) - 1
    for j in 0:N-2
        acc = zero(S)
        for i in 1:min(j, dα)
            acc += α[i+1] * (c[j-i+3] * ((j - i + 2) * (j - i + 1)))
        end
        for i in 0:min(j, dβ)
            acc += β[i+1] * (c[j-i+2] * (j - i + 1))
        end
        for i in 0:min(j, dπ)
            acc += πc[i+1] * c[j-i+1]
        end
        c[j+3] = -acc / (α[1] * ((j + 2) * (j + 1)))
    end
    return c
end

"""
    continue_path(P, w, nodes, y0, y1; N, tol)

Continue the solution with psi(nodes[1]) = y0, psi'(nodes[1]) = y1 along the polygon through
`nodes`, by Taylor steps of order N; the step is set by the size of the last coefficients.
"""
function continue_path(P::JostProblem{T}, w::Complex{T}, nodes::Vector{Complex{T}},
                       y0::Complex{T}, y1::Complex{T}; N::Int=150,
                       tol::T=T(10)^(-60)) where {T}
    C = Complex{T}
    nP = max(length(P.Vd), length(P.Vn))
    P0 = w^2 .* pad(C.(P.Vd), nP) .- pad(C.(P.Vn), nP)
    c = Vector{C}(undef, N + 1)
    u = nodes[1]
    nsteps = 0
    for target in nodes[2:end]
        while u != target
            d = target - u
            dist = abs(d)
            α = taylor_shift(P.A, u)
            β = taylor_shift(P.B, u)
            πc = taylor_shift(P0, u)
            abs(α[1]) == 0 && error("Taylor step at a singular point u = $u")
            taylor_coeffs!(c, α, β, πc, y0, y1, N)
            dsing = minimum(abs(ComplexF64(u) - z) for z in P.sing)
            scale = max(abs(c[1]), abs(c[2]) * T(dsing))
            h = T(0.9) * T(dsing)
            for j in N-3:N
                aj = abs(c[j+1])
                aj > 0 && (h = min(h, (tol * scale / aj)^(one(T) / j)))
            end
            if h >= dist
                hs = d
                unew = target
            else
                hs = d * (h / dist)
                unew = u + hs
            end
            # Horner for psi and psi'
            p = c[N+1]
            dp = c[N+1] * N
            for j in N-1:-1:0
                p = p * hs + c[j+1]
            end
            for j in N-1:-1:1
                dp = dp * hs + c[j+1] * j
            end
            y0, y1 = p, dp
            u = unew
            nsteps += 1
            nsteps > 100000 && error("too many Taylor steps")
        end
    end
    return y0, y1, nsteps
end

# ---------------------------------------------------------------------------------------------
# Infinity: the outgoing series
# ---------------------------------------------------------------------------------------------

"""
    outgoing_series(P, w, um; nterms)

psi_out and psi_out' (d/du) at um from the asymptotic series truncated before its smallest term
(or after `nterms` terms when given); also the truncation index and the relative size of the
smallest term.
"""
function outgoing_series(P::JostProblem{T}, w::Complex{T}, um::Complex{T};
                         nterms::Union{Nothing,Int}=nothing, nmax::Int=4000) where {T}
    C = Complex{T}
    iw = im * w
    n1 = max(length(P.Q1a), length(P.Q1b))
    n0 = max(length(P.Q0a), length(P.Q0b), length(P.Q0c))
    q2 = C.(P.Q2)
    q1 = pad(C.(P.Q1a), n1) .+ iw .* pad(C.(P.Q1b), n1)
    q0 = pad(C.(P.Q0a), n0) .+ iw .* pad(C.(P.Q0b), n0) .+ w^2 .* pad(C.(P.Q0c), n0)
    d2, d1, d0 = length(q2) - 1, length(q1) - 1, length(q0) - 1
    a = C[one(C)]
    tmag = T[one(T)]
    umn = one(C)
    best = 0
    bestmag = T(Inf)
    nlim = nterms === nothing ? nmax : nterms
    for n in 0:nlim-1
        acc = zero(C)
        for i in 2:min(n, d2)
            acc += q2[i+1] * (a[n-i+3] * ((n - i + 2) * (n - i + 1)))
        end
        for i in 1:min(n, d1)
            acc += q1[i+1] * (a[n-i+2] * (n - i + 1))
        end
        for i in 0:min(n, d0)
            acc += q0[i+1] * a[n-i+1]
        end
        an = -acc / (q1[1] * (n + 1))
        push!(a, an)
        umn *= um
        mag = abs(an * umn)
        push!(tmag, mag)
        if nterms === nothing
            if mag < bestmag
                bestmag = mag
                best = n + 1
            elseif n + 1 > best + 10 && mag > 1e6 * bestmag
                break
            end
        end
    end
    ntr = nterms === nothing ? best : nterms        # terms a_0 .. a_{ntr-1} are summed
    nterms === nothing || (bestmag = tmag[end])
    phi = zero(C)
    dphi = zero(C)
    for n in ntr-1:-1:0
        phi = phi * um + a[n+1]
    end
    for n in ntr-1:-1:1
        dphi = dphi * um + a[n+1] * n
    end
    E = exp(iw / um - 2iw * log(um))
    Sp = -iw * (1 + 2um) / um^2
    return E * phi, E * (Sp * phi + dphi), ntr, bestmag / abs(phi)
end

# ---------------------------------------------------------------------------------------------
# The Jost function
# ---------------------------------------------------------------------------------------------

"""Recessive direction: theta = pi/2 - arg(w), kept away from the negative real r axis."""
recessive_theta(w) = clamp(pi / 2 - angle(ComplexF64(w)), 0.05, pi - 0.05)

"""Matching radius for which e^{-4|w|R} = 10^{-digits}."""
matching_radius(w, digits) = max(20.0, digits * log(10) / (4abs(ComplexF64(w))))

"""
    jost(P, w; theta, R, rho0, narc, N, tol, x1frac, nterms)

Jost function J(w) = G (psi_H psi_out' - psi_H' psi_out) at the matching point e^{-i theta}/R,
with psi_H normalized by b_0 = 1 at the horizon and psi_out by a_0 = 1 at infinity.  Returns J and
a NamedTuple of diagnostics.
"""
function jost(P::JostProblem{T}, w::Complex{T}; theta=nothing, R=nothing, rho0=0.25,
              narc::Int=12, N::Int=150, tol::T=T(10)^(-60), x1frac=0.25,
              nterms::Union{Nothing,Int}=nothing, digits=60) where {T}
    C = Complex{T}
    θ = theta === nothing ? T(recessive_theta(w)) : T(theta)
    Rm = R === nothing ? T(matching_radius(w, digits)) : T(R)
    k = kphys(w, P.Vh)
    x1 = P.rhoH * T(x1frac)
    yH, dyH, nH = horizon_series(P, w, k, x1; tol=tol)
    ρ0 = T(rho0)
    nodes = C[C(P.uh - x1), C(ρ0)]
    for j in 1:narc
        push!(nodes, ρ0 * cis(-θ * j / narc))
    end
    um = cis(-θ) / Rm
    push!(nodes, um)
    y0, y1, nsteps = continue_path(P, w, nodes, yH, dyH; N=N, tol=tol)
    po, dpo, ntr, small = outgoing_series(P, w, um; nterms=nterms)
    Gm = evalpoly(um, P.G)
    J = Gm * (y0 * dpo - y1 * po)
    return J, (; theta=θ, R=Rm, nH, nsteps, ntr, small, k)
end

# ---------------------------------------------------------------------------------------------
# Root finding and zero counting
# ---------------------------------------------------------------------------------------------

"""
Secant iteration on f from wa, wb; returns the last iterate and the corrections.  An iterate
outside the rectangle `region` (re0, re1, im0, im1) ends the iteration with a NaN.
"""
function secant(f, wa::Complex{T}, wb::Complex{T}, rtol::T, maxit::Int; region=nothing) where {T}
    fa = f(wa)
    fb = f(wb)
    hist = T[]
    for it in 1:maxit
        fb == fa && break
        wc = wb - fb * (wb - wa) / (fb - fa)
        push!(hist, abs(wc - wb))
        if region !== nothing && !inbox(wc, region)
            return Complex{T}(NaN, NaN), hist
        end
        wa, fa = wb, fb
        wb = wc
        fb = f(wb)
        abs(wb - wa) < rtol * abs(wb) && break
    end
    return wb, hist
end

"""
    qnm_secant(P, w0; rtol, maxit, kwargs...)

Secant iteration on J started at w0 and w0 (1 + 1e-7).  During one pass the matching direction,
the matching radius and the truncation index of the series are frozen, so that the iteration
acts on one analytic function; they are then reset at the root found and the root is polished
again, until two passes agree to `rtol`.  Returns the root, the corrections of the last pass and
the diagnostics, among them the change of the root between the last two passes.
"""
function qnm_secant(P::JostProblem{T}, w0::Complex{T}; rtol::T=T(10)^(-45), maxit::Int=40,
                    digits=60, theta=nothing, R=nothing, maxpass::Int=5, region=nothing,
                    kwargs...) where {T}
    w = w0
    hist = T[]
    θ, Rm, info0 = 0.0, 0.0, nothing
    passdiff = T(Inf)
    for pass in 1:maxpass
        θ = theta === nothing ? recessive_theta(w) : theta
        Rm = R === nothing ? matching_radius(w, digits) : R
        _, info0 = jost(P, w; theta=θ, R=Rm, digits=digits, kwargs...)
        f(x) = first(jost(P, x; theta=θ, R=Rm, nterms=info0.ntr, digits=digits, kwargs...))
        wnew, hist = secant(f, w, w * (1 + T(10)^(-7)), rtol, maxit; region=region)
        isfinite(wnew) || return wnew, hist, (; theta=θ, R=Rm, ntr=info0.ntr, small=info0.small,
                                              nH=info0.nH, nsteps=info0.nsteps, passdiff)
        pass > 1 && (passdiff = abs(wnew - w))
        w = wnew
        pass > 1 && passdiff < 10rtol * abs(w) && break
    end
    return w, hist, (; theta=θ, R=Rm, ntr=info0.ntr, small=info0.small, nH=info0.nH,
                     nsteps=info0.nsteps, passdiff)
end

"""
    jost_contour_winding(f, corners; maxdphase, minpts)

Winding number of f along the closed polygon `corners` (counter-clockwise), refining each edge
until consecutive phase increments are below `maxdphase`.  Returns the winding number, its
unrounded value, the smallest log10|f| met and the samples (w, log10|f|, arg f).
"""
function jost_contour_winding(f, corners::Vector{ComplexF64}; maxdphase=0.6, minpts::Int=8,
                              maxpts::Int=4000)
    total = 0.0
    lfmin = Inf
    samples = Tuple{ComplexF64,Float64,Float64}[]      # (w, log10|f|, arg f)
    nc = length(corners)
    for e in 1:nc
        a, b = corners[e], corners[mod1(e + 1, nc)]
        ts = collect(range(0.0, 1.0; length=minpts + 1))
        vals = [f(a + t * (b - a)) for t in ts]
        i = 1
        while i < length(ts)
            dφ = Float64(angle(vals[i+1] / vals[i]))
            if abs(dφ) > maxdphase && length(ts) < maxpts
                tm = (ts[i] + ts[i+1]) / 2
                insert!(ts, i + 1, tm)
                insert!(vals, i + 1, f(a + tm * (b - a)))
            else
                total += dφ
                i += 1
            end
        end
        for (t, v) in zip(ts, vals)
            lv = Float64(log10(abs(v)))
            push!(samples, (a + t * (b - a), lv, Float64(angle(v))))
            lfmin = min(lfmin, lv)
        end
    end
    return round(Int, total / (2pi)), total / (2pi), lfmin, samples
end

inbox(w, b) = b[1] < real(w) < b[2] && b[3] < imag(w) < b[4]

"""Winding number of J around the rectangle b = (re0, re1, im0, im1)."""
function box_winding(P::JostProblem{T}, b; digits=28, spacing=0.01) where {T}
    f(w) = first(jost(P, Complex{T}(w); digits=digits))
    corners = ComplexF64[complex(b[1], b[3]), complex(b[2], b[3]), complex(b[2], b[4]),
                         complex(b[1], b[4])]
    npts = max(12, ceil(Int, max(b[2] - b[1], b[4] - b[3]) / spacing))
    W, Wr, _, samples = jost_contour_winding(f, corners; maxdphase=0.4, minpts=npts)
    abs(Wr - W) > 0.05 && @warn "winding number not close to an integer" Wr b
    return W, length(samples)
end

"""Distinct zeros of J in the rectangle b reached by the secant from a grid of starting points."""
function multistart(P::JostProblem{T}, b; spacing=0.05, digits=28, tolsame=1e-14) where {T}
    nre = max(2, round(Int, (b[2] - b[1]) / spacing))
    nim = max(2, round(Int, (b[4] - b[3]) / spacing))
    starts = [complex(b[1] + (i - 0.5) * (b[2] - b[1]) / nre, b[3] + (j - 0.5) * (b[4] - b[3]) / nim)
              for i in 1:nre for j in 1:nim]
    found = Complex{T}[]
    lk = ReentrantLock()
    δ = 0.15
    region = (max(b[1] - δ, 1e-3), b[2] + δ, b[3] - δ, min(b[4] + δ, -1e-4))
    Threads.@threads for s in starts
        try
            wr, _, inf = qnm_secant(P, Complex{T}(s); digits=digits, rtol=T(10)^(-digits + 4),
                                    maxit=30, region=region)
            if isfinite(wr) && inf.passdiff < T(10)^(-digits + 8) && inbox(wr, b)
                lock(lk) do
                    any(abs(wr - r) < tolsame for r in found) || push!(found, wr)
                end
            end
        catch err
            err isa InterruptException && rethrow()
        end
    end
    return found
end

"""
    zeros_in_box(P, b; spacing, digits, depth)

All zeros of J in the rectangle b: multistart secant, checked against the winding number of J
around b; when the two counts differ the rectangle is split in four and the search repeated in
each quarter with a finer grid.  Returns the zeros, the winding number and a flag telling whether
the count is confirmed.
"""
function zeros_in_box(P::JostProblem{T}, b; spacing=0.05, digits=28, depth::Int=3) where {T}
    W, _ = box_winding(P, b; digits=digits)
    W == 0 && return Complex{T}[], 0, true
    found = multistart(P, b; spacing=spacing, digits=digits)
    length(found) == W && return found, W, true
    depth == 0 && return found, W, false
    rm, im_ = (b[1] + b[2]) / 2, (b[3] + b[4]) / 2
    quarters = ((b[1], rm, b[3], im_), (rm, b[2], b[3], im_), (b[1], rm, im_, b[4]),
                (rm, b[2], im_, b[4]))
    allz = Complex{T}[]
    ok = true
    for q in quarters
        zq, Wq, okq = zeros_in_box(P, q; spacing=spacing / 2, digits=digits, depth=depth - 1)
        append!(allz, zq)
        ok &= okq
    end
    return allz, W, ok && length(allz) == W
end

"""Phase increment of f along the segment a -> b, given fa = f(a) and fb = f(b)."""
function segment_phase(f, a::ComplexF64, b::ComplexF64, fa, fb; maxdphase=0.3, minsub::Int=8,
                       maxpts::Int=4000)
    ts = collect(range(0.0, 1.0; length=minsub + 1))
    vals = [fa; [f(a + t * (b - a)) for t in ts[2:end-1]]; fb]
    total = 0.0
    i = 1
    while i < length(ts)
        dφ = Float64(angle(vals[i+1] / vals[i]))
        if abs(dφ) > maxdphase && length(ts) < maxpts
            tm = (ts[i] + ts[i+1]) / 2
            insert!(ts, i + 1, tm)
            insert!(vals, i + 1, f(a + tm * (b - a)))
        else
            total += dφ
            i += 1
        end
    end
    return total
end

"""
Winding numbers of f around the nre x nim cells of the rectangle b; the phase increment along
each interior edge is computed once and shared by the two cells it bounds.
"""
function grid_windings(f, b, nre::Int, nim::Int)
    res = collect(range(b[1], b[2]; length=nre + 1))
    ims = collect(range(b[3], b[4]; length=nim + 1))
    nodes = [complex(res[i], ims[j]) for i in 1:nre+1, j in 1:nim+1]
    vals = Matrix{Any}(undef, nre + 1, nim + 1)
    Threads.@threads for k in eachindex(nodes)
        vals[k] = f(nodes[k])
    end
    H = zeros(nre, nim + 1)
    V = zeros(nre + 1, nim)
    hjobs = [(i, j) for i in 1:nre for j in 1:nim+1]
    vjobs = [(i, j) for i in 1:nre+1 for j in 1:nim]
    Threads.@threads for k in eachindex(hjobs)
        i, j = hjobs[k]
        H[i, j] = segment_phase(f, nodes[i, j], nodes[i+1, j], vals[i, j], vals[i+1, j])
    end
    Threads.@threads for k in eachindex(vjobs)
        i, j = vjobs[k]
        V[i, j] = segment_phase(f, nodes[i, j], nodes[i, j+1], vals[i, j], vals[i, j+1])
    end
    cells = Tuple{NTuple{4,Float64},Int,Float64}[]
    for i in 1:nre, j in 1:nim
        Wr = (H[i, j] + V[i+1, j] - H[i, j+1] - V[i, j]) / 2pi
        push!(cells, ((res[i], res[i+1], ims[j], ims[j+1]), round(Int, Wr), Wr))
    end
    return cells
end

inbox_closed(w, b; δ=1e-12) = b[1] - δ <= real(w) <= b[2] + δ && b[3] - δ <= imag(w) <= b[4] + δ

"""Zeros of J in a cell of winding number W (secant from the centre, else 2 x 2 subdivision)."""
function cell_zeros(P::JostProblem{T}, f, cb, W::Int, depth::Int, digits) where {T}
    if W == 1
        c = complex((cb[1] + cb[2]) / 2, (cb[3] + cb[4]) / 2)
        mr, mi = 0.25 * (cb[2] - cb[1]), 0.25 * (cb[4] - cb[3])
        region = (cb[1] - mr, cb[2] + mr, cb[3] - mi, cb[4] + mi)
        try
            w, _, inf = qnm_secant(P, Complex{T}(c); digits=digits, rtol=T(10)^(-digits + 4),
                                   maxit=30, region=region)
            if isfinite(w) && inf.passdiff < T(10)^(-digits + 8) && inbox_closed(w, cb)
                return Complex{T}[w], true
            end
        catch err
            err isa InterruptException && rethrow()
        end
    end
    depth == 0 && return Complex{T}[], false
    zs = Complex{T}[]
    ok = true
    for (sb, Ws, Wr) in grid_windings(f, cb, 2, 2)
        (Ws < 0 || abs(Wr - Ws) > 0.05) && (ok = false)
        Ws <= 0 && continue
        z, okc = cell_zeros(P, f, sb, Ws, depth - 1, digits)
        append!(zs, z)
        ok &= okc
    end
    return zs, ok && length(zs) == W
end

"""
    grid_zeros(P, b; nre, nim, depth, digits)

All zeros of J in the rectangle b.  The winding number of J is computed around every cell of an
nre x nim grid (J at 16-digit matching accuracy, which fixes its phase to about 1e-9), the zeros
of the cells of winding one are found by the secant method from the cell centre, and the other
cells are subdivided.  Since J is analytic in the open fourth quadrant, a negative or
non-integer cell winding number signals an under-resolved edge and is reported through the flag.
Returns the zeros, the total winding number and the flag.
"""
function grid_zeros(P::JostProblem{T}, b; nre::Int=10, nim::Int=6, depth::Int=4,
                    digits=28) where {T}
    f(w) = first(jost(P, Complex{T}(w); digits=16, N=60, tol=T(10)^(-22)))
    zs = Complex{T}[]
    ok = true
    total = 0
    for (cb, W, Wr) in grid_windings(f, b, nre, nim)
        (W < 0 || abs(Wr - W) > 0.05) && (ok = false)
        W <= 0 && continue
        total += W
        z, okc = cell_zeros(P, f, cb, W, depth, digits)
        append!(zs, z)
        ok &= okc
    end
    return zs, total, ok && length(zs) == total
end

end # module
