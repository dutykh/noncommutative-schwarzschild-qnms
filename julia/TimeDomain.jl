# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Time-domain evolution of the radial equation  Psi_tt = Psi_{x x} - V Psi,  x = r* the tortoise
# coordinate (dr/dr* = F), for the models of NCQNM.jl, and extraction of complex frequencies.
#
# Tortoise coordinate.  With u = 1/r, dr*/du = -1/(u^2 F(u)), F = (u_h - u) H(u), F(0) = 1,
# F'(0) = -2.  Writing N = 1 - F (1 + 2u) = u^2 P(u) (P a polynomial, since N(0) = N'(0) = 0),
#     1/(u^2 F) = 1/u^2 + 2/u + b/(u_h - u) + Q(u)/H(u),   b = P(u_h)/H(u_h) = 1/(u_h^2 H(u_h)),
# with Q = (P - b H)/(u_h - u) a polynomial.  Hence, exactly,
#     r*(u) = 1/u - 2 log u + b log(u_h - u) - int_0^u Q/H dv,
# which for Schwarzschild (b = 2, Q = 0) is r* = r + 2 log(r/2 - 1).  The integral of the
# smooth function Q/H is done by Gauss-Legendre quadrature, and r* is inverted by safeguarded
# Newton iteration in the variable l = log(u_h - u) near the horizon (dr*/dl = 1/(u^2 H)) and in
# r far from it (dr*/dr = 1/F).  Near the horizon u_h - u = exp(l) is kept as a logarithm, so
# the plateau r* -> -infinity is resolved without loss.
#
# Evolution.  (i) The null-grid scheme of Gundlach, Price and Pullin: with u = t - x, v = t + x
# the equation is 4 Psi_uv + V Psi = 0, and on the diamond S = (u, v), E = (u, v + d),
# W = (u + d, v), N = (u + d, v + d),
#     Psi_N = Psi_E + Psi_W - Psi_S - (d^2/8) V(x_C) (Psi_E + Psi_W),   x_C = (v - u)/2,
# local error O(d^4), global error O(d^2).  Data: a Gaussian on u = 0, a constant on v = 0.
# (ii) Method of lines: sixth-order centred second difference in x, classical Runge-Kutta 4 in
# t, on an interval long enough that the boundaries are causally disconnected from every
# observer until the final time.
#
# Frequencies.  The matrix-pencil method of Hua and Sarkar: samples y_k = sum_j a_j z_j^k,
# z_j = exp(-i w_j dt), Hankel matrix, truncated SVD, eigenvalues of the shifted pencil.

isdefined(Main, :NCQNM) || include(joinpath(@__DIR__, "NCQNM.jl"))

module TimeDomain

using LinearAlgebra, Polynomials, Printf
using ..NCQNM

export Tortoise, tortoise, rstar_of_u, u_of_rstar, potential_at, gpp_evolve, mol_evolve,
       matrix_pencil, pencil_scan, qnm_scan, annihilate, plateau, growth_fit, local_rate,
       write_columns, gauss_legendre, bound_states

# ---------------------------------------------------------------------------------------------
# Gauss-Legendre rule on [0, 1] (Golub-Welsch).
# ---------------------------------------------------------------------------------------------
function gauss_legendre(n::Int)
    beta = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), beta))
    x = (E.values .+ 1) ./ 2
    w = E.vectors[1, :] .^ 2           # weights on [-1, 1] sum to 2; on [0, 1] to 1
    return x, w
end

# ---------------------------------------------------------------------------------------------
# Tortoise coordinate.
# ---------------------------------------------------------------------------------------------
struct Tortoise
    uh::Float64
    b::Float64
    F::Polynomial{Float64}
    H::Polynomial{Float64}
    Q::Polynomial{Float64}
    gh::Float64                 # 1/u_h - 2 log u_h - int_0^{u_h} Q/H
    xq::Vector{Float64}
    wq::Vector{Float64}
end

function tortoise(mod; nq::Int=64)
    uh, H = horizon(mod)
    F = mod.F
    u = Polynomial([0.0, 1.0])
    N = 1 - F * (1 + 2u)
    cN = coeffs(N)
    @assert abs(cN[1]) < 1e-13 && (length(cN) < 2 || abs(cN[2]) < 1e-13) "N must vanish to second order at u = 0"
    P = length(cN) > 2 ? Polynomial(cN[3:end]) : Polynomial([0.0])
    b = P(uh) / H(uh)
    Qd, Qr = divrem(P - b * H, Polynomial([uh, -1.0]))
    @assert abs(Qr(0.0)) < 1e-12 * max(1.0, maximum(abs, coeffs(P)))
    xq, wq = gauss_legendre(nq)
    T = Tortoise(uh, b, F, H, Qd, 0.0, xq, wq)
    gh = 1 / uh - 2log(uh) - qint(T, uh)
    return Tortoise(uh, b, F, H, Qd, gh, xq, wq)
end

"""int_0^u Q(v)/H(v) dv by Gauss-Legendre quadrature."""
function qint(T::Tortoise, u)
    s = 0.0
    @inbounds for k in eachindex(T.xq)
        v = u * T.xq[k]
        s += T.wq[k] * T.Q(v) / T.H(v)
    end
    return u * s
end

"""r*(u) with l = log(u_h - u) supplied (exact near the horizon)."""
rstar_ul(T::Tortoise, u, l) = 1 / u - 2log(u) + T.b * l - qint(T, u)
rstar_of_u(T::Tortoise, u) = rstar_ul(T, u, log(T.uh - u))

"""
    u_of_rstar(T, x) -> (u, l)

Inverse of the tortoise map: u = 1/r and l = log(u_h - u) at r* = x.
"""
function u_of_rstar(T::Tortoise, x::Float64)
    uh = T.uh
    if x < 2.0
        # Newton in l on a bracket [lo, hi] with r*(lo) < x < r*(hi)
        hi = log(0.9uh)
        lo = min((x - T.gh) / T.b, hi) - 2.0
        while rstar_ul(T, uh - exp(lo), lo) > x
            lo -= 10.0
        end
        l = clamp((x - T.gh) / T.b, lo, hi)
        for it in 1:200
            D = exp(l)
            u = uh - D
            f = rstar_ul(T, u, l) - x
            f < 0 ? (lo = max(lo, l)) : (hi = min(hi, l))
            df = 1 / (u^2 * T.H(u))
            lnew = l - f / df
            if !(lo < lnew < hi)
                lnew = (lo + hi) / 2
            end
            if abs(lnew - l) < 1e-15 * max(1.0, abs(l))
                l = lnew
                break
            end
            l = lnew
        end
        return uh - exp(l), l
    else
        # Newton in r on [r_lo, r_hi]
        rlo = 1 / (uh - 1e-3 * uh)
        rhi = x + 10.0
        while rstar_of_u(T, 1 / rhi) < x
            rhi *= 2
        end
        r = clamp(x - 2log(max(x, 3.0) / 2), rlo, rhi)
        for it in 1:200
            u = 1 / r
            f = rstar_of_u(T, u) - x
            f < 0 ? (rlo = max(rlo, r)) : (rhi = min(rhi, r))
            df = 1 / T.F(u)
            rnew = r - f / df
            if !(rlo < rnew < rhi)
                rnew = (rlo + rhi) / 2
            end
            if abs(rnew - r) < 1e-15 * r
                r = rnew
                break
            end
            r = rnew
        end
        u = 1 / r
        return u, log(uh - u)
    end
end

"""Potential of the model at the tortoise points xs."""
function potential_at(mod, T::Tortoise, xs::AbstractVector)
    V = similar(xs, Float64)
    Threads.@threads for k in eachindex(xs)
        u, _ = u_of_rstar(T, Float64(xs[k]))
        V[k] = potential(mod, u)
    end
    return V
end

# ---------------------------------------------------------------------------------------------
# Null-grid evolution (Gundlach-Price-Pullin).
# ---------------------------------------------------------------------------------------------
"""
    gpp_evolve(Vfun, d, Tu, Tv; vc, sigma, obs, snap)

Grid u_i = i d (i = 0..Nu), v_j = j d (j = 0..Nv), Nu = round(Tu/d), Nv = round(Tv/d).
`Vfun(xs)` returns V at the tortoise points xs.  Data: Psi(0, v) = exp(-(v - vc)^2/(2 sigma^2)),
Psi(u, 0) = Psi(0, 0).  `obs` lists observer positions x (2x/d must be an integer); returns
(t, Psi) per observer, sampled every d in t.  With `snap = K > 0`, also returns log10|Psi| on the
sub-grid of every K-th point.
"""
function gpp_evolve(Vfun, d::Float64, Tu::Float64, Tv::Float64; vc::Float64=20.0,
                    sigma::Float64=3.0, obs::Vector{Float64}=Float64[], snap::Int=0)
    Nu = round(Int, Tu / d)
    Nv = round(Int, Tv / d)
    # centre of the diamond with south corner (i-1, j-1): x = (j - i) d / 2, m = j - i
    ms = collect(1-Nu:Nv-1)
    Vlat = Vfun(ms .* (d / 2))
    off = Nu                                   # Vlat[m + off] for m in 1-Nu:Nv-1
    c = d^2 / 8
    prev = [exp(-(j * d - vc)^2 / (2sigma^2)) for j in 0:Nv]
    cur = similar(prev)
    edge = prev[1]
    mobs = [round(Int, 2x / d) for x in obs]
    for (x, m) in zip(obs, mobs)
        @assert abs(m * d / 2 - x) < 1e-9 "observer $x not on the null grid"
    end
    ts = [Float64[] for _ in obs]
    ps = [Float64[] for _ in obs]
    # observer on the initial line u = 0
    for (k, m) in enumerate(mobs)
        if 0 <= m <= Nv
            push!(ts[k], m * d / 2); push!(ps[k], prev[m+1])
        end
    end
    S = snap > 0 ? fill(NaN, Nu ÷ snap + 1, Nv ÷ snap + 1) : zeros(0, 0)
    if snap > 0
        for j in 0:snap:Nv
            S[1, j÷snap+1] = log10(abs(prev[j+1]) + 1e-300)
        end
    end
    for i in 1:Nu
        cur[1] = edge
        @inbounds for j in 1:Nv
            W = cur[j]; E = prev[j+1]; So = prev[j]
            cur[j+1] = E + W - So - c * Vlat[j-i+off] * (E + W)
        end
        for (k, m) in enumerate(mobs)
            j = i + m
            if 0 <= j <= Nv
                push!(ts[k], (2i + m) * d / 2); push!(ps[k], cur[j+1])
            end
        end
        if snap > 0 && i % snap == 0
            for j in 0:snap:Nv
                S[i÷snap+1, j÷snap+1] = log10(abs(cur[j+1]) + 1e-300)
            end
        end
        prev, cur = cur, prev
    end
    return (; t=ts, psi=ps, snap=S, d, Nu, Nv)
end

# ---------------------------------------------------------------------------------------------
# Method of lines: sixth-order second difference, RK4.
# ---------------------------------------------------------------------------------------------
const C6 = (-49 / 18, 3 / 2, -3 / 20, 1 / 90)

function rhs!(dP, dQ, P, Q, V, ih2)
    n = length(P)
    @inbounds dP .= Q
    Threads.@threads for ch in 1:Threads.nthreads()
        lo = 1 + (ch - 1) * n ÷ Threads.nthreads()
        hi = ch * n ÷ Threads.nthreads()
        @inbounds for i in lo:hi
            if 4 <= i <= n - 3
                s = C6[1] * P[i] + C6[2] * (P[i-1] + P[i+1]) + C6[3] * (P[i-2] + P[i+2]) +
                    C6[4] * (P[i-3] + P[i+3])
            else
                g(k) = (1 <= k <= n) ? P[k] : 0.0      # zero values beyond both ends
                s = C6[1] * P[i] + C6[2] * (g(i - 1) + g(i + 1)) + C6[3] * (g(i - 2) + g(i + 2)) +
                    C6[4] * (g(i - 3) + g(i + 3))
            end
            dQ[i] = s * ih2 - V[i] * P[i]
        end
    end
    return nothing
end

"""
    mol_evolve(V, a, h, dt, T; x0, sigma, ingoing, obs, every)

Grid x_i = a + (i-1) h with V given there.  Data Psi = exp(-(x - x0)^2/(2 sigma^2)) and
Psi_t = 0 (or Psi_t = Psi_x, an ingoing pulse).  Records Psi at the grid points `obs` every
`every` steps, and (with `snap` > 0) the field on every `sstride`-th point every `snap` steps.
"""
function mol_evolve(V::Vector{Float64}, a::Float64, h::Float64, dt::Float64, T::Float64;
                    x0::Float64=10.0, sigma::Float64=2.0, ingoing::Bool=false,
                    obs::Vector{Float64}=Float64[], every::Int=1, snap::Int=0, sstride::Int=1)
    n = length(V)
    x = [a + (i - 1) * h for i in 1:n]
    P = @. exp(-(x - x0)^2 / (2sigma^2))
    Q = ingoing ? @.(-(x - x0) / sigma^2 * P) : zeros(n)
    io = [round(Int, (xo - a) / h) + 1 for xo in obs]
    for (xo, i) in zip(obs, io)
        @assert abs(x[i] - xo) < 1e-9 "observer $xo not on the grid"
    end
    nsteps = round(Int, T / dt)
    ts = Float64[0.0]
    ps = [Float64[P[i]] for i in io]
    k1P = similar(P); k1Q = similar(P); k2P = similar(P); k2Q = similar(P)
    k3P = similar(P); k3Q = similar(P); k4P = similar(P); k4Q = similar(P)
    tP = similar(P); tQ = similar(P)
    ih2 = 1 / h^2
    isub = 1:sstride:n
    tsnap = Float64[]
    snaps = Vector{Vector{Float64}}()
    if snap > 0
        push!(tsnap, 0.0); push!(snaps, P[isub])
    end
    for s in 1:nsteps
        rhs!(k1P, k1Q, P, Q, V, ih2)
        @. tP = P + dt / 2 * k1P; @. tQ = Q + dt / 2 * k1Q
        rhs!(k2P, k2Q, tP, tQ, V, ih2)
        @. tP = P + dt / 2 * k2P; @. tQ = Q + dt / 2 * k2Q
        rhs!(k3P, k3Q, tP, tQ, V, ih2)
        @. tP = P + dt * k3P; @. tQ = Q + dt * k3Q
        rhs!(k4P, k4Q, tP, tQ, V, ih2)
        @. P += dt / 6 * (k1P + 2k2P + 2k3P + k4P)
        @. Q += dt / 6 * (k1Q + 2k2Q + 2k3Q + k4Q)
        if s % every == 0
            push!(ts, s * dt)
            for (k, i) in enumerate(io)
                push!(ps[k], P[i])
            end
        end
        if snap > 0 && s % snap == 0
            push!(tsnap, s * dt); push!(snaps, P[isub])
        end
    end
    return (; t=ts, psi=ps, P, x, tsnap, snaps, xsnap=x[isub])
end

# ---------------------------------------------------------------------------------------------
# Matrix pencil (Hua-Sarkar) and window scans.
# ---------------------------------------------------------------------------------------------
"""
    matrix_pencil(y, dt, M; L)

Fits y_k = sum_{j=1}^M a_j exp(-i w_j k dt).  Returns (w, a, sv) with w sorted by |a|
(largest first) and the singular values sv of the Hankel matrix.
"""
function matrix_pencil(y::AbstractVector, dt::Real, M::Int; L::Int=length(y) ÷ 3)
    N = length(y)
    Y = [y[i+j-1] for i in 1:N-L, j in 1:L+1]
    Fs = svd(Y)
    Vr = Fs.V[:, 1:M]
    V1 = Vr[1:end-1, :]
    V2 = Vr[2:end, :]
    z = eigvals(V1 \ V2)
    w = im .* log.(complex.(z)) ./ dt
    Z = [zk^(k) for k in 0:N-1, zk in z]
    a = Z \ complex.(y)
    p = sortperm(abs.(a), rev=true)
    return w[p], a[p], Fs.S
end

"""
Scan of window starts t0 (width W, sample step ds) of the series (t, y): for each window the
frequencies with Re w > 0 from a pencil of order M, ordered by decay rate (fundamental first).
"""
function pencil_scan(t::Vector{Float64}, y::Vector{Float64}, t0s, W::Float64, M::Int;
                     ds::Float64=0.5)
    dtn = t[2] - t[1]
    stride = max(1, round(Int, ds / dtn))
    out = Vector{Vector{ComplexF64}}()
    for t0 in t0s
        i0 = searchsortedfirst(t, t0 - 1e-9)
        i1 = searchsortedlast(t, t0 + W + 1e-9)
        idx = i0:stride:i1
        w, a, _ = matrix_pencil(y[idx], stride * dtn, M)
        wp = w[real.(w) .> 1e-8]
        sort!(wp, by=z -> -imag(z))
        push!(out, wp)
    end
    return out
end

"""
    qnm_scan(t, y, guide; t0s, Ws, Ms, ds, K)

For every window width W in `Ws` and pencil order M in `Ms`, fits the windows [t0, t0 + W]
(t0 in `t0s`) and keeps the frequency closest to `guide`; then finds the most stable run of K
consecutive window starts.  Returns a vector of named tuples (W, M, w, spread, t0) sorted by
spread, the first being the retained estimate.
"""
function qnm_scan(t::Vector{Float64}, y::Vector{Float64}, guide::ComplexF64; t0s, Ws=(40.0, 60.0, 80.0),
                  Ms=(4, 6, 8), ds::Float64=0.5, K::Int=11)
    res = NamedTuple[]
    for W in Ws, M in Ms
        valid = [t0 for t0 in t0s if t0 + W <= t[end]]
        length(valid) < K && continue
        sc = pencil_scan(t, y, valid, W, M; ds=ds)
        z = ComplexF64[isempty(w) ? ComplexF64(NaN, NaN) : w[argmin(abs.(w .- guide))] for w in sc]
        good = .!isnan.(real.(z))
        sum(good) < K && continue
        zm, s, i = plateau(z, K)
        push!(res, (W=W, M=M, w=zm, spread=s, t0=valid[i], series=z, t0s=valid))
    end
    sort!(res, by=r -> r.spread)
    return res
end

"""
Annihilating filter: removes from the real sequence y the pair of poles z, conj(z) (the
fundamental mode exp(-i w t) and its mirror exp(i conj(w) t)), through
y'_k = y_{k+2} - 2 Re(z) y_{k+1} + |z|^2 y_k.
"""
annihilate(y::AbstractVector, z::Complex) =
    [y[k+2] - 2real(z) * y[k+1] + abs2(z) * y[k] for k in 1:length(y)-2]

"""Most stable run of K consecutive values of the complex series z: (mean, spread, first index)."""
function plateau(z::Vector{ComplexF64}, K::Int)
    best = (Inf, 0)
    for i in 1:length(z)-K+1
        seg = z[i:i+K-1]
        s = maximum(abs.(seg .- sum(seg) / K))
        s < best[1] && (best = (s, i))
    end
    i = best[2]
    seg = z[i:i+K-1]
    return sum(seg) / K, best[1], i
end

# ---------------------------------------------------------------------------------------------
# Growth and decay rates.
# ---------------------------------------------------------------------------------------------
"""Least-squares fit of log|y| = c + lam t (p = false) or c + lam t - p log t (p = true)."""
function growth_fit(t::AbstractVector, y::AbstractVector; withpow::Bool=false)
    ly = log.(abs.(y))
    A = withpow ? hcat(ones(length(t)), t, -log.(t)) : hcat(ones(length(t)), t)
    c = A \ ly
    res = maximum(abs.(A * c .- ly))
    return withpow ? (lam=c[2], p=c[3], res=res) : (lam=c[2], p=0.0, res=res)
end

"""Local logarithmic derivative d log|y| / dt by centred differences over a span of K samples."""
function local_rate(t::AbstractVector, y::AbstractVector, K::Int)
    ly = log.(abs.(y))
    n = length(t)
    idx = 1+K:n-K
    return t[idx], [(ly[i+K] - ly[i-K]) / (t[i+K] - t[i-K]) for i in idx]
end

# ---------------------------------------------------------------------------------------------
# Bound states of -d^2/dx^2 + V below the plateau value (Dirichlet box, second differences).
# ---------------------------------------------------------------------------------------------
function bound_states(V::Vector{Float64}, h::Float64, nev::Int)
    n = length(V)
    d = 2 / h^2 .+ V
    e = fill(-1 / h^2, n - 1)
    return eigvals(SymTridiagonal(d, e), 1:nev)
end

# ---------------------------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------------------------
function write_columns(path::String, header::Vector{String}, cols...; comment::String="",
                       digits::Int=15)
    mkpath(dirname(path))
    fm = Printf.Format("%.$(digits)e")
    open(path, "w") do io
        isempty(comment) || println(io, "# ", replace(comment, "\n" => "\n# "))
        println(io, join(header, ","))
        for i in eachindex(cols[1])
            println(io, join([Printf.format(fm, Float64(c[i])) for c in cols], ","))
        end
    end
end

end # module
