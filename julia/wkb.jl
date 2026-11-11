# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Higher-order WKB formula for the quasinormal frequencies of  Psi'' + (w^2 - V(x)) Psi = 0,  x the
# tortoise coordinate: Iyer and Will, Phys. Rev. D 35 (1987) 3621 (third order); Konoplya,
# Phys. Rev. D 68 (2003) 024018 (sixth order); Matyjasek and Opala, Phys. Rev. D 96 (2017) 024011
# (orders up to 13 and Pade approximants); Konoplya, Zhidenko and Zinhailo, Class. Quantum Grav. 36
# (2019) 155002 (Pade approximants P_{nt/mt}, nt + mt = order).
#
# Near the maximum x0 of the barrier put z = x - x0 and V = v0 + v2 z^2 + v3 z^3 + ...  (v2 < 0).
# The WKB series at the top of the barrier is the Rayleigh-Schroedinger series of the anharmonic
# oscillator
#     -phi'' + (v2 z^2 + g v3 z^3 + g^2 v4 z^4 + ...) phi = E phi,      E = w^2 - v0,
# continued to the inverted oscillator by the Gaussian factor exp(-Om z^2/2), Om = -i sqrt(-v2):
# the purely outgoing conditions on both sides of the barrier select the levels
# E0 = Om (2n + 1) = -i (n + 1/2) sqrt(-2 V''(x0)), the first-order formula of Schutz and Will, and
# the coefficient E_m of g^m (zero for odd m) is the correction of WKB order m/2 + 1.  The formula of
# order N is
#     w^2 = v0 + E0 + E2 + ... + E_{2N-2},
# and its last term involves the derivatives of V up to order 2N.  With the order parameter eps of
# Matyjasek-Opala and Konoplya-Zhidenko-Zinhailo, the term of order k carries eps^k,
#     P_N(eps) = v0 + sum_{k=1}^{N} E_{2k-2} eps^k,     w^2 = P_N(1),
# and the Pade approximant P_{nt/mt}(eps) of P_N (numerator degree nt, denominator degree mt,
# nt + mt = N) gives the improved value w^2 = P_{nt/mt}(1).
#
# The potential is given as a rational function V = Vn(u)/Vd(u) of u = 1/r, with dr/dx = F(u), so
# that d/dx = -u^2 F(u) d/du.  The Taylor coefficients of V(x) at the peak follow from the Taylor
# series of u(x), solution of du/dx = -u^2 F(u), composed with V; all arithmetic is carried out on
# truncated power series in the precision of the input (BigFloat in practice).

module WKB

using LinearAlgebra
using Polynomials

export peak_u, rstar_taylor, rs_corrections, wkb_poly, pade1, omega_from_w2, wkb_omegas,
       iyer_will3, series_mul, series_div, series_sqrt, series_compose

# ---------------------------------------------------------------------------------------------
# Truncated power series: c[k+1] is the coefficient of s^k, k = 0..K.
# ---------------------------------------------------------------------------------------------

function series_mul(a::AbstractVector{T}, b::AbstractVector{T}, K::Int) where {T}
    c = zeros(T, K + 1)
    for i in 0:min(K, length(a) - 1)
        ai = a[i+1]
        iszero(ai) && continue
        for j in 0:min(K - i, length(b) - 1)
            c[i+j+1] += ai * b[j+1]
        end
    end
    return c
end

"""Quotient a/b of two truncated series, b[1] nonzero."""
function series_div(a::AbstractVector{T}, b::AbstractVector{T}, K::Int) where {T}
    c = zeros(T, K + 1)
    for k in 0:K
        s = k < length(a) ? a[k+1] : zero(T)
        for j in 1:min(k, length(b) - 1)
            s -= b[j+1] * c[k-j+1]
        end
        c[k+1] = s / b[1]
    end
    return c
end

"""Square root of a truncated series with a positive constant term."""
function series_sqrt(a::AbstractVector{T}, K::Int) where {T}
    c = zeros(T, K + 1)
    c[1] = sqrt(a[1])
    for k in 1:K
        s = k < length(a) ? a[k+1] : zero(T)
        for j in 1:k-1
            s -= c[j+1] * c[k-j+1]
        end
        c[k+1] = s / (2c[1])
    end
    return c
end

"""p(u(s)) for a polynomial p and a truncated series u, by Horner's rule."""
function series_compose(p::Polynomial, u::AbstractVector{T}, K::Int) where {T}
    cs = coeffs(p)
    r = zeros(T, K + 1)
    for c in Iterators.reverse(cs)
        r = series_mul(r, u, K)
        r[1] += T(c)
    end
    return r
end

# ---------------------------------------------------------------------------------------------
# The barrier peak and the Taylor coefficients of V in the tortoise coordinate.
# ---------------------------------------------------------------------------------------------

"""
    peak_u(Vn, Vd, uh; ngrid=20000)

Point u0 in (0, uh) where V = Vn/Vd is largest (the top of the barrier), refined by Newton's method
on Vn' Vd - Vn Vd' = 0; also returns the list of all interior local maxima found on the grid.
"""
function peak_u(Vn::Polynomial{T}, Vd::Polynomial{T}, uh::T; ngrid::Int=20000) where {T}
    V(u) = Vn(u) / Vd(u)
    us = [uh * T(j) / T(ngrid + 1) for j in 1:ngrid]
    vals = V.(us)
    locmax = Int[]
    for j in 2:ngrid-1
        vals[j] >= vals[j-1] && vals[j] >= vals[j+1] && push!(locmax, j)
    end
    isempty(locmax) && error("no interior maximum of the potential on (0, u_h)")
    jbest = locmax[argmax(vals[locmax])]
    P = derivative(Vn) * Vd - Vn * derivative(Vd)
    dP = derivative(P)
    newton(u) = begin
        for _ in 1:100
            du = P(u) / dP(u)
            u -= du
            abs(du) < 10eps(T) * abs(u) && break
        end
        u
    end
    u0 = newton(us[jbest])
    @assert 0 < u0 < uh "peak outside (0, u_h)"
    maxima = [newton(us[j]) for j in locmax]
    return u0, maxima
end

"""
    rstar_taylor(F, Vn, Vd, u0, K)

Taylor coefficients v[k+1] = v_k, k = 0..K, of V(x) = Vn(u)/Vd(u) at the point x0 where u = u0,
x the tortoise coordinate with dx = dr/F, i.e. du/dx = -u^2 F(u).  Also returns the series of u.
"""
function rstar_taylor(F::Polynomial{T}, Vn::Polynomial{T}, Vd::Polynomial{T}, u0::T, K::Int) where {T}
    u = Polynomial([zero(T), one(T)])
    G = u^2 * F
    return rstar_taylor((us, K) -> series_compose(G, us, K),
                        (us, K) -> series_div(series_compose(Vn, us, K), series_compose(Vd, us, K), K),
                        u0, K)
end

"""
    rstar_taylor(Gser, Vser, u0, K)

Same with G(u) = u^2 F(u) and V(u) given as maps from the truncated series of u to the truncated
series of G(u) and V(u).
"""
function rstar_taylor(Gser::Function, Vser::Function, u0::T, K::Int) where {T}
    us = zeros(T, K + 1)
    us[1] = u0
    # Picard iteration on u(x) = u0 - int_0^x G(u): each sweep fixes one more Taylor coefficient.
    for _ in 1:K
        g = Gser(us, K)
        for k in 1:K
            us[k+1] = -g[k] / k
        end
    end
    return Vser(us, K), us
end

# ---------------------------------------------------------------------------------------------
# Rayleigh-Schroedinger recursion at the top of the barrier.
# ---------------------------------------------------------------------------------------------

"""
    rs_corrections(v, n, M)

Coefficients E_0, ..., E_M (vector index m+1) of the formal eigenvalue series of
-phi'' + (v2 z^2 + sum_{k=3}^{M+2} g^{k-2} v_k z^k) phi = E phi on the branch E_0 = Om (2n+1),
Om = -i sqrt(-v2), with v[k+1] = v_k (v_0 and v_1 are not used).

Writing phi = exp(-Om z^2/2) q(z), the operator becomes L q = -q'' + 2 Om z q' + Om q, which maps
z^j to Om (2j+1) z^j - j (j-1) z^(j-2).  At order g^m the polynomial q_m solves
(L - E_0) q_m = - sum_k v_k z^k q_{m-k+2} + sum_{j=1}^{m} E_j q_{m-j},  solved from the top degree
down; the coefficient of z^n gives E_m, and q_m is normalized by a zero coefficient of z^n.
"""
function rs_corrections(v::AbstractVector{T}, n::Int, M::Int) where {T<:Real}
    @assert length(v) >= M + 3 "Taylor coefficients up to order M + 2 are needed"
    @assert v[3] < 0 "the potential has no maximum at z = 0"
    C = Complex{T}
    Om = -im * sqrt(-v[3])
    E = zeros(C, M + 1)
    q = Vector{Vector{C}}(undef, M + 1)
    q0 = zeros(C, n + 1)
    q0[n+1] = one(C)
    for j in n-2:-2:0
        q0[j+1] = (j + 2) * (j + 1) * q0[j+3] / (2Om * (j - n))
    end
    q[1] = q0
    E[1] = Om * (2n + 1)
    for m in 1:M
        D = n + 3m
        R = zeros(C, D + 1)
        for k in 3:m+2
            vk = v[k+1]
            for (i, c) in enumerate(q[m-k+3])          # coefficient of z^(i-1) in q_{m-k+2}
                R[i+k] -= vk * c
            end
        end
        for j in 1:m-1
            Ej = E[j+1]
            for (i, c) in enumerate(q[m-j+1])
                R[i] += Ej * c
            end
        end
        qm = zeros(C, D + 1)
        for j in D:-1:0
            rj = R[j+1] + (j + 2 <= D ? (j + 2) * (j + 1) * qm[j+3] : zero(C))
            if j > n
                qm[j+1] = rj / (2Om * (j - n))
            elseif j == n
                E[m+1] = -rj
            else
                qm[j+1] = (rj + E[m+1] * q0[j+1]) / (2Om * (j - n))
            end
        end
        q[m+1] = qm
    end
    return E
end

"""Coefficients a_0..a_N of P_N(eps) = v0 + sum_k E_{2k-2} eps^k for the overtone n."""
function wkb_poly(v::AbstractVector{T}, n::Int, N::Int) where {T<:Real}
    E = rs_corrections(v, n, 2N - 2)
    a = Vector{Complex{T}}(undef, N + 1)
    a[1] = v[1]
    for k in 1:N
        a[k+1] = E[2k-1]
    end
    return a, E
end

"""
    pade1(a, nt, mt)

Value at eps = 1 of the Pade approximant with numerator degree nt and denominator degree mt of
the polynomial sum_j a[j+1] eps^j (nt + mt at most its degree); mt = 0 gives the partial sum.
"""
function pade1(a::AbstractVector{C}, nt::Int, mt::Int) where {C}
    @assert nt >= 0 && mt >= 0 && nt + mt <= length(a) - 1
    c(i) = i < 0 ? zero(C) : a[i+1]
    mt == 0 && return sum(a[1:nt+1])
    A = zeros(C, mt, mt)
    rhs = zeros(C, mt)
    for (row, i) in enumerate(nt+1:nt+mt)
        for k in 1:mt
            A[row, k] = c(i - k)
        end
        rhs[row] = -c(i)
    end
    B = vcat(one(C), A \ rhs)
    num = zero(C)
    for i in 0:nt, k in 0:min(i, mt)
        num += B[k+1] * c(i - k)
    end
    return num / sum(B)
end

"""Root of w^2 with positive real part."""
omega_from_w2(w2) = sqrt(w2)

"""
    wkb_omegas(F, Vn, Vd, uh, n; N=13)

All WKB values for the overtone n: `plain[k]` the formula of order k (k = 1..N), and
`pade[(nt, mt)]` the Pade value with nt + mt at most N; also the peak u0 and the Taylor
coefficients of V at the peak.
"""
function wkb_omegas(F::Polynomial{T}, Vn::Polynomial{T}, Vd::Polynomial{T}, uh::T, n::Int;
                    N::Int=13) where {T}
    u0, maxima = peak_u(Vn, Vd, uh)
    v, _ = rstar_taylor(F, Vn, Vd, u0, 2N)
    a, E = wkb_poly(v, n, N)
    plain = [omega_from_w2(sum(a[1:k+1])) for k in 1:N]
    pade = Dict{Tuple{Int,Int},Complex{T}}()
    for k in 1:N, mt in 0:k
        pade[(k - mt, mt)] = omega_from_w2(pade1(a, k - mt, mt))
    end
    return (; plain, pade, u0, maxima, v, a, E)
end

"""
Third-order formula of Iyer and Will written with the derivatives V_k = k! v_k of V at the peak
(the form with real second-order and imaginary third-order corrections):
w^2 = V0 + L2 - i al sqrt(-2 V2) (1 + O3),  al = n + 1/2,
L2 = (1/8)(V4/V2)(1/4 + al^2) - (1/288)(V3/V2)^2 (7 + 60 al^2),
O3 = (1/(-2V2)) [ (5/6912)(V3/V2)^4 (77 + 188 al^2) - (1/384)(V3^2 V4/V2^3)(51 + 100 al^2)
      + (1/2304)(V4/V2)^2 (67 + 68 al^2) + (1/288)(V3 V5/V2^2)(19 + 28 al^2)
      - (1/288)(V6/V2)(5 + 4 al^2) ].
"""
function iyer_will3(v::AbstractVector{T}, n::Int) where {T<:Real}
    V0 = v[1]
    V2, V3, V4, V5, V6 = 2v[3], 6v[4], 24v[5], 120v[6], 720v[7]
    al = T(n) + T(1) / 2
    a2 = al^2
    L2 = (V4 / V2) * (T(1) / 4 + a2) / 8 - (V3 / V2)^2 * (7 + 60a2) / 288
    O3 = ((V3 / V2)^4 * (77 + 188a2) * T(5) / 6912 - (V3^2 * V4 / V2^3) * (51 + 100a2) / 384 +
          (V4 / V2)^2 * (67 + 68a2) / 2304 + (V3 * V5 / V2^2) * (19 + 28a2) / 288 -
          (V6 / V2) * (5 + 4a2) / 288) / (-2V2)
    return V0 + L2 - im * al * sqrt(-2V2) * (1 + O3), L2, O3
end

end # module
