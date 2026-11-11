# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Reference quasinormal frequencies of the Schwarzschild black hole for a scalar field, by
# Leaver's continued fraction (Proc. R. Soc. Lond. A 402 (1985) 285) in units 2M = 1, with
# rho = -i w_L.  Three-term recurrence  alpha_n a_{n+1} + beta_n a_n + gamma_n a_{n-1} = 0:
#   alpha_n = n^2 + (2 rho + 2) n + 2 rho + 1,
#   beta_n  = -(2 n^2 + (8 rho + 2) n + 8 rho^2 + 4 rho + l(l+1) + 1),
#   gamma_n = n^2 + 4 rho n + 4 rho^2.
# The n-th inversion of the continued fraction is used for the n-th overtone, evaluated
# bottom-up from depth J with the leading asymptotic remainder; roots by the secant method.  Returned in
# units M = 1 (M w = w_L / 2).

module Leaver

export schwarzschild_qnm

function cf_value(wL::Complex{T}, l::Int, inv::Int, J::Int) where {T}
    rho = -im * wL
    al(n) = n^2 + (2rho + 2) * n + 2rho + 1
    be(n) = -(2n^2 + (8rho + 2) * n + 8rho^2 + 4rho + l * (l + 1) + 1)
    ga(n) = n^2 + 4rho * n + 4rho^2
    # R_n = a_{n+1}/a_n = -gamma_{n+1}/(beta_{n+1} + alpha_{n+1} R_{n+1}).  Substituting
    # R_n = 1 + c n^(-1/2) + O(1/n) in the recurrence gives c^2 = 2 rho; the minimal solution
    # has Re c < 0, so the tail is started at R_J = 1 - sqrt(2 rho / J) (principal root).
    R = 1 - sqrt(2rho / J)
    for n in J-1:-1:inv
        R = -ga(n + 1) / (be(n + 1) + al(n + 1) * R)
    end
    inv == 0 && return be(0) + al(0) * R
    # S_n = a_{n-1}/a_n from the start: S_1 = -alpha_0/beta_0, S_{n+1} = -alpha_n/(beta_n + gamma_n S_n)
    S = -al(0) / be(0)
    for n in 1:inv-1
        S = -al(n) / (be(n) + ga(n) * S)
    end
    return be(inv) + al(inv) * R + ga(inv) * S
end

function schwarzschild_qnm(l::Int, n::Int, guess::Complex{T}; J::Int=200_000, tol=nothing) where {T}
    tol === nothing && (tol = 100eps(T))
    w0 = 2guess                                # to units 2M = 1
    w1 = w0 * (1 + T(1e-4))
    f0 = cf_value(w0, l, n, J)
    f1 = cf_value(w1, l, n, J)
    for _ in 1:200
        w2 = w1 - f1 * (w1 - w0) / (f1 - f0)
        w0, f0 = w1, f1
        w1 = w2
        f1 = cf_value(w1, l, n, J)
        abs(w1 - w0) < tol * abs(w1) && break
    end
    return w1 / 2
end

end # module
