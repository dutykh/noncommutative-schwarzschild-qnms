# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# The discretised quasinormal problem as a nonlinear eigenvalue problem
#     T(w) a = 0,   T(w) = M1 + w Mw + w^2 Mw2 + k(w) (Mk + w Mkw),   k(w) = w sqrt(1 - V_h/w^2),
# where k is analytic off the segment between the branch points +-sqrt(V_h) (on the imaginary axis
# when V_h < 0).  The same collocation matrices are given to three solvers of NEP-PACK: NLEIGS
# (rational Krylov with poles on the cut), Beyn's contour integrals (circles avoiding the cut) and
# Newton's method (augmented Newton) for refinement.

module NEPSolvers

using LinearAlgebra
using NonlinearEigenproblems
using ..NCQNM

export qnm_nep, qnm_nep_rotated, run_nleigs, run_beyn, refine_newton, cut_points

"""Matrix version of k: S sqrt(I - V_h S^{-2}), principal square root (same cut as the scalar k)."""
kmat(S, Vh) = S * sqrt(I - Vh * inv(S * S))

"""SPMF form of T(w) from the collocation matrices of NCQNM.discretize."""
function qnm_nep(d)
    M = d.M
    Vh = d.Vh
    f1(S) = one(S)
    fw(S) = S
    fw2(S) = S * S
    fk(S) = kmat(S, Vh)
    fkw(S) = S * kmat(S, Vh)
    return SPMF_NEP([M[:one], M[:w], M[:w2], M[:k], M[:kw]], [f1, fw, fw2, fk, fkw])
end

"""
SPMF form of the rotated problem T(i lam): the variable lam = -i w puts the cut on the real
segment [-sqrt|V_h|, sqrt|V_h|] (V_h < 0), where the NLEIGS implementation of NEP-PACK requires
its poles.  With w = i lam: k(w) = i lam sqrt(1 + V_h/lam^2).
"""
function qnm_nep_rotated(d)
    M = d.M
    Vh = d.Vh
    kr(S) = im * S * sqrt(I + Vh * inv(S * S))
    f1(S) = one(S)
    fw(S) = im * S
    fw2(S) = -(S * S)
    fk(S) = kr(S)
    fkw(S) = (im * S) * kr(S)
    return SPMF_NEP([M[:one], M[:w], M[:w2], M[:k], M[:kw]], [f1, fw, fw2, fk, fkw])
end

"""Points of the cut (the segment between +-sqrt(V_h)), clustered at the branch points."""
function cut_points(Vh, n)
    b = sqrt(complex(Vh))                       # i sqrt|V_h| when V_h < 0
    t = cos.(range(0, pi, length = n))           # Chebyshev clustering
    return ComplexF64.(b .* t)
end

"""
NLEIGS on a polygonal target set Sigma (vertices, in the w plane) for the rotated problem
(qnm_nep_rotated), with poles on the cut; returns the eigenvalues in the w plane.
"""
function run_nleigs(nep_rot, d, Sigma; npoles = 200, tol = 1e-11, maxdgr = 120, maxit = 300)
    b = sqrt(abs(real(d.Vh)))
    Xi = collect(b .* cos.(range(0, pi, length = npoles)))       # real poles on the rotated cut
    Sigma_rot = ComplexF64.(-im .* Sigma)                          # lam = -i w
    lam, X, res = nleigs(nep_rot, Sigma_rot; Ξ = Xi, tol = tol, maxdgr = maxdgr,
                         maxit = maxit, isfunm = false, leja = 2, logger = 0)
    return im .* lam, X, res
end

"""Beyn's contour integral on the circle |w - c| = rho (which must avoid the cut)."""
function run_beyn(nep, c, rho; neigs = 6, N = 600, k = 12)
    lam, V = contour_beyn(nep; σ = ComplexF64(c), radius = rho, neigs = neigs, N = N, k = k,
                          logger = 0)
    return lam, V
end

"""Augmented Newton refinement from an approximate eigenpair."""
function refine_newton(nep, lam0, v0; tol = 1e-13, maxit = 50)
    lam, v = augnewton(nep; λ = lam0, v = v0, tol = tol, maxit = maxit, logger = 0)
    return lam, v
end

end # module
