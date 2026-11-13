#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""The model of the axis layer: the angular Laplacian of d theta^2 + a^2 d phi^2, a^2 = sin^2 + lam cos^2.

Near the axis the azimuthal circles of the second-order metric keep the radius of the axis term, and
at a fixed radius the angular operator on exp(i m phi) P(theta) becomes
    A P = -(1/a)(a P')' + m^2 P / a^2 = mu P,   0 < theta < pi,   lam = kappa Theta^2 / r^2,
whose ends theta = 0 and pi are circles of radius sqrt(lam).  The limit of polynomial truncations of
the axis term gives the first-order eigenvalue
    mu_F = L - lam (2l+1)(L - m^2)/(4|m|),   L = l(l+1).
This script checks, in exact arithmetic, that A sin = 2 sin for m = 1 and every lam (the Dirichlet
eigenvalue of l = 1 is exactly 2), and computes numerically the first-order coefficients of the
Dirichlet and Neumann eigenvalues, (mu(lam) - L)/lam at lam = 1e-5, 2e-5, 4e-5 extrapolated by
Richardson, against the matched expansion
    mu_D,N = mu_F +- lam L (2l+1)/8   (|m| = 1),      mu_D,N = mu_F to first order   (|m| >= 2).
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

th, lam = sp.symbols('theta lambda', positive=True)
a = sp.sqrt(sp.sin(th)**2 + lam*sp.cos(th)**2)
P = sp.sin(th)
residual = sp.simplify(-sp.diff(a*sp.diff(P, th), th)/a + P/a**2 - 2*P)
print(f"exact: A sin(theta) - 2 sin(theta) = {residual}  (m = 1, every lambda)")


def shoot(mu, lm, bc, m):
    """P(pi) (Dirichlet) or a P'(pi) (Neumann) for the solution started with the condition at 0."""
    def rhs(t, y):
        p, q = y                     # q = a P'
        av = np.sqrt(np.sin(t)**2 + lm*np.cos(t)**2)
        return [q/av, -mu*av*p + m*m*p/av]
    y0 = [0.0, np.sqrt(lm)] if bc == 'D' else [1.0, 0.0]
    sol = solve_ivp(rhs, [0.0, np.pi], y0, method='DOP853', rtol=1e-13, atol=1e-30)
    return sol.y[0, -1] if bc == 'D' else sol.y[1, -1]


def eigenvalue(l, m, lm, bc):
    L = l*(l + 1)
    return brentq(lambda mu: shoot(mu, lm, bc, m), L - 0.3, L + 0.3, xtol=1e-15, rtol=1e-15)


print("first-order coefficients of the eigenvalues (Richardson at lambda = 1e-5, 2e-5, 4e-5)")
print(" l  m  cond   computed      matched expansion   limit of truncations")
for l, m in ((1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 3)):
    L = l*(l + 1)
    muF = -(2*l + 1)*(L - m*m)/(4*m)
    for bc in ('D', 'N'):
        q = [(eigenvalue(l, m, lm, bc) - L)/lm for lm in (1e-5, 2e-5, 4e-5)]
        mu1 = (8*q[0] - 6*q[1] + q[2])/3
        pred = muF + ((L*(2*l + 1)/8 if bc == 'D' else -L*(2*l + 1)/8) if m == 1 else 0.0)
        print(f"{l:2d} {m:2d}   {bc}   {mu1:+11.5f}   {pred:+11.5f}        {muF:+11.5f}")
