#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""The radial equation of a massless scalar field on a diagonally deformed Schwarzschild metric.

Metric, to first order in eps:
    g_tt = -f (1 + eps a),  g_rr = (1 + eps b)/f,  g_thth = r^2 (1 + eps c),
    g_phph = r^2 sin^2(theta) (1 + eps d),        f = 1 - alpha/r,
with a, b, c, d functions of r and x = cos(theta).  For Phi = exp(-i w t + i m phi) R(r) P(x),
P = P_l^m, the Klein-Gordon equation is projected on P (the coupling to other l is of second
order in eps), which gives  (p R')' + (w^2 q - W) R = 0  with

    p = r^2 f (1 + eps <s - b>),   q = (r^2/f) (1 + eps <s - a>),   s = (a + b + c + d)/2,
    W = l(l+1) + eps [ int (1 - x^2)(s - c) P'^2 dx + m^2 int (s - d) P^2/(1 - x^2) dx ] / N,

where <h> = int h P^2 dx / N and N = int P^2 dx.  The Liouville transformation
dr/dr* = sqrt(p/q), R = psi/(p q)^(1/4) turns it into psi'' + (w^2 - V) psi = 0 with

    V = W/q + rho^{-1} d^2 rho/dr*^2,   rho = (p q)^(1/4).

This route is independent of the derivation of Chen, Chiang and Tsao, Phys. Rev. D 106 (2022)
044068; the script compares the two for generic deformations that are polynomials of degree two
in x.
"""
import sympy as sp

r, al, x, eps, w = sp.symbols('r alpha x epsilon omega')
f = 1 - al/r


def legendre(l, m):
    return sp.assoc_legendre(l, m, x)


def ddrs(g, F):
    """Derivative with respect to the tortoise coordinate, dr/dr* = F."""
    return F*sp.diff(g, r)


def first_order(expr):
    return sp.series(sp.expand(expr), eps, 0, 2).removeO()


def potential_direct(l, m, a, b, c, d):
    """Effective potential from the projected Klein-Gordon operator (first order in eps).

    To first order, dr/dr* = sqrt(p/q) = f (1 + eps (beta - alpha_q)/2) and
    rho = (p q)^(1/4) = r (1 + eps (alpha_q + beta)/4), both written out exactly."""
    P = legendre(l, m)
    N = sp.integrate(P**2, (x, -1, 1))
    s = (a + b + c + d)/2
    mean = lambda h: sp.integrate(sp.expand(h*P**2), (x, -1, 1))/N
    alpha_q = sp.expand(mean(s - a))
    beta_p = sp.expand(mean(s - b))
    gam = sp.expand((sp.integrate(sp.expand((1 - x**2)*(s - c)*sp.diff(P, x)**2), (x, -1, 1))
                     + (m**2*sp.integrate(sp.cancel((s - d)*P**2/(1 - x**2)), (x, -1, 1)) if m else 0))/N)
    F = f*(1 + eps*(beta_p - alpha_q)/2)
    rho = r*(1 + eps*(alpha_q + beta_p)/4)
    Wq = f/r**2*(l*(l + 1) + eps*(gam - l*(l + 1)*alpha_q))
    V = Wq + ddrs(ddrs(rho, F), F)/r - eps*ddrs(ddrs(r, f), f)*(alpha_q + beta_p)/(4*r)
    return first_order(V), first_order(F)


def potential_chen(l, m, Aj, Bj, Cj, Dj):
    """Effective potential (4.29) and tortoise relation (4.24) of Chen, Chiang and Tsao,
    for a diagonal metric with deformations  A = sum_j Aj[j] x^j, etc."""
    P = legendre(l, m)
    N = sp.integrate(P**2, (x, -1, 1))
    J = len(Aj)
    acoef = [(m**2*sp.integrate(sp.cancel(x**j*P**2/(1 - x**2)), (x, -1, 1))/N if m else 0)
             for j in range(J)]
    bcoef = [sp.integrate(x**j*P**2, (x, -1, 1))/N for j in range(J)]
    ccoef = [sp.integrate(sp.expand(x**j*P*((1 - x**2)*sp.diff(P, x, 2) - 2*x*sp.diff(P, x))),
                          (x, -1, 1))/N for j in range(J)]
    dcoef = [sp.integrate(sp.expand(P*(1 - x**2)*sp.diff(x**j, x)*sp.diff(P, x)), (x, -1, 1))/N
             for j in range(J)]
    F0 = f                                # zeroth order dr/dr* inside the eps terms
    V = l*(l + 1)*f/r**2 + f/r*sp.diff(f, r)*(1 + eps*sum(bcoef[j]*(Aj[j] - Bj[j]) for j in range(J)))
    V += eps*f/r**2*sum(acoef[j]*(Aj[j] - Dj[j]) - ccoef[j]*(Aj[j] - Cj[j])
                        - dcoef[j]/2*(Aj[j] + Bj[j] - Cj[j] + Dj[j]) for j in range(J))
    V += eps/(4*r**2)*sum(ddrs(bcoef[j]*r**2*ddrs(Aj[j] - Bj[j] + Cj[j] + Dj[j], F0), F0)
                          for j in range(J))
    V -= eps*sum(bcoef[j]/4*ddrs(ddrs(Aj[j] - Bj[j], F0), F0) for j in range(J))
    F = f*(1 + eps/2*sum(bcoef[j]*(Aj[j] - Bj[j]) for j in range(J)))
    return sp.expand(V), sp.expand(F)


if __name__ == '__main__':
    # 1. Generic radial functions, deformations quadratic in x (j = 0, 1, 2).
    J = 3
    Aj = [sp.Function(f'A{j}')(r) for j in range(J)]
    Bj = [sp.Function(f'B{j}')(r) for j in range(J)]
    Cj = [sp.Function(f'C{j}')(r) for j in range(J)]
    Dj = [sp.Function(f'D{j}')(r) for j in range(J)]
    a = sum(Aj[j]*x**j for j in range(J))
    b = sum(Bj[j]*x**j for j in range(J))
    c = sum(Cj[j]*x**j for j in range(J))
    d = sum(Dj[j]*x**j for j in range(J))
    print('Generic deformations a, b, c, d of degree 2 in x = cos(theta):')
    for (l, m) in [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 2)]:
        Vd, Fd = potential_direct(l, m, a, b, c, d)
        Vc, Fc = potential_chen(l, m, Aj, Bj, Cj, Dj)
        dV = sp.cancel(sp.together(sp.expand(Vd - Vc)))
        dF = sp.cancel(sp.together(sp.expand(Fd - Fc)))
        print(f'  l={l} m={m}:  V_direct - V_Chen = {dV},   F_direct - F_Chen = {dF}', flush=True)
