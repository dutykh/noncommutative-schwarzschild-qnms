#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""Closed forms of the effective potential for every l >= 1 and 1 <= |m| <= l, with checks.

Metric of Chaichian, Tureanu and Zet with f = 1 - alpha/r (type I).  Its deformation functions are
A0, B0, C0, D0 (radial) and, for j >= 1, D_j = kappa (1 + (-1)^j)/(32 r^2), where kappa = 1 for
the published metric and kappa = 5 when the Moyal term Theta^2/4 of g_phiphi is restored (then
D0 also gains 1/(4 r^2)).  The angular sums of the master equation are

    sigma1 = r^2 sum_j (a^j + d^j/2) D_j / kappa = (2l+1)(l(l+1) - m^2)/(64 |m|),
    sigma2 = r^2 sum_j  b^j D_j          / kappa = (2l+1-2|m|)/(32 |m|),

and the potential is V = f (l(l+1)/r^2 + alpha/r^3) + Theta^2 (V0 + Vhat), with V0 the j = 0
part and Vhat = kappa f r^-5 (-sigma1 r + sigma2 (r - 2 alpha)/2).  The script checks sigma1 and
sigma2 against the exact sums of the series, the closed form of V against a direct evaluation of
the master equation, and prints the value of V at the horizon.
"""
import sympy as sp

r, al, x, Th, k, eps = sp.symbols('r alpha x Theta k epsilon', positive=True)
j = sp.symbols('j', integer=True, nonnegative=True)
f = 1 - al/r
A0 = al*(8*r - 11*al)/(16*r**3*(r - al))
B0 = -al*(4*r - 3*al)/(16*r**3*(r - al))
C0 = (2*r**2 - 17*al*(r - al))/(32*r**3*(r - al))
D0pub = -al*(2*r - al)/(16*r**3*(r - al))


def sigma1(l, m):
    return sp.Rational((2*l + 1)*(l*(l + 1) - m*m), 64*abs(m))


def sigma2(l, m):
    return sp.Rational(2*l + 1 - 2*abs(m), 32*abs(m))


def V0_closed(l, m):
    """The j = 0 part of the potential for the metric of Chaichian, Tureanu and Zet in closed form,
    checked below against the series."""
    L = l*(l + 1)
    return ((4*r**3 - 270*al*r**2 + 709*al**2*r - 451*al**3)/(128*r**7)
            - L*(2*r**2 - 33*al*r + 39*al**2)/(32*r**6)
            + abs(m)*(2*l + 1)*(2*r**2 - 13*al*r + 15*al**2)/(64*r**6))


def V_closed(l, m, moyal=False):
    kappa = 5 if moyal else 1
    V0 = V0_closed(l, m)
    if moyal:   # D0 -> D0 + 1/(4 r^2): -a0 f/(4 r^4) + (1/(4 r^2)) d/dr*(r^2 d/dr*(1/(4 r^2)))
        V0 += f*(r - 2*al)/(8*r**5) - f*abs(m)*(2*l + 1)/(8*r**4)
    Vhat = kappa*f/r**5*(-sigma1(l, m)*r + sigma2(l, m)*(r - 2*al)/2)
    return f*(l*(l + 1)/r**2 + al/r**3) + Th**2*(V0 + Vhat)


def sum_rational(R, var):
    """Exact sum over var >= 1 of a rational function R with simple poles that decays at least like
    1/var^2: with R = sum_i c_i/(var - p_i) and sum_i c_i = 0, the sum equals
    -sum_i c_i digamma(1 - p_i), and the digamma values at rational points are exact."""
    num, den = sp.fraction(sp.cancel(sp.together(R)))
    assert sp.degree(num, var) <= sp.degree(den, var) - 2
    total = 0
    for p, mult in sp.roots(sp.Poly(den, var)).items():
        assert mult == 1
        total -= sp.cancel(num.subs(var, p)/sp.diff(den, var).subs(var, p))*sp.digamma(1 - p)
    return sp.simplify(sp.expand_func(total))


def series_sums(l, m):
    """Exact sums over j of (a^j + d^j/2) and b^j weighted by r^2 D_j / kappa = (1+(-1)^j)/32."""
    P = sp.assoc_legendre(l, m, x)
    N = sp.integrate(P**2, (x, -1, 1))
    # For even j = 2n the coefficients are rational functions of n: integrate x^(2n) against
    # polynomial densities term by term, int_{-1}^{1} x^(2n+q) dx = 2/(2n+q+1) for even q.
    n = sp.symbols('n', integer=True, positive=True)

    def moment(poly):            # int x^(2n) poly(x) dx over [-1, 1], poly a polynomial in x
        poly = sp.Poly(sp.expand(poly), x)
        return sum(cf*2/(2*n + q + 1) for (q,), cf in poly.terms() if q % 2 == 0)
    dens_a = sp.cancel(m**2*P**2/(1 - x**2))                       # polynomial in x
    dens_b = sp.expand(P**2)
    # d^j = (1/N) int P (1 - x^2) j x^(j-1) P' dx = (j/N) int x^(j-1) (1 - x^2) P P' dx
    dens_d = sp.expand((1 - x**2)*P*sp.diff(P, x))                  # odd polynomial in x
    a_n = moment(dens_a)/N
    b_n = moment(dens_b)/N
    poly_d = sp.Poly(dens_d, x)
    d_n = 2*n*sum(cf*2/(2*n - 1 + q + 1) for (q,), cf in poly_d.terms() if q % 2 == 1)/N
    s1 = sum_rational(sp.simplify(a_n + d_n/2)*sp.Rational(1, 16), n)
    s2 = sum_rational(sp.simplify(b_n)*sp.Rational(1, 16), n)
    return sp.nsimplify(sp.simplify(s1)), sp.simplify(s2)


def V_direct(l, m, moyal=False):
    """The master equation of Chen, Chiang and Tsao evaluated term by term (j = 0 part), with the
    j >= 1 sums taken from series_sums (independent of the closed forms above)."""
    kappa = 5 if moyal else 1
    D0 = D0pub + (sp.Rational(1, 4)/r**2 if moyal else 0)
    P = sp.assoc_legendre(l, m, x)
    N = sp.integrate(P**2, (x, -1, 1))
    a0 = sp.integrate(sp.cancel(m**2*P**2/(1 - x**2)), (x, -1, 1))/N
    c0 = sp.integrate(sp.expand(P*((1 - x**2)*sp.diff(P, x, 2) - 2*x*sp.diff(P, x))), (x, -1, 1))/N
    dd = lambda g: f*sp.diff(g, r)
    V0 = ((f/r)*sp.diff(f, r)*(A0 - B0) + (f/r**2)*(a0*(A0 - D0) - c0*(A0 - C0))
          + 1/(4*r**2)*dd(r**2*dd(A0 - B0 + C0 + D0)) - dd(dd(A0 - B0))/4)
    s1, s2 = series_sums(l, m)
    Vhat = -(f/r**2)*kappa*s1/r**2 + 1/(4*r**2)*dd(r**2*dd(kappa*s2/r**2))
    return f*(l*(l + 1)/r**2 + al/r**3) + Th**2*(V0 + Vhat), (s1, s2)


def horizon_value(V):
    """V at the zero of F = f + Theta^2 alpha (6 r - 7 alpha)/(16 r^4), to order Theta^2."""
    rh = al + Th**2/(16*al)
    return sp.factor(sp.series(V.subs(r, rh), Th, 0, 3).removeO())


if __name__ == '__main__':
    print('l m  | series sums equal closed forms | V_closed - V_direct (published, Moyal) | '
          'V(r_h) published, Moyal')
    for l in range(1, 7):
        for m in range(1, l + 1):
            Vp, (s1, s2) = V_direct(l, m, moyal=False)
            Vm, _ = V_direct(l, m, moyal=True)
            ok = (sp.simplify(s1 - sigma1(l, m)) == 0, sp.simplify(s2 - sigma2(l, m)) == 0)
            dp = sp.simplify(V_closed(l, m) - Vp)
            dm = sp.simplify(V_closed(l, m, True) - Vm)
            hp = horizon_value(V_closed(l, m))
            hm = horizon_value(V_closed(l, m, True))
            pred = -sp.Rational(3*l*(l + 1) - (2*l + 1)*m, 16)*Th**2/al**4
            print(f'{l} {m}  | {ok} | {dp}, {dm} | {hp}, {hm} | prediction {sp.simplify(hp - pred) == 0}',
                  flush=True)
