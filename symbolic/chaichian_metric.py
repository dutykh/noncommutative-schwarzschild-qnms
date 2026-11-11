#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""The deformed Schwarzschild metric of Chaichian, Tureanu and Zet, recomputed from its tetrads.

Coordinates (x^1, x^2, x^3, x^0) = (r, theta, phi, t), noncommutativity [r, theta] = i Theta,
frame metric eta = diag(1, 1, 1, -1).  The deformed tetrad components are equations (4.2)-(4.6)
of Chaichian, Tureanu and Zet, Phys. Lett. B 660 (2008) 573, and the deformed metric is

    g_{mu nu} = (1/2) eta_{ab} ( e^a_mu * e^{b+}_nu + e^b_nu * e^{a+}_mu ),

with the Moyal product truncated at second order,

    f * g = f g + (i/2) Theta (f_r g_th - f_th g_r)
                - (Theta^2/8) (f_rr g_thth - 2 f_rth g_rth + f_thth g_rr) + O(Theta^3).

The script compares the result with their equations (4.7) (general A) and (4.9) (A^2 = 1 - alpha/r),
both with the star product and with the ordinary product, and prints the differences.
"""
import sympy as sp

r, th, Th, al = sp.symbols('r theta Theta alpha', positive=True)
A = sp.Function('A')(r)
A1, A2, A3 = A.diff(r), A.diff(r, 2), A.diff(r, 3)
I = sp.I

# Deformed tetrads e^a_mu, a and mu in (1, 2, 3, 0); equations (4.2)-(4.6).
e = {
    (1, 1): 1/A + A2/8*Th**2,
    (1, 2): -I/4*(A + 2*r*A1)*Th,
    (2, 2): r + sp.Rational(1, 32)*(7*A*A1 + 12*r*A1**2 + 12*r*A*A2)*Th**2,
    (3, 3): r*sp.sin(th) - I/4*sp.cos(th)*Th
            + sp.Rational(1, 8)*(2*r*A1**2 + r*A*A2 + 2*A*A1 - A1/A)*sp.sin(th)*Th**2,
    (0, 0): A + sp.Rational(1, 8)*(2*r*A1**3 + 5*r*A*A1*A2 + r*A**2*A3 + 2*A*A1**2 + A**2*A2)*Th**2,
}
eta = {1: 1, 2: 1, 3: 1, 0: -1}
frames, coords = (1, 2, 3, 0), (1, 2, 3, 0)


def tet(a, mu):
    return e.get((a, mu), sp.Integer(0))


def conj(expr):
    """Complex conjugate of a tetrad component: i -> -i (all other symbols are real)."""
    return expr.subs(I, -I)


def star(f, g, moyal=True):
    """Moyal product of f and g to second order in Theta, with [r, theta] = i Theta."""
    out = f*g
    if moyal:
        out += I/2*Th*(f.diff(r)*g.diff(th) - f.diff(th)*g.diff(r))
        out -= Th**2/8*(f.diff(r, 2)*g.diff(th, 2) - 2*f.diff(r, th)*g.diff(r, th)
                        + f.diff(th, 2)*g.diff(r, 2))
    return sp.expand(out)


def truncate(expr, order=2):
    """Drop powers of Theta above the given order."""
    return sp.series(sp.expand(expr), Th, 0, order + 1).removeO()


def metric(moyal=True):
    g = {}
    for mu in coords:
        for nu in coords:
            s = 0
            for a in frames:
                s += sp.Rational(1, 2)*eta[a]*(star(tet(a, mu), conj(tet(a, nu)), moyal)
                                               + star(tet(a, nu), conj(tet(a, mu)), moyal))
            g[(mu, nu)] = sp.simplify(truncate(s))
    return g


# Their equation (4.7).
paper47 = {
    (1, 1): 1/A**2 + A2/(4*A)*Th**2,
    (2, 2): r**2 + sp.Rational(1, 16)*(A**2 + 11*r*A*A1 + 16*r**2*A1**2 + 12*r**2*A*A2)*Th**2,
    (3, 3): r**2*sp.sin(th)**2 + sp.Rational(1, 16)*(4*(2*r*A*A1 - r*A1/A + r**2*A*A2 + 2*r**2*A1**2)
                                                     * sp.sin(th)**2 + sp.cos(th)**2)*Th**2,
    (0, 0): -A**2 - sp.Rational(1, 4)*(2*r*A*A1**3 + r*A**3*A3 + A**3*A2 + 2*A**2*A1**2
                                       + 5*r*A**2*A1*A2)*Th**2,
}
# Their equation (4.9), for A^2 = 1 - alpha/r.
paper49 = {
    (1, 1): 1/(1 - al/r) - al*(4*r - 3*al)/(16*r**2*(r - al)**2)*Th**2,
    (2, 2): r**2 + (2*r**2 - 17*al*r + 17*al**2)/(32*r*(r - al))*Th**2,
    (3, 3): r**2*sp.sin(th)**2 + ((r**2 + al*r - al**2)*sp.cos(th)**2 - al*(2*r - al))
            / (16*r*(r - al))*Th**2,
    (0, 0): -(1 - al/r) - al*(8*r - 11*al)/(16*r**4)*Th**2,
}

if __name__ == '__main__':
    schw = sp.sqrt(1 - al/r)
    for moyal in (True, False):
        g = metric(moyal)
        label = 'Moyal product' if moyal else 'ordinary product'
        print(f'--- metric from the tetrads with the {label}')
        for (mu, nu), val in g.items():
            if val != 0 and mu != nu:
                print(f'   off-diagonal g_{mu}{nu} =', val)
        for k in paper47:
            d47 = sp.simplify(g[k] - paper47[k])
            gs = g[k].subs(A, schw).doit()
            d49 = sp.simplify(gs - paper49[k])
            print(f'   g_{k[0]}{k[1]}: minus (4.7) = {d47};   minus (4.9) = {d49}')
    print('--- consistency of (4.7) with (4.9) for A^2 = 1 - alpha/r')
    for k in paper47:
        print(f'   g_{k[0]}{k[1]}:', sp.simplify(paper47[k].subs(A, schw).doit() - paper49[k]))
