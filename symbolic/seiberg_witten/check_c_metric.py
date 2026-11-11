# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""
Check (c): the deformed metric
    g_{mu nu} = 1/2 eta_ab ( e_hat^a_mu * e_hat^{b dagger}_nu + e_hat^b_nu * e_hat^{a dagger}_mu )   (JKP (65))
with the Moyal product to second order, for the (r, theta) twist, from:
    'JKP113' : JKP's tetrads (113) (reproduced in check (b) from (51)-(52)),
    'CTZ42'  : CTZ's tetrads (4.2)-(4.6) (reproduced in check (a)),
    'SW32'   : the k -> 0 contraction of the second-order SW map with Chamseddine's prefactor 1/32
               and the gauge-covariant D (the tetrads of 'JKP113' with the Theta^2 terms halved).
Star product variants:  s1 = +1 (CTZ (3.2), Seiberg-Witten) or -1 (JKP (1), (67));
                        s2 = -1 (expansion of the exponential) or +1 (sign printed in JKP (67)).
Also: CTZ's printed (3.14), 1/2 eta_ab (e^a_mu * e^{b+}_nu + e^b_mu * e^{a+}_nu), differs from (65)
only off the diagonal; its off-diagonal part is computed too.
Then A = sqrt(1 - 2M/r) and comparison with JKP (115) and CTZ (4.9) (alpha = 2M).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
from common import A, r, th, Th, M, eta4, to_syms, a0, a1, a2, a3, a4, moyal

I = sp.I
s, c = sp.sin(th), sp.cos(th)
Ap, App, Appp = sp.diff(A, r), sp.diff(A, r, 2), sp.diff(A, r, 3)

# second-order coefficients of JKP (113)
J00 = sp.Rational(1, 4) * (2 * r * Ap**3 + 5 * r * A * Ap * App + r * A**2 * Appp + 2 * A * Ap**2 + A**2 * App)
J11 = App / 4
J22 = sp.Rational(1, 4) * (2 * A * Ap + 3 * r * Ap**2 + 3 * r * A * App)
J33 = sp.Rational(1, 4) * (2 * r * Ap**2 + r * A * App + 2 * A * Ap - Ap / A) * s
# CTZ (4.2)-(4.6)
C00 = sp.Rational(1, 8) * (2 * r * Ap**3 + 5 * r * A * Ap * App + r * A**2 * Appp + 2 * A * Ap**2 + A**2 * App)
C11 = App / 8
C22 = sp.Rational(1, 32) * (7 * A * Ap + 12 * r * Ap**2 + 12 * r * A * App)
C33 = sp.Rational(1, 8) * (2 * r * Ap**2 + r * A * App + 2 * A * Ap - Ap / A) * s
F12 = (A + 2 * r * Ap) / 4    # e_hat^1_theta = -i Theta F12
F33 = c / 4                   # e_hat^3_phi  has  -i Theta F33


def tetrad(which):
    """Return (E, Edag): 4x4 matrices [a, mu] of e_hat and its conjugate, to O(Theta^2)."""
    if which == 'JKP113':
        q = (J00, J11, J22, J33)
    elif which == 'CTZ42':
        q = (C00, C11, C22, C33)
    elif which == 'SW32':
        q = (J00 / 2, J11 / 2, J22 / 2, J33 / 2)
    else:
        raise ValueError(which)
    E = sp.zeros(4, 4)
    E[0, 0] = A + Th**2 * q[0]
    E[1, 1] = 1 / A + Th**2 * q[1]
    E[1, 2] = -I * Th * F12
    E[2, 2] = r + Th**2 * q[2]
    E[3, 3] = r * s - I * Th * F33 + Th**2 * q[3]
    Edag = E.applyfunc(lambda z: z.subs(I, -I))   # all coefficients are real
    return E, Edag


def trunc2(z):
    z = sp.expand(z)
    return sum(z.coeff(Th, j) * Th**j for j in range(3))


def metric(which, s1, s2, form='65'):
    E, Ed = tetrad(which)
    g = sp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            tot = 0
            for a in range(4):
                for b in range(4):
                    if eta4[a, b] == 0:
                        continue
                    if form == '65':
                        tot += eta4[a, b] * (moyal(E[a, mu], Ed[b, nu], s1, s2) + moyal(E[b, nu], Ed[a, mu], s1, s2))
                    else:   # CTZ (3.14) as printed
                        tot += eta4[a, b] * (moyal(E[a, mu], Ed[b, nu], s1, s2) + moyal(E[b, mu], Ed[a, nu], s1, s2))
            g[mu, nu] = trunc2(sp.Rational(1, 2) * tot)
    return g


def th2(z):
    return sp.simplify(to_syms(sp.expand(z).coeff(Th, 2)))


# JKP (114), Theta^2 coefficients
A_, A1, A2, A3 = a0, a1, a2, a3
JKP114 = {
    (0, 0): -sp.Rational(1, 2) * (2 * r * A_ * A1**3 + r * A_**3 * A3 + A_**3 * A2 + 2 * A_**2 * A1**2
                                  + 5 * r * A_**2 * A1 * A2),
    (1, 1): sp.Rational(1, 2) * A2 / A_,
    (2, 2): sp.Rational(1, 16) * (A_**2 + 20 * r * A_ * A1 + 28 * r**2 * A1**2 + 24 * r**2 * A_ * A2),
    (3, 3): sp.Rational(1, 16) * (4 * (4 * r * A_ * A1 - 2 * r * A1 / A_ + 2 * r**2 * A_ * A2 + 4 * r**2 * A1**2 + 1) * s**2
                                  + 5 * c**2),
}
# CTZ (4.7), Theta^2 coefficients
CTZ47 = {
    (1, 1): sp.Rational(1, 4) * A2 / A_,
    (2, 2): sp.Rational(1, 16) * (A_**2 + 11 * r * A_ * A1 + 16 * r**2 * A1**2 + 12 * r**2 * A_ * A2),
    (3, 3): sp.Rational(1, 16) * (4 * (2 * r * A_ * A1 - r * A1 / A_ + r**2 * A_ * A2 + 2 * r**2 * A1**2) * s**2 + c**2),
    (0, 0): -sp.Rational(1, 4) * (2 * r * A_ * A1**3 + r * A_**3 * A3 + A_**3 * A2 + 2 * A_**2 * A1**2
                                  + 5 * r * A_**2 * A1 * A2),
}
# Schwarzschild: JKP (115), CTZ (4.9) with alpha = 2M, Theta^2 coefficients
JKP115 = {
    (0, 0): M * (11 * M - 4 * r) / (2 * r**4),
    (1, 1): M * (3 * M - 2 * r) / (2 * r**2 * (r - 2 * M)**2),
    (2, 2): sp.Rational(1, 16) - 2 * M / r + M / (8 * (r - 2 * M)),
    (3, 3): 5 * c**2 / 16 + s**2 * (2 * M**2 - 4 * M * r + r**2) / (4 * r * (r - 2 * M)),
}
al = 2 * M
CTZ49 = {
    (1, 1): -al * (4 * r - 3 * al) / (16 * r**2 * (r - al)**2),
    (2, 2): (2 * r**2 - 17 * al * r + 17 * al**2) / (32 * r * (r - al)),
    (3, 3): ((r**2 + al * r - al**2) * c**2 - al * (2 * r - al)) / (16 * r * (r - al)),
    (0, 0): -al * (8 * r - 11 * al) / (16 * r**4),
}

SCHW = sp.sqrt(1 - 2 * M / r)


def schw(expr_syms):
    """Substitute A = sqrt(1 - 2M/r) into an expression in the symbols A, A1, A2, A3."""
    rep = {a0: SCHW, a1: sp.diff(SCHW, r), a2: sp.diff(SCHW, r, 2), a3: sp.diff(SCHW, r, 3),
           a4: sp.diff(SCHW, r, 4)}
    return sp.simplify(sp.factor(sp.simplify(expr_syms.subs(rep))))


def main():
    names = {0: 't', 1: 'r', 2: 'th', 3: 'ph'}
    cache = {}
    for which in ('JKP113', 'CTZ42', 'SW32'):
        for s1 in (1, -1):
            for s2 in (-1, 1):
                g = metric(which, s1, s2)
                cache[(which, s1, s2)] = g
                print(f'\n=== tetrads {which}, star signs s1={s1:+d}, s2={s2:+d} ===')
                # first-order part and off-diagonal entries
                for mu in range(4):
                    for nu in range(mu, 4):
                        z = sp.expand(g[mu, nu])
                        if sp.simplify(z.coeff(Th, 1)) != 0:
                            print(f'   Theta^1 term in g_{names[mu]}{names[nu]}:', sp.simplify(z.coeff(Th, 1)))
                        if mu != nu and sp.simplify(z) != 0:
                            print(f'   off-diagonal g_{names[mu]}{names[nu]} =', sp.simplify(to_syms(z)))
                for mu in range(4):
                    q = th2(g[mu, mu])
                    line = f'   g_{names[mu]}{names[mu]} Theta^2 coeff = {sp.factor(q)}'
                    line += f' | minus JKP(114): {sp.factor(sp.simplify(q - JKP114[(mu, mu)]))}'
                    line += f' | minus CTZ(4.7): {sp.factor(sp.simplify(q - CTZ47[(mu, mu)]))}'
                    print(line)

    print('\n=== CTZ (3.14) as printed (only the off-diagonal part can differ from (65)) ===')
    for which in ('CTZ42',):
        for s1 in (1, -1):
            g = metric(which, s1, -1, form='314')
            for mu in range(4):
                for nu in range(4):
                    if mu != nu:
                        z = sp.simplify(to_syms(g[mu, nu]))
                        if z != 0:
                            print(f'   s1={s1:+d}: g_{names[mu]}{names[nu]} =', z)
            print(f'   s1={s1:+d}: diagonal identical to (65):',
                  all(sp.simplify(g[m, m] - metric(which, s1, -1)[m, m]) == 0 for m in range(4)))

    print('\n=== Schwarzschild substitution A = sqrt(1 - 2M/r) ===')
    print('  check of JKP (115) against JKP (114) and of CTZ (4.9) against CTZ (4.7):')
    for mu in range(4):
        d1 = sp.simplify(schw(JKP114[(mu, mu)]) - JKP115[(mu, mu)])
        d2 = sp.simplify(schw(CTZ47[(mu, mu)]) - CTZ49[(mu, mu)])
        print(f'   g_{names[mu]}{names[mu]}: (114)|_Schw - (115) = {sp.factor(d1)};   (4.7)|_Schw - (4.9) = {sp.factor(d2)}')
    for key in [('JKP113', 1, -1), ('JKP113', -1, -1), ('JKP113', -1, 1), ('JKP113', 1, 1),
                ('SW32', 1, -1), ('SW32', -1, -1)]:
        g = cache[key]
        print(f'  tetrads {key[0]}, s1={key[1]:+d}, s2={key[2]:+d}:')
        for mu in range(4):
            q = schw(th2(g[mu, mu]))
            print(f'   g_{names[mu]}{names[mu]} Theta^2 coeff = {sp.factor(q)}'
                  f'   | minus JKP(115): {sp.factor(sp.simplify(q - JKP115[(mu, mu)]))}')


if __name__ == '__main__':
    main()
