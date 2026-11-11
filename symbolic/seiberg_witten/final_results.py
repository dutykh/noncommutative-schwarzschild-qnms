# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""
Final deformed tetrad and metric for the (r, theta) twist, from the second-order Seiberg-Witten
map with Chamseddine's prefactor 1/32 (verified in check_sw_prefactor.py), the full k -> 0
contraction of the SO(4,1) formula (3.7) (which includes the term JKP (49)), and the Moyal
product with the Seiberg-Witten sign f * g = f g + (i/2) Theta {f, g} - (Theta^2/8)(...).

Also checks the reality conditions CTZ (3.5): the first-order SO(4,1) correction is symmetric
in AB and the second-order one antisymmetric, for this background, with k kept symbolic.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
from common import (Om, R5, M, r, th, Th, e, to_syms, a0, a1, a2, a3, a4, d, anti, THETA_PAIRS)
from tetrad_formulas import chamseddine_2nd, contracted_tetrad_from_SO41, first
import check_c_metric as cm

names = {0: 't', 1: 'r', 2: 'th', 3: 'ph'}
SCHW = sp.sqrt(1 - 2 * M / r)


def schw(z):
    rep = {a0: SCHW, a1: sp.diff(SCHW, r), a2: sp.diff(SCHW, r, 2), a3: sp.diff(SCHW, r, 3),
           a4: sp.diff(SCHW, r, 4)}
    return sp.factor(sp.simplify(sp.sympify(z).subs(rep)))


def reality():
    eta5 = sp.diag(-1, 1, 1, 1, 1)
    ok1, ok2 = True, True
    for mu in range(4):
        w1 = sp.zeros(5, 5)
        for (nu, rho), s in THETA_PAIRS:
            w1 += s * anti(Om[nu], d(Om[mu], rho) + R5[rho][mu], eta5) / 4
        ok1 &= sp.simplify(w1 - w1.T) == sp.zeros(5, 5)
        w2 = sp.zeros(5, 5)
        for (nu, rho), s1 in THETA_PAIRS:
            for (lam, tau), s2 in THETA_PAIRS:
                w2 += s1 * s2 * chamseddine_2nd(mu, nu, rho, lam, tau, 'comm')
        ok2 &= sp.simplify(w2 + w2.T) == sp.zeros(5, 5)
    print('Reality conditions (3.5), k symbolic: first order symmetric:', ok1, '; second order antisymmetric:', ok2)


def main():
    reality()
    print('\nFinal tetrad e_hat^a_mu = e^a_mu - i Theta (first) + Theta^2 (second):')
    for a in range(4):
        for mu in range(4):
            f1 = to_syms(first(a, mu))
            lim, _ = contracted_tetrad_from_SO41(a, mu, 'ctz', 'comm')
            s2 = sp.factor(to_syms(lim) / 32)
            if f1 != 0 or s2 != 0:
                print(f'  e^{a}_{names[mu]}: e = {to_syms(e[a, mu])}, first = {sp.factor(f1)}, second = {s2}')
                print(f'      Schwarzschild: first = {schw(f1)}, second = {schw(s2)}')
    print('\nFinal metric (65), Theta^2 coefficients (general A and Schwarzschild):')
    g = cm.metric('SW32', 1, -1)
    for mu in range(4):
        for nu in range(4):
            z = sp.expand(g[mu, nu])
            if mu != nu:
                assert sp.simplify(z) == 0
                continue
            c0 = to_syms(z.coeff(Th, 0))
            c1 = sp.simplify(z.coeff(Th, 1))
            c2 = cm.th2(z)
            print(f'  g_{names[mu]}{names[mu]}: zeroth = {c0}, Theta^1 = {c1}, Theta^2 = {sp.factor(c2)}')
            print(f'      Schwarzschild Theta^2 = {schw(c2)}')


if __name__ == '__main__':
    main()
