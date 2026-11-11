# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""
Check (b): JKP's tetrad formulas (50)-(52) (bracket of CTZ (3.12) with prefactor 1/16,
gauge-covariant D of JKP (43), extra last line (49)), evaluated for the Schwarzschild data
and the (r, theta) twist (111), compared with JKP (113).

Also: the k -> 0 contraction of JKP's SO(4,1) formula (44), and of CTZ (3.7), and the
size of the extra term alone, for both readings of D.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
from common import e, to_syms, a0, a1, a2, a3, r, th, theta2
from tetrad_formulas import (first, second_printed, contracted_tetrad_from_SO41, jkp_extra)

I = sp.I
A, A1, A2, A3 = a0, a1, a2, a3
s, c = sp.sin(th), sp.cos(th)

# JKP (113)
JKP_PRINTED = {
    (0, 0): (A, 0, sp.Rational(1, 4) * (2 * r * A1**3 + 5 * r * A * A1 * A2 + r * A**2 * A3
                                         + 2 * A * A1**2 + A**2 * A2)),
    (1, 1): (1 / A, 0, A2 / 4),
    (1, 2): (0, -I / 4 * (A + 2 * r * A1), 0),
    (2, 2): (r, 0, sp.Rational(1, 4) * (2 * A * A1 + 3 * r * A1**2 + 3 * r * A * A2)),
    (3, 3): (r * s, -I / 4 * c, sp.Rational(1, 4) * (2 * r * A1**2 + r * A * A2 + 2 * A * A1 - A1 / A) * s),
}


def main():
    names = {0: 't', 1: 'r', 2: 'th', 3: 'ph'}
    res = {}
    for mode in ('comm', 'anti'):
        print(f'=== JKP (52): 1/16 * bracket + extra term, D = {mode} ===')
        for a in range(4):
            for mu in range(4):
                s2 = to_syms(second_printed(a, mu, sp.Rational(1, 16), mode, extra=True))
                res[(mode, a, mu)] = s2
                if s2 != 0:
                    print(f'  e^{a}_{names[mu]}: Theta^2 * ({sp.factor(s2)})')
        print('  extra term (49) alone, Theta-contracted:')
        for a in range(4):
            for mu in range(4):
                x = to_syms(sp.simplify(theta2(lambda nu, rho, lam, tau: jkp_extra(a, mu, nu, rho, lam, tau))))
                if x != 0:
                    print(f'    e^{a}_{names[mu]}: {sp.factor(x)}')

    print('\n=== Comparison with JKP (113) ===')
    for (a, mu), (z, f, s2p) in JKP_PRINTED.items():
        assert sp.simplify(to_syms(e[a, mu]) - z) == 0
        f1 = sp.simplify(-I * to_syms(first(a, mu)) - f)
        line = f'  e^{a}_{names[mu]}: first order computed - printed = {f1};'
        for mode in ('comm', 'anti'):
            line += f'  2nd order D={mode}: computed - printed = {sp.factor(sp.simplify(res[(mode, a, mu)] - s2p))};'
        print(line)
    for mode in ('comm', 'anti'):
        for a in range(4):
            for mu in range(4):
                if (a, mu) not in JKP_PRINTED and res[(mode, a, mu)] != 0:
                    print(f'  extra nonzero component D={mode}: e^{a}_{names[mu]}', res[(mode, a, mu)])

    print('\n=== k -> 0 contraction of JKP (44) (prefactor 1/16) and of CTZ (3.7) (prefactor 1/32) ===')
    for mode in ('comm', 'anti'):
        for (a, mu), (z, f, s2p) in JKP_PRINTED.items():
            lim44, _ = contracted_tetrad_from_SO41(a, mu, 'jkp', mode)
            lim37, _ = contracted_tetrad_from_SO41(a, mu, 'ctz', mode)
            lim44 = to_syms(lim44) / 16
            lim37 = to_syms(lim37) / 32
            print(f'  D={mode} e^{a}_{names[mu]}: (44)/16 - (113) = {sp.factor(sp.simplify(lim44 - s2p))};'
                  f'  (44)/16 - (52) = {sp.factor(sp.simplify(lim44 - res[(mode, a, mu)]))};'
                  f'  (44)/16 / [(3.7)/32] = {sp.simplify(lim44 / lim37) if lim37 != 0 else "0/0"}')


if __name__ == '__main__':
    main()
