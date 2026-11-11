# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""
Supplement to check (c): which product reproduces CTZ (4.7), and the effect of the sign
convention of the first-order Moyal term.

  1. CTZ tetrads (4.2)-(4.6) with the metric (65) and
       (s1, s2) = (0, 0)  : ordinary pointwise product,
       (s1, s2) = (+1, 0) : first-order Moyal term only,
       (s1, s2) = (+1, -1): full second-order Moyal product.
  2. Consistent change of convention: star product with s1 = -1 AND first-order Seiberg-Witten
     terms of the opposite sign (equivalent to Theta -> -Theta) gives the same metric as s1 = +1.
  3. Breakdown of the Moyal contributions to g_phiphi: s1 sin^2(theta)/4 - s2 cos^2(theta)/4.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
from common import r, th, Th, eta4, to_syms, moyal
import check_c_metric as cm

names = {0: 't', 1: 'r', 2: 'th', 3: 'ph'}


def metric_from(E, Ed, s1, s2):
    g = sp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            tot = 0
            for a in range(4):
                if eta4[a, a] == 0:
                    continue
                tot += eta4[a, a] * (moyal(E[a, mu], Ed[a, nu], s1, s2) + moyal(E[a, nu], Ed[a, mu], s1, s2))
            g[mu, nu] = cm.trunc2(sp.Rational(1, 2) * tot)
    return g


def main():
    E, Ed = cm.tetrad('CTZ42')
    print('1. CTZ tetrads, metric (65): Theta^2 coefficient minus CTZ (4.7)')
    for s1, s2 in [(0, 0), (1, 0), (-1, 0), (1, -1)]:
        g = metric_from(E, Ed, s1, s2)
        diffs = [sp.factor(sp.simplify(cm.th2(g[m, m]) - cm.CTZ47[(m, m)])) for m in range(4)]
        print(f'   s1={s1:+d}, s2={s2:+d}:', dict(zip(['tt', 'rr', 'thth', 'phph'], diffs)))

    print('2. Consistent flip of conventions (s1 = -1 and first-order SW terms reversed), all tetrad sets')
    for which in ('JKP113', 'CTZ42', 'SW32'):
        E, Ed = cm.tetrad(which)
        Ef = E.applyfunc(lambda z: sp.expand(z).subs(Th, -Th))      # Theta -> -Theta in the tetrad
        Edf = Ed.applyfunc(lambda z: sp.expand(z).subs(Th, -Th))
        g_plus = metric_from(E, Ed, 1, -1)
        g_flip = metric_from(Ef, Edf, -1, -1)
        g_mixed = metric_from(E, Ed, -1, -1)
        same = all(sp.simplify(g_plus[m, n] - g_flip[m, n]) == 0 for m in range(4) for n in range(4))
        mixed = {names[m] + names[m]: sp.simplify(cm.th2(g_mixed[m, m]) - cm.th2(g_plus[m, m])) for m in range(4)}
        print(f'   {which}: consistent flip reproduces the s1=+1 metric: {same};'
              f'  mixed convention (star s1=-1, tetrads unchanged) minus s1=+1: {mixed}')

    print('3. Moyal contributions to g_phiphi (any of the tetrad sets): ')
    for which in ('JKP113', 'CTZ42', 'SW32'):
        E, Ed = cm.tetrad(which)
        g0 = metric_from(E, Ed, 0, 0)
        out = []
        for s1, s2 in [(1, 0), (0, -1), (1, -1)]:
            g = metric_from(E, Ed, s1, s2)
            out.append(((s1, s2), sp.simplify(cm.th2(g[3, 3]) - cm.th2(g0[3, 3]))))
        print(f'   {which}:', out)


if __name__ == '__main__':
    main()
