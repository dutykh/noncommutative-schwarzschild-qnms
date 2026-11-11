# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""
Check (a): CTZ's tetrad formulas (3.10)-(3.12), evaluated for the Schwarzschild data
(2.5), (2.6), (2.9) and the (r, theta) twist (4.1), compared with CTZ (4.2)-(4.6).

Two readings of the covariant derivative in (3.12):
  'anti' : CTZ (3.9) as printed (an anticommutator),
  'comm' : the gauge-covariant derivative d R + [w, R].
Also: is the printed (3.12) the k -> 0 contraction of the printed SO(4,1) formula (3.7)?
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
from common import e, Th, to_syms, a0, a1, a2, a3, r, th
from tetrad_formulas import (first, second_printed, contracted_tetrad_from_SO41,
                             first_from_SO41, bracket_contracted)

I = sp.I
A, A1, A2, A3 = a0, a1, a2, a3
s, c = sp.sin(th), sp.cos(th)

# CTZ (4.2)-(4.6): coefficients (zeroth, first, second) with e_hat = e0 + Theta*first + Theta^2*second
CTZ_PRINTED = {
    (1, 1): (1 / A, 0, A2 / 8),
    (1, 2): (0, -I / 4 * (A + 2 * r * A1), 0),
    (2, 2): (r, 0, sp.Rational(1, 32) * (7 * A * A1 + 12 * r * A1**2 + 12 * r * A * A2)),
    (3, 3): (r * s, -I / 4 * c, sp.Rational(1, 8) * (2 * r * A1**2 + r * A * A2 + 2 * A * A1 - A1 / A) * s),
    (0, 0): (A, 0, sp.Rational(1, 8) * (2 * r * A1**3 + 5 * r * A * A1 * A2 + r * A**2 * A3
                                         + 2 * A * A1**2 + A**2 * A2)),
}


def show(label, expr):
    print(f'    {label}: {sp.factor(sp.simplify(expr))}')


def main():
    names = {0: 't', 1: 'r', 2: 'th', 3: 'ph'}
    print('=== First order, -i Theta^{nu rho} e^a_{mu nu rho} (CTZ (3.11)) ===')
    first_c = {}
    for a in range(4):
        for mu in range(4):
            f1 = to_syms(first(a, mu))
            first_c[(a, mu)] = f1
            if f1 != 0:
                print(f'  e^{a}_{names[mu]}: -i Theta * ({sp.factor(f1)})')
            # consistency with SO(4,1) contraction
            assert sp.simplify(to_syms(first_from_SO41(a, mu)) - f1) == 0
    print('  (identical to the k->0 contraction of the SO(4,1) first-order term (3.6))')

    results = {}
    for mode in ('anti', 'comm'):
        print(f'\n=== Second order from printed (3.12), prefactor 1/32, D = {mode} ===')
        for a in range(4):
            for mu in range(4):
                s2 = to_syms(second_printed(a, mu, sp.Rational(1, 32), mode, extra=False))
                results[(mode, a, mu)] = s2
                if s2 != 0:
                    print(f'  e^{a}_{names[mu]}: Theta^2 * ({sp.factor(s2)})')

    print('\n=== Comparison with CTZ (4.2)-(4.6) ===')
    for (a, mu), (z, f, s2p) in CTZ_PRINTED.items():
        print(f'  component e^{a}_{names[mu]}:')
        assert sp.simplify(to_syms(e[a, mu]) - z) == 0
        df = sp.simplify(-I * first_c[(a, mu)] - f)
        print(f'    first order: printed - computed = {df}')
        for mode in ('anti', 'comm'):
            diff = sp.simplify(results[(mode, a, mu)] - s2p)
            print(f'    second order, D={mode}: computed - printed = {sp.factor(diff)}')
    # components printed as zero by CTZ but computed nonzero?
    for mode in ('anti', 'comm'):
        for a in range(4):
            for mu in range(4):
                if (a, mu) not in CTZ_PRINTED and results[(mode, a, mu)] != 0:
                    print(f'  extra nonzero component D={mode}: e^{a}_{names[mu]}', results[(mode, a, mu)])

    print('\n=== Term-by-term values of the 9 bracket terms of (3.12) (Theta-contracted, D=comm) ===')
    for (a, mu) in CTZ_PRINTED:
        tl = [sp.factor(to_syms(x)) for x in bracket_contracted(a, mu, 'comm')]
        print(f'  e^{a}_{names[mu]}:', tl)
    print('  (D=anti differs only in T2, T3)')
    for (a, mu) in CTZ_PRINTED:
        tl = [sp.factor(to_syms(x)) for x in bracket_contracted(a, mu, 'anti')]
        print(f'  e^{a}_{names[mu]} anti:', tl[1], tl[2])

    print('\n=== Is (3.12) the k->0 contraction of the SO(4,1) formula (3.7)? ===')
    for mode in ('anti', 'comm'):
        for (a, mu) in CTZ_PRINTED:
            lim, full = contracted_tetrad_from_SO41(a, mu, 'ctz', mode)
            lim = to_syms(lim) / 32
            diff = sp.simplify(lim - results[(mode, a, mu)])
            print(f'  D={mode} e^{a}_{names[mu]}: contraction(3.7)/32 = {sp.factor(lim)};'
                  f'  minus printed (3.12) = {sp.factor(diff)}')


if __name__ == '__main__':
    main()
