# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""
Check (d): the (t, theta) twist of JKP section IV.B, Theta^{t theta} = +Theta (JKP (87), (88)).

  1. Deformed tetrad to second order:
       'SW32'   : full k -> 0 contraction of the SO(4,1) formula CTZ (3.7) with prefactor 1/32
                  and the gauge-covariant D (the anticommutator reading of CTZ (3.9) is also
                  evaluated: unlike the (r, theta) twist, it changes the result here),
       'JKP52'  : JKP's printed (51)-(52) (prefactor 1/16, extra term (49), D of (43)),
       'CTZ312' : the printed bracket CTZ (3.12) with prefactor 1/32 and the gauge-covariant D,
     compared with the printed tetrads JKP (89).
  2. The Moyal terms for this twist: every term carries a derivative d_t of a t-independent
     field, so f * g = f g exactly for the fields involved; checked explicitly.
  3. Metric (65) from the SW32 tetrads and from JKP (89), compared with JKP (90), and, with
     A = sqrt(1 - 2M/r), with JKP (91).
  4. Deformation functions, with f = 1 - 2M/r:
       g_tt = -f (1 + Theta^2 A0),  g_rr = (1 + Theta^2 B0) / f,  g_thth = r^2 (1 + Theta^2 C0),
       g_phph = r^2 sin^2(theta) (1 + Theta^2 D0) + Theta^2 kappa cos^2(theta),
     and their behaviour at r = 2M; the same functions for the (r, theta) twist for comparison.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
import common
from common import (A, r, th, Th, M, eta4, to_syms, a0, a1, a2, a3, a4, moyal, set_twist, e)
from tetrad_formulas import first, second_printed, contracted_tetrad_from_SO41

I = sp.I
names = {0: 't', 1: 'r', 2: 'th', 3: 'ph'}
s, c = sp.sin(th), sp.cos(th)
A_, A1, A2, A3 = a0, a1, a2, a3
SCHW = sp.sqrt(1 - 2 * M / r)
fS = 1 - 2 * M / r

# JKP (89): (zeroth, first-order coefficient of Theta, second-order coefficient of Theta^2)
JKP89 = {
    (0, 0): (A_, 0, -sp.Rational(1, 16) * (A_**5 * A1**2 - 2 * r * A_**4 * A3)),
    (0, 2): (0, -I / 4 * r * A_**2 * A1, 0),
    (1, 1): (1 / A_, 0, sp.Rational(1, 8) * A_**2 * (-A1**2 * A_ - r * A1**3 + A_**2 * A2 - 2 * r * A_ * A1 * A2)),
    (2, 0): (0, -I / 4 * A_**3 * A1, 0),
    (2, 2): (r, 0, sp.Rational(1, 16) * (2 * A_**5 * A1 - r * A_**4 * A1**2)),
    (3, 3): (r * s, 0, -sp.Rational(1, 8) * A_**3 * A1 * (-2 + A_**2 + r * A_ * A1) * s),
}
# JKP (90), Theta^2 coefficients
JKP90 = {
    0: sp.Rational(1, 16) * (3 * A_**6 * A1**2 - 4 * r * A_**5 * A3),
    1: -sp.Rational(1, 4) * (A_**2 * A1**2 + r * A_ * A1**3 - A_**3 * A2 + 2 * r * A_**2 * A1 * A2),
    2: sp.Rational(1, 16) * (4 * r * A_**5 * A1 - 3 * r**2 * A_**4 * A1**2),
    3: sp.Rational(1, 4) * (2 * r * A_**3 * A1 - r * A_**5 * A1 - r * A_**4 * A1**2) * s**2,
}
# JKP (91), Theta^2 coefficients
JKP91 = {
    0: M**2 * (r - 2 * M) * (9 * r - 30 * M) / (48 * r**6),
    1: -M * (11 * M**2 - 10 * M * r + 2 * r**2) / (4 * r**4 * (r - 2 * M)),
    2: M * (11 * M - 4 * r) * (2 * M - r) / (16 * r**3),
    3: M * (r - 2 * M) * (M + r) * s**2 / (4 * r**3),
}


def schw(z):
    rep = {a0: SCHW, a1: sp.diff(SCHW, r), a2: sp.diff(SCHW, r, 2), a3: sp.diff(SCHW, r, 3),
           a4: sp.diff(SCHW, r, 4)}
    return sp.factor(sp.simplify(sp.sympify(z).subs(rep)))


def tetrad_coefficients():
    """first-order and second-order coefficients of every component, several prescriptions."""
    out = {}
    for a in range(4):
        for mu in range(4):
            f1 = to_syms(first(a, mu))
            sw32 = {}
            for mode in ('comm', 'anti'):
                lim, _ = contracted_tetrad_from_SO41(a, mu, 'ctz', mode)
                sw32[mode] = sp.factor(to_syms(lim) / 32)
            danti = sp.factor(sp.simplify(sw32['anti'] - sw32['comm']))
            jkp52 = sp.factor(to_syms(second_printed(a, mu, sp.Rational(1, 16), 'comm', extra=True)))
            ctz312 = sp.factor(to_syms(second_printed(a, mu, sp.Rational(1, 32), 'comm', extra=False)))
            out[(a, mu)] = (to_syms(e[a, mu]), f1, sw32['comm'], jkp52, ctz312)
            DANTI[(a, mu)] = danti
    return out


DANTI = {}


def build(coeffs, which):
    """e_hat and e_hat^dagger (all coefficients real) from a coefficient table."""
    E = sp.zeros(4, 4)
    for (a, mu), val in coeffs.items():
        if which == 'JKP89':
            z0, f1, q = val
            E[a, mu] = common.from_syms(z0 + f1 * Th + q * Th**2)
        else:
            z0, f1, sw32, jkp52, ctz312 = val
            q = {'SW32': sw32, 'JKP52': jkp52}[which]
            E[a, mu] = common.from_syms(z0 - I * f1 * Th + q * Th**2)
    Ed = E.applyfunc(lambda z: z.subs(I, -I))
    return E, Ed


def metric(E, Ed, star=True):
    g = sp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            tot = 0
            for a in range(4):
                if star:
                    tot += eta4[a, a] * (moyal(E[a, mu], Ed[a, nu]) + moyal(E[a, nu], Ed[a, mu]))
                else:
                    tot += eta4[a, a] * (E[a, mu] * Ed[a, nu] + E[a, nu] * Ed[a, mu])
            z = sp.expand(tot / 2)
            g[mu, nu] = sum(z.coeff(Th, j) * Th**j for j in range(3))
    return g


def deformation_functions(g):
    gtt = schw(to_syms(g[0, 0].coeff(Th, 2)))
    grr = schw(to_syms(g[1, 1].coeff(Th, 2)))
    gthth = schw(to_syms(g[2, 2].coeff(Th, 2)))
    gphph = schw(to_syms(g[3, 3].coeff(Th, 2)))
    A0 = sp.factor(sp.simplify(-gtt / fS))
    B0 = sp.factor(sp.simplify(grr * fS))
    C0 = sp.factor(sp.simplify(gthth / r**2))
    P = sp.simplify(gphph.subs(th, sp.pi / 2))
    Q = sp.simplify(gphph.subs(th, 0))
    assert sp.simplify(P * s**2 + Q * c**2 - gphph) == 0
    D0 = sp.factor(sp.simplify(P / r**2))
    kap = sp.factor(Q)
    return {'A0': A0, 'B0': B0, 'C0': C0, 'D0': D0, 'kappa': kap}


def main():
    set_twist(0, 2)
    print('Twist pairs now:', common.THETA_PAIRS, ' (Theta^{t theta} = +Theta)')
    coeffs = tetrad_coefficients()
    print('\n1. Tetrad: e_hat = e - i Theta (first) + Theta^2 (second); nonzero components')
    for (a, mu), (z0, f1, sw32, jkp52, ctz312) in coeffs.items():
        if f1 == 0 and sw32 == 0 and jkp52 == 0:
            continue
        print(f'  e^{a}_{names[mu]}: e = {z0}, first = {sp.factor(f1)}')
        print(f'     second, SW32 (contraction of (3.7)/32, gauge-covariant D) = {sw32}')
        print(f'     (the anticommutator reading of D, CTZ (3.9), would add {DANTI[(a, mu)]})')
        print(f'     second, JKP (52) = {jkp52};  ratio (52)/SW32 = '
              f'{sp.simplify(jkp52 / sw32) if sw32 != 0 else "-"}')
        print(f'     second, printed CTZ (3.12)/32 with D=comm = {ctz312};  minus SW32 = {sp.factor(sp.simplify(ctz312 - sw32))}')
        if (a, mu) in JKP89:
            z, f, q = JKP89[(a, mu)]
            print(f'     JKP (89): first-order printed - computed = {sp.simplify(f - (-I * f1))};'
                  f'  second printed - SW32 = {sp.factor(sp.simplify(q - sw32))};'
                  f'  printed / SW32 = {sp.simplify(q / sw32) if sw32 != 0 else "-"};'
                  f'  printed - JKP(52) = {sp.factor(sp.simplify(q - jkp52))}')
        print(f'     Schwarzschild: first = {schw(f1)}, second SW32 = {schw(sw32)}')
    for key in JKP89:
        if key not in coeffs or (coeffs[key][1] == 0 and coeffs[key][2] == 0 and coeffs[key][3] == 0):
            print('  printed in (89) but computed zero:', key)

    E, Ed = build(coeffs, 'SW32')
    print('\n2. Moyal terms for the (t, theta) twist')
    g_star = metric(E, Ed, star=True)
    g_plain = metric(E, Ed, star=False)
    print('   metric with the Moyal product minus pointwise metric = 0 :',
          all(sp.simplify(g_star[m, n] - g_plain[m, n]) == 0 for m in range(4) for n in range(4)))

    print('\n3. Metric (65), Theta^2 coefficients (no Theta^1 terms, no off-diagonal terms)')
    tables = {'SW32': g_star}
    E2, Ed2 = build(JKP89, 'JKP89')
    tables['from JKP (89)'] = metric(E2, Ed2)
    E3, Ed3 = build(coeffs, 'JKP52')
    tables['from JKP (52)'] = metric(E3, Ed3)
    for name, g in tables.items():
        print(f'  tetrads {name}:')
        for m in range(4):
            for n in range(4):
                z = sp.expand(g[m, n])
                if m != n and sp.simplify(z) != 0:
                    print(f'    off-diagonal g_{names[m]}{names[n]} =', sp.simplify(to_syms(z)))
                if m == n and sp.simplify(z.coeff(Th, 1)) != 0:
                    print(f'    Theta^1 in g_{names[m]}{names[m]} =', sp.simplify(z.coeff(Th, 1)))
        for m in range(4):
            q = sp.factor(sp.simplify(to_syms(g[m, m].coeff(Th, 2))))
            print(f'    g_{names[m]}{names[m]}: {q}   | minus JKP (90): {sp.factor(sp.simplify(q - JKP90[m]))}')
        for m in range(4):
            q = schw(to_syms(g[m, m].coeff(Th, 2)))
            print(f'    Schwarzschild g_{names[m]}{names[m]}: {q}   | minus JKP (91): {sp.factor(sp.simplify(q - JKP91[m]))}')
    print('  check: JKP (90) at A = sqrt(1-2M/r) minus JKP (91):',
          [sp.factor(sp.simplify(schw(JKP90[m]) - JKP91[m])) for m in range(4)])

    print('\n4. Deformation functions (t, theta) twist, SW32 tetrads, f = 1 - 2M/r')
    df = deformation_functions(g_star)
    for key, val in df.items():
        lim = sp.limit(val.subs(th, sp.pi / 3), r, 2 * M) if val != 0 else 0
        print(f'   {key} = {val}    value at r = 2M: {sp.simplify(lim)}')
    print('   (for comparison) the same functions from JKP (89):')
    for key, val in deformation_functions(tables['from JKP (89)']).items():
        print(f'     {key} = {val}')

    set_twist(1, 2)
    import check_c_metric as cm
    g_rt = cm.metric('SW32', 1, -1)
    print('\n5. Deformation functions (r, theta) twist, SW32 tetrads, for comparison')
    for key, val in deformation_functions(g_rt).items():
        lim = sp.limit(val, r, 2 * M, '+') if val != 0 else 0
        print(f'   {key} = {val}    limit r -> 2M+: {lim}')


if __name__ == '__main__':
    main()
