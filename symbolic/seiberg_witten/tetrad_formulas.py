# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""
Second-order Seiberg-Witten corrections of the tetrad, implemented exactly as printed.

  (1) the first-order coefficient e^a_{mu nu rho}: CTZ (3.11) = JKP (51);
  (2) the bracket of CTZ (3.12) (identical to the bracket of JKP (52)), with a
      choice of covariant derivative:
         'comm' : D_rho R = d_rho R + [w_rho, R]    (JKP (43), the gauge-covariant one)
         'anti' : D_rho R = d_rho R + {w_rho, R}    (CTZ (3.9) read literally:
                   omega^{AC} R^{DB} + omega^{BC} R^{DA} is the anticommutator)
  (3) the extra term of JKP (49) = last line of (52);
  (4) the parent SO(4,1) second-order formulas, CTZ (3.7) (Chamseddine, prefactor 1/32)
      and JKP (44) (prefactor 1/16), evaluated with 5x5 matrices, from which the tetrad
      correction is obtained as the (a5) component divided by k, limit k -> 0.

All functions return the full Theta-contracted coefficient for the current twist
(common.set_twist; default (r, theta)), i.e.
   first(a, mu)  = Theta^{nu rho} e^a_{mu nu rho}                     / Theta
   second(a, mu) = Theta^{nu rho} Theta^{lam tau} e^a_{mu nu rho lam tau} / Theta^2
so that  e_hat^a_mu = e^a_mu - i Theta first + Theta^2 second  (CTZ (3.10), JKP (50)).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
from common import (e, W, R, R5, Om, eta4, eta5, k, d, anti, comm, theta1, theta2, THETA_PAIRS)

ETA = eta4


def DR(rho, tau, mu, mode):
    """Covariant derivative D_rho R_{tau mu} (4x4 Lorentz block)."""
    if mode == 'comm':
        return d(R[tau][mu], rho) + comm(W[rho], R[tau][mu])
    if mode == 'anti':
        return d(R[tau][mu], rho) + anti(W[rho], R[tau][mu])
    raise ValueError(mode)


# ---------------------------------------------------------------- first order
def e1_component(a, mu, nu, rho):
    """CTZ (3.11) = JKP (51):
       e^a_{mu nu rho} = 1/4 [ w^{ac}_nu d_rho e^d_mu + (d_rho w^{ac}_mu + R^{ac}_{rho mu}) e^d_nu ] eta_cd"""
    out = 0
    for c in range(4):
        for dd in range(4):
            out += (W[nu][a, c] * d(e[dd, mu], rho)
                    + (d(W[mu][a, c], rho) + R[rho][mu][a, c]) * e[dd, nu]) * ETA[c, dd]
    return sp.Rational(1, 4) * out


def first(a, mu):
    return sp.simplify(theta1(lambda nu, rho: e1_component(a, mu, nu, rho)))


# ---------------------------------------------------------------- second order, printed bracket
def bracket_terms(a, mu, nu, rho, lam, tau, mode):
    """The nine terms inside the square bracket of CTZ (3.12) (= JKP (52) without its last line),
    each already multiplied by eta_bc.  Returned as a list so that the terms can be inspected."""
    terms = [0] * 9
    for b in range(4):
        for c in range(4):
            ebc = ETA[b, c]
            if ebc == 0:
                continue
            # T1:  2 {R_{tau nu}, R_{mu rho}}^{ab} e^c_lam
            terms[0] += 2 * anti(R[tau][nu], R[mu][rho])[a, b] * e[c, lam] * ebc
            # T2: - w^{ab}_lam (D_rho R^{cd}_{tau mu} + d_rho R^{cd}_{tau mu}) e^m_nu eta_dm
            DRm = DR(rho, tau, mu, mode) + d(R[tau][mu], rho)
            s = 0
            for dd in range(4):
                for m in range(4):
                    s += DRm[c, dd] * e[m, nu] * ETA[dd, m]
            terms[1] += -W[lam][a, b] * s * ebc
            # T3: - {w_nu, (D_rho R_{tau mu} + d_rho R_{tau mu})}^{ab} e^c_lam
            terms[2] += -anti(W[nu], DRm)[a, b] * e[c, lam] * ebc
            # T4: - d_tau {w_nu, (d_rho w_mu + R_{rho mu})}^{ab} e^c_lam
            terms[3] += -d(anti(W[nu], d(W[mu], rho) + R[rho][mu]), tau)[a, b] * e[c, lam] * ebc
            # T5: - w^{ab}_lam d_tau ( w^{cd}_nu d_rho e^m_mu + (d_rho w^{cd}_mu + R^{cd}_{rho mu}) e^m_nu ) eta_dm
            s = 0
            for dd in range(4):
                for m in range(4):
                    s += d(W[nu][c, dd] * d(e[m, mu], rho)
                           + (d(W[mu][c, dd], rho) + R[rho][mu][c, dd]) * e[m, nu], tau) * ETA[dd, m]
            terms[4] += -W[lam][a, b] * s * ebc
            # T6: + 2 d_nu w^{ab}_lam d_rho d_tau e^c_mu
            terms[5] += 2 * d(W[lam][a, b], nu) * d(d(e[c, mu], rho), tau) * ebc
            # T7: - 2 d_rho (d_tau w^{ab}_mu + R^{ab}_{tau mu}) d_nu e^c_lam
            terms[6] += -2 * d(d(W[mu][a, b], tau) + R[tau][mu][a, b], rho) * d(e[c, lam], nu) * ebc
            # T8: - {w_nu, (d_rho w_lam + R_{rho lam})}^{ab} d_tau e^c_mu
            terms[7] += -anti(W[nu], d(W[lam], rho) + R[rho][lam])[a, b] * d(e[c, mu], tau) * ebc
            # T9: - (d_tau w^{ab}_mu + R^{ab}_{tau mu}) ( w^{cd}_nu d_rho e^m_lam
            #        + (d_rho w^{cd}_lam + R^{cd}_{rho lam}) e^m_nu ) eta_dm
            #      (eta_dm read as multiplying both terms; the other reading leaves d, m free)
            s = 0
            for dd in range(4):
                for m in range(4):
                    s += (W[nu][c, dd] * d(e[m, lam], rho)
                          + (d(W[lam][c, dd], rho) + R[rho][lam][c, dd]) * e[m, nu]) * ETA[dd, m]
            terms[8] += -(d(W[mu][a, b], tau) + R[tau][mu][a, b]) * s * ebc
    return terms


def jkp_extra(a, mu, nu, rho, lam, tau):
    """Last line of JKP (52) = (49):
       - 1/16 w^{ac}_lam w^{db}_nu e^f_rho R^{gm}_{tau mu} eta_cd eta_fg eta_bm"""
    out = 0
    for c in range(4):
        for dd in range(4):
            if ETA[c, dd] == 0:
                continue
            for b in range(4):
                for m in range(4):
                    if ETA[b, m] == 0:
                        continue
                    for f in range(4):
                        for g in range(4):
                            if ETA[f, g] == 0:
                                continue
                            out += (W[lam][a, c] * W[nu][dd, b] * e[f, rho] * R[tau][mu][g, m]
                                    * ETA[c, dd] * ETA[f, g] * ETA[b, m])
    return -sp.Rational(1, 16) * out


def bracket_contracted(a, mu, mode):
    """Theta-contracted bracket of CTZ (3.12), term by term."""
    tot = [0] * 9
    for (nu, rho), s1 in THETA_PAIRS:
        for (lam, tau), s2 in THETA_PAIRS:
            tl = bracket_terms(a, mu, nu, rho, lam, tau, mode)
            for i in range(9):
                tot[i] += s1 * s2 * tl[i]
    return [sp.simplify(x) for x in tot]


def second_printed(a, mu, prefactor, mode, extra):
    """prefactor * (Theta-contracted bracket) [+ JKP extra term]."""
    br = sum(bracket_contracted(a, mu, mode))
    out = prefactor * br
    if extra:
        out += theta2(lambda nu, rho, lam, tau: jkp_extra(a, mu, nu, rho, lam, tau))
    return sp.simplify(out)


# ---------------------------------------------------------------- parent SO(4,1) formulas
def DR5(rho, tau, mu, mode):
    if mode == 'comm':
        return d(R5[tau][mu], rho) + comm(Om[rho], R5[tau][mu], eta5)
    if mode == 'anti':
        return d(R5[tau][mu], rho) + anti(Om[rho], R5[tau][mu], eta5)
    raise ValueError(mode)


def A5(Xm, Ym):
    return anti(Xm, Ym, eta5)


def C5(Xm, Ym):
    return comm(Xm, Ym, eta5)


def chamseddine_2nd(mu, nu, rho, lam, tau, mode):
    """CTZ (3.7) WITHOUT its prefactor 1/32 (5x5 matrix, both indices up):
       -{w_lam, d_tau {w_nu, d_rho w_mu + R_{rho mu}}} + 2 {w_lam, {R_{tau nu}, R_{mu rho}}}
       -{w_lam, {w_nu, D_rho R_{tau mu} + d_rho R_{tau mu}}}
       -{{w_nu, d_rho w_lam + R_{rho lam}}, (d_tau w_mu + R_{tau mu})}
       + 2 [d_nu w_lam, d_rho (d_tau w_mu + R_{tau mu})]"""
    O, F = Om, R5
    t1 = -A5(O[lam], d(A5(O[nu], d(O[mu], rho) + F[rho][mu]), tau))
    t2 = 2 * A5(O[lam], A5(F[tau][nu], F[mu][rho]))
    t3 = -A5(O[lam], A5(O[nu], DR5(rho, tau, mu, mode) + d(F[tau][mu], rho)))
    t4 = -A5(A5(O[nu], d(O[lam], rho) + F[rho][lam]), d(O[mu], tau) + F[tau][mu])
    t5 = 2 * C5(d(O[lam], nu), d(d(O[mu], tau) + F[tau][mu], rho))
    return t1 + t2 + t3 + t4 + t5


def jkp44_2nd(mu, nu, rho, lam, tau, mode):
    """JKP (44) WITHOUT its prefactor 1/16:
       -{{w_lam, (d_tau w_nu + R_{tau nu})}, (d_rho w_mu + R_{rho mu})}
       -{w_nu, d_rho {w_lam, (d_tau w_mu + R_{tau mu})}} + 2 [d_lam w_nu, d_tau (d_rho w_mu + R_{rho mu})]
       +{w_nu, 2 {R_{rho lam}, R_{mu tau}}} - {w_nu, {w_lam, D_tau R_{rho mu} + d_tau R_{rho mu}}}"""
    O, F = Om, R5
    t1 = -A5(A5(O[lam], d(O[nu], tau) + F[tau][nu]), d(O[mu], rho) + F[rho][mu])
    t2 = -A5(O[nu], d(A5(O[lam], d(O[mu], tau) + F[tau][mu]), rho))
    t3 = 2 * C5(d(O[nu], lam), d(d(O[mu], rho) + F[rho][mu], tau))
    t4 = A5(O[nu], 2 * A5(F[rho][lam], F[mu][tau]))
    t5 = -A5(O[nu], A5(O[lam], DR5(tau, rho, mu, mode) + d(F[rho][mu], tau)))
    return t1 + t2 + t3 + t4 + t5


def contracted_tetrad_from_SO41(a, mu, which, mode):
    """Theta^{nu rho} Theta^{lam tau} (omega^{a5}_{mu nu rho lam tau} / k) at k -> 0,
    with the bracket of CTZ (3.7) ('ctz') or JKP (44) ('jkp'), no prefactor."""
    f = chamseddine_2nd if which == 'ctz' else jkp44_2nd
    tot = 0
    for (nu, rho), s1 in THETA_PAIRS:
        for (lam, tau), s2 in THETA_PAIRS:
            tot += s1 * s2 * f(mu, nu, rho, lam, tau, mode)[a, 4]
    tot = sp.expand(tot / k)
    return sp.simplify(tot.subs(k, 0)), sp.simplify(tot)


def first_from_SO41(a, mu):
    """Theta^{nu rho} (omega^{a5}_{mu nu rho} / k), omega_{mu nu rho} = 1/4 {w_nu, d_rho w_mu + R_{rho mu}}."""
    tot = 0
    for (nu, rho), s in THETA_PAIRS:
        tot += s * sp.Rational(1, 4) * A5(Om[nu], d(Om[mu], rho) + R5[rho][mu])[a, 4]
    tot = sp.expand(tot / k)
    return sp.simplify(tot.subs(k, 0))


if __name__ == '__main__':
    # Self-test for the default (r, theta) twist: first-order coefficients (3.11) and the
    # second-order coefficients from the full k -> 0 contraction of (3.7) with prefactor 1/32,
    # compared with the printed bracket (3.12) / 32 for both readings of D.
    from common import to_syms
    names = {0: 't', 1: 'r', 2: 'th', 3: 'ph'}
    for a in range(4):
        for mu in range(4):
            f1 = to_syms(first(a, mu))
            lim, _ = contracted_tetrad_from_SO41(a, mu, 'ctz', 'comm')
            s2 = sp.factor(to_syms(lim) / 32)
            if f1 == 0 and s2 == 0:
                continue
            pr = {mode: sp.factor(to_syms(second_printed(a, mu, sp.Rational(1, 32), mode, extra=False)))
                  for mode in ('anti', 'comm')}
            print(f'e^{a}_{names[mu]}: first = {sp.factor(f1)}; second (contraction of (3.7))/32 = {s2};'
                  f' printed (3.12)/32 with D=anti: {pr["anti"]}, with D=comm: {pr["comm"]}')
