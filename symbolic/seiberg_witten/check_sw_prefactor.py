# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""
Independent test of the overall factor of the second-order Seiberg-Witten map:
Chamseddine / CTZ (3.7) carry 1/32, JKP (41), (44) carry 1/16 (their footnote 12 calls
1/32 a typo).

Setting: matrix-valued (gl(2)) gauge fields on the plane (x, y), Theta^{xy} = Theta,
Chamseddine's conventions (CTZ (3.2)-(3.9) with the gauge-covariant D; JKP (27)-(34)):
    f * g = f g + (i/2) Theta (d_x f d_y g - d_y f d_x g)
            - (Theta^2/8)(d_x^2 f d_y^2 g - 2 d_xd_y f d_xd_y g + d_y^2 f d_x^2 g)
    R_hat_{mu nu} = d_mu w_hat_nu - d_nu w_hat_mu + w_hat_mu * w_hat_nu - w_hat_nu * w_hat_mu
    delta_hat w_hat = d lam_hat + [w_hat *, lam_hat],   delta w = d lam + [w, lam]
Seiberg-Witten differential equation (JKP (32); SW (3.8) with A -> i w, lambda -> i lam):
    d w_hat_mu / dTheta = -(i/4) eps^{nu rho} {w_hat_nu *, d_rho w_hat_mu + R_hat_{rho mu}}
    d lam_hat  / dTheta = +(i/4) eps^{kl}   {d_k lam_hat *, w_hat_l}
Two ways of solving it to second order:
    ODE : Taylor coefficients, w2 = (1/2) [Theta^1 part of the right-hand side]
    FP  : JKP's reading of (32) as a fixed-point relation w_hat = w - (i/4) Theta {w_hat *, ...},
          which is their (36): w2 = 1 * [Theta^1 part of the right-hand side]
Tests:
  1. closed forms: CTZ (3.7)/32 equals ODE w2; JKP (44)/16 equals FP w2 (= 2 x ODE w2);
  2. the defining property of the map (gauge equivalence)
        w_hat(w + delta w) - w_hat(w) = delta_hat w_hat(w)    to O(Theta^2), first order in lam,
     holds for ODE and fails for FP;  additionally, the FP failure cannot be repaired by any
     choice of lam_hat at second order (checked by solving for a general polynomial lam2);
  3. U(1), constant field strength: SW's exact result R_hat = R / (1 - i Theta R) fixes the
     Theta^2 coefficient of R_hat to -R^3.
Polynomial arithmetic in the sparse ring QQ(i)[x, y, T, E] (T = Theta, E = gauge parameter size).
"""
import random
import sympy as sp
from sympy.polys.domains import QQ_I
from sympy.polys.rings import ring

P, x, y, T, E = ring('x,y,T,E', QQ_I)
VARS = [x, y]
ONE = P(1)
Ihalf = P.domain.from_sympy(sp.I / 2)
Iq = P.domain.from_sympy(sp.I / 4)
EPS = [((0, 1), 1), ((1, 0), -1)]   # eps^{xy} = +1


def tr(p, nT=2, nE=1):
    return P.from_dict({m: c for m, c in p.items() if m[2] <= nT and m[3] <= nE}) if p else P(0)


def cT(p, j):
    """Coefficient of T^j (as a polynomial without T)."""
    return P.from_dict({(m[0], m[1], 0, m[3]): c for m, c in p.items() if m[2] == j}) if p else P(0)


def cE(p, j):
    return P.from_dict({(m[0], m[1], m[2], 0): c for m, c in p.items() if m[3] == j}) if p else P(0)


# ---- 2x2 (or NxN) matrices as lists of lists of ring elements
def mz(N):
    return [[P(0) for _ in range(N)] for _ in range(N)]


def madd(*ms):
    N = len(ms[0])
    return [[sum((m[i][j] for m in ms), P(0)) for j in range(N)] for i in range(N)]


def msc(c, m):
    return [[c * z for z in row] for row in m]


def mmul(a, b):
    N = len(a)
    return [[tr(sum((a[i][l] * b[l][j] for l in range(N)), P(0))) for j in range(N)] for i in range(N)]


def mdiff(m, v):
    return [[z.diff(VARS[v]) for z in row] for row in m]


def mmap(f, m):
    return [[f(z) for z in row] for row in m]


def anti(a, b):
    return madd(mmul(a, b), mmul(b, a))


def comm(a, b):
    return madd(mmul(a, b), msc(-ONE, mmul(b, a)))


def star(f, g):
    d = mdiff
    out = mmul(f, g)
    out = madd(out, msc(Ihalf * T, madd(mmul(d(f, 0), d(g, 1)), msc(-ONE, mmul(d(f, 1), d(g, 0))))))
    q = madd(mmul(d(d(f, 0), 0), d(d(g, 1), 1)), msc(P(-2), mmul(d(d(f, 0), 1), d(d(g, 0), 1))),
             mmul(d(d(f, 1), 1), d(d(g, 0), 0)))
    out = madd(out, msc(P.domain.from_sympy(sp.Rational(-1, 8)) * T**2, q))
    return mmap(tr, out)


def field_strength(w):
    return [[madd(mdiff(w[n], m), msc(-ONE, mdiff(w[m], n)), comm(w[m], w[n])) for n in range(2)]
            for m in range(2)]


def nc_field_strength(wh):
    return [[mmap(tr, madd(mdiff(wh[n], m), msc(-ONE, mdiff(wh[m], n)), star(wh[m], wh[n]),
                           msc(-ONE, star(wh[n], wh[m])))) for n in range(2)] for m in range(2)]


def sw_map(w, lam, mode):
    N = len(w[0])
    Rc = field_strength(w)
    w1 = []
    for mu in range(2):
        s = mz(N)
        for (nu, rho), sg in EPS:
            s = madd(s, msc(P(sg), anti(w[nu], madd(mdiff(w[mu], rho), Rc[rho][mu]))))
        w1.append(msc(-Iq * ONE, s))
    s = mz(N)
    for (kk, ll), sg in EPS:
        s = madd(s, msc(P(sg), anti(mdiff(lam, kk), w[ll])))
    lam1 = msc(Iq * ONE, s)
    wh = [madd(w[mu], msc(T, w1[mu])) for mu in range(2)]
    lh = madd(lam, msc(T, lam1))
    Rh = nc_field_strength(wh)
    fac = P.domain.from_sympy(sp.Rational(1, 2)) if mode == 'ODE' else P.domain.one
    w2 = []
    for mu in range(2):
        s = mz(N)
        for (nu, rho), sg in EPS:
            Y = madd(mdiff(wh[mu], rho), Rh[rho][mu])
            s = madd(s, msc(P(sg), madd(star(wh[nu], Y), star(Y, wh[nu]))))
        w2.append(mmap(lambda z: fac * cT(-Iq * z, 1), s))
    s = mz(N)
    for (kk, ll), sg in EPS:
        s = madd(s, msc(P(sg), madd(star(mdiff(lh, kk), wh[ll]), star(wh[ll], mdiff(lh, kk)))))
    lam2 = mmap(lambda z: fac * cT(Iq * z, 1), s)
    what = [mmap(tr, madd(w[mu], msc(T, w1[mu]), msc(T**2, w2[mu]))) for mu in range(2)]
    lhat = mmap(tr, madd(lam, msc(T, lam1), msc(T**2, lam2)))
    return what, lhat, w1, w2


def closed_form_2nd(w, which):
    """Theta^{nu rho} Theta^{lam tau} w_{mu nu rho lam tau} / Theta^2 from CTZ (3.7) or JKP (44)
    (bracket only, no prefactor), gauge-covariant D."""
    Rc = field_strength(w)
    d = mdiff
    neg = lambda m: msc(-ONE, m)

    def DR(rho, tau, mu):
        return madd(d(Rc[tau][mu], rho), comm(w[rho], Rc[tau][mu]))
    out = []
    for mu in range(2):
        tot = mz(len(w[0]))
        for (nu, rho), s1 in EPS:
            for (lam, tau), s2 in EPS:
                if which == 'ctz':
                    t = madd(neg(anti(w[lam], d(anti(w[nu], madd(d(w[mu], rho), Rc[rho][mu])), tau))),
                             msc(P(2), anti(w[lam], anti(Rc[tau][nu], Rc[mu][rho]))),
                             neg(anti(w[lam], anti(w[nu], madd(DR(rho, tau, mu), d(Rc[tau][mu], rho))))),
                             neg(anti(anti(w[nu], madd(d(w[lam], rho), Rc[rho][lam])),
                                      madd(d(w[mu], tau), Rc[tau][mu]))),
                             msc(P(2), comm(d(w[lam], nu), d(madd(d(w[mu], tau), Rc[tau][mu]), rho))))
                else:
                    t = madd(neg(anti(anti(w[lam], madd(d(w[nu], tau), Rc[tau][nu])),
                                      madd(d(w[mu], rho), Rc[rho][mu]))),
                             neg(anti(w[nu], d(anti(w[lam], madd(d(w[mu], tau), Rc[tau][mu])), rho))),
                             msc(P(2), comm(d(w[nu], lam), d(madd(d(w[mu], rho), Rc[rho][mu]), tau))),
                             anti(w[nu], msc(P(2), anti(Rc[rho][lam], Rc[mu][tau]))),
                             neg(anti(w[nu], anti(w[lam], madd(DR(tau, rho, mu), d(Rc[rho][mu], tau))))))
                tot = madd(tot, msc(P(s1 * s2), t))
        out.append(tot)
    return out


def is_zero(m):
    return all(z == 0 for row in m for z in row)


def rand_poly(rng, deg=2):
    return sum((rng.randint(-3, 3) * x**i * y**j for i in range(deg + 1) for j in range(deg + 1 - i)), P(0))


def rand_mat(rng, N=2, deg=2):
    return [[rand_poly(rng, deg) for _ in range(N)] for _ in range(N)]


def gauge_residual(w, lam, mode):
    """Residual of w_hat(w + E delta w) - w_hat(w) - E delta_hat w_hat(w), returned per mu
    as the coefficient of E (a polynomial in x, y, T)."""
    dw = [madd(mdiff(lam, mu), comm(w[mu], lam)) for mu in range(2)]
    wh, lh, _, _ = sw_map(w, lam, mode)
    whp, _, _, _ = sw_map([madd(w[mu], msc(E, dw[mu])) for mu in range(2)], lam, mode)
    res = []
    for mu in range(2):
        lhs = mmap(lambda z: cE(z, 1), whp[mu])
        rhs = mmap(tr, madd(mdiff(lh, mu), star(wh[mu], lh), msc(-ONE, star(lh, wh[mu]))))
        res.append(madd(lhs, msc(-ONE, rhs)))
    return res


def main():
    rng = random.Random(20261020)
    N = 2
    w = [rand_mat(rng, N), rand_mat(rng, N)]
    lam = rand_mat(rng, N)

    print('Test 1: closed forms against the integrated SW equation (random gl(2) fields, degree 2)')
    _, _, _, w2_ode = sw_map(w, lam, 'ODE')
    _, _, _, w2_fp = sw_map(w, lam, 'FP')
    c37 = closed_form_2nd(w, 'ctz')
    c44 = closed_form_2nd(w, 'jkp')
    h32 = P.domain.from_sympy(sp.Rational(1, 32))
    h16 = P.domain.from_sympy(sp.Rational(1, 16))
    for mu in range(2):
        d1 = is_zero(madd(msc(h32 * ONE, c37[mu]), msc(-ONE, w2_ode[mu])))
        d2 = is_zero(madd(msc(h16 * ONE, c44[mu]), msc(-ONE, w2_fp[mu])))
        d3 = is_zero(madd(msc(h16 * ONE, c44[mu]), msc(-2 * h32 * ONE, c37[mu])))
        print(f'  mu={mu}: (3.7)/32 = w2[ODE]: {d1};  (44)/16 = w2[FP]: {d2};  (44)/16 = 2 (3.7)/32: {d3}')

    print('Test 2: gauge equivalence  w_hat(w + delta w) - w_hat(w) = delta_hat w_hat(w)')
    for mode in ('ODE', 'FP'):
        res = gauge_residual(w, lam, mode)
        for mu in range(2):
            orders = [is_zero(mmap(lambda z: cT(z, j), res[mu])) for j in range(3)]
            print(f'  {mode}, mu={mu}: residual vanishes at Theta^0, Theta^1, Theta^2: {orders}')

    # Can the FP failure be absorbed into a different second-order gauge parameter lam2'?
    # The Theta^2 residual would have to equal D_mu L = d_mu L + [w_mu, L] for one matrix L
    # (polynomial in x, y; degree up to 8 is ample).  Solve the linear system for L.
    print('Test 2b: can the FP residual at Theta^2 be written as d_mu L + [w_mu, L] for some L?')
    res = gauge_residual(w, lam, 'FP')
    r2 = [mmap(lambda z: cT(z, 2), res[mu]) for mu in range(2)]
    deg = 8
    syms = []
    Lm = [[None] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            expr = 0
            for a in range(deg + 1):
                for b in range(deg + 1 - a):
                    s = sp.Symbol(f'L{i}{j}_{a}_{b}')
                    syms.append(s)
                    expr += s * sp.Symbol('x')**a * sp.Symbol('y')**b
            Lm[i][j] = expr
    xs, ys = sp.symbols('x y')
    wexp = [[[w[mu][i][j].as_expr() for j in range(N)] for i in range(N)] for mu in range(2)]
    eqs = []
    for mu in range(2):
        for i in range(N):
            for j in range(N):
                DL = sp.diff(Lm[i][j], [xs, ys][mu])
                for l in range(N):
                    DL += wexp[mu][i][l] * Lm[l][j] - Lm[i][l] * wexp[mu][l][j]
                target = r2[mu][i][j].as_expr()
                diffp = sp.Poly(sp.expand(DL - target), xs, ys)
                eqs.extend(diffp.coeffs())
    sol = sp.linsolve(eqs, syms)
    print('  solvable:', sol != sp.EmptySet)
    res_ode = gauge_residual(w, lam, 'ODE')
    print('  (for comparison, the ODE residual at Theta^2 is identically zero:',
          all(is_zero(mmap(lambda z: cT(z, 2), res_ode[mu])) for mu in range(2)), ')')

    print('Test 3: U(1), constant field strength (w_x = -R y / 2, w_y = R x / 2), R a number')
    for Rv in (sp.Rational(3, 2), sp.Integer(-2)):
        Rd = P.domain.from_sympy(Rv)
        wu = [[[-Rd * y / 2]], [[Rd * x / 2]]]
        exact = sp.series(Rv / (1 - sp.I * sp.Symbol('Theta') * Rv), sp.Symbol('Theta'), 0, 3).removeO()
        for mode in ('ODE', 'FP'):
            wh, _, _, _ = sw_map(wu, [[P(0)]], mode)
            Rh = nc_field_strength(wh)[0][1][0][0]
            print(f'  R={Rv}, {mode}: R_hat_xy = {Rh.as_expr()}   (SW exact: {sp.expand(exact)})')


if __name__ == '__main__':
    main()
