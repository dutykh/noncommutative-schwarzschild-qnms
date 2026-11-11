#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""The deformed Schwarzschild tetrad from the Seiberg-Witten differential equation itself.

Independent of the printed second-order formulas (Chamseddine; Chaichian, Tureanu, Zet (3.7),
(3.12); Juric, Kumara, Pozar (44), (49), (52)): the de Sitter gauge fields of the Schwarzschild
solution are inserted in

    d w_hat_mu / dTheta = -(i/4) eps^{nu rho} { w_hat_nu *, d_rho w_hat_mu + R_hat_{rho mu} },

with  {X *, Y} = X * eta Y + Y * eta X,  R_{rho mu} = d_rho w_mu - d_mu w_rho + w_rho eta w_mu
- w_mu eta w_rho,  the Moyal product  f * g = f g + (i/2) eps^{ab} d_a f d_b g + O(Theta^2)  and
eps^{r theta} = 1 = -eps^{theta r}.  The Taylor coefficients of the solution are

    w1 = G(w, 0),     w2 = (1/2) d/dTheta G(w_hat(Theta), Theta) at Theta = 0,

so that w_hat = w + Theta w1 + Theta^2 w2 + O(Theta^3); the tetrad is the coefficient of k in
w_hat^{a5}, in the limit k -> 0.  The metric then follows from
g_{mu nu} = (1/2) eta_ab (e^a_mu * e^{b+}_nu + e^b_nu * e^{a+}_mu) with the Moyal product to
second order.  Both are compared with the theorems of the article.

Frame order (1, 2, 3, 0, 5) = (r, theta, phi, t, de Sitter), eta = diag(1, 1, 1, -1, 1).
"""
import sys
import sympy as sp

TWIST = sys.argv[1] if len(sys.argv) > 1 else 'r-theta'

r, th, Th, k, M = sp.symbols('r theta Theta k M', positive=True)
t, ph = sp.symbols('t phi')
A = sp.Function('A')(r)
Ap, App, Appp = [sp.diff(A, r, j) for j in (1, 2, 3)]
COORD = {'t': t, 'r': r, 'th': th, 'ph': ph}
MU = ['t', 'r', 'th', 'ph']
ETA = sp.diag(1, 1, 1, -1, 1)
I = sp.I


def zero():
    return sp.zeros(5, 5)


def antisym(entries):
    W = zero()
    for (a, b), v in entries.items():
        W[a, b] = v
        W[b, a] = -v
    return W


# Commutative de Sitter fields: spin connection and omega^{a5} = k e^a.
W = {
    't': antisym({(0, 3): -A*Ap, (3, 4): k*A}),
    'r': antisym({(0, 4): k/A}),
    'th': antisym({(0, 1): -A, (1, 4): k*r}),
    'ph': antisym({(0, 2): -A*sp.sin(th), (1, 2): -sp.cos(th), (2, 4): k*r*sp.sin(th)}),
}


def d(X, mu):
    return X.applyfunc(lambda e: sp.diff(e, COORD[mu]))


def mul(X, Y):
    return X*ETA*Y


TW1, TW2 = ('r', 'th') if TWIST == 'r-theta' else ('t', 'th')


def poisson(X, Y):
    """Antisymmetrised first-order Moyal term eps^{ab} d_a X eta d_b Y (matrix order kept)."""
    return mul(d(X, TW1), d(Y, TW2)) - mul(d(X, TW2), d(Y, TW1))


EPS = [(TW1, TW2, 1), (TW2, TW1, -1)]


def field_strength(Wd, W1=None):
    """R_{rho mu} at order 0, and its first-order part when W1 (the Theta coefficient) is given."""
    R0, R1 = {}, {}
    for rho in MU:
        for mu in MU:
            R0[rho, mu] = d(Wd[mu], rho) - d(Wd[rho], mu) + mul(Wd[rho], Wd[mu]) - mul(Wd[mu], Wd[rho])
            if W1 is not None:
                R1[rho, mu] = (d(W1[mu], rho) - d(W1[rho], mu)
                               + mul(W1[rho], Wd[mu]) + mul(Wd[rho], W1[mu])
                               - mul(W1[mu], Wd[rho]) - mul(Wd[mu], W1[rho])
                               + I/2*(poisson(Wd[rho], Wd[mu]) - poisson(Wd[mu], Wd[rho])))
    return R0, R1


R0, _ = field_strength(W)
X0 = {(rho, mu): d(W[mu], rho) + R0[rho, mu] for rho in MU for mu in MU}

# First Taylor coefficient: w1 = -(i/4) eps^{nu rho} {w_nu, X_{rho mu}}.
W1 = {}
for mu in MU:
    S = zero()
    for nu, rho, s in EPS:
        S += s*(mul(W[nu], X0[rho, mu]) + mul(X0[rho, mu], W[nu]))
    W1[mu] = (-I/4*S).applyfunc(sp.simplify)

# Theta^1 part of the right-hand side along the solution.
_, R1 = field_strength(W, W1)
W2 = {}
for mu in MU:
    S = zero()
    for nu, rho, s in EPS:
        X1 = d(W1[mu], rho) + R1[rho, mu]
        S += s*(mul(W1[nu], X0[rho, mu]) + mul(X0[rho, mu], W1[nu])
                + mul(W[nu], X1) + mul(X1, W[nu])
                + I/2*(poisson(W[nu], X0[rho, mu]) + poisson(X0[rho, mu], W[nu])))
    W2[mu] = (-I/4*S/2).applyfunc(sp.expand)   # Taylor coefficient: one half of the derivative


def tetrad_coeff(Wmat, a, mu):
    return sp.simplify(sp.diff(Wmat[mu][a, 4], k).subs(k, 0))


FR = {0: '1', 1: '2', 2: '3', 3: '0'}
claimed = {
    (3, 't'): (0, (2*r*Ap**3 + 5*r*A*Ap*App + r*A**2*Appp + 2*A*Ap**2 + A**2*App)/8),
    (0, 'r'): (0, App/8),
    (0, 'th'): (-(A + 2*r*Ap)/4, 0),
    (1, 'th'): (0, (2*A*Ap + 3*r*Ap**2 + 3*r*A*App)/8),
    (2, 'ph'): (-sp.cos(th)/4, (2*r*Ap**2 + r*A*App + 2*A*Ap - Ap/A)*sp.sin(th)/8),
}
print('Tetrad e^a_mu = e + Theta e1 + Theta^2 e2 from the Seiberg-Witten equation:')
E = {}
ok = True
for a in range(4):
    for mu in MU:
        e0 = tetrad_coeff(W, a, mu)
        e1 = tetrad_coeff(W1, a, mu)
        e2 = tetrad_coeff(W2, a, mu)
        E[a, mu] = (e0, e1, e2)
        c1, c2 = claimed.get((a, mu), (0, 0))
        d1 = sp.simplify(e1 - I*c1) if c1 != 0 else sp.simplify(e1)
        d2 = sp.simplify(e2 - c2)
        if e0 != 0 or e1 != 0 or e2 != 0 or c1 != 0 or c2 != 0:
            print(f'  e^{FR[a]}_{mu}: zeroth {e0};  Theta {e1};  Theta^2 {sp.factor(e2)}')
            if TWIST == 'r-theta':
                print(f'      difference with the theorem: Theta {d1}, Theta^2 {d2}')
        ok = ok and d1 == 0 and d2 == 0
if TWIST == 'r-theta':
    print('Tetrad of the theorem reproduced:', ok)


# Metric with the Moyal product to second order.
def star(f, g):
    u, v = COORD[TW1], COORD[TW2]
    p1 = sp.diff(f, u)*sp.diff(g, v) - sp.diff(f, v)*sp.diff(g, u)
    p2 = (sp.diff(f, u, 2)*sp.diff(g, v, 2) - 2*sp.diff(f, u, v)*sp.diff(g, u, v)
          + sp.diff(f, v, 2)*sp.diff(g, u, 2))
    return f*g + I*Th/2*p1 - Th**2/8*p2


def full(a, mu, conj=False):
    e0, e1, e2 = E[a, mu]
    s = -1 if conj else 1
    # e1 is the coefficient of Theta, purely imaginary here; conjugation flips its sign.
    return e0 + s*Th*e1 + Th**2*e2


etad = [1, 1, 1, -1]
print('\nMetric, coefficients of Theta^1 and Theta^2 (A^2 = 1 - 2M/r):')
schw = {A: sp.sqrt(1 - 2*M/r)}
claimed_g = {
    ('t', 't'): -M*(4*r - 11*M)/(4*r**4),
    ('r', 'r'): -M*(2*r - 3*M)/(4*r**2*(r - 2*M)**2),
    ('th', 'th'): sp.Rational(1, 16) - M/r + M/(8*(r - 2*M)),
    ('ph', 'ph'): 5*sp.cos(th)**2/16 + (r**2 - 3*M*r + M**2)/(4*r*(r - 2*M))*sp.sin(th)**2,
}
okg = True
for i, mu in enumerate(MU):
    for nu in MU[i:]:
        g = 0
        for a in range(4):
            g += etad[a]*(star(full(a, mu), full(a, nu, True)) + star(full(a, nu), full(a, mu, True)))/2
        g = sp.expand(g)
        g1 = g.coeff(Th, 1)
        g2 = g.coeff(Th, 2)
        g2s = sp.simplify(g2.subs(schw).doit())
        g1s = sp.simplify(g1.subs(schw).doit())
        target = claimed_g.get((mu, nu), 0)
        diff = sp.simplify(g2s - target)
        if TWIST != 'r-theta':
            target, diff = None, None
        if g1s != 0 or g2s != 0 or target:
            tail = f';  difference with the theorem {diff}' if TWIST == 'r-theta' else ''
            print(f'  g_{mu}{nu}: Theta^1 {g1s};  Theta^2 {sp.factor(g2s)}' + tail)
        okg = okg and g1s == 0 and (diff == 0 or TWIST != 'r-theta')
if TWIST == 'r-theta':
    print('Metric of the theorem reproduced:', okg)
else:
    print('No component of first order in Theta:', okg)

if TWIST != 'r-theta':
    f = 1 - 2*M/r
    G = {}
    for i, mu in enumerate(MU):
        for nu in MU[i:]:
            g = 0
            for a in range(4):
                g += etad[a]*(star(full(a, mu), full(a, nu, True)) + star(full(a, nu), full(a, mu, True)))/2
            G[mu, nu] = sp.simplify(sp.expand(g).coeff(Th, 2).subs(schw).doit())
    A0 = sp.factor(-G['t', 't']/f)
    B0 = sp.factor(f*G['r', 'r'])
    C0 = sp.factor(G['th', 'th']/r**2)
    D0 = sp.factor(sp.simplify(G['ph', 'ph']/(r**2*sp.sin(th)**2)))
    print('Deformation functions for the (t, theta) twist:')
    print('  A0 =', A0, '\n  B0 =', B0, '\n  C0 =', C0, '\n  D0 =', D0)
    off = {kk: v for kk, v in G.items() if kk[0] != kk[1] and v != 0}
    print('  off-diagonal Theta^2 components:', off if off else 'none')
    claimed_tt = [-M**2*(r - 3*M)/(8*r**5), -M*(2*r**2 - 10*M*r + 11*M**2)/(8*r**5),
                  M*(r - 2*M)*(r - 3*M)/(8*r**5), M*(r - 2*M)*(r + M)/(8*r**5)]
    print('  difference with the article:',
          [sp.simplify(x - y) for x, y in zip((A0, B0, C0, D0), claimed_tt)])
