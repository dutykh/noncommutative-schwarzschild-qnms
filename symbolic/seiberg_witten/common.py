# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""
Common commutative data for the second-order Seiberg-Witten (Moyal) deformation
of the Schwarzschild tetrad with the (r, theta) twist.

Conventions used throughout (both papers agree on the labels):
  coordinate index mu = 0, 1, 2, 3  <->  (t, r, theta, phi)
  frame index      a  = 0, 1, 2, 3  <->  e^0 = A dt, e^1 = dr / A, e^2 = r dtheta,
                                          e^3 = r sin(theta) dphi
  so that CTZ's labels (1, 2, 3, 0) = (r, theta, phi, t) and JKP's labels
  (0, 1, 2, 3) = (t, r, theta, phi) are the same numbers.
  eta_ab = diag(-1, 1, 1, 1); the de Sitter index 5 is stored at position 4,
  eta_55 = +1.
  Gauge fields are stored as matrices with BOTH group indices UP:
  W[mu][a, b] = omega^{ab}_mu.  The bracket conventions (CTZ (3.8), JKP (28)):
      {X, Y}^{AB} = X^{AC} eta_CD Y^{DB} + Y^{AC} eta_CD X^{DB} = (X eta Y + Y eta X)^{AB}
      [X, Y]^{AB} = (X eta Y - Y eta X)^{AB}
  Field strength (CTZ (2.2), JKP (7)):
      R_{mu nu} = d_mu w_nu - d_nu w_mu + [w_mu, w_nu].
  Twist: Theta^{r theta} = +Theta = -Theta^{theta r}, all other entries zero
  (CTZ (4.1) and JKP (111) agree).
"""
import sympy as sp

t, r, th, ph = sp.symbols('t r theta phi', real=True)
X = [t, r, th, ph]
Th = sp.Symbol('Theta', real=True)
k = sp.Symbol('k')
M = sp.Symbol('M', positive=True)
A = sp.Function('A')(r)

# symbols standing for A, A', A'', A''', A'''' in the final comparison
a0, a1, a2, a3, a4 = sp.symbols('A A1 A2 A3 A4')


def to_syms(expr):
    """Replace A(r) and its derivatives by plain symbols A, A1, A2, A3, A4."""
    expr = sp.sympify(expr)
    rep = {sp.Derivative(A, (r, 4)): a4, sp.Derivative(A, (r, 3)): a3,
           sp.Derivative(A, (r, 2)): a2, sp.Derivative(A, r): a1}
    expr = expr.subs(rep)
    return expr.subs(A, a0)


def from_syms(expr):
    rep = {a4: sp.diff(A, r, 4), a3: sp.diff(A, r, 3), a2: sp.diff(A, r, 2),
           a1: sp.diff(A, r), a0: A}
    return sp.sympify(expr).subs(rep)


eta4 = sp.diag(-1, 1, 1, 1)
eta5 = sp.diag(-1, 1, 1, 1, 1)

# commutative tetrad e[a][mu]
e = sp.zeros(4, 4)
e[0, 0] = A
e[1, 1] = 1 / A
e[2, 2] = r
e[3, 3] = r * sp.sin(th)


def d(expr, mu):
    """Partial derivative with respect to coordinate mu (matrices allowed)."""
    return sp.diff(expr, X[mu])


def solve_spin_connection():
    """Solve the null-torsion condition
       T^a_{mu nu} = d_mu e^a_nu - d_nu e^a_mu + (w^{ab}_mu e^c_nu - w^{ab}_nu e^c_mu) eta_bc = 0
    for the antisymmetric omega^{ab}_mu (24 unknowns, linear system)."""
    unknowns = {}
    W = [sp.zeros(4, 4) for _ in range(4)]
    for mu in range(4):
        for a in range(4):
            for b in range(a + 1, 4):
                s = sp.Symbol(f'w_{a}{b}_{mu}')
                unknowns[s] = None
                W[mu][a, b] = s
                W[mu][b, a] = -s
    eqs = []
    for a in range(4):
        for mu in range(4):
            for nu in range(mu + 1, 4):
                T = d(e[a, nu], mu) - d(e[a, mu], nu)
                for b in range(4):
                    for c in range(4):
                        T += (W[mu][a, b] * e[c, nu] - W[nu][a, b] * e[c, mu]) * eta4[b, c]
                eqs.append(sp.expand(T))
    sol = sp.solve(eqs, list(unknowns.keys()), dict=True)
    assert len(sol) == 1
    sol = sol[0]
    Wsol = [sp.simplify(W[mu].subs(sol)) for mu in range(4)]
    return Wsol


W = solve_spin_connection()


def anti(Xm, Ym, eta=eta4):
    return Xm * eta * Ym + Ym * eta * Xm


def comm(Xm, Ym, eta=eta4):
    return Xm * eta * Ym - Ym * eta * Xm


def field_strength(Wl, eta):
    n = Wl[0].shape[0]
    R = [[sp.zeros(n, n) for _ in range(4)] for _ in range(4)]
    for mu in range(4):
        for nu in range(4):
            R[mu][nu] = (d(Wl[nu], mu) - d(Wl[mu], nu) + comm(Wl[mu], Wl[nu], eta)).applyfunc(sp.simplify)
    return R


# Lorentz curvature (k -> 0)
R = field_strength(W, eta4)

# de Sitter (SO(4,1)) gauge field with contraction parameter k:
#   Omega^{ab}_mu = omega^{ab}_mu, Omega^{a5}_mu = k e^a_mu, Omega^{5a}_mu = -k e^a_mu
Om = []
for mu in range(4):
    O = sp.zeros(5, 5)
    O[:4, :4] = W[mu]
    for a in range(4):
        O[a, 4] = k * e[a, mu]
        O[4, a] = -k * e[a, mu]
    Om.append(O)
R5 = field_strength(Om, eta5)

# Twist: Theta^{mu0 nu0} = +1, Theta^{nu0 mu0} = -1 (in units of Theta), all others zero.
# Default (r, theta): Theta^{12} = +1 (CTZ (4.1), JKP (111)).  set_twist(0, 2) gives the
# (t, theta) twist of JKP (87).  The list is changed in place, so that every module that
# imported it sees the current twist.
TWIST = [1, 2]
THETA_PAIRS = [((1, 2), 1), ((2, 1), -1)]


def set_twist(mu0, nu0):
    TWIST[:] = [mu0, nu0]
    THETA_PAIRS[:] = [((mu0, nu0), 1), ((nu0, mu0), -1)]


def theta1(f):
    """Theta^{nu rho} f(nu, rho), coefficient of Theta^1."""
    return sum(s * f(nu, rho) for (nu, rho), s in THETA_PAIRS)


def theta2(f):
    """Theta^{nu rho} Theta^{lam tau} f(nu, rho, lam, tau), coefficient of Theta^2."""
    out = 0
    for (nu, rho), s1 in THETA_PAIRS:
        for (lam, tau), s2 in THETA_PAIRS:
            out += s1 * s2 * f(nu, rho, lam, tau)
    return out


def moyal(f, g, s1=1, s2=-1, order=2):
    """Moyal product to second order for the current twist (u, v) = (X[mu0], X[nu0]),
    Theta^{uv} = Theta (default u = r, v = theta):
         f * g = f g + s1 (i/2) Theta (d_u f d_v g - d_v f d_u g)
                 + s2 (Theta^2/8) (d_u^2 f d_v^2 g - 2 d_u d_v f d_u d_v g + d_v^2 f d_u^2 g)
       s1 = +1: CTZ (3.2) and Seiberg-Witten; s1 = -1: JKP printed (1), (67).
       s2 = -1: expansion of the exponential (correct for either s1);
       s2 = +1: the sign printed in JKP (67)."""
    u, v = X[TWIST[0]], X[TWIST[1]]
    out = f * g
    if order >= 1:
        out += s1 * sp.I / 2 * Th * (sp.diff(f, u) * sp.diff(g, v) - sp.diff(f, v) * sp.diff(g, u))
    if order >= 2:
        out += s2 * Th**2 / 8 * (sp.diff(f, u, 2) * sp.diff(g, v, 2)
                                 - 2 * sp.diff(f, u, v) * sp.diff(g, u, v)
                                 + sp.diff(f, v, 2) * sp.diff(g, u, 2))
    return out


def report_connection():
    names = ['t', 'r', 'th', 'ph']
    print('Spin connection omega^{ab}_mu from null torsion (nonzero, a<b):')
    for mu in range(4):
        for a in range(4):
            for b in range(a + 1, 4):
                if W[mu][a, b] != 0:
                    print(f'  omega^{{{a}{b}}}_{names[mu]} =', W[mu][a, b])
    print('Nonzero Riemann components R^{ab}_{mu nu} (a<b, mu<nu):')
    for mu in range(4):
        for nu in range(mu + 1, 4):
            for a in range(4):
                for b in range(a + 1, 4):
                    if R[mu][nu][a, b] != 0:
                        print(f'  R^{{{a}{b}}}_{names[mu]}{names[nu]} =', sp.simplify(R[mu][nu][a, b]))
    # torsion components of the SO(4,1) curvature vanish identically
    ok = all(sp.simplify(R5[mu][nu][a, 4]) == 0 for mu in range(4) for nu in range(4) for a in range(4))
    print('R^{a5}_{mu nu} = k T^a_{mu nu} vanishes identically:', ok)


if __name__ == '__main__':
    report_connection()
