#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""Exact horizon structure of the master equation, with l and |m| kept symbolic.

The master equation of Chen, Chiang and Tsao for g_tt = -f(1+T A0), g_rr = (1+T B0)/f,
g_thth = r^2 (1+T C0), g_phph = r^2 sin^2 (1+T D0) + T kax cos^2 (T = Theta^2, f = 1 - 2M/r) gives
    dr/dr* = F = f (1 + T (A0 - B0)/2),
    V = f (L/r^2 + 2M/r^3) + T [ (f/r) f' (A0 - B0) + (f/r^2)(a0 (A0 - D0) - c0 (A0 - C0))
          + (1/4r^2) d*(r^2 d*(A0 - B0 + C0 + D0)) - (1/4) d*d*(A0 - B0) + Vhat ],
with d* = f d/dr inside the bracket, a0 = |m|(2l+1)/2, c0 = a0 - L, L = l(l+1), and
Vhat = 16 kax f r^-5 (-s1 r + s2 (r - 4M)/2), which vanishes at r = 2M.

Claims checked here, symbolically in L, |m| (written am) and in the residues:
  (1) if A0, B0, C0, D0 have at most simple poles at r = 2M with residues a, b, c, d, then the bracket
      has at most a simple pole there, with residue (c + d)/(16 M^2);
  (2) for the metric of Chaichian-Tureanu-Zet and for the correct metric ('swc'), c + d = 0, the
      horizon r_h of F satisfies r_h = 2M + T/(16M) + O(T^2), and
          V(r_h) = -(3L - (2l+1) am) T / (256 M^4) + O(T^2),
      which is negative for every l >= 1 and 1 <= am <= l, since 3L - (2l+1) am >= l(l+2);
  (3) for the metric of Juric-Kumara-Pozar, c + d = -1/(32M), the bracket has the residue
      -1/(512 M^3), and V(r_h) = -1/(32 M^2) + O(T).
"""
import sympy as sp

r, M, T, L, am, l = sp.symbols('r M T L a_m l', positive=True)
a, b, c, d = sp.symbols('a b c d')
f = 1 - 2*M/r
x = r - 2*M


def bracket(A0, B0, C0, D0):
    dd = lambda g: f*sp.diff(g, r)
    a0 = am*(2*l + 1)/2
    c0 = a0 - L
    return ((f/r)*sp.diff(f, r)*(A0 - B0) + (f/r**2)*(a0*(A0 - D0) - c0*(A0 - C0))
            + 1/(4*r**2)*dd(r**2*dd(A0 - B0 + C0 + D0)) - dd(dd(A0 - B0))/4)


# (1) generic simple poles: A0 = a/(r-2M) + alpha(r) etc., with regular parts written as Taylor
# polynomials of degree 3 with arbitrary coefficients (at most two derivatives enter the bracket)
X = sp.symbols('X')                      # X = r - 2M
co = {n: sp.symbols(f'{n}0:4') for n in ('p', 'q', 's', 't')}
reg = {n: sum(co[n][k]*x**k for k in range(4)) for n in co}
gen = bracket(a/x + reg['p'], b/x + reg['q'], c/x + reg['s'], d/x + reg['t'])
lau = sp.series(sp.expand(gen.subs(r, 2*M + X)), X, 0, 1).removeO()
print('(1) coefficient of (r - 2M)^-2 in the bracket:', sp.factor(lau.coeff(X, -2)))
print('    coefficient of (r - 2M)^-1 in the bracket:', sp.factor(lau.coeff(X, -1)))


def model(name):
    al = 2*M
    if name in ('ctz', 'swc'):
        A0 = al*(8*r - 11*al)/(16*r**3*(r - al))
        B0 = -al*(4*r - 3*al)/(16*r**3*(r - al))
    if name == 'ctz':
        C0 = (2*r**2 - 17*al*r + 17*al**2)/(32*r**3*(r - al))
        D0 = -al*(2*r - al)/(16*r**3*(r - al))
    elif name == 'swc':
        C0 = (r**2 - 16*M*r + 32*M**2)/(16*r**3*(r - 2*M))
        D0 = (r**2 - 3*M*r + M**2)/(4*r**3*(r - 2*M))
    elif name == 'jkp_rth':
        A0 = M*(4*r - 11*M)/(2*r**3*(r - 2*M))
        B0 = M*(3*M - 2*r)/(2*r**3*(r - 2*M))
        C0 = (r**2 - 32*M*r + 64*M**2)/(16*r**3*(r - 2*M))
        D0 = (2*M**2 - 4*M*r + r**2)/(4*r**3*(r - 2*M))
    return A0, B0, C0, D0


for name in ('ctz', 'swc', 'jkp_rth'):
    A0, B0, C0, D0 = model(name)
    resid = [sp.simplify(sp.limit(g*x, r, 2*M)) for g in (A0, B0, C0, D0)]
    Br = bracket(A0, B0, C0, D0)
    res = sp.simplify(sp.limit(sp.expand(Br*x), r, 2*M))
    print(f'\n{name}: residues of A0, B0, C0, D0 at 2M: {resid};  residue of the bracket: {res}')
    # horizon: F = f + T f (A0 - B0)/2 is regular; solve F(2M + h) = 0 to first order in T
    F = sp.cancel(f*(1 + T*(A0 - B0)/2))
    h = sp.symbols('h')
    F1 = sp.series(F.subs(r, 2*M + h), h, 0, 2).removeO()
    hsol = sp.solve(sp.Eq(F1, 0), h)[0]
    h1 = sp.series(hsol, T, 0, 2).removeO()
    print('   r_h - 2M =', sp.simplify(h1), '+ O(T^2)')
    if res == 0:
        # V(r_h) to first order: V_Sch(r_h) + T * bracket(2M); Vhat vanishes at 2M
        Vsch = f*(L/r**2 + 2*M/r**3)
        vs = sp.series(Vsch.subs(r, 2*M + h1), T, 0, 2).removeO()
        b2M = sp.simplify(sp.limit(Br, r, 2*M))
        Vh = sp.factor(sp.simplify(vs + T*b2M))
        print('   bracket at 2M =', sp.factor(b2M))
        print('   V(r_h) =', Vh, '+ O(T^2)')
        print('   equals -(3L - (2l+1) am) T/(256 M^4):',
              sp.simplify(Vh + (3*L - (2*l + 1)*am)*T/(256*M**4)) == 0)
    else:
        # the pole dominates at the horizon: V(r_h) = T*res/(r_h - 2M) + O(T)
        lead = sp.simplify(sp.limit(T*res/h1, T, 0))
        print('   V(r_h) = ', lead, '+ O(T)')
