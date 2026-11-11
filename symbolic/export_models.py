#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""Exact tortoise factors and potentials of the deformed Schwarzschild models, written as Julia code.

Units M = 1 (alpha = 2), u = 1/r.  A diagonal metric deformed at second order,
    g_tt = -f (1 + T A0),  g_rr = (1 + T B0)/f,  g_thth = r^2 (1 + T C0),
    g_phph = r^2 sin^2(theta) (1 + T D0) + T kax cos^2(theta),        T = Theta^2,
gives through the master equation of Chen, Chiang and Tsao (checked independently in
master_equation.py)
    dr/dr* = F = f (1 + T (A0 - B0)/2),
    V = f (l(l+1) u^2 + 2 u^3) + T [ W00 + a0 Wa + c0 Wc + kax16 (s1 Ws1 + s2 Ws2) ],
with a0 = |m|(2l+1)/2, c0 = a0 - l(l+1), s1 = (2l+1)(l(l+1) - m^2)/(64|m|),
s2 = (2l+1-2|m|)/(32|m|) and kax16 = 16 kax (the sums over j >= 1 of the axis term exist only for
m != 0).  Each W is a rational function of u; numerators and denominators are exported with exact
rational coefficients.  Models:
  ctz        Chaichian, Tureanu, Zet, Phys. Lett. B 660 (2008) 573, eq. (4.9)   [(r,theta) twist]
  jkp_rth    Juric, Kumara, Pozar, Nucl. Phys. B 1017 (2025) 116950, eq. (115)  [(r,theta) twist]
  swc        second-order Seiberg-Witten map with prefactor 1/32, contraction term and Moyal
             product (this work; differs from both papers above)          [(r,theta) twist]
  swc_tth    the same construction for the (t,theta) twist (this work)          [(t,theta) twist]
  jkp_tth    Juric, Kumara, Pozar, eq. (91)                                     [(t,theta) twist]
"""
import pathlib
import sympy as sp

r, u, M = sp.symbols('r u M', positive=True)
f = 1 - 2*M/r


def deformations(model):
    if model == 'ctz':
        al = 2*M
        A0 = al*(8*r - 11*al)/(16*r**3*(r - al))
        B0 = -al*(4*r - 3*al)/(16*r**3*(r - al))
        C0 = (2*r**2 - 17*al*(r - al))/(32*r**3*(r - al))
        D0 = -al*(2*r - al)/(16*r**3*(r - al))
        kax = sp.Rational(1, 16)
    elif model == 'jkp_rth':
        A0 = M*(4*r - 11*M)/(2*r**3*(r - 2*M))
        B0 = M*(3*M - 2*r)/(2*r**3*(r - 2*M))
        C0 = (r**2 - 32*M*r + 64*M**2)/(16*r**3*(r - 2*M))
        D0 = (2*M**2 - 4*M*r + r**2)/(4*r**3*(r - 2*M))
        kax = sp.Rational(5, 16)
    elif model == 'swc':
        # Seiberg-Witten map at second order with the prefactor 1/32 (Chamseddine; confirmed by the
        # exact constant-field solution and by gauge equivalence), the contraction term of e^2_theta
        # (Juric, Kumara, Pozar) and the Moyal product in the metric: g_tt, g_rr as in Chaichian et al.,
        # g_thth = r^2 + (1/16 - M/r + M/(8(r - 2M))) T, g_phph = r^2 sin^2 + (5cos^2/16
        # + (r^2 - 3Mr + M^2) sin^2/(4r(r - 2M))) T.
        al = 2*M
        A0 = al*(8*r - 11*al)/(16*r**3*(r - al))
        B0 = -al*(4*r - 3*al)/(16*r**3*(r - al))
        C0 = (r**2 - 16*M*r + 32*M**2)/(16*r**3*(r - 2*M))
        D0 = (r**2 - 3*M*r + M**2)/(4*r**3*(r - 2*M))
        kax = sp.Rational(5, 16)
    elif model == 'swc_tth':
        # (t,theta) twist, Seiberg-Witten map at second order with the prefactor 1/32, the full
        # contraction and the Moyal product (which adds nothing here: the fields do not depend on t);
        # symbolic/seiberg_witten/check_d_t_theta.py.  Bounded at r = 2M, A0 = B0 there.
        A0 = -M**2*(r - 3*M)/(8*r**5)
        B0 = -M*(2*r**2 - 10*M*r + 11*M**2)/(8*r**5)
        C0 = M*(r - 2*M)*(r - 3*M)/(8*r**5)
        D0 = M*(r - 2*M)*(r + M)/(8*r**5)
        kax = sp.Integer(0)
    elif model == 'jkp_tth':
        # g_tt = -(1 - 2M/r) + M^2 (r - 2M)(9r - 30M) T/(48 r^6), etc. (eq. (91))
        A0 = -M**2*(9*r - 30*M)/(48*r**5)
        B0 = -M*(11*M**2 - 10*M*r + 2*r**2)/(4*r**5)
        C0 = M*(11*M - 4*r)*(2*M - r)/(16*r**5)
        D0 = M*(r - 2*M)*(M + r)/(4*r**5)
        kax = sp.Integer(0)
    else:
        raise ValueError(model)
    return A0, B0, C0, D0, kax


def check_metric(model):
    """Recover the published metric components from the deformation functions."""
    A0, B0, C0, D0, kax = deformations(model)
    T, th = sp.symbols('T theta')
    g = {'tt': -f*(1 + T*A0), 'rr': (1 + T*B0)/f, 'thth': r**2*(1 + T*C0),
         'phph': r**2*sp.sin(th)**2*(1 + T*D0) + T*kax*sp.cos(th)**2}
    if model == 'swc_tth':
        A = sp.sqrt(1 - 2*M/r); A1 = sp.diff(A, r); A2 = sp.diff(A, r, 2)
        pub = {'tt': -A**2 + A**5*A1**2*(A - r*A1)*T/8,
               'rr': 1/A**2 + A*(A**2*A2 - A*A1**2 - 2*r*A*A1*A2 - r*A1**3)*T/8,
               'thth': r**2 + r*A**4*A1*(A - r*A1)*T/8,
               'phph': r**2*sp.sin(th)**2 + r*A**3*A1*(2 - A**2 - r*A*A1)*sp.sin(th)**2*T/8}
    elif model == 'swc':
        pub = {'tt': -(1 - 2*M/r) - M*(4*r - 11*M)*T/(4*r**4),
               'rr': 1/(1 - 2*M/r) - M*(2*r - 3*M)*T/(4*r**2*(r - 2*M)**2),
               'thth': r**2 + (sp.Rational(1, 16) - M/r + M/(8*(r - 2*M)))*T,
               'phph': r**2*sp.sin(th)**2 + (5*sp.cos(th)**2/16
                                              + (r**2 - 3*M*r + M**2)*sp.sin(th)**2/(4*r*(r - 2*M)))*T}
    elif model == 'jkp_tth':
        pub = {'tt': -(1 - 2*M/r) + M**2*(r - 2*M)*(9*r - 30*M)*T/(48*r**6),
               'rr': 1/(1 - 2*M/r) - M*(11*M**2 - 10*M*r + 2*r**2)*T/(4*r**4*(r - 2*M)),
               'thth': r**2 + M*(11*M - 4*r)*(2*M - r)*T/(16*r**3),
               'phph': r**2*sp.sin(th)**2 + M*(r - 2*M)*(M + r)*sp.sin(th)**2*T/(4*r**3)}
    elif model == 'jkp_rth':
        pub = {'tt': -(1 - 2*M/r) + M*(11*M - 4*r)*T/(2*r**4),
               'rr': 1/(1 - 2*M/r) + M*(3*M - 2*r)*T/(2*r**2*(r - 2*M)**2),
               'thth': r**2 + (sp.Rational(1, 16) - 2*M/r + M/(8*(r - 2*M)))*T,
               'phph': r**2*sp.sin(th)**2 + (5*sp.cos(th)**2/16
                                              + sp.sin(th)**2*(2*M**2 - 4*M*r + r**2)/(4*r*(r - 2*M)))*T}
    else:
        al = 2*M
        pub = {'tt': -(1 - al/r) - al*(8*r - 11*al)*T/(16*r**4),
               'rr': 1/(1 - al/r) - al*(4*r - 3*al)*T/(16*r**2*(r - al)**2),
               'thth': r**2 + (2*r**2 - 17*al*r + 17*al**2)*T/(32*r*(r - al)),
               'phph': r**2*sp.sin(th)**2 + ((r**2 + al*r - al**2)*sp.cos(th)**2 - al*(2*r - al))
                       * T/(16*r*(r - al))}
    return {k: sp.simplify(g[k] - pub[k]) for k in g}


def pieces(model):
    """F and the W pieces as rational functions of u (M = 1)."""
    A0, B0, C0, D0, kax = deformations(model)
    dd = lambda g: f*sp.diff(g, r)
    F2 = f*(A0 - B0)/2
    W00 = ((f/r)*sp.diff(f, r)*(A0 - B0) + 1/(4*r**2)*dd(r**2*dd(A0 - B0 + C0 + D0))
           - dd(dd(A0 - B0))/4)
    Wa = (f/r**2)*(A0 - D0)
    Wc = -(f/r**2)*(A0 - C0)
    Ws1 = -(f/r**2)/r**2                                   # times s1 (sum of (a^j + d^j/2) r^2 D_j / kax16)
    Ws2 = 1/(4*r**2)*dd(r**2*dd(1/r**2))                    # times s2
    out = {}
    for name, e in (('F2', F2), ('W00', W00), ('Wa', Wa), ('Wc', Wc), ('Ws1', Ws1), ('Ws2', Ws2)):
        e = sp.cancel(sp.together(e.subs(M, 1).subs(r, 1/u)))
        num, den = sp.fraction(sp.factor(e))
        num = sp.Poly(sp.expand(num), u)
        den = sp.Poly(sp.expand(den), u)
        lc = den.LC()
        out[name] = ([num.coeff_monomial(u**k)/lc for k in range(num.degree() + 1)],
                     [den.coeff_monomial(u**k)/lc for k in range(den.degree() + 1)])
    return out, kax


def julia_rational(c):
    c = sp.Rational(c)
    return f'{c.p}//{c.q}'


if __name__ == '__main__':
    lines = ['# Generated by symbolic/export_models.py: exact pieces of the models, units M = 1,',
             '# u = 1/r.  Each entry is (numerator coefficients, denominator coefficients) in powers of u.',
             'const MODEL_PIECES = Dict{String,Any}(']
    for model in ('ctz', 'swc', 'swc_tth', 'jkp_rth', 'jkp_tth'):
        print(model, 'metric check:', check_metric(model))
        out, kax = pieces(model)
        for k, (n, d) in out.items():
            print(f'   {k}: numerator degree {len(n) - 1}, denominator {d}')
        entries = [f'    "{model}" => Dict{{String,Any}}(', f'        "kax16" => {julia_rational(16*kax)},']
        for k, (n, d) in out.items():
            entries.append(f'        "{k}" => (Rational{{BigInt}}[{", ".join(julia_rational(x) for x in n)}],'
                           f' Rational{{BigInt}}[{", ".join(julia_rational(x) for x in d)}]),')
        entries.append('    ),')
        lines += entries
    lines.append(')')
    target = pathlib.Path(__file__).resolve().parents[1] / 'julia' / 'model_pieces.jl'
    target.write_text('\n'.join(lines) + '\n')
    print('written', target)
