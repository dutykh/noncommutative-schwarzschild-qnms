#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""The integrals of the associated Legendre functions used for the angular sums, in exact arithmetic.

For P = P_l^m and 1 <= m <= l:
    N  = int P^2 dx                = 2 (l+m)! / ((2l+1)(l-m)!),
    I1 = int P^2 / (1-x^2) dx      = (l+m)! / (m (l-m)!),
and for 2 <= m <= l:
    I2 = int P^2 / (1-x^2)^2 dx    = (l+m)!/(l-m)! (L + m^2 - 1) / (2m(m^2 - 1)),   L = l(l+1).
Each integrand is a polynomial, since P^2 = (1-x^2)^m Q^2, so the integrals are exact rationals.
Every pair with l <= 12 is checked.
"""
import sympy as sp

x = sp.symbols('x')
bad = 0
count = 0
for l in range(1, 13):
    for m in range(1, l + 1):
        P2 = sp.expand(sp.assoc_legendre(l, m, x)**2)
        fac = sp.factorial(l + m) / sp.factorial(l - m)
        L = l*(l + 1)
        N = sp.integrate(P2, (x, -1, 1))
        I1 = sp.integrate(sp.cancel(P2/(1 - x**2)), (x, -1, 1))
        ok = N == 2*fac/(2*l + 1) and I1 == fac/m
        if m >= 2:
            I2 = sp.integrate(sp.cancel(P2/(1 - x**2)**2), (x, -1, 1))
            ok = ok and I2 == fac*sp.Rational(L + m*m - 1, 2*m*(m*m - 1))
            count += 1
        bad += not ok
print(f'N and I1 checked for every 1 <= m <= l <= 12, I2 for the {count} pairs with 2 <= m <= l <= 12;',
      'all exact' if bad == 0 else f'{bad} failures')
