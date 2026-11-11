#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""Rows of the table that replaces the published tones of Heidari et al. (2024): for each mode
(l, m, n) with n <= 1 and each Theta^2 of their tables, the published value, the exact tone of the
same equation (metric of Chaichian, Tureanu and Zet) and the exact tone of the correct metric, as
LaTeX rows on standard output."""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
pub = [r for r in csv.DictReader(l for l in open(DATA / "wkb" / "ctz_heidari_comparison.csv")
                                 if not l.startswith("#"))]
exact = list(csv.DictReader(open(DATA / "production" / "published_points.csv")))


def fmt(re, im, d):
    sign = "-" if im < 0 else "+"
    return f"${re:.{d}f}{sign}{abs(im):.{d}f}\\,\\ii$"


def as_printed(x):
    """A published value with the digits of its table: trailing zeros dropped, five decimals at least."""
    t = x.lstrip("-")
    whole, frac = t.split(".")
    frac = frac.rstrip("0")
    frac = frac + "0" * max(0, 5 - len(frac))
    return whole + "." + frac


def fmt_printed(re, im):
    sign = "-" if im.startswith("-") else "+"
    return f"${as_printed(re)}{sign}{as_printed(im)}\\,\\ii$"


def value(model, l, m, n, t2):
    for r in exact:
        if (r["model"], r["l"], r["m"], r["n"]) == (model, str(l), str(m), str(n)) and abs(float(r["Theta2"]) - t2) < 1e-9:
            assert float(r["digits"]) > 10, r
            return float(r["re"]), float(r["im"])
    raise KeyError((model, l, m, n, t2))


lines = []
for (l, m, n) in ((1, 1, 0), (1, 1, 1), (2, 1, 0), (2, 1, 1), (2, 2, 0), (2, 2, 1)):
    first = True
    for t2 in (0.2, 0.4, 0.6, 0.8, 1.0):
        p = [r for r in pub if (r["l"], r["m"], r["n"]) == (str(l), str(m), str(n)) and abs(float(r["Theta2"]) - t2) < 1e-9][0]
        label = f"$({l},{m},{n})$" if first else ""
        first = False
        lines.append(" & ".join([label, f"${t2:g}$", fmt_printed(p["pub_re"], p["pub_im"]),
                                 fmt(*value("ctz", l, m, n, t2), 8), fmt(*value("swc", l, m, n, t2), 8)]) + r"\\")
    lines.append(r"\hline")
print("\n".join(lines[:-1]))
