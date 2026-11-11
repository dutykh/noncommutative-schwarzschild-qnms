#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""The methods compared (units M = 1).

(a) Spectral convergence of the collocation for the correct metric, l = m = 1, Theta = 0.5:
error of the fundamental and of the first overtone against the zeros of the Jost function, for
the quartic pencil in double precision and for Newton's method in 256-bit arithmetic.
(b) Error of every method on five benchmark tones against the zeros of the Jost function.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manuscript_style as hs  # noqa: E402

DATA = HERE.parent / "data"
OUT = HERE / "pdf"
OUT.mkdir(parents=True, exist_ok=True)

conv = list(csv.DictReader(open(DATA / "figures" / "convergence_swc_l1m1_Th0p5.csv")))
comp = list(csv.DictReader(open(DATA / "comparison" / "methods.csv")))

hs.configure(family="latin-modern")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(hs.WIDTH, 3.0), width_ratios=[1.0, 1.25])

for n, color in ((0, hs.NAVY), (1, hs.WINE)):
    rows = [r for r in conv if int(r["n"]) == n]
    N = np.array([int(r["N"]) for r in rows])
    ed = np.array([float(r["err_double_qz"]) for r in rows])
    eb = np.array([float(r["err_newton_256bit"]) for r in rows])
    ax1.plot(N, eb, color=color, linestyle="-", marker="o", markersize=3.2, markerfacecolor="white",
             markeredgecolor=color, label=rf"$n={n}$, 256 bits")
    ax1.plot(N, ed, color=color, linestyle=hs.DASHES[1], marker="s", markersize=3.0,
             markerfacecolor="white", markeredgecolor=color, label=rf"$n={n}$, double")
ax1.set_yscale("log")
ax1.set_ylim(1e-23, 1e0)
ax1.set_xlim(12, 104)
ax1.set_xlabel(r"degree $N$ of the Chebyshev expansion")
ax1.set_ylabel(r"$M|\omega-\omega_{\mathrm{Jost}}|$")
hs.grid(ax1)
hs.panel(ax1, "(a)")
hs.legend_outside(ax1, "above", ncol=2)

methods = [
    ("newton_256bit_N100", "Newton, 256 bits"),
    ("beyn_double_N80", "contour integral"),
    ("nleigs_double_N80", "NLEIGS, first"),
    ("nleigs_second_realisation_double_N100", "NLEIGS, second"),
    ("joukowski_qz_double_N80", "Joukowski pencil"),
    ("complex_scaling", "complex scaling"),
    ("time_domain", "time evolution"),
    ("wkb_pade66", "WKB 12, Pad\\'e"),
    ("wkb_pade33", "WKB 6, Pad\\'e"),
    ("wkb_wkb6", "WKB 6"),
    ("wkb_wkb3", "WKB 3"),
]
tones = [("swc", "1", "1", "0"), ("swc", "1", "1", "1"), ("swc", "2", "1", "0"), ("swc", "2", "2", "0"),
         ("ctz", "1", "1", "0")]
names = [r"$(1,1)$, $n=0$", r"$(1,1)$, $n=1$", r"$(2,1)$, $n=0$", r"$(2,2)$, $n=0$",
         r"$(1,1)$, $n=0$, CTZ"]
y = np.arange(len(methods))[::-1]
offset = np.linspace(-0.26, 0.26, len(tones))
for k, (tone, label) in enumerate(zip(tones, names)):
    xs, ys = [], []
    for j, (key, _) in enumerate(methods):
        for r in comp:
            if (r["model"], r["l"], r["m"], r["n"]) == tone and r["method"] == key:
                xs.append(max(float(r["abs_err"]), 1e-30))
                ys.append(y[j] + offset[k])
    style = hs.series(k)
    ax2.plot(xs, ys, linestyle="none", marker=style["marker"], markersize=3.6, markerfacecolor="white",
             markeredgecolor=style["color"], markeredgewidth=0.9, label=label)
ax2.set_xscale("log")
ax2.set_xlim(1e-29, 1e-1)
ax2.set_yticks(y)
ax2.set_yticklabels([m[1] for m in methods])
ax2.set_ylim(-0.7, len(methods) - 0.3)
ax2.tick_params(axis="y", length=0)
for yy in y:
    ax2.axhline(yy, color=hs.MIST, linewidth=0.4, zorder=0)
ax2.set_xlabel(r"$M|\omega-\omega_{\mathrm{Jost}}|$")
hs.panel(ax2, "(b)")
hs.legend_outside(ax2, "above", ncol=3, columnspacing=0.8, handlelength=1.0)

hs.save(fig, OUT / "methods_comparison.pdf")
