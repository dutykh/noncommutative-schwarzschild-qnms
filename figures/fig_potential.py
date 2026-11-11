#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""The effective potential near the horizon (l = m = 1, units M = 1).

(a) Schwarzschild, the correct second-order metric and the doubled metric at Theta = 0.5, against
the tortoise coordinate: the correct potential tends to a small negative value V_h, the doubled one
to the plateau -1/32.  (b) The doubled metric for Theta = 0.5, 0.25, 0.1, 0.05 against the distance
xi to the mid-point of its step, with the logistic limit -1/(32 (1 + exp(xi/2))).
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manuscript_style as hs  # noqa: E402

DATA = HERE.parent / "data" / "timedomain"
OUT = HERE / "pdf"
OUT.mkdir(parents=True, exist_ok=True)


def table(name, column):
    rows = [r for r in csv.reader(l for l in open(DATA / name) if not l.startswith("#"))]
    head = rows[0]
    i, j = head.index("rstar"), head.index(column)
    data = np.array([[float(r[i]), float(r[j])] for r in rows[1:]])
    return data[:, 0], data[:, 1]


def step_midpoints():
    rows = [r for r in csv.reader(l for l in open(DATA / "jkp_rth_step.csv") if not l.startswith("#"))]
    return {float(r[0]): float(r[4]) for r in rows[1:] if r[1] == "1" and r[2] == "1"}


hs.configure(family="latin-modern")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(hs.WIDTH, 2.55))

x, v = table("tortoise_schwarzschild.csv", "V_l1m0")
ax1.plot(x, v, color=hs.SLATE, linestyle=hs.DASHES[2], label="Schwarzschild")
x, v = table("tortoise_swc_Th0.50.csv", "V_l1m1")
ax1.plot(x, v, color=hs.NAVY, linestyle=hs.DASHES[0], label="correct metric")
x, v = table("tortoise_jkp_rth_Th0.50.csv", "V_l1m1")
ax1.plot(x, v, color=hs.WINE, linestyle=hs.DASHES[1], label="doubled metric")
ax1.axhline(0.0, color=hs.MIST, linewidth=0.6, zorder=0)
ax1.set_xlim(-40, 30)
ax1.set_ylim(-0.045, 0.115)
ax1.set_xlabel(r"$r_{*}/M$")
ax1.set_ylabel(r"$M^{2}V_{\mathrm{eff}}$")
ax1.annotate(r"$M^{2}V_{h}=-0.0028$", xy=(-28, -0.0035), xytext=(-37, 0.035), color=hs.SLATE,
             arrowprops=dict(arrowstyle="-", color=hs.SLATE, linewidth=0.6, shrinkA=1, shrinkB=1))
ax1.annotate(r"$M^{2}V_{h}=-0.0376$", xy=(-33, -0.0368), xytext=(-39, -0.026), color=hs.SLATE,
             arrowprops=dict(arrowstyle="-", color=hs.SLATE, linewidth=0.6, shrinkA=1, shrinkB=1))
hs.panel(ax1, "(a)")
hs.legend_outside(ax1, "above", ncol=2)

mid = step_midpoints()
xi = np.linspace(-20, 15, 400)
ax2.plot(xi, -1 / (32 * (1 + np.exp(xi / 2))), color=hs.INK, linewidth=0.9, label="logistic limit")
for k, th in enumerate((0.5, 0.25, 0.1, 0.05)):
    x, v = table(f"tortoise_jkp_rth_Th{th:.2f}.csv", "V_l1m1")
    style = hs.series(k)
    style.pop("marker")
    ax2.plot(x - mid[th], v, **style, label=rf"$\Theta={th:g}\,M$")
ax2.set_xlim(-20, 15)
ax2.set_ylim(-0.045, 0.115)
ax2.set_xlabel(r"$\xi/M=(r_{*}-r_{*}^{\mathrm{step}})/M$")
ax2.set_ylabel(r"$M^{2}V_{\mathrm{eff}}$")
hs.panel(ax2, "(b)")
h, lab = ax2.get_legend_handles_labels()
order = [0, 3, 1, 4, 2]        # rows: limit, 0.5, 0.25 / 0.1, 0.05 (columns are filled first)
hs.legend_outside(ax2, "above", ncol=3, handles=[h[i] for i in order], labels=[lab[i] for i in order],
                  columnspacing=0.9, handlelength=1.8)

hs.save(fig, OUT / "potential_step.pdf")
