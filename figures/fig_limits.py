#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""A change of the potential confined to the horizon layer (l = m = 1, units M = 1).

(a) The extra potential Delta V = c Theta^4 (2/r)^3/((r - 2) + Theta^2) of julia/scripts/layer_test.jl
in the tortoise coordinate of the correct metric, divided by its value at the horizon, for
Theta = 0.1, 0.02, 0.005, with the Schwarzschild dipole barrier scaled to its maximum.
(b) The change |dw| of the fundamental and of the first overtone caused by the term with c = 1/256,
from data/limits/layer_n*_Th*.csv at the degree N = 120, against the whole second-order shift
Theta^2 |omega_2|, and the law dw ~ Theta^(2 - 8 i wS) through the point of smallest Theta.
(c) The phase of dw, with the slope -8 Re wS of the same law.
"""
import csv
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manuscript_style as hs  # noqa: E402

DATA = HERE.parent / "data" / "limits"
OUT = HERE / "pdf"
OUT.mkdir(parents=True, exist_ok=True)

WS = {0: complex(0.2929361332672827, -0.0976599889135782), 1: complex(0.2644486506048139, -0.3062573915590669)}
W2 = {0: complex(-0.000281210569696518, -0.005648147701350282),
      1: complex(0.002442297338829383, -0.017456496713672230)}
C = 1 / 256


def F(s, T):
    return 1 - 2 / s + (3 * s - 7) * T**2 / (4 * s**4)


def horizon(T):
    r = 2 + T * T / 32
    for _ in range(60):
        dF = 2 / r**2 + T * T * (3 * r - 4 * (3 * r - 7)) / (4 * r**5)
        r -= F(r, T) / dF
    return r


def tortoise(r, T):
    """r* = r + 2 ln(r/2 - 1) - int_r^inf (1/F - 1/f) ds, by the trapezoid rule in log(s - r_h)."""
    rh = horizon(T)
    out = []
    for x in r:
        q = np.linspace(np.log(x - rh), np.log(1e5), 20001)
        s = rh + np.exp(q)
        g = (1 / F(s, T) - 1 / (1 - 2 / s)) * np.exp(q)
        out.append(x + 2 * np.log(x / 2 - 1) - np.sum((g[1:] + g[:-1]) * np.diff(q)) / 2)
    return np.array(out)


def shifts():
    """dw at the degree N = 120 for c = 1/256, by overtone and Theta."""
    d = {0: {}, 1: {}}
    for path in glob.glob(str(DATA / "layer_n*_Th*.csv")):
        tones = {}
        for row in csv.DictReader(open(path)):
            if int(row["N"]) == 120:
                tones[float(row["c"])] = complex(float(row["re"]), float(row["im"]))
        if 0.0 in tones and C in tones:
            d[int(row["n"])][float(row["Theta"])] = tones[C] - tones[0.0]
    return {n: sorted(d[n].items()) for n in d}


data = shifts()
hs.configure(family="latin-modern")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(hs.WIDTH, 2.35))

r = 2 + np.logspace(-8, np.log10(60), 800)
barrier = (1 - 2 / r) * (2 / r**2 + 2 / r**3)
ax1.plot(r + 2 * np.log(r / 2 - 1), barrier / barrier.max(), color=hs.SLATE, linestyle=hs.DASHES[2],
         linewidth=0.9, label="barrier")
for k, T in enumerate((0.1, 0.02, 0.005)):
    rh = horizon(T)
    rr = rh + np.logspace(np.log10(T * T * 1e-4), np.log10(60), 600)
    step = (2 / rr)**3 / ((rr - 2) + T * T) / ((2 / rh)**3 / ((rh - 2) + T * T))
    style = hs.series(k)
    style.pop("marker", None)
    ax1.plot(tortoise(rr, T), step, **style, label=rf"$\Theta={T:g}\,M$")
ax1.set_xlim(-32, 12)
ax1.set_ylim(-0.05, 1.12)
ax1.set_xlabel(r"$r_{*}/M$")
ax1.set_ylabel(r"$\Delta V/\Delta V(r_{h})$")
hs.panel(ax1, "(a)")
hs.legend_outside(ax1, "above", ncol=2, columnspacing=0.8, handlelength=1.6)

grid = np.logspace(np.log10(0.004), np.log10(0.25), 100)
for n in (0, 1):
    T = np.array([t for t, _ in data[n]])
    dw = np.array([w for _, w in data[n]])
    style = hs.series(n)
    ax2.plot(T, np.abs(dw), linestyle="none", marker=style["marker"], markerfacecolor="white",
             markeredgecolor=style["color"], markersize=4.2, label=rf"$|\delta\omega|$, $n={n}$")
    ax2.plot(grid, grid**2 * abs(W2[n]), color=style["color"], linestyle=hs.DASHES[0], linewidth=1.0,
             label=rf"$\Theta^{{2}}|\omega_{{2}}|$, $n={n}$")
    ax2.plot(grid, abs(dw[0]) * (grid / T[0])**(2 + 8 * WS[n].imag), color=style["color"],
             linestyle=hs.DASHES[1], linewidth=0.8)
    phase = np.unwrap(np.angle(dw))
    ax3.plot(T, phase, linestyle="none", marker=style["marker"], markerfacecolor="white",
             markeredgecolor=style["color"], markersize=4.2, label=rf"$n={n}$")
    slope = -8 * WS[n].real
    ax3.plot(grid, np.mean(phase - slope * np.log(T)) + slope * np.log(grid), color=style["color"],
             linestyle=hs.DASHES[1], linewidth=0.8)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(0.004, 0.25)
ax2.set_xlabel(r"$\Theta/M$")
ax2.set_ylabel(r"$M\,|\delta\omega|$, $M\Theta^{2}|\omega_{2}|$")
hs.panel(ax2, "(b)")
hs.legend_outside(ax2, "above", ncol=2, columnspacing=0.6, handlelength=1.4)
ax3.set_xscale("log")
ax3.set_xlim(0.004, 0.25)
ax3.set_xlabel(r"$\Theta/M$")
ax3.set_ylabel(r"$\arg\delta\omega$")
hs.panel(ax3, "(c)")
hs.legend_outside(ax3, "above", ncol=2)

hs.save(fig, OUT / "layer_test.pdf")
