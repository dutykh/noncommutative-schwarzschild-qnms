#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""Ringdown and growth in the time domain, correct metric, l = m = 1 (units M = 1).

(a) |Psi| at r* = 10 for Schwarzschild and for Theta = 0.5 and 1, with the late-time law
A exp(mu t) t^(-3/2), mu = sqrt|V_h|, fitted for A only.  (b) The field for Theta = 0.5, each time
slice divided by its maximum, log10(|Psi(t, r*)| / max |Psi(t, .)|): the pulse and its ringdown
leave along the light cone, and the growing part spreads over the plateau near the horizon like
a solution of the heat equation, with its maximum near r* = -sqrt(t/mu), the dashed curve.
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


def series(name, col="psi_10"):
    rows = list(csv.DictReader(l for l in open(DATA / name) if not l.startswith("#")))
    t = np.array([float(r["t"]) for r in rows])
    y = np.array([float(r[col]) for r in rows])
    ok = np.isfinite(y)
    return t[ok], y[ok]


MU = {0.5: 0.05307721, 1.0: 0.10088112}

hs.configure(family="latin-modern")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(hs.WIDTH, 2.8), width_ratios=[1.0, 1.15])

t, y = series("series_schwarzschild_l1.csv")
ax1.plot(t, np.abs(y), color=hs.SLATE, linestyle=hs.DASHES[2], linewidth=1.0, label="Schwarzschild")
for k, th in enumerate((0.5, 1.0)):
    t, y = series(f"series_swc_l1m1_Th{th:.2f}.csv")
    color = (hs.NAVY, hs.WINE)[k]
    ax1.plot(t, np.abs(y), color=color, linewidth=1.0, label=rf"$\Theta={th:g}\,M$")
    mu = MU[th]
    late = t > 0.5 * t[-1]
    loga = np.mean(np.log(np.abs(y[late])) - mu * t[late] + 1.5 * np.log(t[late]))
    tt = np.linspace(60, 600, 200)
    ax1.plot(tt, np.exp(loga + mu * tt - 1.5 * np.log(tt)), color=color, linestyle=hs.DASHES[1],
             linewidth=0.9)
ax1.set_yscale("log")
ax1.set_xlim(0, 600)
ax1.set_ylim(1e-16, 1e8)
ax1.set_xlabel(r"$t/M$")
ax1.set_ylabel(r"$|\Psi(t,r_{*}=10M)|$")
hs.grid(ax1)
hs.panel(ax1, "(a)")
hs.legend_outside(ax1, "above", ncol=3)

rows = np.loadtxt(DATA / "spacetime_swc_l1m1_Th0.50.csv", delimiter=",", skiprows=2)
ts = np.unique(rows[:, 0])
xs = np.unique(rows[:, 1])
P = rows[:, 2].reshape(len(ts), len(xs))
A = np.abs(P)
L = np.log10(np.maximum(A / np.maximum(A.max(axis=1, keepdims=True), 1e-300), 1e-300))
image = ax2.pcolormesh(xs, ts, L, shading="nearest", cmap=hs.DENSITY_COLORMAP, vmin=-8, vmax=0,
                       rasterized=True)
tc = np.linspace(40, 1500, 200)
ax2.plot(-np.sqrt(tc / MU[0.5]), tc, color=hs.PAPER, linestyle=hs.DASHES[1], linewidth=1.0)
ax2.set_xlim(-600, 100)
ax2.set_ylim(0, 1500)
ax2.set_xlabel(r"$r_{*}/M$")
ax2.set_ylabel(r"$t/M$")
hs.panel(ax2, "(b)")
hs.colorbar(fig, image, ax2, r"$\log_{10}\left(|\Psi|/\max_{r_{*}}|\Psi|\right)$")

hs.save(fig, OUT / "growth.pdf")
