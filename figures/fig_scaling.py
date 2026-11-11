#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""Exterior complex scaling, l = m = 1, Theta = 0.5 (units M = 1), frequency plane.

For three pairs of scaling angles (horizon end, far end), the eigenvalues that discretise the two
rotated continua (the one issued from w = 0 and the one issued from the branch point
w = -i sqrt|V_h|) and the eigenvalues they uncover, which do not move with the angles.
(a) The correct metric; (b) the doubled metric, whose branch point lies much lower.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manuscript_style as hs  # noqa: E402

DATA = HERE.parent / "data" / "scaling"
OUT = HERE / "pdf"
OUT.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(DATA / "continua.csv")))
PAIRS = [("0.30", "0.30"), ("0.60", "0.60"), ("1.20", "1.20")]

hs.configure(family="latin-modern")
fig, axes = plt.subplots(1, 2, figsize=(hs.WIDTH, 2.9))
for ax, model, title in ((axes[0], "swc", "(a)"), (axes[1], "jkp_rth", "(b)")):
    sel = [r for r in rows if r["model"] == model and r["l"] == "1" and r["m"] == "1" and r["Theta"] == "0.5000"]
    vh = float(sel[0]["Vh"])
    for k, (tm, tp) in enumerate(PAIRS):
        style = hs.series(k)
        for cls, marker, size in (("far", "o", 2.0), ("horizon", "s", 2.0)):
            pts = [complex(float(r["Re_w"]), float(r["Im_w"])) for r in sel
                   if r["thm"] == tm and r["thp"] == tp and r["class"] == cls]
            pts = np.array(pts)
            ax.plot(pts.real, pts.imag, linestyle="none", marker=marker, markersize=size,
                    markerfacecolor="white", markeredgecolor=style["color"], markeredgewidth=0.5)
    q = np.array([complex(float(r["Re_w"]), float(r["Im_w"])) for r in sel if r["class"] == "qnm"])
    ax.plot(q.real, q.imag, linestyle="none", marker="D", markersize=5.0, markerfacecolor="white",
            markeredgecolor=hs.INK, markeredgewidth=1.0, label="uncovered tones")
    nu = np.sqrt(-vh)
    ax.plot([0], [-nu], linestyle="none", marker="x", markersize=5, color=hs.WINE, markeredgewidth=1.2,
            label=r"branch point $-\mathrm{i}\sqrt{|V_{h}|}$")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.45, 0.02)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$M\operatorname{Re}\omega$")
    ax.set_ylabel(r"$M\operatorname{Im}\omega$")
    hs.panel(ax, title)

from matplotlib.lines import Line2D  # noqa: E402

handles = [Line2D([], [], color=hs.series(k)["color"], linewidth=1.6) for k in range(len(PAIRS))]
labels = [rf"$\vartheta_{{\pm}}={float(tm):.1f}$" for tm, _ in PAIRS]
handles += [Line2D([], [], linestyle="none", marker="o", markersize=3.5, markerfacecolor="white",
                   markeredgecolor=hs.INK, markeredgewidth=0.7),
            Line2D([], [], linestyle="none", marker="s", markersize=3.5, markerfacecolor="white",
                   markeredgecolor=hs.INK, markeredgewidth=0.7)]
labels += [r"continuum from $\omega=0$", "continuum from the branch point"]
h, lab = axes[0].get_legend_handles_labels()
handles += h
labels += lab
order = [0, 3, 6, 1, 4, 2, 5]   # rows: the three angles / the two continua and the tones / the branch point
hs.legend_outside(fig, "upper", ncol=3, handles=[handles[i] for i in order], labels=[labels[i] for i in order])
hs.save(fig, OUT / "complex_scaling.pdf")
