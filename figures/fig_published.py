#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""Fundamental tones against Theta^2 (units M = 1): the published values against the exact ones.

Top row l = m = 1, bottom row l = 2 with m = 1 and m = 2; left the real part, right the imaginary
part.  Lines: exact tones of the metric of Chaichian, Tureanu and Zet and of the correct metric.
Markers: the values published by Heidari et al. (2024) for the first metric, and third-order WKB
applied to its potential.
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

curves = list(csv.DictReader(open(DATA / "figures" / "fundamentals_vs_theta2.csv")))
pub = list(csv.DictReader(l for l in open(DATA / "wkb" / "ctz_heidari_comparison.csv") if not l.startswith("#")))


def curve(model, l, m):
    rows = [r for r in curves if r["model"] == model and r["l"] == str(l) and r["m"] == str(m)]
    return (np.array([float(r["Theta2"]) for r in rows]),
            np.array([complex(float(r["re"]), float(r["im"])) for r in rows]))


def published(l, m, key):
    rows = [r for r in pub if r["l"] == str(l) and r["m"] == str(m) and r["n"] == "0"]
    return (np.array([float(r["Theta2"]) for r in rows]),
            np.array([complex(float(r[key + "_re"]), float(r[key + "_im"])) for r in rows]))


hs.configure(family="latin-modern")
fig, axes = plt.subplots(2, 2, figsize=(hs.WIDTH, 4.6))
cases = [((1, 1),), ((2, 1), (2, 2))]
labels_done = set()
for row, group in enumerate(cases):
    for col, part in enumerate((np.real, np.imag)):
        ax = axes[row, col]
        for j, (l, m) in enumerate(group):
            dash = hs.DASHES[0] if j == len(group) - 1 else hs.DASHES[1]
            x, z = curve("ctz", l, m)
            ax.plot(x, part(z), color=hs.WINE, linestyle=dash, linewidth=1.1,
                    label=f"exact, Chaichian--Tureanu--Zet, $m={m}$")
            x, z = curve("swc", l, m)
            ax.plot(x, part(z), color=hs.NAVY, linestyle=dash, linewidth=1.1,
                    label=f"exact, correct metric, $m={m}$")
            x, z = published(l, m, "pub")
            ax.plot(x, part(z), linestyle="none", marker="s", markersize=4.2,
                    markerfacecolor="white", markeredgecolor=hs.WINE, markeredgewidth=0.9,
                    label=f"published, $m={m}$")
            x, z = published(l, m, "wkb3")
            ax.plot(x, part(z), linestyle="none", marker="+", markersize=4.5, color=hs.OCHRE,
                    markeredgewidth=0.9, label=f"third-order WKB, $m={m}$")
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel(r"$\Theta^{2}/M^{2}$")
        ax.set_ylabel(r"$M\operatorname{Re}\omega$" if col == 0 else r"$M\operatorname{Im}\omega$")
        hs.panel(ax, "(" + "abcd"[2 * row + col] + rf") $\ell={group[0][0]}$")

from matplotlib.lines import Line2D  # noqa: E402

handles = [Line2D([], [], color=hs.WINE, linewidth=1.1), Line2D([], [], color=hs.NAVY, linewidth=1.1),
           Line2D([], [], color=hs.SLATE, linewidth=1.1, linestyle=hs.DASHES[1]),
           Line2D([], [], linestyle="none", marker="s", markersize=4.2, markerfacecolor="white",
                  markeredgecolor=hs.WINE, markeredgewidth=0.9),
           Line2D([], [], linestyle="none", marker="+", markersize=4.5, color=hs.OCHRE, markeredgewidth=0.9),
           Line2D([], [], color=hs.SLATE, linewidth=1.1)]
labels = ["exact, Chaichian--Tureanu--Zet metric", "exact, correct metric", r"$m=1$ (dashed), $m=\ell$ (solid)",
          "published values", "third-order WKB", None]
handles, labels = handles[:5], labels[:5]
order = [0, 3, 1, 4, 2]
hs.legend_outside(fig, "upper", ncol=3, handles=[handles[i] for i in order], labels=[labels[i] for i in order])
hs.save(fig, OUT / "published_tones.pdf")
