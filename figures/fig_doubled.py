#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""The spectrum of the doubled metric, l = m = 1 (units M = 1), from exterior complex scaling.

(a) The tones for Theta from 0.5 down to 0.01 (colour: log10 Theta) with the Schwarzschild tones:
a ladder of cavity tones slides towards the real axis and towards zero frequency as Theta
decreases.  (b) The real part and minus the imaginary part of the two tones that continue, by
continuity in Theta, the fundamental and the first overtone at Theta = 0.5, with the
Schwarzschild fundamental for comparison.
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
rows = [r for r in csv.DictReader(open(DATA / "jkp_rth_tracks.csv")) if r["l"] == "1" and r["m"] == "1"]
th = np.array([float(r["Theta"]) for r in rows])
w = np.array([complex(float(r["Re_w"]), float(r["Im_w"])) for r in rows])
fund = np.array([r["fundamental"] == "1" for r in rows])
least = np.array([r["least_damped"] == "1" for r in rows])
SCHW = np.array([0.292936133267283 - 0.097659988913578j, 0.264448650604833 - 0.306257391559047j,
                 0.229539334931302 - 0.540133425019107j])

hs.configure(family="latin-modern")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(hs.WIDTH, 2.9), width_ratios=[1.15, 1.0])

order = np.argsort(-th)
image = ax1.scatter(w.real[order], w.imag[order], c=np.log10(th[order]), cmap=hs.SECOND_COLORMAP,
                    vmin=-2, vmax=np.log10(0.5), s=7, marker="o", linewidths=0.4, edgecolors="none")
ax1.plot(SCHW.real, SCHW.imag, linestyle="none", marker="D", markersize=5.0, markerfacecolor="white",
         markeredgecolor=hs.NAVY, markeredgewidth=1.0)
ax1.set_xlim(0, 0.8)
ax1.set_ylim(-0.6, 0.0)
ax1.set_xlabel(r"$M\operatorname{Re}\omega$")
ax1.set_ylabel(r"$M\operatorname{Im}\omega$")
hs.panel(ax1, "(a)")
bar = hs.colorbar(fig, image, ax1, r"$\log_{10}(\Theta/M)$")

thetas = np.unique(th)[::-1]


def branch(start):
    """Follow a tone from Theta = 0.5 downwards, taking at each step the nearest tone."""
    z, out = start, []
    for t in thetas:
        cand = w[th == t]
        z = cand[np.argmin(np.abs(cand - z))]
        out.append(z)
    return np.array(out)


at05 = w[th == thetas[0]]
starts = [at05[np.argmax(at05.imag)], at05[np.argsort(-at05.imag)[1]]]
for k, z0 in enumerate(starts):
    b = branch(z0)
    color = (hs.WINE, hs.TEAL)[k]
    ax2.plot(thetas, b.real, color=color, linestyle=hs.DASHES[0], linewidth=1.1)
    ax2.plot(thetas, -b.imag, color=color, linestyle=hs.DASHES[1], linewidth=1.1)
ax2.axhline(SCHW[0].real, color=hs.NAVY, linewidth=0.8, linestyle=hs.DASHES[2])
ax2.axhline(-SCHW[0].imag, color=hs.NAVY, linewidth=0.8, linestyle=hs.DASHES[2])
ax2.set_xscale("log")
ax2.set_xlim(0.01, 0.5)
ax2.set_ylim(0.0, 0.35)
ax2.set_xlabel(r"$\Theta/M$")
ax2.set_ylabel(r"$M\omega$")
hs.panel(ax2, "(b)")
from matplotlib.lines import Line2D  # noqa: E402

handles = [Line2D([], [], linestyle="none", marker="D", markersize=5.0, markerfacecolor="white",
                  markeredgecolor=hs.NAVY, markeredgewidth=1.0),
           Line2D([], [], color=hs.WINE, linewidth=1.1), Line2D([], [], color=hs.TEAL, linewidth=1.1),
           Line2D([], [], color=hs.SLATE, linewidth=1.1), Line2D([], [], color=hs.SLATE, linewidth=1.1,
                                                               linestyle=hs.DASHES[1]),
           Line2D([], [], color=hs.NAVY, linewidth=0.8, linestyle=hs.DASHES[2])]
labels = ["Schwarzschild tones", r"tone continuing the fundamental at $\Theta=0.5M$",
          "tone continuing the first overtone", r"$M\operatorname{Re}\omega$", r"$-M\operatorname{Im}\omega$",
          "Schwarzschild fundamental"]
order = [0, 3, 1, 4, 2, 5]
hs.legend_outside(fig, "upper", ncol=3, handles=[handles[i] for i in order], labels=[labels[i] for i in order])
hs.save(fig, OUT / "doubled_spectrum.pdf")
