#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""The quartic Joukowski pencil of the correct metric, l = m = 1, Theta = M, N = 80 (units M = 1).

(a) The eigenvalues Lambda in polar form, modulus relative to sqrt|V_h| on a logarithmic scale
against the argument: the circle |Lambda| = sqrt|V_h|, image of the cut, separates the physical
sheet (above) from the second sheet (below); the eigenvalues that discretise the cut lie on it,
and the tones lie above.  (b) The eigenvalues of the physical sheet in the frequency plane: the cut
between the branch points, its discretisation, and the tones, which converge as N grows
(distance to the eigenvalues at N = 60 below 1e-5).
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manuscript_style as hs  # noqa: E402

DATA = HERE.parent / "data" / "figures"
OUT = HERE / "pdf"
OUT.mkdir(parents=True, exist_ok=True)

lines = open(DATA / "joukowski_swc_l1m1_Th1.csv").read().splitlines()
vh = float(lines[0].split("V_h =")[1])
rows = [r for r in csv.reader(lines[1:])][1:]
lam = np.array([complex(float(r[0]), float(r[1])) for r in rows])
w = np.array([complex(float(r[2]), float(r[3])) for r in rows])
phys = np.array([r[4] == "1" for r in rows])
dist = np.array([float(r[5]) for r in rows])
nu = np.sqrt(-vh)
rho = np.abs(lam) / nu
oncut = np.abs(rho - 1) < 1e-3
tone = (dist < 1e-5) & phys & ~oncut
other = ~oncut & ~tone

hs.configure(family="latin-modern")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(hs.WIDTH, 2.75))

arg = np.angle(lam) / np.pi
ax1.axhline(1.0, color=hs.SLATE, linewidth=0.8, label=r"$|\Lambda|=\sqrt{|V_{h}|}$")
ax1.plot(arg[other], rho[other], linestyle="none", marker="o", markersize=2.2,
         markerfacecolor="white", markeredgecolor=hs.MIST, markeredgewidth=0.6, label="other eigenvalues")
ax1.plot(arg[oncut], rho[oncut], linestyle="none", marker="s", markersize=2.6,
         markerfacecolor="white", markeredgecolor=hs.TEAL, markeredgewidth=0.7, label="discretised cut")
ax1.plot(arg[tone], rho[tone], linestyle="none", marker="o", markersize=5,
         markerfacecolor="white", markeredgecolor=hs.NAVY, markeredgewidth=1.1, label="tones")
ax1.set_yscale("log")
ax1.set_xlim(-1, 1)
ax1.set_ylim(1e-6, 3e3)
ax1.text(0.98, 40, "physical sheet", color=hs.SLATE, ha="right", va="center")
ax1.text(0.98, 4e-6, "second sheet", color=hs.SLATE, ha="right", va="center")
ax1.set_xlabel(r"$\arg\Lambda/\pi$")
ax1.set_ylabel(r"$|\Lambda|/\sqrt{|V_{h}|}$")
hs.grid(ax1)
hs.panel(ax1, "(a)")

sel = phys
ax2.plot([0, 0], [-nu, nu], color=hs.WINE, linewidth=2.0, solid_capstyle="butt", label="cut")
ax2.plot(w[sel & other].real, w[sel & other].imag, linestyle="none", marker="o", markersize=2.2,
         markerfacecolor="white", markeredgecolor=hs.MIST, markeredgewidth=0.6)
ax2.plot(w[sel & oncut].real, w[sel & oncut].imag, linestyle="none", marker="s", markersize=2.4,
         markerfacecolor="white", markeredgecolor=hs.TEAL, markeredgewidth=0.6)
ax2.plot(w[tone].real, w[tone].imag, linestyle="none", marker="o", markersize=5,
         markerfacecolor="white", markeredgecolor=hs.NAVY, markeredgewidth=1.1)
ax2.set_xlim(-0.5, 0.5)
ax2.set_ylim(-0.6, 0.15)
ax2.set_aspect("equal")
ax2.set_xlabel(r"$M\operatorname{Re}\omega$")
ax2.set_ylabel(r"$M\operatorname{Im}\omega$")
hs.panel(ax2, "(b)")

handles, labels = [], []
for a in (ax1, ax2):
    h, lab = a.get_legend_handles_labels()
    handles += h
    labels += lab
hs.legend_outside(fig, "upper", ncol=5, handles=handles, labels=labels)
hs.save(fig, OUT / "joukowski_planes.pdf")
