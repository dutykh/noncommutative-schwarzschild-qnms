#!/usr/bin/env python3
# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
"""The table of first-order coefficients omega_2, in units of 10^-3 / M^3, for the correct metric,
the metric of Chaichian, Tureanu and Zet, and the corrected twist [t, theta] = i Theta (which
alone admits m = 0), printed as a LaTeX table."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE.parent / "data" / "production" / "first_order.csv")))


def get(model, l, m, n):
    for r in rows:
        if (r["model"], r["l"], r["m"], r["n"]) == (model, str(l), str(m), str(n)):
            return complex(float(r["re_omega2"]), float(r["im_omega2"])) * 1e3
    return None


def fmt(z):
    if z is None:
        return ""
    sign = "-" if z.imag < 0 else "+"
    return f"${z.real:.4f}{sign}{abs(z.imag):.4f}\\,\\ii$"


def schw(l, n):
    for r in rows:
        if r["l"] == str(l) and r["n"] == str(n):
            z = complex(float(r["re_omegaS"]), float(r["im_omegaS"]))
            return f"${z.real:.6f}-{abs(z.imag):.6f}\\,\\ii$"


body = []
for l in (1, 2, 3):
    for m in range(0, l + 1):
        for n in (0, 1):
            label = f"$({l},{m})$" if n == 0 else ""
            body.append(" & ".join([label, f"${n}$", schw(l, n), fmt(get("swc", l, m, n)),
                                    fmt(get("ctz", l, m, n)), fmt(get("swc_tth", l, m, n))]) + r"\\")
    if l < 3:
        body.append(r"\hline")

table = r"""\begin{table*}
\caption{\label{tab:firstorder}First-order coefficients $\omega_{2}$ of $\omega(\Theta)=\omega_{\mathrm{S}}+\Theta^{2}\omega_{2}+O(\Theta^{4})$, in units of $10^{-3}M^{-3}$, for the correct metric with the twist $[r,\theta]=\ii\Theta$, for the metric of Chaichian, Tureanu and Zet, and for the twist $[t,\theta]=\ii\Theta$, which alone admits $m=0$ (for the twist $[r,\theta]=\ii\Theta$ these cells are empty, by \cref{prop:mzero}); $\omega_{\mathrm{S}}$ is the Schwarzschild tone, in units of $M^{-1}$. Each coefficient is the Richardson extrapolation of the difference quotients at $\Theta^{2}/M^{2}=10^{-4}$, $2\times10^{-4}$ and $4\times10^{-4}$ of tones computed in 256-bit arithmetic, and is correct to the digits printed; the coefficients of the correct metric, those of the metric of Chaichian, Tureanu and Zet for $\ell\leqslant2$, and those of the twist $[t,\theta]=\ii\Theta$ for $(\ell,m)=(1,0)$, $(1,1)$, $(2,0)$ and $(2,2)$ agree with the difference quotients of the zeros of the Jost function at $\Theta=0.01\,M$ up to the correction of order $\Theta^{2}$.}
\begin{ruledtabular}
\begin{tabular}{cclccc}
$(\ell,m)$ & $n$ & $M\omega_{\mathrm{S}}$ & correct metric & Chaichian--Tureanu--Zet & twist $[t,\theta]=\ii\Theta$\\
\hline
""" + "\n".join(body) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}
"""
print(table)
