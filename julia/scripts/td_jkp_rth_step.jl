# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# The step of the "jkp_rth" potential, seen in the time domain.  The pole of V at r = 2M, a
# distance of order Theta^2 inside the horizon, turns into a step of V in the tortoise coordinate:
# V ~ V_h ~ -1/32 for r* well below r*_s ~ 4 ln(Theta) - 4.93 (the plateau), V ~ 0 above it,
# then the usual barrier near r* ~ 1.6.  Measured here, for (l,m) = (1,1), (2,2), (2,1) and
# Theta = 0.5, 0.25, 0.1, 0.05:
#   (a) the barrier top x_b, the mid-point x_s of the step (V = V_h/2) and the cavity round trip
#       2(x_b - x_s);
#   (b) the echo of the step, isolated by causality: the signal Psi of the full potential minus
#       the signal Psi_0 of the potential continued below x_c = x_0 + 4 (x_0 the zero of V between
#       step and barrier) by the exponential that matches its value and slope at x_c (no step,
#       no plateau).  Both signals coincide until
#       waves that reached r* < x_c come back; the difference contains the reflection on the
#       step and the growth born on the plateau;
#   (c) the tone that survives after the causal time, from matrix-pencil windows that include
#       the non-oscillating growing terms, at two resolutions and two observers.
include(joinpath(@__DIR__, "..", "TimeDomain.jl"))
using .NCQNM, .TimeDomain, Printf

const DATA = joinpath(@__DIR__, "..", "..", "data", "timedomain")
const SCHW = Dict(1 => 0.292936133267283 - 0.097659988913578im, 2 => 0.483643872210713 - 0.096758775978288im)

function step_mid(Vx, Vh)
    xsf = collect(-80.0:0.01:0.0)
    vf = [Vx(x) for x in xsf]
    level = Vh / 2
    k = findlast(i -> (vf[i] - level) * (vf[i+1] - level) <= 0, 1:length(xsf)-1)
    a, b = xsf[k], xsf[k+1]
    for _ in 1:60
        c = (a + b) / 2
        (Vx(a) - level) * (Vx(c) - level) <= 0 ? (b = c) : (a = c)
    end
    return (a + b) / 2
end

rows = String[]
echorows = String[]
lms = isempty(ARGS) ? [(1, 1), (2, 2), (2, 1)] : [(parse(Int, ARGS[1]), parse(Int, ARGS[2]))]
for (l, m) in lms, Th in (0.5, 0.25, 0.1, 0.05)
    mod = model("jkp_rth", l, m, Th)
    T = tortoise(mod)
    Vh = potential(mod, T.uh)
    mu = sqrt(-Vh)
    Vx(x) = potential(mod, u_of_rstar(T, x)[1])
    xs_ = step_mid(Vx, Vh)
    xb = let g = collect(-5.0:0.001:10.0); g[argmax([Vx(x) for x in g])] end
    tau = 2 * (xb - xs_)
    @printf("(%d,%d) Theta=%.2f: V_h=%.6e mu=%.5f  barrier top r*=%.3f (V=%.5f)  step mid r*=%.3f  round trip 2(x_b - x_s)=%.2f\n",
            l, m, Th, Vh, mu, xb, Vx(xb), xs_, tau)
    # (b) echo by the causal difference, narrow ingoing pulse from r* = 40, observer r* = 10
    h = 0.05; a, b = -300.0, 400.0
    xs = collect(a:h:b)
    V = potential_at(mod, T, xs)
    # cut point: 4 units above the zero x_0 of V between the step and the barrier, where V > 0
    # rises like the near-horizon tail of the barrier and the step is already small
    x0 = let g = collect(xs_:0.01:xb), vg = [Vx(x) for x in g]
        g[findlast(i -> vg[i] < 0 && vg[i+1] >= 0, 1:length(g)-1)]
    end
    xc = x0 + 4.0
    ic = round(Int, (xc - a) / h) + 1
    Vc = V[ic]; kap = (V[ic+1] - V[ic-1]) / (2h) / Vc
    @assert Vc > 0 && kap > 0 "the continuation must decay below the cut point"
    V0 = copy(V)
    for i in 1:ic-1
        V0[i] = Vc * exp(kap * (xs[i] - xs[ic]))
    end
    rf = mol_evolve(V, a, h, h / 4, 200.0; x0=40.0, sigma=1.0, ingoing=true, obs=[10.0], every=4)
    r0 = mol_evolve(V0, a, h, h / 4, 200.0; x0=40.0, sigma=1.0, ingoing=true, obs=[10.0], every=4)
    t = rf.t; yf = rf.psi[1]; y0 = r0.psi[1]; dy = yf .- y0
    # barrier echo of the pulse at r* = 10: maximum of |Psi_0| after the pulse has passed
    ib = findfirst(t .> 40.0) - 1 + argmax(abs.(y0[t .> 40.0]))
    tb = t[ib]; Ab = abs(y0[ib])
    ion = findfirst(abs.(dy) .> 1e-6 * Ab)
    ton = t[ion]
    # first local maximum of |dy| after the onset, and the ratio to |Psi_0| there
    ae = abs.(dy)
    ie = ion
    while ie < length(t) - 1 && !(ae[ie] >= ae[ie-1] && ae[ie] > ae[ie+1])
        ie += 1
    end
    te = t[ie]
    # the growth part of dy: its local rate at the end of the run
    tl, lr = local_rate(t, dy, 50)
    # time at which the full signal stops decaying: minimum after t_b of the envelope
    # max|Psi| over [t - 5, t + 5] (independent of the cut point)
    K = round(Int, 5 / (t[2] - t[1]))
    env = [maximum(abs.(yf[max(1, i - K):min(length(t), i + K)])) for i in eachindex(t)]
    ig = ib - 1 + argmin(env[ib:length(t)-K])
    tg = t[ig]
    # size of the difference one round trip after the barrier echo
    iq = searchsortedfirst(t, tb + tau)
    dq = abs(dy[iq]) / Ab
    push!(echorows, @sprintf("%d,%d,%.3f,%.6e,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3e,%.3e,%.5f,%.3f,%.3e", l, m, Th, Vh, xb, xs_, tau, x0, xc,
                             tb, ton, te, ae[ie] / Ab, ae[ie] / max(abs(y0[ie]), 1e-300), lr[end], tg, dq))
    @printf("   growth takes over at t=%.2f (t - t_b = %.2f vs round trip %.2f); |dPsi(t_b + round trip)|/|Psi_0(t_b)| = %.2e\n", tg, tg - tb, tau, dq)
    @printf("   echo (causal difference, cut at r*=%.2f): pulse back from the barrier at t=%.2f; difference > 1e-6 at t=%.2f; first maximum at t=%.2f (t - t_b = %.2f vs round trip %.2f), |dPsi|/|Psi_0 peak| = %.2e, |dPsi|/|Psi_0| there = %.2e; late rate of dPsi = %.5f\n",
            xc, tb, ton, te, te - tb, tau, ae[ie] / Ab, ae[ie] / max(abs(y0[ie]), 1e-300), lr[end])
    tag = @sprintf("jkp_rth_l%dm%d_Th%.2f", l, m, Th)
    if l == 1
        sel = 1:5:length(t)
        write_columns(joinpath(DATA, "echo_$(tag).csv"), ["t", "psi_full", "psi_nostep", "difference"], t[sel], yf[sel], y0[sel], dy[sel];
                      comment="jkp_rth (l,m)=($l,$m) Theta=$Th at r* = 10; MOL h = 0.05; ingoing Gaussian (sigma 1) from r* = 40; potential cut at r* = $(round(xc, digits=3)) for psi_nostep",
                      digits=9)
    end
    # (c) tone after the causal time (two resolutions, two observers)
    for hh in (0.1, 0.05)
        xs2 = collect(a:hh:b)
        V2 = potential_at(mod, T, xs2)
        r = mol_evolve(V2, a, hh, hh / 4, 280.0; x0=40.0, sigma=2.0, ingoing=true, obs=[10.0, 50.0], every=round(Int, 0.1 / (hh / 4)))
        for (k, xo) in enumerate((10.0, 50.0))
            tc = 38.4 + (xo - 1.6) + tau
            res = qnm_scan(r.t, r.psi[k], SCHW[l]; t0s=tc .+ (5.0:1.0:60.0), Ws=(40.0, 60.0, 80.0), Ms=(8, 10, 12), ds=0.2, K=11)
            q = res[1]
            push!(rows, @sprintf("%d,%d,%.3f,%.3f,%.1f,%.10f,%.10f,%.3e,%.1f,%d,%.1f,%.10f,%.10f", l, m, Th, hh, xo, real(q.w), imag(q.w),
                                 q.spread, q.W, q.M, q.t0, real(q.w - SCHW[l]), imag(q.w - SCHW[l])))
            @printf("   late tone h=%.2f r*=%3.0f: w = %.8f %+.8fi  spread %.1e (W=%g M=%d t0=%.1f)  w - w_Schw = %+.2e %+.2ei\n",
                    hh, xo, real(q.w), imag(q.w), q.spread, q.W, q.M, q.t0, real(q.w - SCHW[l]), imag(q.w - SCHW[l]))
        end
    end
end
tag = isempty(ARGS) ? "all" : join(ARGS, "_")
open(joinpath(DATA, "jkp_rth_echo_$(tag).csv"), "w") do io
    println(io, "# jkp_rth step seen from r* = 10: barrier top x_b, step mid-point x_s (V = V_h/2), round trip 2(x_b - x_s); zero x_0 of V; causal difference: cut point x_c,")
    println(io, "# return of the pulse from the barrier t_b, onset (|dPsi| > 1e-6 |Psi_0(t_b)|) and first maximum t_e of |dPsi|, its size relative to the barrier echo and to Psi_0 at t_e, late rate of dPsi;")
    println(io, "# t_growth = minimum after t_b of the envelope of the full signal (start of growth at r* = 10); |dPsi| one round trip after t_b relative to |Psi_0(t_b)|")
    println(io, "l,m,Theta,V_h,x_b,x_s,round_trip,x_0,x_c,t_b,t_onset,t_first_max,ratio_to_barrier_echo,ratio_to_psi0,late_rate,t_growth,diff_after_round_trip")
    foreach(r -> println(io, r), echorows)
end
open(joinpath(DATA, "jkp_rth_late_tone_$(tag).csv"), "w") do io
    println(io, "# oscillating tone of jkp_rth after the causal time (windows starting 5 to 60 after it; pencil orders 8-12 include the growing terms)")
    println(io, "l,m,Theta,h,rstar_obs,Re_w,Im_w,spread,W,M,t0,Re_minus_Schw,Im_minus_Schw")
    foreach(r -> println(io, r), rows)
end
