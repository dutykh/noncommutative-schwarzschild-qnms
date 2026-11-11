# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Tables of r(r*) and of the potentials V(r*) on r* in [-600, 600], from the exact tortoise
# integral inverted by Newton iteration, checked against an independent integration of
# dr/dr* = F by the classical Runge-Kutta method (in l = log(u_h - u) on the horizon side,
# where dl/dr* = u^2 H(u), and in u on the far side, where du/dr* = -u^2 F(u)), started from the
# common value at r* = 0 and run with two step sizes (the error must drop by 2^4 = 16).  Also locates, for "jkp_rth", the step of the
# potential between the plateau V_h and the near-horizon Schwarzschild value.
include(joinpath(@__DIR__, "..", "TimeDomain.jl"))
using .NCQNM, .TimeDomain, Printf

const DATA = joinpath(@__DIR__, "..", "..", "data", "timedomain")

function rk4_check(T::Tortoise, h::Float64)
    uh = T.uh
    u0, l0 = u_of_rstar(T, 0.0)
    fl(l) = (u = uh - exp(l); u^2 * T.H(u))
    fu(u) = -u^2 * T.F(u)
    xs = Float64[]; us = Float64[]; ls = Float64[]
    # horizon side: l(x) for x from 0 down to -600
    # increments accumulated with compensated (Kahan) summation, so that rounding does not
    # build up over the 10^5 to 10^6 steps
    l = l0; cl = 0.0; x = 0.0
    n = round(Int, 600 / h)
    for s in 1:n
        k1 = fl(l); k2 = fl(l - h / 2 * k1); k3 = fl(l - h / 2 * k2); k4 = fl(l - h * k3)
        y = -h / 6 * (k1 + 2k2 + 2k3 + k4) - cl
        tt = l + y; cl = (tt - l) - y; l = tt
        x = -s * h
        if s % round(Int, 1 / h) == 0
            push!(xs, x); push!(ls, l); push!(us, uh - exp(l))
        end
    end
    u = u0; cu = 0.0
    for s in 1:n
        k1 = fu(u); k2 = fu(u + h / 2 * k1); k3 = fu(u + h / 2 * k2); k4 = fu(u + h * k3)
        y = h / 6 * (k1 + 2k2 + 2k3 + k4) - cu
        tt = u + y; cu = (tt - u) - y; u = tt
        x = s * h
        if s % round(Int, 1 / h) == 0
            push!(xs, x); push!(us, u); push!(ls, log(uh - u))
        end
    end
    return xs, us, ls
end

cases = [("schwarzschild", 0.0, [(1, 0), (2, 0)]),
         ("ctz", 0.1, [(1, 1)]), ("ctz", 0.25, [(1, 1)]), ("ctz", 0.5, [(1, 1), (2, 2)]),
         ("ctz", 1.0, [(1, 1)]),
         ("jkp_rth", 0.05, [(1, 1), (2, 2), (2, 1)]), ("jkp_rth", 0.1, [(1, 1), (2, 2), (2, 1)]),
         ("jkp_rth", 0.25, [(1, 1), (2, 2), (2, 1)]), ("jkp_rth", 0.5, [(1, 1), (2, 2), (2, 1)]),
         ("jkp_tth", 0.5, [(1, 1)]),
         ("swc", 0.1, [(1, 1), (2, 2), (2, 1)]), ("swc", 0.25, [(1, 1), (2, 2), (2, 1)]),
         ("swc", 0.5, [(1, 1), (2, 2), (2, 1)]), ("swc", 1.0, [(1, 1), (2, 2), (2, 1)])]
if !isempty(ARGS)
    cases = [c for c in cases if c[1] in ARGS]
end

mk(name, l, m, Th) = name == "schwarzschild" ? schwarzschild(l) : model(name, l, m, Th)

println("Tortoise tables: exact inversion against Runge-Kutta 4 (steps 1e-2 and 5e-3)")
steprows = String[]
for (name, Th, lms) in cases
    mod1 = mk(name, lms[1]..., Th)
    T = tortoise(mod1)
    xs = collect(-600.0:1.0:600.0)
    ul = [u_of_rstar(T, x) for x in xs]
    u = first.(ul); l = last.(ul)
    # Runge-Kutta check
    errs = Float64[]
    for h in (1e-2, 5e-3)
        xr, ur, lr = rk4_check(T, h)
        e = 0.0
        for (k, x) in enumerate(xr)
            ue, le = u_of_rstar(T, x)
            e = max(e, x < 0 ? abs(lr[k] - le) : abs(ur[k] - ue) / ue)
        end
        push!(errs, e)
    end
    @printf("%-14s Th=%.2f  u_h=%.14f  r_h=%.12f  b=%.12f   max error of RK4 (|dl| horizon side, |du|/u far side): h=1e-2 %.1e, h=5e-3 %.1e\n",
            name, Th, T.uh, 1 / T.uh, T.b, errs...)
    Vcols = [[potential(mk(name, lm..., Th), uu) for uu in u] for lm in lms]
    hdr = vcat(["rstar", "r", "log_uh_minus_u"], ["V_l$(lm[1])m$(lm[2])" for lm in lms])
    tag = name == "schwarzschild" ? "schwarzschild" : @sprintf("%s_Th%.2f", name, Th)
    write_columns(joinpath(DATA, "tortoise_$(tag).csv"), hdr, xs, 1 ./ u, l, Vcols...;
                  comment="r(r*) and V(r*), units M = 1, convention r* = 1/u - 2 log u + b log(u_h - u) - int_0^u Q/H " *
                          @sprintf("(u_h = %.15f, b = %.15f)", T.uh, T.b))
    if name == "jkp_rth"
        # step of the potential: where V crosses the mid-value between V_h and 0, located by
        # bisection on a fine grid, and the width over which V goes from 0.1 V_h to 0.9 V_h
        for lm in lms
            mod = mk(name, lm..., Th)
            Vh = potential(mod, T.uh)
            Vx(x) = potential(mod, u_of_rstar(T, x)[1])
            function cross(level)
                a, b = -80.0, 0.0
                # V increases from V_h through level on the way out; find the last crossing
                # below the barrier by scanning outward
                xsf = collect(-80.0:0.01:0.0)
                vf = [Vx(x) for x in xsf]
                k = findlast(i -> (vf[i] - level) * (vf[i+1] - level) <= 0, 1:length(xsf)-1)
                a, b = xsf[k], xsf[k+1]
                for _ in 1:60
                    c = (a + b) / 2
                    (Vx(a) - level) * (Vx(c) - level) <= 0 ? (b = c) : (a = c)
                end
                return (a + b) / 2
            end
            xmid = cross(Vh / 2)
            x10 = cross(0.1Vh); x90 = cross(0.9Vh)
            vmin = minimum(Vx(x) for x in -200.0:0.01:0.0)
            push!(steprows, @sprintf("%.2f,%d,%d,%.10e,%.10f,%.10f,%.10f,%.10e", Th, lm[1], lm[2], Vh, xmid, x10, x90, vmin))
            @printf("    step (l,m)=(%d,%d): V_h=%.6e  V=V_h/2 at r*=%.4f, 90%%->10%% of V_h over [%.3f, %.3f] (width %.3f), min V on [-200,0] = %.6e, 4 ln(Th) - 4.93 = %.3f\n",
                    lm..., Vh, xmid, x90, x10, x10 - x90, vmin, 4log(Th) - 4.9318)
        end
    end
end
isempty(steprows) || open(joinpath(DATA, "jkp_rth_step.csv"), "w") do io
    println(io, "# step of the jkp_rth potential in r*: mid-level crossing V = V_h/2, crossings of 0.1 V_h and 0.9 V_h, minimum of V on [-200, 0]")
    println(io, "Theta,l,m,V_h,rstar_mid,rstar_10pct,rstar_90pct,Vmin")
    foreach(r -> println(io, r), steprows)
end
