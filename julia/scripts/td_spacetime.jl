# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# The field Psi(t, r*) of the correct metric, l = m = 1, Theta = 0.5 M, by the method of lines
# (h = 0.2, RK4 with dt = h/2), for the space-time picture of the growing waves.  The interval
# [-700, 480] is long enough that its ends cannot influence the window -600 <= r* <= 100 before
# t = 1500.  Data: an ingoing Gaussian pulse centred at r* = 40.  Output: data/timedomain/
# spacetime_swc_l1m1_Th0.50.csv with the columns t, r*, Psi (every 5 time units, every 2 in r*).
include(joinpath(@__DIR__, "..", "TimeDomain.jl"))
using .NCQNM, .TimeDomain, Printf

const DATA = joinpath(@__DIR__, "..", "..", "data", "timedomain")
mod = model("swc", 1, 1, 0.5)
T = tortoise(mod)
h = 0.2
a, b = -700.0, 480.0
xs = collect(a:h:b)
V = potential_at(mod, T, xs)
dt = h / 2
Tend = 1500.0
r = mol_evolve(V, a, h, dt, Tend; x0=40.0, sigma=2.0, ingoing=true, snap=round(Int, 5 / dt),
               sstride=round(Int, 2 / h))
open(joinpath(DATA, "spacetime_swc_l1m1_Th0.50.csv"), "w") do io
    println(io, "# method of lines, h = 0.2, dt = 0.1, ingoing Gaussian at r* = 40 (sigma = 2); mu = sqrt(-V_h) = ",
            sqrt(-potential(mod, T.uh)))
    println(io, "t,rstar,psi")
    for (t, P) in zip(r.tsnap, r.snaps), (x, p) in zip(r.xsnap, P)
        -600.0 <= x <= 100.0 || continue
        @printf(io, "%.1f,%.1f,%.6e\n", t, x, p)
    end
end
println("done: ", length(r.tsnap), " snapshots")
