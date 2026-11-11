# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# The eight methods on common benchmark tones.  The reference is the zero of the Jost function at
# 256 bits (two independent settings agree to 59 digits).  The three solvers of the collocation
# problem (quartic Joukowski pencil, NLEIGS with poles on the cut, Beyn's contour integral) run in
# double precision on the same matrices; Newton's method runs in 256-bit arithmetic; the values
# of exterior complex scaling, of the time evolution, of the second NLEIGS realisation and of
# WKB are read from their own tables.  Output: data/comparison/methods.csv.
include(joinpath(@__DIR__, "..", "NCQNM.jl"))
include(joinpath(@__DIR__, "..", "nepsolvers.jl"))
using .NCQNM, .NEPSolvers, LinearAlgebra, Printf

setprecision(BigFloat, 256)
const DATA = joinpath(@__DIR__, "..", "..", "data")
const OUT = joinpath(DATA, "comparison")
mkpath(OUT)

# benchmark tones: (model, l, m, Theta, overtone)
const CASES = [("swc", 1, 1, 0.5, 0), ("swc", 1, 1, 0.5, 1), ("swc", 2, 1, 0.5, 0),
               ("swc", 2, 2, 0.5, 0), ("ctz", 1, 1, 0.5, 0)]

csvrows(file) = [split(l, ',') for l in eachline(file) if !startswith(l, '#')]

function jost_ref(model, l, m, Th, n)
    rows = csvrows(joinpath(DATA, "jost", "$(model)_l$(l)m$(m).csv"))
    for r in rows[2:end]
        if isapprox(parse(Float64, r[4]), Th) && r[5] == "n$(n)"
            return Complex{BigFloat}(parse(BigFloat, r[6]), parse(BigFloat, r[7]))
        end
    end
    error("no Jost value for $model $l $m $Th $n")
end

closest(ws, z) = ws[argmin(abs.(ws .- z))]

function table_value(file, keycols, key, recols; filter = r -> true)
    rows = csvrows(file)
    head = rows[1]
    idx = [findfirst(==(c), head) for c in keycols]
    ire, iim = findfirst(==(recols[1]), head), findfirst(==(recols[2]), head)
    for r in rows[2:end]
        filter(r) || continue
        ok = all(i -> begin
                     a, b = r[idx[i]], key[i]
                     b isa Number ? isapprox(parse(Float64, a), b; atol = 1e-9) : a == b
                 end, eachindex(idx))
        ok && return complex(parse(Float64, r[ire]), parse(Float64, r[iim]))
    end
    return missing
end

function matlab_nleigs(model, l, m, Th, ref)
    tag = "$(model)_l$(l)_m$(m)_Th" * replace(string(Th), "." => "p")
    files = filter(f -> startswith(f, tag), readdir(joinpath(DATA, "nleigs")))
    vals = ComplexF64[]
    for f in files, line in eachline(joinpath(DATA, "nleigs", f))
        startswith(line, '%') && continue
        p = split(strip(line))
        length(p) >= 2 && push!(vals, complex(parse(Float64, p[1]), parse(Float64, p[2])))
    end
    isempty(vals) && return missing
    z = closest(vals, ComplexF64(ref))
    return abs(z - ref) < 1e-4 ? z : missing
end

rows = String["model,l,m,Theta,n,method,re,im,abs_err"]
push_row!(c, method, z, ref) = push!(rows, @sprintf("%s,%d,%d,%.2f,%d,%s,%.16f,%.16f,%.3e",
                                                   c[1], c[2], c[3], c[4], c[5], method,
                                                   Float64(real(z)), Float64(imag(z)),
                                                   Float64(abs(Complex{BigFloat}(z) - ref))))

for c in CASES
    name, l, m, Th, n = c
    ref = jost_ref(name, l, m, Th, n)
    zr = ComplexF64(ref)
    @printf("%s (%d,%d) Theta=%.2f n=%d, reference %.16f %+.16fi\n", name, l, m, Th, n, real(zr), imag(zr))

    mod = model(name, l, m, Th)
    s = solve_joukowski(mod, 80)
    zj = closest(s.w[s.phys], zr)
    push_row!(c, "joukowski_qz_double_N80", zj, ref)

    d = discretize(mod, 80)
    Sigma = ComplexF64[zr - 0.06 + 0.04im, zr + 0.06 + 0.04im, zr + 0.06 - 0.04im, zr - 0.06 - 0.04im]
    lam, _, _ = run_nleigs(qnm_nep_rotated(d), d, Sigma)
    if !isempty(lam)
        push_row!(c, "nleigs_double_N80", closest(ComplexF64.(lam), zr), ref)
    end
    lb, _ = run_beyn(qnm_nep(d), zr, 0.03; neigs = 4, N = 600)
    if !isempty(lb)
        push_row!(c, "beyn_double_N80", closest(ComplexF64.(lb), zr), ref)
    end

    modb = model(name, l, m, BigFloat(Th); T = BigFloat)
    db = discretize(modb, 100)
    wb, _, _ = newton_refine(db, Complex{BigFloat}(zj))
    push_row!(c, "newton_256bit_N100", wb, ref)

    zm = matlab_nleigs(name, l, m, Th, zr)
    zm === missing || push_row!(c, "nleigs_second_realisation_double_N100", zm, ref)

    ze = table_value(joinpath(DATA, "scaling", "modes.csv"), ["model", "l", "m", "Theta", "index"],
                     (name, string(l), string(m), Th, string(n)), ["Re_w", "Im_w"])
    ze === missing || push_row!(c, "complex_scaling", ze, ref)

    tdfile = joinpath(DATA, "timedomain", name == "swc" ? "qnm_ringdown_summary_swc.csv" :
                                                          "qnm_ringdown_summary_ctz_jkp_tth.csv")
    if n == 0
        zt = table_value(tdfile, ["model", "l", "m", "Theta"], (name, string(l), string(m), Th),
                         ["Re_w", "Im_w"])
        zt === missing || push_row!(c, "time_domain", zt, ref)
    end

    for appr in ("wkb3", "wkb6", "pade33", "pade66")
        zw = table_value(joinpath(DATA, "wkb", "models_wkb.csv"),
                         ["model", "l", "m", "n", "Theta2", "approximation"],
                         (name, string(l), string(m), string(n), Th^2, appr), ["re", "im"])
        zw === missing || push_row!(c, "wkb_" * appr, zw, ref)
    end
    push!(rows, @sprintf("%s,%d,%d,%.2f,%d,jost_256bit_reference,%.16f,%.16f,0", name, l, m, Th, n,
                         Float64(real(ref)), Float64(imag(ref))))
end

open(joinpath(OUT, "methods.csv"), "w") do io
    foreach(r -> println(io, r), rows)
end
foreach(println, rows)
