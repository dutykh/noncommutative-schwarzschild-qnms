# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Summary of the zeros of the Jost function: every table of data/jost, printed to 25
# digits, and the number of digits on which the collocation (Joukowski pencil) frequencies of
# double precision agree with them.
# Output: data/jost/jost_vs_collocation.csv
using Printf

const DATA = joinpath(@__DIR__, "..", "..", "data", "jost")

# collocation frequencies (M w), model, l, m, Theta, label
const COLLOCATION = [
    ("ctz", 1, 1, 0.1, "n0", big"0.292980685" - big"0.097721078" * im),
    ("ctz", 1, 1, 0.1, "n1", big"0.264525617" - big"0.306442989" * im),
    ("ctz", 1, 1, 0.5, "n0", big"0.293994252" - big"0.099172782" * im),
    ("ctz", 1, 1, 0.5, "n1", big"0.266089779" - big"0.310857250" * im),
    ("ctz", 2, 2, 0.5, "n0", big"0.485334084" - big"0.098236978" * im),
    ("ctz", 2, 1, 0.5, "n0", big"0.485050502" - big"0.098558200" * im),
    ("jkp_tth", 1, 1, 0.5, "n0", big"0.292867760" - big"0.097586697" * im),
    ("jkp_rth", 1, 1, 0.5, "z0", big"0.292365174" - big"0.096848229" * im),
    ("jkp_rth", 2, 2, 0.5, "z0", big"0.485279618" - big"0.099957259" * im),
    ("jkp_rth", 2, 1, 0.5, "z0", big"0.484718411" - big"0.100376842" * im),
    ("swc", 1, 1, 0.5, "n0", big"0.2928268754" - big"0.0990516347" * im),
    ("swc", 1, 1, 0.25, "n0", big"0.2929159992" - big"0.0980117193" * im),
    ("swc", 1, 1, 0.25, "n1", big"0.2645869194" - big"0.3073449155" * im),
    ("swc", 1, 1, 0.1, "n0", big"0.2929332547" - big"0.0977164376" * im),
    ("swc", 2, 2, 0.5, "n0", big"0.4835721421" - big"0.0981167760" * im),
    ("swc", 2, 2, 0.5, "n1", big"0.4640571004" - big"0.2997518205" * im),
    ("swc", 2, 1, 0.5, "n0", big"0.4825060395" - big"0.0983904184" * im),
]

setprecision(BigFloat, 256)
rows = Dict{Tuple{String,Int,Int,Float64,String},Tuple{Complex{BigFloat},Float64}}()
for f in sort(readdir(DATA))
    (endswith(f, ".csv") && !occursin("tracks", f) && !occursin("schwarzschild", f) &&
     !occursin("collocation", f)) || continue
    lines = readlines(joinpath(DATA, f))
    for ln in lines[2:end]
        c = split(ln, ",")
        key = (String(c[1]), parse(Int, c[2]), parse(Int, c[3]), parse(Float64, c[4]), String(c[5]))
        rows[key] = (parse(BigFloat, c[6]) + parse(BigFloat, c[7]) * im, parse(Float64, c[8]))
    end
end

println("model    (l,m)  Theta   label  Re(M w)                       Im(M w)                       digits")
for key in sort(collect(keys(rows)))
    w, d = rows[key]
    @printf("%-8s (%d,%d)  %-6.3f  %-5s  %.25f  %.25f  %.0f\n", key[1], key[2], key[3], key[4], key[5],
            real(w), imag(w), d)
end

out = ["model,l,m,Theta,label,re_collocation,im_collocation,re_jost,im_jost,digits_agreement"]
println("\nagreement of the collocation frequencies with the zeros of J")
for (mo, l, m, Th, lab, wc) in COLLOCATION
    key = (mo, l, m, Th, lab)
    haskey(rows, key) || (println("   missing ", key); continue)
    w = rows[key][1]
    d = -log10(Float64(abs(w - wc) / abs(w)))
    @printf("   %-8s (%d,%d) Theta=%-5.2f %s  collocation %.10f %+.10fi  Jost %.12f %+.12fi  digits %.1f\n",
            mo, l, m, Th, lab, real(wc), imag(wc), real(w), imag(w), d)
    push!(out, join([mo, l, m, Th, lab, @sprintf("%.10f", real(wc)), @sprintf("%.10f", imag(wc)),
                     @sprintf("%.25f", real(w)), @sprintf("%.25f", imag(w)), @sprintf("%.1f", d)], ","))
end
open(joinpath(DATA, "jost_vs_collocation.csv"), "w") do io
    foreach(r -> println(io, r), out)
end
println("written ", joinpath(DATA, "jost_vs_collocation.csv"))
