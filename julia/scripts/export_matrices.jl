# Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
# and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
# Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
#
# Writes the collocation matrices of  T(w) = M1 + w Mw + w^2 Mw2 + k(w) (Mk + w Mkw),
# k(w) = w sqrt(1 - V_h/w^2), as plain text (real and imaginary parts, 17 significant digits),
# with V_h, for the rational Krylov solver in MATLAB (matlab/).
include(joinpath(@__DIR__, "..", "NCQNM.jl"))
using .NCQNM, Printf, DelimitedFiles

outdir = joinpath(@__DIR__, "..", "..", "data", "matrices")
mkpath(outdir)
cases = [("ctz", 1, 1, 0.5, nothing), ("swc", 1, 1, 0.5, nothing), ("swc", 1, 1, 0.1, nothing),
         ("swc", 2, 1, 0.5, nothing), ("jkp_rth", 1, 1, 0.1, :exp)]
for (name, l, m, Th, mp) in cases, N in (80, 100)
    mod = model(name, l, m, Th)
    uh = horizon(mod)[1]
    ep = mp === :exp ? 0.5 - uh : nothing
    d = discretize(mod, N; eps = ep)
    tag = @sprintf("%s_l%d_m%d_Th%s_N%d", name, l, m, replace(string(Th), "." => "p"), N)
    for key in (:one, :w, :w2, :k, :kw)
        writedlm(joinpath(outdir, "$(tag)_$(key)_re.txt"), real.(d.M[key]), ' ')
        writedlm(joinpath(outdir, "$(tag)_$(key)_im.txt"), imag.(d.M[key]), ' ')
    end
    open(joinpath(outdir, "$(tag)_Vh.txt"), "w") do io
        @printf(io, "%.17e\n", real(d.Vh))
    end
    println("written ", tag)
end
