% Authors: Davide Batic, Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science
% and Technology, Abu Dhabi, UAE), Stefan Güttel (Department of Mathematics, The University of
% Manchester, UK) and Narges Heidari (Faculty of Physics, Shahrood University of Technology, Iran)
%
% Quasinormal frequencies as eigenvalues of the nonlinear matrix problem
%     T(w) a = 0,   T(w) = M1 + w Mw + w^2 Mw2 + k(w) (Mk + w Mkw),   k(w) = sqrt(w^2 - V_h),
% by the fully rational Krylov method NLEIGS (Guettel, Van Beeumen, Meerbergen, Michiels 2014)
% of the Rational Krylov Toolbox.  The principal square root has its cuts on the imaginary
% rays from +-i sqrt|V_h| to +-i infinity (V_h < 0); in the right half-plane it coincides with
% the physical sheet (k ~ w at infinity).  T is sampled at Leja-Bagby points on the boundary of
% a polygonal target set Sigma, with poles on the cuts; the rational interpolant is linearised
% and its eigenvalues inside Sigma are computed by the rational Krylov method with a repeated
% shift.
%
% Usage: nleigs_driver('swc_l1_m1_Th0p5_N100', [0.2-0.05i, 0.4-0.05i, 0.4-0.15i, 0.2-0.15i], 0.3-0.1i)
function nleigs_driver(tag, vertices, shift, nshifts, tol, Nmax)
if nargin < 4, nshifts = 150; end
if nargin < 5, tol = 1e-12; end
if nargin < 6, Nmax = 100; end
here = fileparts(mfilename('fullpath'));
src = fullfile(here, '..', 'data', 'matrices');
dst = fullfile(here, '..', 'data', 'nleigs');
rd = @(key) readmatrix(fullfile(src, [tag '_' key '_re.txt']), 'FileType', 'text', 'Delimiter', ' ') + ...
       1i*readmatrix(fullfile(src, [tag '_' key '_im.txt']), 'FileType', 'text', 'Delimiter', ' ');
M1 = rd('one'); Mw = rd('w'); Mw2 = rd('w2'); Mk = rd('k'); Mkw = rd('kw');
Vh = readmatrix(fullfile(src, [tag '_Vh.txt']), 'FileType', 'text');
n = size(M1, 1);
k = @(z) sqrt(z^2 - Vh);
fun = @(z) M1 + z*Mw + z^2*Mw2 + k(z)*(Mk + z*Mkw);

% poles on the two cuts, logarithmically spaced from the branch points
npts = 1000;
b = sqrt(abs(Vh));
Xi = 1i*logspace(log10(b), 4, npts/2);
Xi = [Xi, conj(Xi)];

vertices = vertices(:).';
if vertices(end) ~= vertices(1), vertices = [vertices, vertices(1)]; end
Sigma = util_discretise_polygon(vertices, npts);
QN = util_nleigs(fun, Sigma, Xi, tol, Nmax);
LN = linearize(QN);
rng(1); v = randn(LN.N*n, 1);
[V, K, H] = rat_krylov(LN, v, repmat(shift, 1, nshifts));
[X, D] = eig(H(1:end-1, :), K(1:end-1, :));
ritz = diag(D);
% the discretised boundary returned by util_discretise_polygon is sorted, not ordered along
% the polygon, so the inside test uses the vertices
inside = inpolygon(real(ritz), imag(ritz), real(vertices), imag(vertices));
ritz = ritz(inside);
vec = V(1:n, :)*(H*X(:, inside));
vec = vec./sqrt(sum(abs(vec).^2, 1));
res = arrayfun(@(j) norm(fun(ritz(j))*vec(:, j)), 1:numel(ritz));
nrm = arrayfun(@(j) norm(M1) + abs(ritz(j))*norm(Mw) + abs(ritz(j))^2*norm(Mw2) + ...
               abs(k(ritz(j)))*(norm(Mk) + abs(ritz(j))*norm(Mkw)), 1:numel(ritz));
[~, order] = sort(-imag(ritz));
out = fullfile(dst, sprintf('%s_target_%.3f%+.3fi.txt', tag, real(shift), imag(shift)));
fid = fopen(out, 'w');
fprintf(fid, '%% NLEIGS degree %d, relative residual of each Ritz pair; target vertices: %s\n', ...
        LN.N, mat2str(vertices, 4));
for j = order(:).'
    fprintf(fid, '%.15f %+.15f  %.2e\n', real(ritz(j)), imag(ritz(j)), res(j)/nrm(j));
end
fclose(fid);
fprintf('%s: degree %d, %d Ritz values in the target set\n', tag, LN.N, numel(ritz));
type(out);
end
