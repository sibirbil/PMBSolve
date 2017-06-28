include("matfac.jl")
include("pmbsolve.jl")

X = readdlm("X_3883_6040.dat")
Y = sparse(round(Int32, X[:,1]), round(Int32, X[:,2]), X[:,3])
nrows, ncols = size(Y)

lat = 50;
n = (nrows + ncols)*lat;
x0 = sqrt(rand(1:5, n)/lat);
datasize = length(nonzeros(Y));
mfac(x) = matfac(x, Y, lat, datasize);
pars.display = true;
pmbout = pmbsolve(mfac, x0);
