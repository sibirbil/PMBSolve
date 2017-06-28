type Toutput
  x::Array{Float64,1}
  fval::Float64
  g::Array{Float64,1}
  fhist::Array{Float64,1}
  nghist::Array{Float64,1}
  fcalls::Int32
  niter::Int32
  nmbs::Int32
  exit::Int
  time::Float64
end
output = Toutput([],0.0,[],[],[],0,0,0,0,0.0);

type Tpars
  M::Int
  gtol::Float64
  ftol::Float64
  display::Bool
  message::Bool
  history::Bool
  maxiter::Int32
  maxiniter::Int32
  maxfcalls::Int32
  maxtime::Int32 # in seconds
end
pars = Tpars(5, 1.0e-5, 1.0e-8, false, true, false, 1000, 100, 1000, 3600);
