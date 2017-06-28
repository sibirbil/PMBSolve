function nonscomp(x)

  n = length(x)
  dfx = zeros(n)

  fx = (x[1]-1.0)^2.0
  for i=2:n
      fx = fx + 4.0*(x[i]-x[i-1]^2.0)^2.0
  end

  dfx[1] = 2.0*(x[1]-1.0)-16.0*x[1]*(x[2]-x[1]^2.0)
  for i=2:n-1
      dfx[i] = 8.0*(x[i]-x[i-1]^2.0)-16*x[i]*(x[i+1]-x[i]^2.0)
  end
  dfx[n] = 8.0*(x[n]-x[n-1]^2.0)

  return fx, dfx

end # function ends
