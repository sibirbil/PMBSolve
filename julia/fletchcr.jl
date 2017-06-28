function fletchcr(x)

  n = length(x)
  dfx = zeros(n)

  fx = 0.0
  for i=1:n-1
    fx = fx + 100.0*(x[i+1]-x[i]+1.0-x[i]^2.0)^2.0
  end

  dfx[1] = -2.0*(2.0*x[1]+1.0)*(x[2]-x[1]+1.0-x[1]^2.0)
  for i=2:n-1
    dfx[i] = -2.0*(2.0*x[i]+1.0)*(x[i+1]-x[i]+1.0-x[i]^2.0)+2.0*(x[i]-x[i-1]+1.0-x[i-1]^2.0)
  end
  dfx[n] = 2.0*(x[n]-x[n-1]+1.0-x[n-1]^2.0)

  return fx, dfx

end # function ends
