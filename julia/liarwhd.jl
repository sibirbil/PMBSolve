function liarwhd(x)

  n = length(x)
  dfx = zeros(n)

  fx = 0.0
  for i=1:n
      fx = fx + 4.0*(x[i]^2.0-x[1])^2.0+(x[i]-1.0)^2.0
  end

  dfx[1] = 8.0*(2.0*x[1]-1.0)*(x[1]^2.0-x[1])+2.0*(x[1]-1.0)
  for i=2:n
      dfx[1] = dfx[1] - 8.0*(x[i]^2.0-x[1])
      dfx[i] = 16.0*x[i]*(x[i]^2.0-x[1])+2.0*(x[i]-1.0)
  end

  return fx, dfx

end # function ends
