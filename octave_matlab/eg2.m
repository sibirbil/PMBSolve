function [fx, dfx] = eg2(x)

  n = length(x);
  dfx = zeros(n,1);

  fx = 0.0;
  for i=1:n-1
      fx = fx + sin(x(1)+x(i)^2.0-1.0);
  end
  fx = fx + 0.5*sin(x(n)^2.0);

  dfx(1) = (2.0*x(1)+1.0)*cos(x(1)+x(1)^2.0-1.0);
  for i=2:n-1
      dfx(1) = dfx(1) + cos(x(1)+x(i)^2.0-1.0);
      dfx(i) = 2.0*x(i)*cos(x(1)+x(i)^2.0-1.0);
  end
  dfx(n) = cos(x(n)^2.0);

end % function ends
