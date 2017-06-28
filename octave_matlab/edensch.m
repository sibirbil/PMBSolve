function [fx, dfx] = edensch(x)

  n = length(x);
  dfx = zeros(n,1);

  fx = 16.0;
  for i=1:n-1
      fx = fx + (x(i)-2.0)^4.0+(x(i)*x(i+1)-2.0*x(i+1))^2.0+(x(i+1)+1.0)^2.0;
  end

  dfx(1) = 4.0*(x(1)-2.0)^3.0+2.0*x(2)*(x(1)*x(2)-2.0*x(2));
  for i=2:n-1
      dfx(i) = 4.0*(x(i)-2.0)^3.0+2.0*x(i+1)*(x(i)*x(i+1)-2.0*x(i+1))+...
          2.0*(x(i-1)-2.0)*(x(i-1)*x(i)-2.0*x(i))+2.0*(x(i)+1.0);
  end
  dfx(n) = 2.0*(x(n-1)-2.0)*(x(n-1)*x(n)-2.0*x(n))+2.0*(x(n)+1.0);

end % function ends
