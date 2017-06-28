function [f, df] = matfac(x, Y, latent_dim, datasize)

  [n,m] = size(Y);
  Z1 = reshape(x(1:n*latent_dim),n,latent_dim);
  Z2 = reshape(x(n*latent_dim+1:latent_dim*(m+n)),latent_dim,m);

  E = Y - (Z1*Z2).*(Y>0);
  f = full(0.5*sum(sum(E.*E)));
  f = f/datasize;

  G1 = -E*Z2';
  G2 = -Z1'*E;
  df = full([G1(:);G2(:)]);
  df = df/datasize;

  end
