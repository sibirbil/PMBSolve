#include <iostream>
#include <random>
#include "pmb_driver.hpp"
#include "common.h"

using namespace std;

struct Data {
  int n;
  Data(int n) : n(n) {} 
};

void rosenbrock(opt_prec_t* x, opt_prec_t &f, opt_prec_t* g, Data* d) {
  int n = d->n;
  
  f = 0.0;
  for (int i=0; i<n/2; i++) {
    f += 100.0*pow(x[2*i+1] - pow(x[2*i], 2.0), 2.0) + pow((1.0 - x[2*i]), 2.0);
  }
  
  g[0] = -400.0*x[0]*(x[1] - pow(x[0], 2.0)) - 2.0*(1.0 - x[0]);
  for (int i=1; i<n-1; i++) {
    g[i] = 200.0*(x[i] - pow(x[i-1], 2.0)) - 400.0*x[i]*(x[i+1] - pow(x[i], 2.0)) - 2.0*(1.0-x[i]);
  }
  g[n-1] = 200.0*(x[n-1] - pow(x[n-2], 2.0));
}

int main(int argc, char * argv[]) {
  if(argc != 2) {
    cout << "Usage: executable dimension" << endl;
    return 0;
  }

  int n = atoi(argv[1]);
  Data* d = new Data(n);

  //initial solution
  std::random_device r;
  std::default_random_engine eng(r());
  std::uniform_real_distribution<> unif(5, 10);

  opt_prec_t* x_0 = new opt_prec_t[n];
  for (int i = 0; i < n; i++) {
    x_0[i] = unif(eng);
  }

  //options
  Options options;
  options.gtol = 1e-05;
  options.maxiter = 500;
  options.maxinneriter = 100;
  options.M = 5;
  options.display = true;
  options.history = true;

  Output output(n, options.maxiter);

  void (*fun)(opt_prec_t*, opt_prec_t&, opt_prec_t*, Data*) = rosenbrock;
  pmb_driver<Data>(x_0, options, output, n, fun, d);
  
  cout << "Exit: " << output.exit << endl;
  cout << "Fval: " << output.fval << endl;

  opt_prec_t ngf = fabs(output.g[0]);
  for (int i = 1; i < n; i++) {
    ngf = max(ngf, fabs(output.g[i]));
  }

  cout << "Norm: " << ngf << endl;
  cout << "Iterations: " << output.niter << endl;
  cout << "Evaluations: " << output.fcalls << endl;
  cout << "Models built: " << output.nmbs << endl;
  cout << "Time Spent in seconds: " << output.time << endl;

  return 0;
}
