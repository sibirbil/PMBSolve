#ifndef COMMON_H 
#define COMMON_H

#include <vector>

#define max(a,b)				\
  ({ __typeof__ (a)_a = (a);			\
    __typeof__ (b)_b = (b);			\
    _a > _b ? _a : _b; })

#define min(a,b)				\
  ({ __typeof__ (a)_a = (a);			\
    __typeof__ (b)_b = (b);			\
    _a < _b ? _a : _b; })

#ifdef SP
#define opt_prec_t float
#endif

#ifdef DP
#define opt_prec_t double
#endif

struct Output {
  int fcalls;
  int nmbs;
  int exit;
  double time;
  opt_prec_t* x;
  opt_prec_t* g;
  opt_prec_t fval;
  int niter;
  opt_prec_t* fhist;
  opt_prec_t* nghist;

  Output(int n, int maxiter) {
    x = new opt_prec_t[n];
    g = new opt_prec_t[n];
    fhist = new opt_prec_t[maxiter];
    nghist = new opt_prec_t[maxiter];
    nmbs = 0;
    fcalls = 0;
    exit = -5;
    niter = 0;
  }
};

struct Options {
  int M = 5;
  opt_prec_t gtol = 1.0e-5;
  opt_prec_t ftol = 1.0e-8;
  bool display = false;
  bool message = true;
  bool history = false;
  int maxiter = 1000;
  int maxinneriter = 100;
  int maxtime = 3600;
  int maxfcalls = 1000;
};
#endif
