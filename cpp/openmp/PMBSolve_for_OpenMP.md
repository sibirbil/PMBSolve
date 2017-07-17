# Preconditioned Model Building (PMB)

PMB is a method for solving unconstrained optimization problems. Suppose that we want to solve the ```rosenbrock``` function with PMB. First you need to create a file that includes two header files corresponding to the PMB driver and the auxiliary functions, respectively.
```cpp
#include "pmb_driver.hpp"
#include "common.h"
```
Then you write a new ```struct``` that includes the function to be solved as an operator.
```cpp
struct Rosenbrock {
  int n;
  Rosenbrock(int n) : n(n) {}

  void operator()(opt_prec_t* x, opt_prec_t &f, opt_prec_t* g) {
    f = 0.0;
    #pragma omp parallel for schedule(static)
    for (int i=0; i<n/2; i++) {
      f += 100.0*pow(x[2*i+1] - pow(x[2*i], 2.0), 2.0) + pow((1.0 - x[2*i]), 2.0);
    }

    g[0] = -400.0*x[0]*(x[1] - pow(x[0], 2.0)) - 2.0*(1.0 - x[0]);
    #pragma omp parallel for schedule(static)
    for (int i=1; i<n-1; i++) {
      g[i] = 200.0*(x[i] - pow(x[i-1], 2.0)) - 400.0*x[i]*(x[i+1] - pow(x[i], 2.0)) - 2.0*(1.0-x[i]);
    }
    g[n-1] = 200.0*(x[n-1] - pow(x[n-2], 2.0));
  }
};
```
Note that the operator takes a vector (solution) as the first argument and returns the function value and the gradient vector. Here, the term ```opt_prec_t``` shows the precision level like ```float``` or ```double```. This precision is specified during the compilation.

PMB solver can take several options. These options with their default values are given below.
```cpp
Options options;
options.M = 5; // The memory size for the preconditioners
options.gtol = 1e-05; // The tolerance for the first-order optimality
options.ftol = 1.0e-8; // The normalized difference between two consecutive function values
options.display = false; // To display information about the progress in every iteration
options.message = true; // Shows a message about the exit condition or errors
options.history = false; // Stores the function values and the first-order errors throughout the iterations
options.maxiter = 1000; // Maximum number of iterations
options.maxinneriter = 100; // Maximum number of inner iterations for model building
options.maxfcalls = 1000; // Maximum number of function calls
options.maxtime = 3600; // Maximum computation time in seconds
```

Then, we need to write the ```main``` function and create an instance of the ```Rosenbrock``` struct. The details of the main function is given in file "rosenbrock_solver.cpp".


After obtaining a solution to the problem, the solver provides an output with the following structure.
```cpp
Output* output;
cout << "Total number of function calls to solve the problem: " << output->fcalls << endl;
cout << "Total number of times a model is built during the inner iterations: " output->nmbs << endl;
cout << "Exit status: " << output->exit << endl;
/*
Exit status (The associated parameters with each status is given in parantheses)
  1: First order condition is met (options.gtol)
  0: Maximum number of inner iterations is reached (options.maxiniter)
  -1: Maximum number of iterations is reached (options.maxiter)
  -2: Maximum number of function calls is reached (options.maxfcalls)
  -3: Time limit is reached (options.maxtime)
  -4: Change in function value between two consecutive iterations is below tolerance (options.ftol)
*/
cout << "Time to solve the problem in seconds: " << output->time << endl;
// Final solution (n-dimensional vector)
// output->x;
// Final gradient at options->x (n-dimensional vector)
// output->g;
cout << "Final objective function value: " << output->fval << endl;
cout << "Number of iterations: " << output->niter << endl;
// History of function values (iteration 1 to output->niter)
// output->fhist;
// History of gradient norms (iteration 1 to output->niter)
// output->nghist;
```

PMB driver supports [OpenMP](http://www.openmp.org). In addition to the ```openmp``` flag, we also specify the precision for compilation.
```
$ g++ rosenbrock_solver.cpp -O3 -std=c++11 -fopenmp -D"DP" -o rose_dp
```
The argument ```-D"DP"``` stands for ```double``` precision. If you want single precision, then replace this argument with ```-D"SP"```. You can also check the ```Makefile``` in the repository.

We are now ready to solve the ```rosenbrock``` function. First, we need to specify the number of threads with the following command. How about 16?
```
$ export OMP_NUM_THREADS=16
```
Then we can solve a quite large-scale instance of the ```rosenbrock``` function. Let's try **one million-dimensional** case.
```
$ rose_dp 1000000
Maximum number of iterations (maxiter) is reached
Exit: -1
Fval: 2.56411e-09
Norm: 5.16755e-05
Iterations: 500
Evaluations: 549
Models built: 48
Time Spent in seconds: 6.63962
```
This is a very difficult problem. We have exhausted the maximum number of iterations but obtained a very good solution. This is a lucky run, where we have found the optimal solution. Please note that the solution depends highly on the initial starting solution for this particular problem.

A slightly more complicated implementation for sparse matrix factorization is also available in the repository. This example can also be compiled with the existing ```Makefile```. For our numerical experiment here, we can use the [MovieLens data set](https://grouplens.org/datasets/movielens/) with one million ratings. The data file is available with this repository as a zipped file ("1M.dat.zip"). After unzipping the file, you need to specify the data file as the first argument, and the second argument is the latent dimension.

```
$ pmb_mf_dp 1M.dat 50
rows/cols/ratings: 3883 6040 1000209
File is read. Allocating memory for matrix
Memory is allocated; creating crs and ccs
        crs is created
        ccs is created
First order condition is within gtol
Exit: 1
Fval: 0.132988
Norm: 9.80845e-06
Iterations: 346
Evaluations: 365
Models built: 18
Time Spent in seconds: 5.63203
RMSE: 0.515728
Latent Dim: 50
```
