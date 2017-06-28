# Preconditioned Model Building Solver (PMBSolve)

PMBSolve is an optimization solver for solving unconstrained problems of the form
```
min f(x),
```
where f is a differentiable real-valued function defined on n-dimensional Euclidean space. Below, we have implementations for various programming languages.

## PMBSolve for Octave/MATLAB

This implementation has been tested on Octave 4.2.1 and MATLAB 2105b. In addition to basic usage on several test functions, we also present a **matrix factorization** example on a movie recommendation data set. Here is the [notebook](octave_matlab/PMBSolve_for_Octave_MATLAB.ipynb).

## PMBSolve for Julia

This implementation has been tested on Julia 0.4.6. We demonstrate ```pmbsolve``` method on several test functions. Here is the [notebook](julia/PMBSolve_for_Julia.ipynb).

## PMBSolve for Python

Coming soon with a simple **machine learning** example using **logistic regression**.

## PMBSolve for C++11

Coming soon with a **shared-memory implementation** of a **matrix factorization** example on a movie recommendation data set.
