# Preconditioned Model Building Solver (PMBSolve)

PMBSolve is an optimization solver for solving unconstrained problems of the form
```
min f(x),
```
where f is a differentiable real-valued function defined on n-dimensional Euclidean space. The details of the algorithm is given in the arXiv paper by [Oztoprak and Birbil, 2017](http://www.tandfonline.com/doi/abs/10.1080/02331934.2017.1401070?journalCode=gopt20). 

Below, we have implementations for various programming languages. We have used the L-BFGS approach for preconditioning the initial trial step. Our implementation of the preconditioning method (```precond```) borrows heavily from the wonderful [minFunc](http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html) package of [Mark Schmidt](http://www.cs.ubc.ca/~schmidtm/).


## PMBSolve for Octave/MATLAB

This implementation has been tested on Octave 4.2.1 and MATLAB 2105b. In addition to basic usage on several test functions, we also present a **matrix factorization** example on a movie recommendation data set. Here is the [notebook](octave_matlab/PMBSolve_for_Octave_MATLAB.ipynb).

## PMBSolve for Julia

This implementation has been tested on Julia 0.4.6. We demonstrate ```pmbsolve``` method on several test functions. Here is the [notebook](julia/PMBSolve_for_Julia.ipynb).

## PMBSolve for Python

An implementation in [```scipy.optimize```](https://docs.scipy.org/doc/scipy/reference/optimize.html) and an implementation as a stand-alone method. We also present a simple **machine learning** example using **logistic regression**. Here is the [notebook](python/PMBSolve_for_Python.ipynb) comparing ```pmbsolve``` against other ```scipy.optimize``` methods.

Note that the logistic regression example requires the MNIST data set. It could not be included in this repository because its size exceeds the GitHub limit. In order to replicate the results, please download the CSV files [here](https://pjreddie.com/projects/mnist-in-csv/).

## PMBSolve for C++11

A **shared-memory implementation** of ```pmbsolve``` with a large-scale **matrix factorization** example on a movie recommendation data set. Here is a simple [how-to document](cpp/openmp/PMBSolve_for_OpenMP.md).
