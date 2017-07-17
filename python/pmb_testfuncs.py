# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 15:32:16 2017

@author: kaan
"""

import numpy as np
from pmb_scipy import pmbsolve
from scipy.optimize import minimize
from time import time

def objfun():
    pass
def objfun_deriv():
    pass

def edensch(x):
    fx = 16.0
    for i in range(len(x)-1):
        fx = fx + (x[i]-2.0)**4.0+(x[i]*x[i+1]-2.0*x[i+1])**2.0 + (x[i+1]+1.0)**2.0
    return fx

def edensch_deriv(x):
    n = len(x)    
    dfx = np.zeros(n)

    dfx[0] = 4.0*(x[0]-2.0)**3.0+2.0*x[1]*(x[0]*x[1]-2.0*x[1])
    for i in range(1,n-1):
        dfx[i] = 4.0*(x[i]-2.0)**3.0 + 2.0*x[i+1]*(x[i]*x[i+1]-2.0*x[i+1])+ 2.0*(x[i-1]-2.0)*(x[i-1]*x[i]-2.0*x[i])+2.0*(x[i]+1.0)
    dfx[-1] = 2.0*(x[-2]-2.0)*(x[-2]*x[-1]-2.0*x[-1])+2.0*(x[-1]+1.0)

    return dfx

n = 100000
x0 = 5.0 + np.random.rand(n)*10.0

objfun = edensch
objfun_deriv = edensch_deriv
# Call the minimizer
minimizer = "pmb"

if minimizer=="lbfgs":
    t0 = time()
    res = minimize(objfun, x0, jac=objfun_deriv, method="L-BFGS-B",
                   options={"maxcor":5})
    print(res.message)
    print("Final function value:", res.fun)
    print("Number of function evaluations:", res.nfev)
    print("Number of iterations:", res.nit)
    print("Wallclock time:", time()-t0)

if minimizer == "pmb":
    res = minimize(objfun, x0, jac=objfun_deriv, method=pmbsolve,
                   options={"eta":0.01})  
    print(res.message)
    print("Final function value:", res.fun)
    print("Number of function evaluations:", res.nfev)
    print("Number of outer iterations:", res.nit)
    print("Number of inner iterations:", res.nmbs)
    print("Wallclock time:", res.time)
