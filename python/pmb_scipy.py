# -*- coding: utf-8 -*-
# pmb_scipy.py
# This version of PMB has the same algorithm. The interface is modified
# so that it can be used seamlessly with scipy.optimize.minimize()
# See the end of the code for a usage example.
# 2.7.2017 -- Kaan

from scipy.optimize import OptimizeResult
import numpy as np
from time import time

def precond(g, g_old, s, Hdiag, S, Y, YS, M, mem_start, mem_end):
    y = g-g_old
    ys = np.dot(y,s)
    if ys > 1e-10:
        if mem_end < M-1:
            mem_end += 1;
            if mem_start != 0 :
                if mem_start == M-1:
                    mem_start = 0
                else:
                    mem_start += 1
        else:
            mem_start = min(1,M-1)
            mem_end = 0
        S[:,mem_end] = np.copy(s)
        Y[:,mem_end] = np.copy(y)
        YS[mem_end] = np.copy(ys)
        Hdiag = ys/np.dot(y,y);
    if mem_start == 0:
        ind = list(range(mem_end))
        nMem = mem_end-mem_start+1
    else:
        ind = list(range(mem_start,M))+list(range(mem_end))
        nMem = M
    al = np.zeros(nMem)
    be = np.zeros(nMem)
    s = -g;
    for i in ind[::-1]:
        al[i] = np.dot(S[:,i],s)/YS[i]
        s = s - al[i]*Y[:,i]
    s = Hdiag*s
    for i in ind:
        be[i] = np.dot(Y[:,i],s)/YS[i]
        s = s + S[:,i]*(al[i]-be[i])

    return s, Hdiag, S, Y, YS, mem_start, mem_end


def pmbsolve(fun, x_0, args=(), jac=None, **options):
    """Minimize function fun starting at x_0.
    
    Input
    =====
    
    fun(x, *args)  : The objective function.
                     Input  : n-dimensional array x, and related parameters.
                     Output : The function value at x (scalar).
    x_0  : Initial position (ndarray)
    args : Any arguments that are passed to the objective function (sequence).
    jac(x, *args)  : A function that returns the gradient of the objective function.
                     Input  : n-dimensional array x, an
                     Output : The gradient at x (n-dim array).
    
    options : Parameters
              ftol         : Stop when the normalized difference between two consecutive
                             function values falls below this (1e-5 by default)
              gtol         : Stop when the maximum component of the gradient falls below
                             this (1e-5 by default). 
              keephistory  : (Boolean) Whether to keep the function and gradient norm values at 
                             each iteration (False by default).
              M            : The memory parameter for LBFGS preconditioning (5 by default).
              maxiter      : Maximum number of outer iterations; 0 for no limit (0 by default).
              maxiniter    : Maximum number of inner iterations; 0 for no limit (0 by default).
              maxfcalls    : Maximum number of function calls; 0 for no limit (0 by default).
              maxtime      : Running time limit in seconds; 0 for no limit (0 by default).
              display      : (Boolean) Whether to print intermediate results (False by default).
         
    Output
    ======
    An OptimizeResult object with the following attributes.
    
    fun       : The final objective function value.
    jac       : The final gradient vector.
    nit       : Number of outer iterations until convergence.
    nmbs      : Total number of inner iterations (model-building steps) until convergence.
    x         : The solution (the vector x minimizing fun(x) )
    nfev      : Number of function calls until convergence.
    message   : A string describing the exit condition.
    time      : Total wallclock time until convergence.
    fhist     : History of objective function values at each outer iteration. 
                (if keephistory is True)
    nghist    : History of maximum-elements of the gradient (infinite norm) at 
                each outer iteration. (if keephistory is True)

    """
    fhist=[] # holds the history of function values
    nghist=[] # holds the history of inf. norm of derivative
    fcalls = 0  # count of function calls
    nmbs = 0  # count of total inner iterations (model building steps)
    
    x = x_0
    f = fun(x,*args)
    fold = np.inf  # to check stopping condition
    g = jac(x,*args)
    fcalls += 1
    
    # Parameters and default values
    M = options.get("M",5)
    maxiter = options.get("maxiter",0)
    maxiniter = options.get("maxiniter",0)
    maxfcalls = options.get("maxfcalls",0)
    display = options.get("display",False)
    ftol = options.get("ftol",1e-8)
    gtol = options.get("gtol",1e-5)
    keephistory=options.get("keephistory",False)
    maxtime = options.get("maxtime",0)
    
    n = len(x)
    S = np.zeros((n, M))
    Y = np.zeros((n, M))
    YS = np.zeros(M)
    mem_start = 0
    mem_end = -1
    Hdiag = 1
    
    iteration = 1
    tstart = time()
    while True:  # outer iterations
        # Stopping conditions
        ngf = np.max(g) # the inf-norm of the derivative vector

        if maxiter > 0 and iteration >= maxiter:
            exitcode = "Maximum number of iterations (maxiter) is reached."
            break
        if maxfcalls > 0 and fcalls>= maxfcalls:
            exitcode = "Maximum number of function calls (maxfcalls) is reached."
            break
        if maxtime > 0 and time()-tstart >= maxtime:
            exitcode = "Maximum time limit (maxtime) is reached."
            break
        if ngf < gtol:
            exitcode = "Gradient norm converged to zero within gtol."
            break
        if iteration > 1 and (fold-f)/max((abs(fold),abs(f),1)) < ftol:
            exitcode = "Function value decreases less than ftol."
            break
        else:
            fold = f
        # end stopping conditions
        
        if keephistory:
            fhist.append(f)
            nghist.append(ngf)        
        if display:
            print('PMB - Iter: %d ===> f = %.10f \t norm(g) = %f\n' % (iteration, f, ngf))

        # L-BFGS preconditioning
        if iteration > 1:
            s, Hdiag, S, Y, YS, mem_start, mem_end = precond(g, g_old, s, Hdiag, S, Y, YS, M, mem_start, mem_end)
        else:
            s = -g/ngf
        g_old = np.copy(g)
        # end L-BFGS preconditioning
        initer = 0
        while maxiniter==0 or initer < maxiniter:
            xt = x + s # x^k_t
            ft = fun(xt, *args) # f^k_t
            gt = jac(xt, *args) # g^k_t
            fcalls += 1            
            
            sg = np.dot(s,g)  # (v6)
            if f - ft > -1e-4*sg:
                x = xt
                f = ft
                g = gt
                break
            sgt = np.dot(s,gt)
            y = gt - g  # y^k_t
            ys = sgt - sg # v1
            ss = np.dot(s,s) # v2
            yy = np.dot(y,y) # v3
            yg = np.dot(y,g) # v4
            gg = np.dot(g,g) # v5
            
            # Guess eta
            fdiff = abs(f-ft)
            if abs(sg) > 1e-8:
                eta1 = fdiff/abs(sg)
            else:
                eta1 = 1.0
            if abs(sgt) > 1e-8:
                eta2 = fdiff/abs(sgt)
            else:
                eta2 = 1.0
            eta = min(eta1, eta2)/(eta1+eta2)
            # end guess eta
            
            sigma = 0.5*(np.sqrt(ss)*(np.sqrt(yy)+np.sqrt(gg)/eta)-ys)
            theta = (ys + 2*sigma)**2 - ss*yy
            cg = -ss/(2*sigma) # cg(sigma)
            cs = cg/theta*(-(ys + 2*sigma)*yg+yy*sg) # cs(sigma)
            cy = cg/theta*(-(ys + 2*sigma)*sg+ss*yg) # cy(sigma)
            s = cg*g + cs*s + cy*y  # step
            initer += 1
            # inner iterations end
        
        nmbs += initer
        if maxiniter > 0 and initer >= maxiniter:
            exitcode = 'Maximum number of inner iterations (maxiniter) is reached'
            break
        iteration += 1
    # outer iterations end

    duration = time()-tstart
    if keephistory:
        fhist.append(f)
        nghist.append(np.max(g))
        retobj = OptimizeResult(fun=f, jac=g, x=x, nfev=fcalls,
                                message=exitcode,nit=iteration,
                                nmbs = nmbs, time=duration,
                                fhist=fhist, nghist=nghist)
    else:
        retobj = OptimizeResult(fun=f, jac=g, x=x, nfev=fcalls,
                                message=exitcode,nit=iteration,
                                nmbs = nmbs, time=duration)
    return retobj
    
if __name__=="__main__":
    
    # Here is an example where PMB is used with SciPy minimizer.
    
    # In a script, write:
    # import pmbsolve from pmb_scipy
    
    from scipy.optimize import minimize
    import matplotlib.pylab as plt
    
    # define the function to be minimized, and its gradient
    def rosenbrock(x, *args):
        return np.sum( 100*(x[1:]-x[:-1]**2)**2 + (x[:-1]-1)**2 )
        
    def rosenbrock_der(x, *args):
        g = np.zeros(len(x))
        g[1:-1] = 200*(x[1:-1] - x[:-2]**2) - 400*(x[2:] - x[1:-1]**2)*x[1:-1] + 2*(x[1:-1]-1)
        g[0] = -400*x[0]*(x[1]-x[0]**2) + 2*(x[0]-1)
        g[-1] = 200*(x[-1] - x[-2]**2)
        return g

    # Initialize    
    n = 100 # dimension of the problem
    x0 = 5.0 + np.random.rand(n)*10.0

    # Call the minimizer
    minimizer = "pmb"
    if minimizer=="lbfgs":
        t0 = time()
        res = minimize(rosenbrock, x0, jac=rosenbrock_der, method="L-BFGS-B",
                       options={"maxcor":5})
        print(res.message)
        print("Final function value:", res.fun)
        print("Number of function evaluations:", res.nfev)
        print("Number of iterations:", res.nit)
        print("Wallclock time:", time()-t0)
    
    if minimizer == "pmb":
        
        res = minimize(rosenbrock, x0, jac=rosenbrock_der, method=pmbsolve,
                       options={"keephistory":True, "display":True})  
        print(res.message)
        print("Final function value:", res.fun)
        print("Number of function evaluations:", res.nfev)
        print("Number of outer iterations:", res.nit)
        print("Number of model-building steps:", res.nmbs)
        print("Wallclock time:", res.time)

        # The function values vs iteration
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.grid(True)
        plt.semilogy(res.fhist,".-")
        plt.xlabel("Iterations")
        plt.ylabel("Objective function value")
        plt.title("Function value")
        
        # The gradient norm values vs. iteration    
        plt.subplot(1,2,2)
        plt.grid(True)
        plt.semilogy(res.nghist,".-")
        plt.xlabel("Iterations")
        plt.ylabel("Gradient norm value")
        plt.title("Gradient inf-norm")