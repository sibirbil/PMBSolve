# -*- coding: utf-8 -*-
# pmb.py
# Adapted to Python based on the Matlab code.
# See the end of the code for a usage example.
# 2.7.2017 -- Kaan

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

def pmbsolve(fun, x_0, **pars):
    """Minimize function fun starting at x_0.
    
    Input
    =====
    
    fun : The objective function.
          Input  : n-dimensional array x
          Output : A pair f,g where f is the function value at x (scalar)
                  and g is the gradient at x (n-dim array).
    x_0 : Initial position.
    pars: Parameters
          M            : The memory parameter for LBFGS preconditioning (5 by default).
          maxiter      : Maximum number of outer iterations; 0 for no limit (0 by default).
          maxiniter    : Maximum number of inner iterations; 0 for no limit (0 by default).
          maxfcalls    : Maximum number of function calls; 0 for no limit (0 by default).
          maxtime      : Running time limit in seconds; 0 for no limit (0 by default).
          display      : (Boolean) Whether to print intermediate results (False by default).
         
    Output
    ======
    A dictionary with the following keys.
    
    x         : The solution.
    fval      : The final objective function value.
    g         : The final gradient vector.
    niter     : Number of outer iterations until convergence.
    exit      : A string describing the exit condition.
    fcalls    : Number of function calls until convergence.
    nmbs      : Total number of model-building steps until convergence.
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
    f, g = fun(x)
    fcalls += 1
    
    # Parameters and default values
    M = pars.get("M",5)
    maxiter = pars.get("maxiter",0)
    maxiniter = pars.get("maxiniter",0)
    maxfcalls = pars.get("maxfcalls",0)
    display = pars.get("display",False)
    ftol = pars.get("ftol",1e-8)
    gtol = pars.get("gtol",1e-5)
    keephistory=pars.get("keephistory",False)
    maxtime = pars.get("maxtime",0)
        
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
        if ngf < gtol :
            exitcode = "First order condition is within gtol."
            break
        if iteration>1 and (fold-f)/max(abs(fold), abs(f), 1) < ftol:
            exitcode = "Function value decreases less than ftol."
        else:
            fold = f
        
        # end stopping conditions
        
        if keephistory:
            fhist.append(f)
            nghist.append(ngf)
        if display:
            print('PMB - Iter: %d ===> f = %f \t norm(g) = %f\n' % (iteration, f, ngf))
            
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
            ft, gt = fun(xt) # f^k_t and g^k_t
            fcalls += 1            
            
            sg = np.dot(s,g)  # (v6)
            if f - ft > -1e-4*sg:
                x = xt
                f = ft
                g = gt
                break
            sgt = np.dot(s,gt)
            y = gt - g  # y^k_t
            ys = np.dot(y,s) # v1
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

    retdict = {"x":x, "fval":f, "g":g, "niter":iteration, "exit":exitcode,
               "fcalls":fcalls, "time":duration, "nmbs":nmbs}
    if keephistory:
        fhist.append(f)
        nghist.append(np.max(g))
        retdict["fhist"]=fhist
        retdict["nghist"]=nghist
        
    return retdict
 
if __name__=="__main__":
    def rosenbrock(x):
        f = np.sum( 100*(x[1:]-x[:-1]**2)**2 + (x[:-1]-1)**2 )
        g = np.zeros(len(x))
        g[1:-1] = 200*(x[1:-1] - x[:-2]**2) - 400*(x[2:] - x[1:-1]**2)*x[1:-1] + 2*(x[1:-1]-1)
        g[0] = -400*x[0]*(x[1]-x[0]**2) + 2*(x[0]-1)
        g[-1] = 200*(x[-1] - x[-2]**2)
        return f,g
        
    n = 10000
    x0 = 5.0 + np.random.rand(n)*10.0

    pmb_out = pmbsolve(rosenbrock, x0, keephistory=True)
    print ('PMB Objective Function Value: ', pmb_out["fval"])
    print ('PMB Final Gradient Norm Value: ', max(abs(pmb_out["g"])))
    print ('PMB Exit Condition: ', pmb_out["exit"]);

    import matplotlib.pylab as plt
    plt.semilogy(pmb_out["fhist"])
    plt.xlabel("Iterations")    
    plt.ylabel("Objective function value")