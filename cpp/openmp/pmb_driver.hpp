//  pmtob_driver.cpp
//  Name: pmb_driver
//  Author: kamer

#include <iostream>
#include <stdlib.h>
#include <limits>
#include <math.h>
#include <algorithm>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "omp.h"
#include "common.h"
#include "pmb_precond.hpp"

using namespace std;

template<class Func>
void pmb_driver(opt_prec_t* x_0, Options& pars, Output*& output,  Func& func) {

        int n = func.n;
        output = new Output(n, pars.maxiter);

        opt_prec_t* x = new opt_prec_t[n];
        opt_prec_t* g = new opt_prec_t[n];
        opt_prec_t* gt = new opt_prec_t[n];
        opt_prec_t* g_old = new opt_prec_t[n];
        opt_prec_t* s = new opt_prec_t[n];
        opt_prec_t* y = new opt_prec_t[n];
        opt_prec_t* xt = new opt_prec_t[n];
        opt_prec_t* S = new opt_prec_t[n * pars.M];
        opt_prec_t* Y = new opt_prec_t[n * pars.M];
        opt_prec_t* YS = new opt_prec_t[pars.M];
        opt_prec_t* al = new opt_prec_t [pars.M];
        opt_prec_t* be = new opt_prec_t [pars.M];
        int* ind = new int[pars.M];

        opt_prec_t ys = 1.0, yy = 1.0, ss = 1.0, yg = 1.0, sg = 1.0, gg = 1.0, cg, f,
                   fold = (std::numeric_limits<opt_prec_t>::max)(),
                   ft, Hdiag;
        int evals = 0, iter = 0, mem_end = 0, mem_start = 1;

        double startTime = omp_get_wtime();

#pragma omp parallel for schedule(static)
        for(int k = 0; k < n; k++) {
                x[k] = x_0[k];
                g[k] = gt[k] = s[k] = y[k] = xt[k] = g_old[k] = 0;
        }

        for(int i = 0; i < pars.M; i++) {
                opt_prec_t* dest_S = S + i*n;
                opt_prec_t* dest_Y = Y + i*n;
#pragma omp parallel for schedule(static)
                for(int k = 0; k < n; k++) {
                        dest_S[k] = dest_Y[k] = 0;
                }
        }
        memset(YS, 0, (pars.M *sizeof(opt_prec_t)));

        func(x, f, g);
        evals++;

        while (true) {
                opt_prec_t ngf = fabs(g[0]);
                for (int i = 1; i < n; i++) {
                        ngf = max(ngf, fabs(g[i]));
                }

                if (ngf < pars.gtol) {
                        output->exit = 1;
                        if(pars.message) {
                                printf("First order condition is within gtol\n");
                        }
                        break;
                }

                if(iter >= pars.maxiter) {
                        output->exit = -1;
                        if(pars.message) {
                                printf("Maximum number of iterations (maxiter) is reached\n");
                        }
                        break;
                }

                if(evals >= pars.maxfcalls) {
                        output->exit = -2;
                        if(pars.message) {
                                printf("Maximum number of function calss (maxfcalls) is reached\n");
                        }
                        break;
                }

                if(omp_get_wtime() - startTime > pars.maxtime) {
                        output->exit = -3;
                        if(pars.message) {
                                printf("Maximum time limit (maxtime) is reached\n");
                        }
                        break;
                }

                if((fold - f)/max(max(fabs(fold), fabs(f)), 1.0f) < pars.ftol) {
                        output->exit = -4;
                        if(pars.message) {
                                printf("Function value decreases less than ftol");
                        }
                        break;
                }

                if(pars.history) {
                        output->fhist[iter] = f;
                        output->nghist[iter] = ngf;
                }

                if (pars.display) {
                        printf("Iter: %d ===> f = %f \t norm(g) = %f\n", iter+1, f, ngf);
                }

                if(iter > 1) {
                        precond(s, y, g, g_old,
                                Hdiag, mem_start, mem_end, ind,
                                S, Y, YS, al, be, n, pars.M, iter);
                } else {
#pragma omp parallel for schedule(static)
                        for(int k = 0; k < n; k++) {
                                s[k] = -g[k]/ngf;
                        }
                }


#pragma omp parallel for schedule(static)
                for(int k = 0; k < n; k++) {
                        g_old[k] = g[k];
                }

                int initer = 0;
                while (initer < pars.maxinneriter) {
#pragma omp parallel for schedule(static)
                        for (int k = 0; k < n; k++) {
                                xt[k] = s[k] + x[k];
                        }

                        func(xt, ft, gt);
                        evals++;
                        ys = yg = sg = yy = ss = gg = 0;
#pragma omp parallel for schedule(static) reduction(+:ys,yg,sg,yy,ss,gg)
                        for (int k = 0; k < n; k++) {
                                opt_prec_t yv =  gt[k] - g[k];
                                opt_prec_t sv = s[k];
                                opt_prec_t gv = g[k];
                                y[k] = yv;
                                ys += yv * sv;  yg += yv * gv;  sg += sv * gv;  yy += yv * yv;  ss += sv * sv;  gg += gv * gv;
                        }

                        if (f - ft > -1.0e-4 * sg) {
                                f = ft;
#pragma omp parallel for schedule(static)
                                for(int k = 0; k < n; k++) {
                                        x[k] = xt[k];
                                        g[k] = gt[k];
                                }
                                break;
                        }

                        opt_prec_t eta, eta1, eta2;
                        opt_prec_t fdiff = fabs(f - ft);
                        if (fabs(sg) > 1.0e-8) {
                                eta1 = fdiff/fabs(sg);
                        } else {
                                eta1 = 1.0;
                        }
                        opt_prec_t sgt = ys + sg;
                        if (fabs(sgt) > 1.0e-8) {
                                eta2 = fdiff/fabs(sgt);
                        } else {
                                eta2 = 1.0;
                        }
                        if (-sgt < sg) {
                                eta = max(eta1, eta2)/(eta1 + eta2);
                        } else {
                                eta = min(eta1, eta2)/(eta1 + eta2);
                        }


                        opt_prec_t sigma = 0.5 * (sqrt(ss) * (sqrt(yy) + 1.0 / eta * sqrt(gg)) - ys);
                        opt_prec_t theta = pow((ys + 2.0 * sigma), 2.0) - (ss * yy);

                        cg = -ss / (2.0 * sigma);
                        opt_prec_t cs = cg / theta * (-(ys + 2.0 * sigma) * yg + (yy * sg));
                        opt_prec_t cy = cg / theta * (-(ys + 2.0 * sigma) * sg + (ss * yg));

#pragma omp parallel for schedule(static)
                        for (int k = 0; k < n; k++) {
                                s[k] = g[k] * cg + s[k] * cs + y[k] * cy;
                        }
                        initer++;
                }
                output->nmbs += initer;

                if (initer >= pars.maxinneriter) {
                        output->exit = 0;
                        if (pars.message) {
                                printf("Maximum number of inner iterations (maxiniter) is reached\n");
                        }
                        break;
                }
                iter++;
        }

        output->time = omp_get_wtime() - startTime;
        for (int k = 0; k < n; k++) {
                output->g[k] = g[k];
                output->x[k] = x[k];
        }
        output->fval = f;
        output->niter = iter;
        output->fcalls = evals;

        delete[] x;
        delete[] g;
        delete[] gt;
        delete[] g_old;
        delete[] s;
        delete[] y;
        delete[] xt;
        delete[] S;
        delete[] Y;
        delete[] YS;
        delete[] al;
        delete[] be;
        delete[] ind;
}
