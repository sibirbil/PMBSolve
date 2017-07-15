#ifndef PRECOND_H
#define PRECOND_H

#include <iostream>
#include "common.h"

using namespace std;

void precond(opt_prec_t* s, opt_prec_t* y, opt_prec_t* g, opt_prec_t* g_old, 
	     opt_prec_t &Hdiag, int &mem_start, int &mem_end, int* ind,
	     opt_prec_t* S, opt_prec_t* Y, opt_prec_t* YS, opt_prec_t* al, opt_prec_t* be,
	     int n, int M, int iteration) { 
  
  opt_prec_t ys, yy, val, ma, *sTemp, *dest_S, *dest_Y;
  int ind_length, nMem = 1;
  
  ys = 0;
#pragma omp parallel for schedule(static) reduction(+:ys)
  for (int k = 0; k < n; k++) {
    opt_prec_t tmp = g[k] - g_old[k];
    y[k] = tmp; 
    ys += tmp * s[k];
  }
  
  if (ys > 1e-10) {
    if (mem_end < M) {
      mem_end = mem_end + 1;
      if (mem_start != 1) {
	if (mem_start == M)
	  mem_start = 1;
	else
	  mem_start = mem_start + 1;
      }
    } else{
      mem_start = min(2, M);
      mem_end = 1;
    }
    
    dest_S = S + (n * (mem_end-1));
    dest_Y = Y + (n * (mem_end-1));
    
    yy = 0;
#pragma omp parallel for schedule(static) reduction(+:yy)
    for(int k = 0; k < n; k++) {
      opt_prec_t tmp = y[k];
      dest_S[k] = s[k];
      dest_Y[k] = tmp;
      yy += tmp * tmp;
    }
    
    YS[mem_end - 1] = ys;
    Hdiag = ys / yy;
  }
  
  if (mem_start == 1) {
    for (int i = 0; i < mem_end; i++) {
      ind[i] = i+1; 
    }
    ind_length = mem_end;
    nMem = mem_end - mem_start + 1;
  } else {
    ind_length = M - mem_start + mem_end;
    for (int i = mem_start; i <= M; i++) {
      ind [i-mem_start] = i; 
    }
    for (int i = 1; i <= mem_end; i++) {
      ind [M - mem_start + i] = i; 
    }
    nMem = M;
  }
  
  memset (al, 0, sizeof(opt_prec_t) * nMem);
  memset (be, 0, sizeof(opt_prec_t) * nMem);
  
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
      s[i] = -g[i];
    }

    for (int j = 1; j <= ind_length; j++) {
      int i = ind [ind_length - j] - 1;
      sTemp = S + i*n;
      val = 0;
#pragma omp parallel for schedule(static) reduction(+:val)
      for(int k = 0; k < n; k++) {
	val += sTemp[k] * s[k];
      }
      al[i] =  val / YS[i];
      
      sTemp = Y + i*n;
      ma = al[i];
#pragma omp parallel for schedule(static)
      for (int k = 0; k < n; k++) {
	s[k] -= sTemp[k] * ma;
      }
    }
    
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
      s[i] = Hdiag * s[i];
    }
    
    for (int j = 0; j < ind_length; j++) {
      int i = ind[j] - 1;
      sTemp = Y + i*n;
      
      val = 0;
#pragma omp parallel for schedule(static) reduction(+:val)
      for(int k = 0; k < n; k++) {
	val += sTemp[k] * s[k];
      }
      be [i] = val / YS[i];
      val = al[i] - be[i];
      sTemp = S + i*n;
      
#pragma omp parallel for schedule(static)
      for (int k = 0; k < n; k++) {
	s[k] += sTemp[k] * val;
      }
    }
}
#endif

