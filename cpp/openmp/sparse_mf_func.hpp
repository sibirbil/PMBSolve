#ifndef SMF_F_H
#define SMF_F_H

#include "common.h"
#include <iostream>

using namespace std;

#define data_coord_t int 
#define data_point_t int

#ifdef SP
#define data_val_t float
#endif
#ifdef DP
#define data_val_t double
#endif

class SparseMatrix {
public:  
  data_coord_t nrows;
  data_coord_t ncols;
  data_point_t nnnz;
  
  int n;
  int ldim;
  double scfac;
  
  data_point_t* crs_ptrs;
  data_coord_t* crs_colids;
  
  data_point_t* ccs_ptrs;
  data_coord_t* ccs_rowids;
  data_point_t* ccs_translator;
  
  data_val_t* crs_values;

  opt_prec_t* div_term;
    
  SparseMatrix (char* fileName, int _ldim) : ldim(_ldim) {
    FILE *mat_file;
    
    if((mat_file = fopen(fileName, "r")) == NULL) {
      cout << "Cannot read file\n";
      exit(1);
    }
    
    fscanf(mat_file, "%d %d %d", &nrows, &ncols, &nnnz);
    cout << "rows/cols/ratings: " << nrows << " " << ncols << " " << nnnz << endl;

    data_coord_t* I = new data_coord_t[nnnz];
    data_coord_t* J = new data_coord_t[nnnz];
    data_val_t* V = new data_val_t[nnnz];
    
    for (int i = 0; i < nnnz; i++) {
      double temp_i, temp_j, temp_val;
      fscanf(mat_file, "%lf %lf %lf", &temp_i, &temp_j, &temp_val);
      I[i] = (data_coord_t)(temp_i - 1);
      J[i] = (data_coord_t)(temp_j - 1);
      V[i] = (data_val_t)(temp_val);
    }
    fclose(mat_file);

    cout << "File is read. Allocating memory for matrix " << endl;

    n = (ldim * nrows) + (ldim * ncols);
    scfac = ((double)(1.0f)) / nnnz;

    crs_ptrs = new data_point_t[nrows + 1];
    crs_colids = new data_coord_t[nnnz];
    crs_values = new data_val_t[nnnz];
  
    ccs_ptrs = new data_point_t[ncols + 1];
    ccs_rowids = new data_coord_t[nnnz];
    ccs_translator = new data_point_t[nnnz];

    div_term = new opt_prec_t[nnnz];

    cout << "Memory is allocated; creating crs and ccs" << endl;

    memset(crs_ptrs, 0, (nrows + 1) * sizeof(data_point_t));

    // in each cell of the array crs_ptrs, we have number of elements in that row in the matrix, but crs_ptrs is one cell ahead.
    for(data_point_t i = 0; i < nnnz; i++) {
      crs_ptrs[I[i]+1]++; //increase the counts
    }

    //Now we have cumulative ordering of crs_ptrs.
    for(data_coord_t i = 1; i <= nrows; i++) {
      crs_ptrs[i] += crs_ptrs[i-1]; //prefix sum
    }

    //here we set crs_colids such that for each element, it holds the related column of that element
    for(data_point_t i = 0; i < nnnz; i++) {
      data_coord_t rowid = I[i];
      data_point_t index = crs_ptrs[rowid];

      crs_colids[index] = J[i];
      crs_values[index] = V[i];

      crs_ptrs[rowid] = crs_ptrs[rowid] + 1;
    }

    //forward shift and assign for fixing the ptrs array
    for(data_coord_t i = nrows; i > 0; i--) {
      crs_ptrs[i] = crs_ptrs[i-1];
    }
    crs_ptrs[0] = 0;

    cout << "\tcrs is created" << endl;

    memset(ccs_ptrs, 0, (ncols + 1) * sizeof(data_point_t));

    for(data_point_t i = 0; i < nnnz; i++) {
      ccs_ptrs[J[i]+1]++;
    }

    for(data_coord_t i = 1; i <= ncols; i++) {
      ccs_ptrs[i] += ccs_ptrs[i-1];
    }

    for(data_coord_t i = 0; i < nrows; i++) {
      for(data_point_t ptr = crs_ptrs[i]; ptr < crs_ptrs[i+1]; ptr++) {
	data_coord_t colid  = crs_colids[ptr];

	data_point_t index = ccs_ptrs[colid];
	ccs_rowids[index] = i;
	ccs_translator[index] = ptr;

	ccs_ptrs[colid] = ccs_ptrs[colid] + 1;
      }
    }

    for(data_coord_t i = ncols; i > 0; i--) {
      ccs_ptrs[i] = ccs_ptrs[i-1];
    }
    ccs_ptrs[0] = 0;

    cout << "\tccs is created" << endl;

    delete[] I;
    delete[] J;
    delete[] V;
  }
    
  opt_prec_t dTerm_Z1update(opt_prec_t* Z1, opt_prec_t* Z1update, opt_prec_t* Z2) {
    opt_prec_t scfac = ((double)(1.0f)) / nnnz;
    
    opt_prec_t totalcost = 0;  
#pragma omp parallel
    {
      opt_prec_t diff_1, diff_2, diff_3, diff_4;
      opt_prec_t temp1, temp2, temp3, temp4;
      const opt_prec_t *myZ2_1, *myZ2_2, *myZ2_3, *myZ2_4;

#pragma omp for schedule(runtime) reduction(+:totalcost)
      for (data_coord_t i = 0; i < nrows; i++) {
	const opt_prec_t *myZ1 = Z1 + (i * ldim);
	opt_prec_t *myZ1U = Z1update + (i * ldim);
      
	memset(myZ1U, 0, sizeof(opt_prec_t) * ldim);
      
	data_point_t start = crs_ptrs[i];
	data_point_t end = crs_ptrs[i + 1];

	data_point_t ptr;      
	for (ptr = start; ptr < end - 3; ptr += 4) {
	  myZ2_1  = Z2 + (crs_colids[ptr] * ldim);
	  myZ2_2  = Z2 + (crs_colids[ptr + 1] * ldim);
	  myZ2_3  = Z2 + (crs_colids[ptr + 2] * ldim);
	  myZ2_4  = Z2 + (crs_colids[ptr + 3] * ldim);

	  diff_1 = -(crs_values[ptr]);
	  diff_2 = -(crs_values[ptr + 1]);
	  diff_3 = -(crs_values[ptr + 2]);
	  diff_4 = -(crs_values[ptr + 3]);

	  int k = 0;
	  for (; k < ldim - 3; k += 4) {
	    temp1 = myZ1[k];
	    temp2 = myZ1[k+1];
	    temp3 = myZ1[k+2];
	    temp4 = myZ1[k+3];

	    diff_1 += (temp1 * myZ2_1[k] + temp2 * myZ2_1[k+1] + temp3 * myZ2_1[k+2] + temp4 * myZ2_1[k+3]);
	    diff_2 += (temp1 * myZ2_2[k] + temp2 * myZ2_2[k+1] + temp3 * myZ2_2[k+2] + temp4 * myZ2_2[k+3]);
	    diff_3 += (temp1 * myZ2_3[k] + temp2 * myZ2_3[k+1] + temp3 * myZ2_3[k+2] + temp4 * myZ2_3[k+3]);
	    diff_4 += (temp1 * myZ2_4[k] + temp2 * myZ2_4[k+1] + temp3 * myZ2_4[k+2] + temp4 * myZ2_4[k+3]);
	  }

	  for (; k < ldim; k++) {
	    opt_prec_t temp = myZ1[k];
	    diff_1 += temp * myZ2_1[k];
	    diff_2 += temp * myZ2_2[k];
	    diff_3 += temp * myZ2_3[k];
	    diff_4 += temp * myZ2_4[k];
	  }

	  div_term[ptr] = diff_1;
	  div_term[ptr + 1] = diff_2;
	  div_term[ptr + 2] = diff_3;
	  div_term[ptr + 3] = diff_4;
	
	  for (int k = 0; k < ldim; k++) {
	    myZ1U[k] += (myZ2_1[k] * diff_1 +  myZ2_2[k] * diff_2 + myZ2_3[k] * diff_3 + myZ2_4[k] * diff_4) * scfac;
	  }
	
	  totalcost += diff_1 * diff_1 + diff_2 * diff_2 + diff_3 * diff_3 + diff_4 * diff_4;
	}
      
	for (; ptr < end; ptr++) {
	  opt_prec_t *myZ2_1  = Z2 + (crs_colids[ptr] * ldim);
	
	  diff_1 = -(crs_values[ptr]);
	  for (int k = 0; k < ldim; k++) {
	    diff_1 += myZ1[k] * myZ2_1[k];
	  }
	
	  div_term[ptr] = diff_1;
	  opt_prec_t coef = diff_1 * scfac;
	  for (int k = 0; k < ldim; k++) {
	    myZ1U[k] += myZ2_1[k] * coef;
	  }
	  totalcost += diff_1 * diff_1;
	}
      }
    }
    return totalcost * 0.5 * scfac;
  }

  void Z2update(opt_prec_t* Z1, opt_prec_t* Z2update) {
#pragma omp parallel for schedule(runtime)
    for (data_coord_t j = 0; j < ncols; j++) {
      opt_prec_t *  myZ2U = Z2update + (j * ldim);
      memset (myZ2U, 0, sizeof(opt_prec_t) * ldim);
    
      data_point_t start = ccs_ptrs[j];
      data_point_t end = ccs_ptrs[j + 1];
      data_point_t ptr;
      for (ptr = start; ptr < end - 3; ptr += 4) {
	const opt_prec_t *myZ1_1  = Z1 + (ccs_rowids[ptr] * ldim);
	const opt_prec_t *myZ1_2  = Z1 + (ccs_rowids[ptr + 1] * ldim);
	const opt_prec_t *myZ1_3  = Z1 + (ccs_rowids[ptr + 2] * ldim);
	const opt_prec_t *myZ1_4  = Z1 + (ccs_rowids[ptr + 3] * ldim);
      
	const opt_prec_t cv_1 = div_term[ccs_translator[ptr]];
	const opt_prec_t cv_2 = div_term[ccs_translator[ptr + 1]];
	const opt_prec_t cv_3 = div_term[ccs_translator[ptr + 2]];
	const opt_prec_t cv_4 = div_term[ccs_translator[ptr + 3]];

	for (int k = 0; k < ldim; k++) {
	  myZ2U[k] += (myZ1_1[k] * cv_1 + myZ1_2[k] * cv_2 + myZ1_3[k] * cv_3 + myZ1_4[k] * cv_4) * scfac;
	}
      }

      for (; ptr < end; ptr++) {
	opt_prec_t *myZ1 = Z1 + (ccs_rowids[ptr] * ldim);
      
	opt_prec_t cv = div_term[ccs_translator[ptr]] * scfac;
	for (int k = 0; k < ldim; k++) {
	  myZ2U[k] += myZ1[k] * cv;
	}
      }
    }
  }

  //x is the solution, g is the gradient
  void operator()(opt_prec_t* x, opt_prec_t& f, opt_prec_t* g) {

    opt_prec_t * Z1 = &(x[0]);
    opt_prec_t * Z2 = &(x[nrows * ldim]);

    //divide g into G1 and G2
    opt_prec_t * G1 = &(g[0]);
    opt_prec_t * G2 = &(g[nrows * ldim]);
  
    //Compute the gradient and the objective function
    f = dTerm_Z1update(Z1, G1, Z2);

    Z2update(Z1, G2);
  }

  ~SparseMatrix() {
    delete[] crs_ptrs;
    delete[] crs_colids;
    delete[] crs_values;
    
    delete[] ccs_ptrs;
    delete[] ccs_rowids;
    delete[] ccs_translator;

    delete[] div_term;
  }
};
#endif
