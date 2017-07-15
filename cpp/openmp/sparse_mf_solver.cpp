#include <iostream>
#include <random>
#include "pmb_driver.hpp"
#include "sparse_mf_func.hpp"

using namespace std;

int main(int argc, char * argv[]) {
  if(argc != 3) {
    cout << "Usage: executable filename latent_dimension" << endl;
    return 0;
  }

  //load data
  char fileName[80];
  strcpy (fileName, argv[1]);
  int ldim = atoi(argv[2]);

  SparseMatrix func(fileName, ldim);
  //this is the place we store errors; global scope

  //initial solution
  std::random_device r;
  std::default_random_engine eng(r());
  std::uniform_int_distribution<int> uniform_dist(1, 5);
  std::uniform_real_distribution<> unif(1,5);
  opt_prec_t* x_0 = new opt_prec_t[func.n];
  for (int i = 0; i < func.n; i++) {
    x_0[i] = sqrt(unif(eng)/ldim);
  }

  //options
  Options options;
  options.gtol = 1e-05;
  options.maxiter = 500;
  options.maxinneriter = 100;
  options.M = 5;
  options.display = true;
  options.history = true;

  double tt = omp_get_wtime();

  Output* output;
  pmb_driver<SparseMatrix>(x_0, options, output, func);

  double timeSpent = omp_get_wtime() - tt;

  cout << "Exit: " << output->exit << endl;
  cout << "Fval: " << output->fval << endl;

  opt_prec_t ngf = fabs(output->g[0]);
  for (int i = 1; i < func.n; i++) {
    ngf = max(ngf, fabs(output->g[i]));
  }

  cout << "Norm: " << ngf << endl;
  cout << "Iterations: " << output->niter << endl;
  cout << "Evaluations: " << output->fcalls << endl;
  cout << "Models built: " << output->nmbs << endl;
  cout << "Time Spent in seconds: " << output->time << endl;
  cout << "RMSE: " << pow(((output->fval * 2) ), 0.5) << endl;
  cout << "Latent Dim: " << ldim << endl;

  // left factor matrix: from index 0 to index nrow*ldim-1 of output->x array
  char s[100];
  strcpy(s,fileName);
  FILE *f;
  f = fopen(strcat(s,".leftfactor"),"w");
  for (int i=0; i< (func.nrows)*ldim; i++){
	fprintf(f, "%f ", output->x[i]);
	if ((i+1)%ldim == 0){
		fprintf(f,"\n");
	}
  }
  fclose(f);
    
  strcpy(s,fileName);
  // right factor matrix: from index nrow*ldim to index (nrow+ncol)*ldim-1 of output->x array
  f = fopen(strcat(s,".rightfactor"),"w");
  for (int i=(func.nrows)*ldim; i<(func.nrows + func.ncols)*ldim; i++){
	fprintf(f, "%f ", output->x[i]);
	if ((i+1)%(func.ncols) == 0){
		fprintf(f,"\n");
	}
  }
  fclose(f);
  return 0;
}
