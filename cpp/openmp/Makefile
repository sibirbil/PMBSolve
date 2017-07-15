all:
	g++ sparse_mf_solver.cpp -O3 -std=c++11 -fopenmp -D"DP" -o pmb_mf_dp
	g++ sparse_mf_solver.cpp -O3 -std=c++11 -fopenmp -D"SP" -o pmb_mf_sp
	g++ rosenbrock_solver.cpp -O3 -std=c++11 -fopenmp -D"DP" -o rose_dp
	g++ rosenbrock_solver.cpp -O3 -std=c++11 -fopenmp -D"SP" -o rose_sp