////////////////////////////////////////////
// INCLUDES
////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <vector>
#include "floyd_warshall_algo.cuh"
#include "RRT.cuh"
using namespace std;

////////////////////////////////////////////
// MAIN
////////////////////////////////////////////
int main(void) {

	// Allocate memory for Random Number Generator kernel
	curandState *device_state;
	cudaMalloc(&device_state, NUM_THREADS * NUM_BLOCKS * sizeof(curandState)); // allocate device memory to store RNG states

	// Initialize variables to store results on PC (host) and GPU (device) (i.e.: adjacency matrix, paths & control actions)
	int *device_adjacency_matrix, *host_adjacency_matrix;
	double *device_path_solns, *host_path_solns;
	double *device_control_solns, *host_control_solns;

	// Used to get the start state for each thread
	//double *dev_start_state_0, *host_start_state_0;
	//host_start_state_0 = (double *) malloc(2 * 2016 * sizeof(double));
	//cudaMalloc(&dev_start_state_0, 2 * 2016 * sizeof(double));

	// Allocate host and device memory to store adjacency matrix
	host_adjacency_matrix = (int *) malloc(
			NUM_RESULTS_PER_THREAD * NUM_THREADS * NUM_BLOCKS * sizeof(int));// allocate host memory to store adjacency matrix
	cudaMalloc(&device_adjacency_matrix,
			NUM_RESULTS_PER_THREAD * NUM_THREADS * NUM_BLOCKS * sizeof(int));// allocate device memory to store adjacency matrix
	init_adj_matrix_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(device_adjacency_matrix);	// initialize adjacency matrix in parallel

	// Allocate host and device memory to store path solution for each tree
	host_path_solns = (double *) malloc(
			NUM_RESULTS_PER_THREAD_2 * NUM_THREADS * NUM_BLOCKS * sizeof(double));// allocate host memory to store path solutions
	cudaMalloc(&device_path_solns,
			NUM_RESULTS_PER_THREAD_2 * NUM_THREADS * NUM_BLOCKS * sizeof(double));// allocate device memory to store path solutions
	cudaMemset(device_path_solns, 0,
			NUM_RESULTS_PER_THREAD_2 * NUM_THREADS * NUM_BLOCKS * sizeof(double));// initialize device results array to 0

	// Allocate host and device memory to store control solutions for each tree
	host_control_solns = (double *) malloc(
			NUM_OF_GOAL_STATES* LENGTH_OF_SOLN_PATH * NUM_THREADS * NUM_BLOCKS * sizeof(double));// allocate host memory to store control solutions
	cudaMalloc(&device_control_solns,
			NUM_OF_GOAL_STATES* LENGTH_OF_SOLN_PATH * NUM_THREADS * NUM_BLOCKS * sizeof(double));// allocate device memory to store control solutions
	cudaMemset(device_control_solns, 0,
			NUM_OF_GOAL_STATES* LENGTH_OF_SOLN_PATH * NUM_THREADS * NUM_BLOCKS * sizeof(double));// initialize device results array to 0

	// Initialize RNG in parallel
	RNG_setup_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(device_state);

	// Run parallel RRT algorithm
	RRT_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(device_state, device_adjacency_matrix,
			device_path_solns, device_control_solns, NULL);

	// Copy results from GPU (device) to PC (host)
	cudaMemcpy(host_adjacency_matrix, device_adjacency_matrix,
			NUM_RESULTS_PER_THREAD * NUM_THREADS * NUM_BLOCKS * sizeof(int),	// copy adjacency matrix from device to host
			cudaMemcpyDeviceToHost);

	cudaMemcpy(host_path_solns, device_path_solns,
			NUM_RESULTS_PER_THREAD_2 * NUM_THREADS * NUM_BLOCKS * sizeof(double),// copy path solutions from device to host
			cudaMemcpyDeviceToHost);

	cudaMemcpy(host_control_solns, device_control_solns,
			NUM_OF_GOAL_STATES* LENGTH_OF_SOLN_PATH * NUM_THREADS * NUM_BLOCKS * sizeof(double),// copy control solutions from device to host
			cudaMemcpyDeviceToHost);

	// copy start state of each thread and output to console
	//cudaMemcpy(host_start_state_0, dev_start_state_0, 2 * 2016 * sizeof(double), cudaMemcpyDeviceToHost);
	//for(int i=0; i < 2016; i++)
	//		printf("idx: %d State:%f, %f\n",i,host_start_state_0[2*i],host_start_state_0[2*i+1]);


	// Initialize variables used to obtain shortest path solution using Floyd-Warshall algorithm
	int *device_roots_idx_on_path=(int *)malloc(NUM_RESULTS_PER_THREAD * NUM_THREADS * NUM_BLOCKS * sizeof(int));
	int *root_indices = (int *)calloc(30, sizeof(int));

	for(int i=0;i<RANDOM_GSIZE*RANDOM_GSIZE;i++){
		device_roots_idx_on_path[i]=-1;//set to all negative ones for use in path construction
	}

	// Call host function which will copy all info to device and run Floyd-Warshall CUDA kernels
	_GPU_Floyd(host_adjacency_matrix,device_roots_idx_on_path,RANDOM_GSIZE);

	// Find out exact step-by-step shortest paths between vertices(if such a path exists)
	_get_full_paths(host_adjacency_matrix,device_roots_idx_on_path,RANDOM_GSIZE, root_indices);

	/* output root indexes on solution path
	for(int i=0; i < 30;i++){
		printf("%d\n",root_indices[i]);
	}
	//*/

	//* Root indices on solution path (using this since I already know the solution from previous run)
	root_indices[0] = 975;
	root_indices[1] = 1017;
	root_indices[2] = 1059;
	root_indices[3] = 1101;
	root_indices[4] = 1143;
	root_indices[5] = 1185;
	root_indices[6] = 1228;
	root_indices[7] = 1271;
	root_indices[8] = 1314;
	root_indices[9] = 1315;
	root_indices[10] = 1316;
	root_indices[11] = 1317;
	root_indices[12] = 1318;
	root_indices[13] = 1319;
	root_indices[14] = 1320;
	root_indices[15] = 1321;
	root_indices[16] = 1322;
	root_indices[17] = 1323;
	root_indices[18] = 1282;
	root_indices[19] = 1241;
	root_indices[20] = 1200;
	root_indices[21] = 1159;
	root_indices[22] = 1118;
	root_indices[23] = 1077;
	root_indices[24] = 1036;
	root_indices[25] = 995;
	root_indices[26] = 1038;
	root_indices[27] = 1039;
	root_indices[28] = 997;
	//*/

	// Construct full solution path
	std::vector< std::pair<std::pair<double,double>, double> > solution_path;
	int root_index;
	int next_root_index;
	int start_of_path, start_of_next_path;
	double root_x, root_y, goal_x, goal_y;
	for(int i=0; i < 29; i++){
		root_index = root_indices[i];
		next_root_index = root_indices[i+1];
		for(int goal_index=0; goal_index < 7; goal_index++){
			start_of_path = 2*NUM_OF_GOAL_STATES*LENGTH_OF_SOLN_PATH*root_index + goal_index*2*LENGTH_OF_SOLN_PATH;
			start_of_next_path = 2*NUM_OF_GOAL_STATES*LENGTH_OF_SOLN_PATH*next_root_index + goal_index*2*LENGTH_OF_SOLN_PATH;
			root_x = host_path_solns[start_of_next_path-4+(2*LENGTH_OF_SOLN_PATH)];
			root_y = host_path_solns[start_of_next_path-3+(2*LENGTH_OF_SOLN_PATH)];
			goal_x = host_path_solns[start_of_path-2+(2*LENGTH_OF_SOLN_PATH)];
			goal_y = host_path_solns[start_of_path-1+(2*LENGTH_OF_SOLN_PATH)];

			if((root_x+root_y != 0.000000) && (fabs(root_x-goal_x) < 0.000001) && (fabs(root_y-goal_y) < 0.000001)){

				//solution_path.push_back(std::make_pair(host_path_solns[start_of_path-4+40], host_path_solns[start_of_path-3+40]));

				for(int i=0; i < LENGTH_OF_SOLN_PATH-2; i++){
					if((host_path_solns[start_of_path+i] == 0.000000) && (host_path_solns[start_of_path+i+1] == 0.000000))
						break;

					//printf("%f\n", host_path_solns[start_of_path+i]);
					solution_path.push_back(std::make_pair(std::make_pair(host_path_solns[start_of_path+(2*i)], host_path_solns[start_of_path+(2*i+1)]), host_control_solns[(start_of_path/2)+i] ));
				}
			}
		}
	}

	/* output full solution path
	for(int i=0; i < solution_path.size(); ++i){
		if(solution_path[i].first.first + solution_path[i].first.second != 0)
			std::cout << solution_path[i].first.first << " " << solution_path[i].first.second << " " << solution_path[i].second << std::endl;
	}
	//*/

	//*
	// Output results to files
	FILE *adjacency_matrix_file = fopen("adjacency_matrix.txt", "w");
	FILE *path_solutions_file = fopen("path_solutions.txt", "w");
	FILE *control_solutions_file = fopen("control_solutions.txt", "w");
	FILE *pendulum_estimates_file = fopen("pendulum_estimates.txt", "w");

	for (int i = 0; i < NUM_THREADS * NUM_BLOCKS; i++) {
		for (int j = 0; j < NUM_RESULTS_PER_THREAD; j++) {
			fprintf(adjacency_matrix_file, "%d ", host_adjacency_matrix[i * NUM_THREADS * NUM_BLOCKS + j]);
		}
		fprintf(adjacency_matrix_file, "\n");
	}

	for (int i = 0; i < NUM_RESULTS_PER_THREAD_2 * NUM_THREADS * NUM_BLOCKS; i++)
		fprintf(path_solutions_file, "%f,\n", host_path_solns[i]);

	for (int i = 0; i < NUM_OF_GOAL_STATES* LENGTH_OF_SOLN_PATH * NUM_THREADS * NUM_BLOCKS; i++)
		fprintf(control_solutions_file, "%f,\n", host_control_solns[i]);

	for(int i=0; i < solution_path.size(); ++i){
		if(solution_path[i].first.first + solution_path[i].first.second != 0)
			fprintf(pendulum_estimates_file, "%f %f %f\n", solution_path[i].first.first, solution_path[i].first.second, solution_path[i].second);
	}

	fclose(adjacency_matrix_file);
	fclose(path_solutions_file);
	fclose(control_solutions_file);
	fclose(pendulum_estimates_file);
	//*/

	// Free memory
	free(host_path_solns);
	free(host_control_solns);
	free(host_adjacency_matrix);
	free(device_roots_idx_on_path);
	cudaFree(device_path_solns);
	cudaFree(device_control_solns);
	cudaFree(device_adjacency_matrix);
	cudaFree(device_state);

	return 0;
}
