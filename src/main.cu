////////////////////////////////////////////
// INCLUDES
////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "floyd_warshall_algo.cuh"
#include "RRT.cuh"
using namespace std;

////////////////////////////////////////////
// MAIN
////////////////////////////////////////////
int main(void) {
	//*
	curandState *device_state;
	cudaMalloc(&device_state, NUM_THREADS * NUM_BLOCKS * sizeof(curandState)); // allocate device memory to store RNG states

	int *device_adjacency_matrix, *host_adjacency_matrix;	// pointers to results of computation
	double *device_path_solns, *host_path_solns;
	double *device_control_solns, *host_control_solns;

	host_adjacency_matrix = (int *) malloc(
			NUM_RESULTS_PER_THREAD * NUM_THREADS * NUM_BLOCKS * sizeof(int));// allocate host memory to store results
	cudaMalloc(&device_adjacency_matrix,
			NUM_RESULTS_PER_THREAD * NUM_THREADS * NUM_BLOCKS * sizeof(int));// allocate device memory to store results
	init_adj_matrix_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(device_adjacency_matrix);

	host_path_solns = (double *) malloc(
			NUM_RESULTS_PER_THREAD_2 * NUM_THREADS * NUM_BLOCKS * sizeof(double));// allocate host memory to store results
	cudaMalloc(&device_path_solns,
			NUM_RESULTS_PER_THREAD_2 * NUM_THREADS * NUM_BLOCKS * sizeof(double));// allocate device memory to store results
	cudaMemset(device_path_solns, 0,
			NUM_RESULTS_PER_THREAD_2 * NUM_THREADS * NUM_BLOCKS * sizeof(double));// initialize device results array to 0

	host_control_solns = (double *) malloc(
			NUM_OF_GOAL_STATES* LENGTH_OF_SOLN_PATH * NUM_THREADS * NUM_BLOCKS * sizeof(double));// allocate host memory to store results
	cudaMalloc(&device_control_solns,
			NUM_OF_GOAL_STATES* LENGTH_OF_SOLN_PATH * NUM_THREADS * NUM_BLOCKS * sizeof(double));// allocate device memory to store results
	cudaMemset(device_control_solns, 0,
			NUM_OF_GOAL_STATES* LENGTH_OF_SOLN_PATH * NUM_THREADS * NUM_BLOCKS * sizeof(double));// initialize device results array to 0

	RNG_setup_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(device_state);	// run GPU to initialize RNG

	RRT_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(device_state, device_adjacency_matrix,
			device_path_solns, device_control_solns);	// run main GPU algorithm

	cudaMemcpy(host_adjacency_matrix, device_adjacency_matrix,
			NUM_RESULTS_PER_THREAD * NUM_THREADS * NUM_BLOCKS * sizeof(int),	// copy results from device to host
			cudaMemcpyDeviceToHost);

	cudaMemcpy(host_path_solns, device_path_solns,
			NUM_RESULTS_PER_THREAD_2 * NUM_THREADS * NUM_BLOCKS * sizeof(double),// copy results from device to host
			cudaMemcpyDeviceToHost);

	cudaMemcpy(host_control_solns, device_control_solns,
			NUM_OF_GOAL_STATES* LENGTH_OF_SOLN_PATH * NUM_THREADS * NUM_BLOCKS * sizeof(double),// copy results from device to host
			cudaMemcpyDeviceToHost);


	int *D_Gpath=(int *)malloc(NUM_RESULTS_PER_THREAD * NUM_THREADS * NUM_BLOCKS * sizeof(int));
	int *roots = (int *)calloc(30, sizeof(int));

	for(int i=0;i<RANDOM_GSIZE*RANDOM_GSIZE;i++){
		D_Gpath[i]=-1;//set to all negative ones for use in path construction
	}

	//call host function which will copy all info to device and run CUDA kernels
	_GPU_Floyd(host_adjacency_matrix,D_Gpath,RANDOM_GSIZE);

	_get_full_paths(host_adjacency_matrix,D_Gpath,RANDOM_GSIZE, roots);//find out exact step-by-step shortest paths between vertices(if such a path exists)

	//*/
	// output results
	for(int i=0; i < 30;i++){
		printf("%d\n",roots[i]);
	}
	/*
	 printf("Bin:    Count: \n");
	 for (int i = 0; i < N * NUM_RESULTS_PER_THREAD * NUM_THREADS; i++)
	 printf("%d    %f\n", i, host_result[i]);
	 //*/

	/*
	int roots_on_path[50];
	for(int x = 0; x < 50; x++)
		roots_on_path[x] = 0;

	int number_of_path_segments;
	for(int x = 0; x < 50; x++)
	{
		if(roots_on_path[x] == 0)
		{
			number_of_path_segments = x;
			break;
		}
		printf(" %d ", roots_on_path[x]);
	}

	printf("\n");
	printf("Number of path segments: %d\n", number_of_path_segments);
	double individual_tree_roots[50][2];
	double individual_tree_goals[50*8][2];
	int root_index, goal_index;
	for(int i = 0; i < number_of_path_segments; i++){
		for(int j = 0; j < 8; j++)
		{
			goal_index = (roots_on_path[i]*2*8*20) - (j*2*20) - 1;
			root_index = goal_index - 2;
			individual_tree_roots[i][0] = host_path_solns[root_index-1];
			individual_tree_roots[i][1] = host_path_solns[root_index];

			individual_tree_goals[j][0] = host_path_solns[goal_index-1];
			individual_tree_goals[j][1] = host_path_solns[goal_index];

			/*
			printf("%f\n", host_path_solns[root_index-1]);
			printf("%f\n", host_path_solns[root_index]);
			printf("%f\n", host_path_solns[goal_index-1]);
			printf("%f\n\n", host_path_solns[goal_index]);
			//*
		}
	}

	number_of_path_segments = 1;
	for(int i = 0; i < number_of_path_segments; i++)
	{
		printf("%f\n", individual_tree_roots[i][0]);
		printf("%f\n", individual_tree_roots[i][1]);
		for(int j = 0; j < 8; j++)
		{
			printf("%f\n", individual_tree_goals[i*8 + j][0]);
			printf("%f\n", individual_tree_goals[i*8 + j][1]);
		}
	}
	//*/
	//*
	FILE *dataFile = fopen("data.txt", "w");
	FILE *dataFile2 = fopen("data2.txt", "w");
	FILE *dataFile3 = fopen("data3.txt", "w");

	for (int i = 0; i < NUM_THREADS * NUM_BLOCKS; i++) {
		for (int j = 0; j < NUM_RESULTS_PER_THREAD; j++) {
			fprintf(dataFile, "%d ", host_adjacency_matrix[i * NUM_THREADS * NUM_BLOCKS + j]);
		}
		fprintf(dataFile, "\n");
	}

	for (int i = 0; i < NUM_RESULTS_PER_THREAD_2 * NUM_THREADS * NUM_BLOCKS; i++)
		fprintf(dataFile2, "%f,\n", host_path_solns[i]);

	for (int i = 0; i < NUM_OF_GOAL_STATES* LENGTH_OF_SOLN_PATH * NUM_THREADS * NUM_BLOCKS; i++)
		fprintf(dataFile3, "%f,\n", host_control_solns[i]);

	fclose(dataFile);
	fclose(dataFile2);
	fclose(dataFile3);
	//*/

	free(host_path_solns);
	free(host_control_solns);
	free(host_adjacency_matrix);
	free(D_Gpath);
	cudaFree(device_path_solns);
	cudaFree(device_control_solns);
	cudaFree(device_adjacency_matrix);
	cudaFree(device_state);

	return 0;
}
