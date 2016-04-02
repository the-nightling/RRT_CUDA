/*
 * RRT.cuh
 *
 *  Created on: 28 Feb 2016
 *      Author: thenightling
 */

#ifndef RRT_CUH_
#define RRT_CUH_

////////////////////////////////////////////
// INCLUDES
////////////////////////////////////////////

// includes from CUDA
#include <curand.h>
#include <curand_kernel.h>

////////////////////////////////////////////
// MACROS
////////////////////////////////////////////

// grid macros
#define ANG_POS_MIN -M_PI		// minimum angular position
#define ANG_POS_MAX M_PI		// maximum angular position
#define ANG_VEL_MIN -8.0		// minimum angular velocity
#define ANG_VEL_MAX 8.0		// maximum angular velocity
#define GRID_X 42
#define GRID_Y 6
#define DELTA_X (ANG_POS_MAX-ANG_POS_MIN)/(2*GRID_X + 4)
#define DELTA_Y (ANG_VEL_MAX-ANG_VEL_MIN)/(2*GRID_Y*NUM_BLOCKS + 4)
#define FIRST_X0_X 4*DELTA_X + ANG_POS_MIN
#define FIRST_X0_Y 4*DELTA_Y + ANG_VEL_MIN

// RRT algorithm macros
#define NUM_OF_ITERATIONS 300
#define LENGTH_OF_SOLN_PATH 20
#define NUM_OF_GOAL_STATES 8

// thread macros
#define NUM_BLOCKS 8
#define NUM_THREADS GRID_X * GRID_Y
#define NUM_RESULTS_PER_THREAD GRID_X * GRID_Y * NUM_BLOCKS
#define NUM_RESULTS_PER_THREAD_2 2 * LENGTH_OF_SOLN_PATH * NUM_OF_GOAL_STATES

// for graph shortest path finding algorithm
#define NUM_NODES 2016
#define MAX_COST 9999

////////////////////////////////////////////
// FUNCTION HEADERS
////////////////////////////////////////////
__device__ void euclidianDistSquare(double* A, double B[][2], int lengthOfB,
		double* listOfDistSq);
__device__ int findMin(double array[], int lengthOfArray);
__device__ void pendulumDynamics(double* x, double u, double* xd);


////////////////////////////////////////////
// KERNEL HEADERS
////////////////////////////////////////////
__global__ void RNG_setup_kernel(curandState *state);
__global__ void init_adj_matrix_kernel(int * adjacency_matrix);
__global__ void RRT_kernel(curandState *my_curandstate, int *adjacency_matrix, double * path_solutions, double * control_solutions, double * tmp);



#endif /* RRT_CUH_ */
