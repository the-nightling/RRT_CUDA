////////////////////////////////////////////
// INCLUDES
////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "RRT.cuh"


////////////////////////////////////////////
// CUDA KERNELS
////////////////////////////////////////////

/*
 * Initializes CUDA RNG
 */
__global__ void RNG_setup_kernel(curandState *state) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;		// thread id
	curand_init(1234, idx, 0, &state[idx]);	// using seed 1234 (change to time at a later stage)
}

/*
 * Initializes adjacent matrix
 */
__global__ void init_adj_matrix_kernel(int * adjacency_matrix){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for(int i=0; i < NUM_THREADS*NUM_BLOCKS; i++){
		int index = idx * NUM_THREADS*NUM_BLOCKS + i;
		if(index % (NUM_THREADS*NUM_BLOCKS + 1) == 0){
			adjacency_matrix[index] = 0;
		}else{
			adjacency_matrix[index] = 9999;
			//adjacency_matrix[index] = 0;
		}
	}
}

/*
 * Main kernel; Contains RRT algorithm
 */
__global__ void RRT_kernel(curandState *my_curandstate, int *adjacency_matrix,
		double * result2) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;		// thread id

	// computing initial state
	double x0[] = { FIRST_X0_X, FIRST_X0_Y }; // initial state; angle position measured from x-axis
	x0[0] += (idx % GRID_X) * 2 * DELTA_X;
	x0[1] += (idx / GRID_X) * 2 * DELTA_Y;

	// TODO: automate goal placement around initial state
	double xG[8][2] =
	{ { x0[0] - 2 * DELTA_X, x0[1] + 2 * DELTA_Y },
			{ x0[0], x0[1] + 2 * DELTA_Y },
			{ x0[0] + 2 * DELTA_X, x0[1] + 2 * DELTA_Y },
			{ x0[0] - 2 * DELTA_X, x0[1] },
			{ x0[0] + 2 * DELTA_X, x0[1] },
			{ x0[0] - 2 * DELTA_X, x0[1] - 2 * DELTA_Y },
			{ x0[0], x0[1] - 2 * DELTA_Y },
			{ x0[0] + 2 * DELTA_X, x0[1] - 2 * DELTA_Y }
	};

	double xlimits[2][2] = { { x0[0] - 3 * DELTA_X, x0[0] + 3 * DELTA_X }, {
			x0[1] - 3 * DELTA_Y, x0[1] + 3 * DELTA_Y } }; // state limits; angular position between -pi & pi rad; angular velocity between -10 & 10 rad/s
	//double xlimits[2][2] = { { x0[0] - DELTA_X, x0[0] + DELTA_X }, { x0[1] - DELTA_Y, x0[1] + DELTA_Y } }; // state limits; angular position between -pi & pi rad; angular velocity between -10 & 10 rad/s

	// control torques to be used: linspace(-5,5,20)
	//*
	double U[] = { -5.0000, -4.4737, -3.9474, -3.4211, -2.8947, -2.3684,
			-1.8421, -1.3158, -0.7895, -0.2632, 5.0000, 4.4737, 3.9474, 3.4211,
			2.8947, 2.3684, 1.8421, 1.3158, 0.7895, 0.2632 };
	//*/
	/*
	double U[] = { -1.0000, -0.8947, -0.7895, -0.6842, -0.5789, -0.4737, -0.3684, -0.2632, -0.1579, -0.0526,
			1.0000, 0.8947, 0.7895, 0.6842, 0.5789, 0.4737, 0.3684, 0.2632, 0.1579, 0.0526};
	//*/
	int lengthOfU = (int) (sizeof(U) / sizeof(U[0]));

	double dt = 0.02; // time interval between application of subsequent control torques

	// static memory allocation
	double xn[2];        // stores a state
	double xd[2];

	double G[N][2];	// stores tree
	int x, y;
	for (x = 0; x < N; x++) {	// initialize tree to initial state
		G[x][0] = x0[0];
		G[x][1] = x0[1];
	}

	//int adjMatrix[NUM_THREADS][NUM_THREADS];
	//memset(adjMatrix, 0, sizeof(int)*NUM_THREADS*NUM_THREADS);

	int P[N]; // stores index of parent state for each state in graph G
	int Ui[N]; // stores index of control actions in U (each state will use a control action value in U)
	double u_path[M]; // stores sequence of control actions (solution to problem)
	double x_path[8][M][2];
	for (y = 0; y < 8; y++) {
		for (x = 0; x < M; x++) {	// initialize tree to initial state
			x_path[y][x][0] = 0;
			x_path[y][x][1] = 0;
			u_path[x] = 0;
		}
	}
	int xbi = 0;    // stores sequence of states joining initial to goal state
	double xn_c[20][2]; // stores temporary achievable states from a particular vertex; 20 is length of U

	double dsq[N];  // stores distance square values

	int goal_index;
	int not_found[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
	int weight = 0;
	// keep growing RRT until goal found or run out of iterations
	int n;
	for (n = 1; n < N; n++) {
		// get random state
		xn[0] = curand_uniform(my_curandstate + idx)
																												* (xlimits[0][1] - xlimits[0][0]) + xlimits[0][0];
		xn[1] = curand_uniform(my_curandstate + idx)
																												* (xlimits[1][1] - xlimits[1][0]) + xlimits[1][0];

		// find distances between that state point and every vertex in RRT
		euclidianDistSquare(xn, G, n, dsq);

		// select RRT vertex closest to the state point
		int minIndex = findMin(dsq, n);

		// from the closest RRT vertex, compute all the states that can be reached,
		// given the pendulum dynamics and available torques
		int ui;
		for (ui = 0; ui < lengthOfU; ui++) {
			pendulumDynamics(G[minIndex], U[ui], xd);
			xn_c[ui][0] = G[minIndex][0] + dt * xd[0];
			xn_c[ui][1] = G[minIndex][1] + dt * xd[1];
		}

		// select the closest reachable state point
		euclidianDistSquare(xn, xn_c, lengthOfU, dsq);
		ui = findMin(dsq, lengthOfU);
		xn[0] = xn_c[ui][0];
		xn[1] = xn_c[ui][1];

		// if angular position is greater than pi rads, wrap around
		if (xn[0] > M_PI || xn[0] < -M_PI)
			xn[0] = fmod((xn[0] + M_PI), (2 * M_PI)) - M_PI;

		// link reachable state point to the nearest vertex in the tree
		G[n][0] = xn[0];
		G[n][1] = xn[1];
		P[n] = minIndex;
		Ui[n] = ui;

		// if tree has grown near enough to one of the surrounding goal states
		// set that particular goal state to 'found'
		// save path from initial state to that goal state
		for (goal_index = 0; goal_index < 8; goal_index++) {
			if (not_found[goal_index] == 1
					&& (xn[0] <= xG[goal_index][0] + 0.05)
					&& (xn[0] >= xG[goal_index][0] - 0.05)) {
				if ((xn[1] <= xG[goal_index][1] + 0.25)
						&& (xn[1] >= xG[goal_index][1] - 0.25)) {

					not_found[goal_index] = 0;
					xbi = n;
					int index = 0;
					while (xbi != 0) {
						u_path[index] = U[Ui[xbi]];
						x_path[goal_index][index][0] = G[xbi][0];
						x_path[goal_index][index][1] = G[xbi][1];
						index++;

						xbi = P[xbi];
					}
				}
			}
		}
	}

	weight = 1;

	// Update adjacency matrix:
	// for each goal state surrounding an initial state,
	// if the goal state has been reached,
	// if tree is growing near border of phase space, check if tree is growing within state space limits
	// set respective flag in adjacency matrix to 1 (or to a weight)
	int k;
	for (k = 0; k < 8; k++) {
		if (not_found[k] == 0) {
			if (k == 0 && idx % GRID_X != 0) {
				if (idx + GRID_X - 1 <= NUM_THREADS * NUM_BLOCKS - 1) {
					adjacency_matrix[idx * NUM_THREADS * 8 + idx + GRID_X - 1] = weight;
				}
			} else if (k == 1) {
				if (idx + GRID_X <= NUM_THREADS * NUM_BLOCKS - 1) {
					adjacency_matrix[idx * NUM_THREADS * 8 + idx + GRID_X] = weight;
				}
			} else if (k == 2 && (idx + 1) % GRID_X != 0) {
				if (idx + GRID_X + 1 <= NUM_THREADS * NUM_BLOCKS - 1) {
					adjacency_matrix[idx * NUM_THREADS * 8 + idx + GRID_X + 1] = weight;
				}
			} else if (k == 3 && idx % GRID_X != 0) {
				if (idx - 1 >= 0) { // don't need that line
					adjacency_matrix[idx * NUM_THREADS * 8 + idx - 1] = weight;
				}
			} else if (k == 4 && (idx + 1) % GRID_X != 0) {
				if (idx + 1 <= NUM_THREADS * NUM_BLOCKS - 1) { // don't need that line
					adjacency_matrix[idx * NUM_THREADS * 8 + idx + 1] = weight;
				}
			} else if (k == 5 && idx % GRID_X != 0) {
				if (idx - GRID_X - 1 >= 0) {
					adjacency_matrix[idx * NUM_THREADS * 8 + idx - GRID_X - 1] = weight;
				}
			} else if (k == 6) {
				if (idx - GRID_X >= 0) {
					adjacency_matrix[idx * NUM_THREADS * 8 + idx - GRID_X] = weight;
				}
			} else if (k == 7 && (idx + 1) % GRID_X != 0) {
				if (idx - GRID_X + 1 >= 0) {
					adjacency_matrix[idx * NUM_THREADS * 8 + idx - GRID_X + 1] = weight;
				}
			}
		}
	}
	//*/

	//* copy path results of algorithm to device results array
	int i, j;
	int L = 8;
	for (j = 0; j < L; j++) {
		for (i = 0; i < M; i++) {
			result2[idx * 2 * L * M + j * 2 * M + 2 * i] = x_path[j][i][0];
			result2[idx * 2 * L * M + j * 2 * M + 2 * i + 1] = x_path[j][i][1];
			if (not_found[j] == 0) {
				if (i == M - 2) {
					result2[idx * 2 * L * M + j * 2 * M + 2 * i] = x0[0];
					result2[idx * 2 * L * M + j * 2 * M + 2 * i + 1] = x0[1];
				} else if (i == M - 1) {
					result2[idx * 2 * L * M + j * 2 * M + 2 * i] = xG[j][0];
					result2[idx * 2 * L * M + j * 2 * M + 2 * i + 1] = xG[j][1];
				}
			}
		}
	}
	//*/





	/*
	 int i;
	 for (i = 0; i < NUM_RESULTS_PER_THREAD; i++)
	 result[idx * NUM_RESULTS_PER_THREAD + i] = x0[i];
	 //*/
	/*
	 result[idx * NUM_RESULTS_PER_THREAD + 0] = x0[0];
	 result[idx * NUM_RESULTS_PER_THREAD + 1] = x0[1];
	 //*/

}



////////////////////////////////////////////
// HELPER FUNCTIONS
////////////////////////////////////////////
/*
 * computes the Euclidian distances squared from point A to every point in array B
 */
__device__ void euclidianDistSquare(double* A, double B[][2], int lengthOfB,
		double* listOfDistSq) {
	int i;
	for (i = 0; i < lengthOfB; i++)
		listOfDistSq[i] = pow((B[i][0] - A[0]), 2) + pow((B[i][1] - A[1]), 2);
}

/*
 * finds the index of the minimum in an array
 */
__device__ int findMin(double array[], int lengthOfArray) {
	int minIndex = 0;

	int i;
	for (i = 0; i < lengthOfArray; i++) {
		if (array[i] < array[minIndex])
			minIndex = i;
	}

	return minIndex;
}

/*
 * Computes x_dot of the pendulum, given x and a control input u
 */
__device__ void pendulumDynamics(double* x, double u, double* xd) {
	// pendulum parameters
	int m = 1;                  // mass
	int l = 1;                  // length of pendulum link
	int I = m * l * l;              // moment of inertia
	double g = 9.8;              // acceleration due to gravity
	double b = 0.1;              // damping factor

	xd[0] = x[1];
	xd[1] = (u - m * g * l * sin((M_PI / 2) - x[0]) - b * x[1]) / I;
}
