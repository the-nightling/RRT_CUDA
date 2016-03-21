#ifndef FLOYD_WARSHALL_ALGO_CUH_
#define FLOYD_WARSHALL_ALGO_CUH_

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <cmath>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//these can be altered on user depending on data set and type of operation(random test, read from file etc)
//#define BLOCK_SIZE 256
#define BLOCK_SIZE 256
#define RANGE 1
#define RANDOM_GSIZE 2016
//#define INF (1<<22)
#define INF 9999

//typedef for vector used in path reconstruction
typedef std::pair<std::pair<int,int>,int> Piii;

//forward function declarations
void _showPath(int start,int end,const std::vector<Piii> &path,const int *D,const int N);
bool _getPath(int curEdge, int nxtEdge,std::vector<Piii> &path,const int *D, const int *Dpath,const int N);
void _get_full_paths(const int *D, const int *Dpath, const int N);

//CUDA GPU kernel/functions forward declaration
__global__ void _Wake_GPU(int reps);
__global__ void _GPU_Floyd_kernel(int k, int *G,int *P, int N);
void _GPU_Floyd(int *H_G, int *H_Gpath, const int N);

//other optional utility functions
void _generateCustomGraph(int *G, int N);


#endif /* FLOYD_WARSHALL_ALGO_CUH_ */
