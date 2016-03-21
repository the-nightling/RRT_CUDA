#include "floyd_warshall_algo.cuh"

using namespace std;

void _showPath(int start,int end,const vector<Piii> &path,const int *D,const int N){
	cout<<"\nHere is the shortest cost path from "<<start<< " to "<<end<<", at a total cost of "<<D[start*N+end]<<".\n";
	for(int i=path.size()-1;i>=0;--i){
		cout<<"From "<<path[i].first.first<<" to "<<path[i].first.second<<" at a cost of "<<path[i].second<<'\n';
	}
	cout<<'\n';
}

bool _getPath(int curEdge, int nxtEdge,vector<Piii> &path,const int *D, const int *Dpath,const int N){
	int curIdx=curEdge*N+nxtEdge;
	if(D[curIdx]>=INF)return false;
	if(Dpath[curIdx]==-1){//end of backwards retracement
		path.push_back(make_pair(make_pair(curEdge,nxtEdge),D[curIdx]));
		return true;
	}else{//record last edge cost and move backwards
		path.push_back(make_pair(make_pair(Dpath[curIdx],nxtEdge),D[Dpath[curIdx]*N+nxtEdge]));
		return _getPath(curEdge,Dpath[curIdx],path,D,Dpath,N);
	}
}

void _get_full_paths(const int *D, const int *Dpath, const int N){
	int start_vertex=-1,end_vertex=-1;
	vector<Piii> path;

	//*
	path.clear();
	start_vertex = 976;
	end_vertex = 998;

	if(_getPath(start_vertex, end_vertex,path,D,Dpath,N)){
		_showPath(start_vertex,end_vertex,path,D,N);

	}else{
		cout<<"\nThere does not exist valid a path between "<<start_vertex<<" , and "<<end_vertex<<'\n';
	}
	//*/

	/*
	do{
		path.clear();
		cout<<"Enter start vertex #:";
		cin>>start_vertex;
		cout<<"Enter dest vertex(enter negative number to exit) #:";
		cin>>end_vertex;
		if(start_vertex<0 || start_vertex>=N || end_vertex<0 || end_vertex>=N)return;

		if(_getPath(start_vertex, end_vertex,path,D,Dpath,N)){
			_showPath(start_vertex,end_vertex,path,D,N);

		}else{
			cout<<"\nThere does not exist valid a path between "<<start_vertex<<" , and "<<end_vertex<<'\n';

		}
	}while(1);
	//*/
}

__global__ void _Wake_GPU(int reps){
	int idx=blockIdx.x*blockDim.x + threadIdx.x;
	if(idx>=reps)return;
}

__global__ void _GPU_Floyd_kernel(int k, int *G,int *P, int N){//G will be the adjacency matrix, P will be path matrix
	int col=blockIdx.x*blockDim.x + threadIdx.x;
	if(col>=N)return;
	int idx=N*blockIdx.y+col;

	__shared__ int best;
	if(threadIdx.x==0)
		best=G[N*blockIdx.y+k];
	__syncthreads();
	if(best==INF)return;
	int tmp_b=G[k*N+col];
	if(tmp_b==INF)return;
	int cur=best+tmp_b;
	if(cur<G[idx]){
		G[idx]=cur;
		P[idx]=k;
	}
}

void _GPU_Floyd(int *H_G, int *H_Gpath, const int N){
	//allocate device memory and copy graph data from host
	int *dG,*dP;
	int numBytes=N*N*sizeof(int);
	cudaError_t err=cudaMalloc((int **)&dG,numBytes);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((int **)&dP,numBytes);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	//copy from host to device graph info
	err=cudaMemcpy(dG,H_G,numBytes,cudaMemcpyHostToDevice);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(dP,H_Gpath,numBytes,cudaMemcpyHostToDevice);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	dim3 dimGrid((N+BLOCK_SIZE-1)/BLOCK_SIZE,N);

	for(int k=0;k<N;k++){//main loop
		_GPU_Floyd_kernel<<<dimGrid,BLOCK_SIZE>>>(k,dG,dP,N);
		err = cudaThreadSynchronize();
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	}
	//copy back memory
	err=cudaMemcpy(H_G,dG,numBytes,cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(H_Gpath,dP,numBytes,cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	//free device memory
	err=cudaFree(dG);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(dP);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
}

void _generateCustomGraph(int *G, int N){
	FILE *dataFile = fopen("data.txt", "r");
	if(dataFile != NULL){
		cout<<"Successfully opened file.\n";
	} else {
		cout<<"File not found.\n";
	}

	for(int i=0; i < 2016*2016; i++){
		fscanf(dataFile, "%d", &G[i]);
	}

	fclose(dataFile);
}
