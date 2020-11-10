#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
//Code written by Alan Fleming

//CONSTANTS
#define MATRIXSIZE 131072
#define BLOCKSIZE 1024


//Code to preform sum reduction using the cpu
int SumReductionCPU(int* x, int N){
	int sum = 0;
	for(int i = 0; i < N; i++){
		sum += x[i];
	}
	return sum;
}

__global__ void sumReductionKernal(int* arr) {

	//initialize Partial Result for thread	
	__shared__ int partialResult[2 * BLOCKSIZE];
	unsigned int start = 2*blockIdx.x * blockDim.x;
	partialResult[threadIdx.x] = arr[start + threadIdx.x];
	partialResult[blockDim.x + threadIdx.x] = arr[start +blockDim.x + threadIdx.x];
	
	//Preform sum reduction
	for(unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
		__syncthreads();
		if (threadIdx.x < stride){
			partialResult[threadIdx.x] += partialResult[threadIdx.x + stride];
		}
	}
	
	__syncthreads();
	if(threadIdx.x == 0){
		//write block sum to global memory
		arr[blockIdx.x] = partialResult[0];
	}
}

int main() {
	
	int *a = (int *)malloc(sizeof(int) * MATRIXSIZE); //allocate space for array
	//initialize array
	int init = 1325;
	for(int i=0; i<MATRIXSIZE;i++){
		init = 3125 * init % 6553;
		a[i] = (init - 1000) % 97;
	}

	//Test CPU reduction
	//Get start time
	clock_t t1 = clock();
	//Calculate reduction
	int cpuResult = SumReductionCPU(a, MATRIXSIZE);
	//Get stop time
	clock_t t2 = clock();
	//Calculate runtime
	float cpuTime= (float(t2-t1)/CLOCKS_PER_SEC*1000);

	//Allocate memory on GPU compution
	int *dev_a;
	cudaMalloc((void **)(&dev_a), MATRIXSIZE *sizeof(int));

	//copy memory to gpu
	cudaMemcpy(dev_a,a, MATRIXSIZE * sizeof(int), cudaMemcpyHostToDevice);

	//calculate dimentions for gpu
	dim3 dimBlock(BLOCKSIZE);
	dim3 dimGrid(ceil(double(MATRIXSIZE)/dimBlock.x));

	//Set up cuda events for recording runtime
	cudaEvent_t start,stop;
	float gpuTime; 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	//Calculate GPU Reduction for each block
	sumReductionKernal<<<dimGrid, dimBlock>>>(dev_a);
	//Calculate GPU Recuction for block results
	sumReductionKernal<<<dimGrid, dimBlock>>>(dev_a);

	//calculate runtime 
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime,start,stop);

	//destroy cuda events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//copy sum from gpu
	cudaMemcpy(a, dev_a, sizeof(int), cudaMemcpyDeviceToHost);

	//print speedup
	printf("CPU Runtime: %f\nGpu Runtime: %f\nSpeedup: %f\n", (double)cpuTime, (double)gpuTime, double(cpuTime / gpuTime));

	//print reduction results
	printf("CPU Result: %d\nGPU Result: %d\n", cpuResult, a[0]);
	//verify results
	if(cpuResult == a[0]) {
		printf("TEST PASSED\n");
	} else {
		printf("TEST FAILED\n");
	}

	//free memory
	free(a);
	cudaFree(dev_a);
	
	return 0;
}
