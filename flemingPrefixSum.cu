#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
//Code written by Alan Fleming

//CONSTANTS
#define MATRIXSIZE 2048
#define BLOCKSIZE 1024


//Code to prefix sum using the cpu
void prefixSumCPU(int* x, int* y,  int N){
	y[0] = x[0];
	for(int i = 1; i < N; i++){
		y[i] = y[i-1] + x[i];
	}
}

__global__ void parallelPrefixSum(int* x, int* y, int inputSize) {
	//allocate shared memory for block
	__shared__ int scan_array[2*BLOCKSIZE];
	
	//initialize shared memory
	unsigned int start = 2 * blockIdx.x * blockDim.x;
	scan_array[threadIdx.x] = x[start + threadIdx.x];
	scan_array[blockDim.x + threadIdx.x] = x[start + blockDim.x + threadIdx.x];

	__syncthreads();

	//Reduction step

	for(int stride = 1; stride <= BLOCKSIZE; stride *= 2){
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if(index < 2 * BLOCKSIZE) {
			scan_array[index] += scan_array[index - stride];
		}
		__syncthreads();
	}
	
	
	//Post Scan
	for(int stride = BLOCKSIZE/2; stride > 0; stride /= 2) {
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if(index + stride < 2*BLOCKSIZE) {
			scan_array[index + stride] += scan_array[index];
		}
		__syncthreads();
	}

	//Output array
	y[start + threadIdx.x] = scan_array[threadIdx.x];
	y[start + blockDim.x + threadIdx.x] = scan_array[blockDim.x + threadIdx.x];
}

int main() {
	
	int *a = (int *)malloc(sizeof(int) * MATRIXSIZE); //allocate space for array
	int *cpuResult = (int *)malloc(sizeof(int) * MATRIXSIZE); //allocate space for cpu output array
	int *gpuResult = (int *)malloc(sizeof(int) * MATRIXSIZE); //allocate space for gpu output array

	//initialize array
	int init = 1325;
	for(int i=0; i<MATRIXSIZE;i++){
		init = 3125 * init % 6553;
		a[i] = (init - 1000) % 97;
		gpuResult[i] = 0;
	}

	//Test CPU reduction
	//Get start time
	clock_t t1 = clock();
	//Calculate reduction
	
	prefixSumCPU(a, cpuResult, MATRIXSIZE);

	//Get stop time
	clock_t t2 = clock();
	//Calculate runtime
	float cpuTime= (float(t2-t1)/CLOCKS_PER_SEC*1000);

	//Allocate memory on GPU compution. dev_b is used to store the results of the first pass of reduction
	int *dev_a, *dev_b;
	cudaMalloc((void **)(&dev_a), MATRIXSIZE *sizeof(int));
	cudaMalloc((void **)(&dev_b), MATRIXSIZE *sizeof(int));

	//copy memory to gpu
	cudaMemcpy(dev_a,a, MATRIXSIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,gpuResult, MATRIXSIZE * sizeof(int), cudaMemcpyHostToDevice);

	//calculate dimentions for gpu
	dim3 dimBlock(BLOCKSIZE);
	dim3 dimGrid(ceil(double(MATRIXSIZE)/dimBlock.x/2));

	//Set up cuda events for recording runtime
	cudaEvent_t start,stop;
	float gpuTime; 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	//calculate prefix sum
	parallelPrefixSum<<<dimGrid, dimBlock>>>(dev_a, dev_b, MATRIXSIZE);
	
	//calculate runtime 
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime,start,stop);

	//destroy cuda events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//copy sum from gpu
	cudaMemcpy(gpuResult, dev_b, sizeof(int), cudaMemcpyDeviceToHost);

	//print speedup
	printf("CPU Runtime: %f\nGpu Runtime: %f\nSpeedup: %f\n", (double)cpuTime, (double)gpuTime, double(cpuTime / gpuTime));

	//verify results
	bool valid = true;
	for(int i = 0; i < MATRIXSIZE; i++) {	
		if(cpuResult[i] != gpuResult[i]) {
			valid = false;
			break;
		}
	}
	if(valid) {
		printf("TEST PASSED\n");
	} else {
		printf("TEST FAILED\n");
	}
		
	//free memory
	free(a);
	free(cpuResult);
	free(gpuResult);
	cudaFree(dev_a);
	cudaFree(dev_b);
	
	return 0;
}
