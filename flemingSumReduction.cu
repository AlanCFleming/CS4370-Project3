#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
//Code written by Alan Fleming

//CONSTANTS
#define MATRIXSIZE 8
#define BLOCKSIZE 4


//Code to preform sum reduction using the cpu.
//Overwrites x[0] with the sum of X and returns it as an int
int SumReduction(int* x, int N){
	for(i = 1; i < N; i++)
		x[0] += x[i];
	return x[0];
}
