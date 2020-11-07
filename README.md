# CS4370-Project3
This is a cuda program that covers "parallel sum reduction and parallel prefix sum" for class.

## Editing BLOCKSIZE and MATRIXSIZE
* A define statement for MATRIXSIZE can be found on line 8 of the .cu file
* A define statement for BLOCKSIZE can be found on line 9 of the .cu file


## Compiling
nvcc was used to compile these programs. This will create an executable program.
* Command for compiling sum reduction: nvcc flemingSumReduction.cu -o sumReduction
* Command for compiling prefix sum: nvcc flemingPrefixSum.cu -o prefixSum

## Running
These programs can be run directly from the command line. 
* Command for parallel sum reduction: {path}/sumReduction
* Command for parallel prefix sum: {path}/prefixSum
