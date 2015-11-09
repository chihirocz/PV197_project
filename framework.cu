#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "kernel.cu"
#include "kernel_CPU.C"

#define N 128
#define ITERS 10

void createRandomCells(int *cells, int n) {
	for (int i = 0; i < n*n*n; i++)
		if ((float)rand() / (float)RAND_MAX > 0.5)
			cells[i] = 1;
		else
			cells[i] = 0;
}

int main(int argc, char **argv){
	int *cells = NULL; 	// cells computed by CPU
	int *cellsGPU = NULL;	// CPU buffer for GPU results
	int *dCells = NULL;	// cells computed by GPU

	// parse command line
	int device = 0;
	if (argc == 2) 
		device = atoi(argv[1]);
	if (cudaSetDevice(device) != cudaSuccess){
		fprintf(stderr, "Cannot set CUDA device!\n");
		exit(1);
	}
	cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Using device %d: \"%s\"\n", device, deviceProp.name);

	// create events for timing
	cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

	// allocate and set host memory
	cells = (int*)malloc(N*N*N*sizeof(cells[0]));
	cellsGPU = (int*)malloc(N*N*N*sizeof(cells[0]));
	createRandomCells(cells, N);
 
	// allocate and set device memory
	if (cudaMalloc((void**)&dCells, N*N*N*sizeof(dCells[0])) != cudaSuccess) {
		fprintf(stderr, "Device memory allocation error!\n");
		goto cleanup;
	}
	cudaMemcpy(dCells, cells, N*N*N*sizeof(dCells[0]), cudaMemcpyHostToDevice);

	// solve on CPU
        printf("Solving on CPU...\n");
	cudaEventRecord(start, 0);
	solveCPU(&cells, N, ITERS);
	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        printf("CPU performance: %f megacells/s\n",
                float(N*N*N)*float(ITERS)/time/1e3f);

	// dummy copy, just to awake GPU
        cudaMemcpy(cellsGPU, dCells, N*N*N*sizeof(dCells[0]), cudaMemcpyDeviceToHost);

	// solve on GPU
	printf("Solving on GPU...\n");
	cudaEventRecord(start, 0);
	solveGPU(&dCells, N, ITERS);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
	printf("GPU performance: %f megacells/s\n",
                float(N*N*N)*float(ITERS)/time/1e3f);

	// check GPU results
	cudaMemcpy(cellsGPU, dCells, N*N*N*sizeof(dCells[0]), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < N; k++)
				if (cellsGPU[i*N*N + j*N + k] != cells[i*N*N + j*N + k]){
					printf("Error detected at [%i, %i, %i]: %i should be %i.\n", i, j, k, cellsGPU[i*N*N + j*N + k], cells[i*N*N + j*N + k]);
					goto cleanup; // exit after the first error
				}

	printf("Test OK.\n");

cleanup:
	cudaEventDestroy(start);
        cudaEventDestroy(stop);

	if (dCells) cudaFree(dCells);

	if (cells) free(cells);
	if (cellsGPU) free(cellsGPU);

	return 0;
}

