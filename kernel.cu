// write your code into this file

#define TILE_WIDTH 3
#define TILE_LENGTH 32

__global__ void compute_cell(int* in_array, int* out_array, int dim);

void solveGPU(int **dCells, int dim, int iters)
{
	dim3 threadsPerBlock(TILE_LENGTH, 1, 1);
	dim3 numBlocks(dim/threadsPerBlock.x, dim/threadsPerBlock.y, dim/threadsPerBlock.z);
	
	int* result_array;
	cudaMalloc((void**)&result_array, dim*dim*dim*sizeof(int));
    int* tmp;
	int* array_in;
	int* array_out;

    array_in = *dCells;
    array_out = (int*)result_array;
	
	for (int i = 0; i < iters; i++)
	{
		compute_cell<<<numBlocks, threadsPerBlock>>>(array_in, array_out, dim);
		result_array = array_out;
        tmp = array_in;
        array_in = array_out;
        array_out = tmp;
	}

    *dCells = result_array; // result array from loop above
    cudaFree(array_out);
}


__global__ void compute_cell(int* in_array, int* out_array, int dim)
{
	// using subsegment of 3x3x32 (+2 bordering) cells from entire cube to use coalesced global mem access, i.e. 9 segments
	
	__shared__ int tile[TILE_WIDTH][TILE_WIDTH][TILE_LENGTH];
	__shared__ int results[TILE_WIDTH+2][TILE_WIDTH+2][TILE_LENGTH+2];

    int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
    int idx_z = blockIdx.z*blockDim.z + threadIdx.z;

	int dim2 = dim*dim;
	
	tile[threadIdx.x][threadIdx.y][threadIdx.z] = in_array[(idx_x*dim2)+(idx_y*dim)+idx_z];
	

	results[threadIdx.x][threadIdx.y][threadIdx.z] = in_array[(idx_x*dim2)+(idx_y*dim)+idx_z];
	__syncthreads();
	// TODO redesign to avoid 32-way bank conflict
	out_array[idx_z+(idx_y*dim)+(idx_x*dim2)] = results[threadIdx.x][threadIdx.y][threadIdx.z];
	__syncthreads();
}
