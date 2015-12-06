// write your code into this file

#define TILE_X 3
#define TILE_Y 3
#define TILE_Z 16
#define PADDING 0

__global__ void compute_cell(int* in_array, int* out_array, int dim);

void solveGPU(int **dCells, int dim, int iters)
{
	dim3 threadsPerBlock(TILE_X, TILE_Y, TILE_Z);
	dim3 numBlocks((int)ceil(dim/(float)(TILE_X-2)), (int)ceil(dim/(float)(TILE_Y-2)), (int)ceil(dim/(float)(TILE_Z-2)));
	
	int* result_array;
	cudaMalloc((void**)&result_array, dim*dim*dim*sizeof(int));
    int* tmp;
	int* array_in;
	int* array_out;

    array_in = *dCells;
    array_out = (int*)result_array;

	cudaFuncSetCacheConfig(compute_cell, cudaFuncCachePreferShared);

	
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
	__shared__ int tile[TILE_X][TILE_Y][TILE_Z];
	
	//int tx = threadIdx.x;

    int idx_x = blockIdx.x*(blockDim.x-2) + threadIdx.x;
    int idx_y = blockIdx.y*(blockDim.y-2) + threadIdx.y;
    int idx_z = blockIdx.z*(blockDim.z-2) + threadIdx.z;
	
    int border_idx_x = blockIdx.x*blockDim.x + threadIdx.x;
    int border_idx_y = blockIdx.y*blockDim.y + threadIdx.y;
    int border_idx_z = blockIdx.z*blockDim.z + threadIdx.z;
	
	int dim2 = dim*dim;

	// loading non-boundary cells into shared
//	int is_border = 0;
//	is_border += 0;

	// loading tile to shared
	tile[threadIdx.x][threadIdx.y][threadIdx.z] = 0;
	if (idx_x < dim && idx_y < dim && idx_z < dim)
	{
		if (border_idx_x > 0 && border_idx_y > 0 && border_idx_z > 0)
		{
			tile[threadIdx.x][threadIdx.y][threadIdx.z] = in_array[(idx_x*dim2)+(idx_y*dim)+idx_z];
		}
	}

	
	__syncthreads();
	
	
	// neighbourhood computation
	// TODO shuffling functions here
	int result = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				result += tile[threadIdx.x+i][threadIdx.y+j][threadIdx.z+k];
			}
		}
	}
	result -= tile[threadIdx.x+1][threadIdx.y+1][threadIdx.z+1];
	__syncthreads();
	
	
	// cell life computation
	if ((result < 4) || (result > 5))
	{
		result = 0;
	}
	else if (result == 5)
	{
		result = 1;
	}
	else
	{
		result = tile[threadIdx.x+1][threadIdx.y+1][threadIdx.z+1];
	}
	__syncthreads();
	
	// TODO redesign to avoid 32-way bank conflict
	out_array[(idx_x*dim2)+(idx_y*dim)+idx_z] = result;
}
