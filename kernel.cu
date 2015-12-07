// write your code into this file

#define TILE_X 4
#define TILE_Y 4
#define TILE_Z 32

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

    int mat_idx_x = blockIdx.x*(blockDim.x-2) + threadIdx.x-1;
    int mat_idx_y = blockIdx.y*(blockDim.y-2) + threadIdx.y-1;
    int mat_idx_z = blockIdx.z*(blockDim.z-2) + threadIdx.z-1;
	/*
    int border_idx_x = blockIdx.x*blockDim.x + threadIdx.x;
    int border_idx_y = blockIdx.y*blockDim.y + threadIdx.y;
    int border_idx_z = blockIdx.z*blockDim.z + threadIdx.z;
	*/
	int dim2 = dim*dim;
	
	unsigned short thread_exceeds_matrix;
	if ((mat_idx_x < dim) && (mat_idx_y < dim) && (mat_idx_z < dim))
	{
		if ((mat_idx_x >= 0) && (mat_idx_y >= 0) && (mat_idx_z >= 0))
		{
			thread_exceeds_matrix = 1;
		}
	}
	// loading tile to shared
//	tile[threadIdx.x][threadIdx.y][threadIdx.z] = 0;
/*	if ((mat_idx_x < dim) && (mat_idx_y < dim) && (mat_idx_z < dim))
	{
		if ((mat_idx_x >= 0) && (mat_idx_y >= 0) && (mat_idx_z >= 0))
		{
			tile[threadIdx.x][threadIdx.y][threadIdx.z] = in_array[(mat_idx_x*dim2)+(mat_idx_y*dim)+mat_idx_z];
		}
	}
	*/
	tile[threadIdx.x][threadIdx.y][threadIdx.z] = thread_exceeds_matrix ? 0: in_array[(mat_idx_x*dim2)+(mat_idx_y*dim)+mat_idx_z];
	__syncthreads();
	
	
	// neighbourhood computation
	// TODO shuffling functions here
	int result = 0;
	int shared_value = tile[threadIdx.x][threadIdx.y][threadIdx.z];
	
	if (threadIdx.x > 0 && threadIdx.y > 0 && threadIdx.z > 0
		&& threadIdx.x < (TILE_X-1) && threadIdx.y < (TILE_Y-1) && threadIdx.z < (TILE_Z-1) )
	{
		for (int i = -1; i < 2; i++)
		{
			for (int j = -1; j < 2; j++)
			{
				for (int k = -1; k < 2; k++)
				{
					result += tile[threadIdx.x+i][threadIdx.y+j][threadIdx.z+k];
				}
			}
		}
		result -= shared_value;
	}
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
		// inspect this!!
		result = shared_value;
	}
	__syncthreads();
	
	// TODO redesign to avoid 32-way bank conflict
	if (!thread_exceeds_matrix)
	{
		out_array[(mat_idx_x*dim2)+(mat_idx_y*dim)+mat_idx_z] = result;
	}
}
