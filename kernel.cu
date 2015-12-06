// write your code into this file

#define TILE_X 2
#define TILE_Y 2
#define TILE_Z 32
#define PADDING 0

__global__ void compute_cell(int* in_array, int* out_array, int dim);

void solveGPU(int **dCells, int dim, int iters)
{
	dim3 threadsPerBlock(TILE_X, TILE_Y, TILE_Z);
	dim3 numBlocks(dim/threadsPerBlock.x, dim/threadsPerBlock.y, dim/threadsPerBlock.z);
	
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
	__shared__ int tile[TILE_X+2][TILE_Y+2][TILE_Z+2+PADDING];
	
	//int tx = threadIdx.x;

    int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
    int idx_z = blockIdx.z*blockDim.z + threadIdx.z;
	
	int dim2 = dim*dim;

	// loading non-boundary cells into shared
	tile[threadIdx.x+1][threadIdx.y+1][threadIdx.z+1] = in_array[(idx_x*dim2)+(idx_y*dim)+idx_z];

	
	// loading non-diagonal boundary cells into shared
	
	// X-coords
	if (threadIdx.x == 0)
	{
		if (idx_x == 0)
		{
			tile[threadIdx.x][threadIdx.y+1][threadIdx.z+1] = 0;
		}
		else
		{
			tile[threadIdx.x][threadIdx.y+1][threadIdx.z+1] = in_array[((idx_x-1)*dim2)+(idx_y*dim)+idx_z];
		}
	}
	if (threadIdx.x == (TILE_X-1))
	{
		if (idx_x == (dim-1))
		{
			tile[threadIdx.x+2][threadIdx.y+1][threadIdx.z+1] = 0;
		}
		else
		{
			tile[threadIdx.x+2][threadIdx.y+1][threadIdx.z+1] = in_array[((idx_x+1)*dim2)+(idx_y*dim)+idx_z];
		}
	}
	
	// Y-coords
	if (threadIdx.y == 0)
	{
		if (idx_y == 0)
		{
			tile[threadIdx.x+1][threadIdx.y][threadIdx.z+1] = 0;
		}
		else
		{
			tile[threadIdx.x+1][threadIdx.y][threadIdx.z+1] = in_array[(idx_x*dim2)+((idx_y-1)*dim)+idx_z];
		}
	}
	if (threadIdx.x == (TILE_Y-1))
	{
		if (idx_y == (dim-1))
		{
			tile[threadIdx.x+1][threadIdx.y+2][threadIdx.z+1] = 0;
		}
		else
		{
			tile[threadIdx.x+1][threadIdx.y+2][threadIdx.z+1] = in_array[(idx_x*dim2)+((idx_y+1)*dim)+idx_z];
		}
	}
	
	// Z-coords
	if (threadIdx.z == 0)
	{
		if (idx_z == 0)
		{
			tile[threadIdx.x+1][threadIdx.y+1][threadIdx.z] = 0;
		}
		else
		{
			tile[threadIdx.x+1][threadIdx.y+2][threadIdx.z] = in_array[(idx_x*dim2)+(idx_y*dim)+idx_z-1];
		}
	}
	if (threadIdx.x == (TILE_Z-1))
	{
		if (idx_z == (dim-1))
		{
			tile[threadIdx.x+1][threadIdx.y+1][threadIdx.z+2] = 0;
		}
		else
		{
			tile[threadIdx.x+1][threadIdx.y+1][threadIdx.z+2] = in_array[(idx_x*dim2)+(idx_y*dim)+idx_z+1];
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
