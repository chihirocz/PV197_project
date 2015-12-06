// write your code into this file

#define TILE_X 2
#define TILE_Y 2
#define TILE_Z 32
#define PADDING 0

__global__ void compute_cell(int* in_array, int* out_array, int dim);
__global__ void copy_to_bordered(int* in_array, int* out_array, int dim);
__global__ void copy_to_raw(int* in_array, int* out_array, int dim);

void solveGPU(int **dCells, int dim, int iters)
{
	dim3 threadsPerBlock(TILE_X, TILE_Y, TILE_Z);
	dim3 numBlocks(dim/threadsPerBlock.x, dim/threadsPerBlock.y, dim/threadsPerBlock.z);
	
	int temp_array_size = (dim+2)*(dim+2)*(dim+2)*sizeof(int);
	int* bordered_array;
	int* result_array;
	cudaMalloc((void**)&bordered_array, temp_array_size);
	cudaMalloc((void**)&result_array,   temp_array_size);
	cudaMemset(bordered_array, 0, temp_array_size);
	cudaMemset(result_array, 0, temp_array_size);
 
    int* tmp;
	int* array_in;
	int* array_out;

	cudaFuncSetCacheConfig(copy_to_bordered, cudaFuncCachePreferL1);
	copy_to_bordered<<<numBlocks, threadsPerBlock>>>(*dCells, bordered_array, dim);

	array_in = bordered_array;
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

		
	cudaFuncSetCacheConfig(copy_to_raw, cudaFuncCachePreferL1);
	copy_to_raw<<<numBlocks, threadsPerBlock>>>(*dCells, result_array, dim);

	cudaFree(bordered_array);
	cudaFree(result_array);
}

__global__ void copy_to_bordered(int* in_array, int* out_array, int dim)
{
	int dim2 = dim*dim;
	int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
    int idx_z = blockIdx.z*blockDim.z + threadIdx.z;
	
	out_array[((idx_x+1)*dim2)+((idx_y+1)*dim)+idx_z+1] = in_array[(idx_x*dim2)+(idx_y*dim)+idx_z];
}

__global__ void copy_to_raw(int* in_array, int* out_array, int dim)
{
	int dim2 = dim*dim;
	int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
    int idx_z = blockIdx.z*blockDim.z + threadIdx.z;
	
	out_array[(idx_x*dim2)+(idx_y*dim)+idx_z] = in_array[((idx_x+1)*dim2)+((idx_y+1)*dim)+idx_z+1];
}

__global__ void compute_cell(int* in_array, int* out_array, int dim)
{
	// using subsegment of 3x3x32 (+2 bordering) cells from entire cube to use coalesced global mem access, i.e. 9 segments
	__shared__ int tile[TILE_X+2][TILE_Y+2][TILE_Z+2+PADDING];
	
	int tx = threadIdx.x;

    int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
    int idx_z = blockIdx.z*blockDim.z + threadIdx.z;
	
	int dim2 = dim*dim;
	
/*	
	for (int i = -1; i < 2; i++)
    {
        for (int j = -1; j < 2; j++)
        {
            for (int k = -1; k < 2; k++)
            {
                int b_min = idx_x+i;
                b_min = min(b_min, idx_y+j);
                b_min = min(b_min, idx_z+k);

                int b_max = idx_x+i;
                b_max = max(b_max, idx_y+j);
                b_max = max(b_max, idx_z+k);

                if ((b_min < 0) || (b_max >= dim))
                {
                    tile[tx+1+i][j+1][k+1] = 0;
                }
                else
                {
                    // Above all this is quite stupid, because a value is stored multiple times into shared
					// and coalesced access is not used unlike the previous solution.
                    // However loading "caps" of the block is guaranteed.
                
                    tile[tx+1+i][j+1][k+1] = in_array[(idx_x+i)+((idx_y+j)*dim)+((idx_z+k)*dim2)];
                }
            }
        }
    }
	*/
	tile[threadIdx.x+1][threadIdx.y+1][threadIdx.z+1] = 0;
	tile[threadIdx.x+1][threadIdx.y+1][threadIdx.z+1] = in_array[(idx_x*dim2)+(idx_y*dim)+idx_z];
	// work out tile bounds for tile[0] and tile[TILE_LENGTH]
	__syncthreads();
	
	
	// neighbourhood computation
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
	
	out_array[(idx_x*dim2)+(idx_y*dim)+idx_z] = result;
}
