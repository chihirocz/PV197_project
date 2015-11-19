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
	__shared__ int cell_tile[TILE_LENGTH+2][TILE_WIDTH][TILE_WIDTH];
	__shared__ int results[TILE_LENGTH];
	
	int tx = threadIdx.x;

    int idx_x = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
    int idx_z = blockIdx.z*blockDim.z + threadIdx.z;
	
	// It is guaranteed that dimension in one line is multiple of 128. There are no residues.
	int dim2 = dim*dim;

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
                    cell_tile[tx+1+i][j+1][k+1] = 0;
                }
                else
                {
                    // Above all this is quite stupid, because a value is stored multiple times into shared
					// and coalesced access is not used unlike the previous solution.
                    // However loading "caps" of the block is guaranteed.
                
                    cell_tile[tx+1+i][j+1][k+1] = in_array[(idx_x+i)+((idx_y+j)*dim)+((idx_z+k)*dim2)];
                }
            }
        }
    }
	// work out tile bounds for cell_tile[0] and cell_tile[TILE_LENGTH]
	__syncthreads();
	
	
	// neighbourhood computation
	// TODO shuffling functions here
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				results[tx] += cell_tile[tx+i][j][k];
			}
		}
	}
	results[tx] -= cell_tile[tx+1][1][1];
	__syncthreads();
	
	
	// cell life computation
	if ((results[tx] < 4) || (results[tx] > 5))
	{
		results[tx] = 0;
	}
	else if (results[tx] == 5)
	{
		results[tx] = 1;
	}
	else
	{
		results[tx] = cell_tile[tx+1][1][1];
	}
	__syncthreads();
	
	// TODO redesign to avoid 32-way bank conflict
	out_array[idx_x+(idx_y*dim)+(idx_z*dim2)] = results[tx];
}
