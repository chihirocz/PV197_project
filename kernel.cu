// write your code into this file

#define TILE_WIDTH 3
#define TILE_LENGTH 32


void solveGPU(int **dCells, int dim, int iters)
{
	dim3 threadsPerBlock(TILE_LENGTH, 1, 1);
	dim3 numBlocks(dim/threadsPerBlock.x, dim/threadsPerBlock.y, dim/threadsPerBlock.z);
	
	int* result_array = (int*) cudaMalloc(dim*dim*dim*sizeof(int));
    int* tmp, array_in, aray_out;

    array_in = *dCells;
    array_out = result_array;
	
	for (int i = 0; i < iters; i++)
	{
		compute_cell<<numBlocks, threadPerBlock)>>(array_in, array_out, dim);
        tmp = array_in;
        array_in = array_out;
        array_out = tmp;
	}

    *dCells = array_in; // result array from loop above
    cudaFree(result_array);
}


__global__ void compute_cell(int* in_array, int* out_array, int dim)
{
	// using subsegment of 3x3x32 (+2 bordering) cells from entire cube to use coalesced global mem access, i.e. 9 segments
	__shared__ int cell_tile[TILE_LENGTH+2][TILE_WIDTH][TILE_WIDTH];
	__shared__ int results[TILE_LENGTH];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bz = blockIdx.z;
	int tx = threadIdx.x;
	
	// It is guaranteed that dimension in one line is multiple of 128. There are no residues.
	ushort blocks_in_line = dim/TILE_LENGTH;
	int dim2 = dim*dim;
	
	for (int i = -1; i < 1; i++)
	{
		int idx_y = by + i;
		if ((idx_y < 0) || (idx_y >=dim))
		{
			cell_tile[tx+1][i+1][j+1] = 0;
		}
		else
		{
			for (int j = -1; j < 1; j++)
			{
				int idx_z = bz + j;
				if ((idx_z < 0) || (idx_z >= dim))
				{
					// work out the other bounds
					cell_tile[tx+1][i+1][j+1] = 0;
				}
				else
				{
					cell_tile[tx+1][i+1][j+1] = in_array[(tx+bx*blockDim.x+by*dim+bz*dim2];
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
				results[tx] += cell_tile[tx+k][i][j];
			}
		}
	}
	results[tx] -= cell_tile[tx+1][1][1];
	
	
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
	
	
	// cudaMemcpy() results to global
	// TODO redesign to avoid 16-way bank conflict
	out_array[tx+bx*blockDim.x+by*dim+bz*dim2] = results[tx];
}
