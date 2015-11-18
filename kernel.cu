// write your code into this file

#define TILE_WIDTH 3
#define TILE_LENGTH 32


void solveGPU(int **dCells, int dim, int iters)
{
	dim3 threadPerBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
	
	for (int i = 0; i < iters; i++)
	{
		compute_cell<<dim*dim*dim/TILE_LENGTH, TILE_LENGHT)>>(*dCells, dim);
	}
}

__global__ void compute_cell(int** cells_array, int dim)
{
	// using subsegment of 3x3x32 (+2 bordering) cells from entire cube to use coalesced global mem access, i.e. 9 segments
	__shared__ int cell_tile[TILE_LENGTH+2][TILE_WIDTH][TILE_WIDTH];
	__shared__ int results[TILE_LENGTH];
	
	uint bx = blockIdx.x;
	uint tx = threadIdx.x;
	
	// It is guaranteed that dimension in one line is multiple of 128. There are no residues.
	ushort blocks_in_line = dim/TILE_LENGTH;
	int dim2 = dim*dim;
	int slice_index = bx * TILE_LENGTH + tx;
	
	for (int i = -1; i < 1; i++)
	{
		for (int j = -1; j < 1; j++)
		{
			int idx = slice_index + (i*dim2) + (j*dim);
			if (idx < 0)
			{
				// work out the other bounds
				cell_tile[tx+1][i+1][j+1] = 0;
			}
			else
			{
				cell_tile[tx+1][i+1][j+1] = (*dCells)[idx];
			}
		}
	}
	__syncthreads();
	
	
	// for i,j,k compute life
	
	// cudaMemcpy() results to global
}