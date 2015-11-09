// naive CPU implementation

void solveCPU(int **in, int n, int iters) {
	int *out = (int*)malloc(n*n*n*sizeof(out[0]));

	for (int it = 0; it < iters; it++) {
		for (int i = 0; i < n; i++) 
			for (int j = 0; j < n; j++)
				for (int k = 0; k < n; k++) {
					int alive = 0;
					for (int ii = max(i-1, 0); ii <= min(i+1, n-1); ii++)
					for (int jj = max(j-1, 0); jj <= min(j+1, n-1); jj++)
					for (int kk = max(k-1, 0); kk <= min(k+1, n-1); kk++)
						alive += (*in)[ii*n*n + jj*n + kk];
					alive -= (*in)[i*n*n + j*n + k];

					if (alive < 4 || alive > 5)
						out[i*n*n + j*n + k] = 0;
					else if (alive == 5)
						out[i*n*n + j*n + k] = 1;
					else
						out[i*n*n + j*n + k] = (*in)[i*n*n + j*n + k];
				}

		// flip in x out
		int *tmp = *in;
		*in = out;
		out = tmp;
	}

	free(out);
}

