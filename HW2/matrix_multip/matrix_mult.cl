__kernel void matrix_multip( __global const int *N,
						__global const float *x,  
						__global const float *y,
                        __global float *restrict z)
{
	// int group_id = get_group_id(0);
	// int group_size = get_local_size(0);
	// int local_id = get_local_id(0);
	int global_id = get_global_id(0);
	int dimension = *N;

	int col = global_id % dimension;
	int row = global_id / dimension;

	z[col + row*dimension] = 0;
	for(int k = 0; k < dimension; ++k) {
		z[col + row*dimension] += x[row*dimension + k] * y[k*dimension + col];
	}
	// printf("Hello, World %d:%f!\n",idx,z[idx]);
}

