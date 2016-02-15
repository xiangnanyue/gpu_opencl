__kernel void videofilter_gpu( __global const int *N,
                        __global const float *x,
                        __global const float *y,
                        __global float *restrict z)
{
    int global_id = get_global_id(0);
    int height = *N;
	int weight = *(N + 1);
	
	// convolution

    int col = global_id % weight;
    int row = global_id / weight;
	int pivol = col + row*weight;
	
    z[pivol] = 0;
    for(int i = 0; i < 3; ++i) 
	{
		for (int j =0; j < 3; ++j) 
		{
			int k = pivol + (i - 1) * weight + (j -1);
			if (k < 0 || k >= weight*height)
			{ 
				z[pivol] += 0; 
			} 
			else   
			{ 
				z[pivol] += x[k] * y[3*i+j]; 
			}
		}
    }
}

