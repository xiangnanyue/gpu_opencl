__kernel void vector_add(__global const float *x,  
                        __global float *restrict z)
{
	int group_id = get_group_id(0);
	int group_size = get_local_size(0);
	int local_id = get_local_id(0);
	// int idx = get_global_id(0);
	
	*z += x[group_id*group_size + local_id];

	// printf("Hello, World %d:%f!\n",idx,z[idx]);
}

