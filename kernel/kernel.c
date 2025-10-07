#include <stdio.h>      /* printf */
#include <math.h>  

void cos_arr(float* a, float* b, float* c, int a_size) {
	for (int i = 0; i < a_size; i++) {
		b[i] = cos(a[i]);
		c[i] = sin(a[i]);
	}
}


__kernel void cos_arr(__global float* a,__global float* b,__global float* c, int a_size) {
      	int gi = get_global_id(0);
		float a_val = a[gi];
        b[gi] = native_cos(a_val);
        c[gi] = native_sin(a_val);
}


int main() {
	int a_size = 100;
	float a[100];	
	for (int i = 0; i < a_size; i++) {
		a[i] = i;
	}

	float b[100];
	float c[100];
	
	cos_arr(a,b,c,a_size);
	for (int i = 0; i < a_size; i++) {
		printf("b = %.2f, c = %.2f\n", b[i], c[i]);
	}
}
