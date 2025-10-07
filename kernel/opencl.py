import pyopencl as cl
import pyopencl.array as clarray
import numpy as np
import time 
kernel="""
__kernel void cos_arr(__global float* a,__global float* b,__global float* c, int a_size) {
      	int gi = get_global_id(0);
        float a_val = a[gi];
        b[gi] = native_cos(a_val);
        c[gi] = native_sin(a_val);

}
"""
kernel22="""
__kernel void cos_arr(__global double* a,__global double* b,__global double* c, int a_size) {
      	int gi = get_global_id(0);
        double a_val = a[gi];
        b[gi] = native_cos(a_val);
        c[gi] = native_sin(a_val);

}
"""
def test_method(context,queue):
	dt = np.float64
	
	n = 100000000
	a = np.arange(n).astype(dt) 
	b = np.zeros(n).astype(dt) 
	c = np.zeros(n).astype(dt)

	a = np.ascontiguousarray(a) 
	b = np.ascontiguousarray(b)
	c = np.ascontiguousarray(c)

	program = cl.Program(context,kernel22).build()
	
	print("Copying arrays")	
	
	a_dev = clarray.to_device(queue, a) 
	b_dev = clarray.to_device(queue, b) 
	c_dev = clarray.to_device(queue, c)
	t = time.time()
	print("Starting kernel")
	
	program.cos_arr.set_scalar_arg_dtypes([None,None,None, np.int32])	
	program.cos_arr(queue,(n,),None, a_dev.data, b_dev.data, c_dev.data, np.int32(n))

	cl.enqueue_copy(queue, b, b_dev.data)
	cl.enqueue_copy(queue, c, c_dev.data)

	tgpu = time.time() - t
	
#	print(b.mean(), c.mean())
	print("Starting other kernel")
	
	t = time.time() 
	
	btest, ctest = np.cos(a), np.sin(a)
	print(btest)
	print(b)

	tcpu = time.time() - t
	
	print("Testing agreements")
	
	print("gpu time:", tgpu)
	print("cpu time:", tcpu)
	
	assert np.allclose(b,btest.astype(np.float32)) 
	assert np.allclose(c,ctest.astype(np.float32))

def get_context_queue(gpu_name="V100"):
#   list the platforms
    platforms = cl.get_platforms()
    print("Found platforms (will use first listed):", platforms)
#   select the gpu
    my_gpu = platforms[0].get_devices(
        device_type=cl.device_type.GPU)
    assert( my_gpu)
    my_gpu = [g for g in my_gpu if gpu_name in str(g)]
    print("Found GPU(s):", my_gpu)
#   create the context for the gpu, and the corresponding queue
    context = cl.Context(devices=my_gpu)
    queue = cl.CommandQueue(context)
    return context, queue

if __name__=="__main__":
    c, q = get_context_queue()
    test_method(c,q)
    print("DONE")
