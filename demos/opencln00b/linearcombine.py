import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as pycl_array  # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)
import numpy as np  # Import Numpy number tools
# james will fix

context = cl.create_some_context()  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue

u = np.random.rand(50000).astype(np.float32)
v = np.random.rand(50000).astype(np.float32)

a = pycl_array.to_device(queue, u)
b = pycl_array.to_device(queue, v)

alpha = -3.3
alpha_buf = pycl_array.to_device(queue, np.asarray([alpha]).astype(np.float32))
beta = 2.5
beta_buf = pycl_array.to_device(queue, np.asarray([beta]).astype(np.float32))

# Create two random pyopencl arrays
c = pycl_array.empty_like(a)  # Create an empty pyopencl destination array

program = cl.Program(context, """
__kernel void sum(__global const float *a, __global const float *b, __global float *c,
    __global const float *alpha_buf, __global const float *beta_buf)
{
    int i = get_global_id(0);
    c[i] = alpha_buf[0] * a[i] + beta_buf[0] * b[i];
}
""").build()  # Create the OpenCL program

program.sum(queue, a.shape, None, a.data, b.data, c.data, alpha_buf.data, beta_buf.data)  # Enqueue the program for execution and store the result in c

print("a: {}".format(a))
print("b: {}".format(b))
print("c: {}".format(c))
print("c: {}".format(alpha * u + beta * v))

# Print all three arrays, to show sum() worked