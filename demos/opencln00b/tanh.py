#!/usr/bin/env python
# -*- coding: utf-8 -*-
# gus

# import PyOpenCL and Numpy. An OpenCL-enabled GPU is not required,
import numpy as np
import pyopencl as cl
import pyopencl.clmath as clm

# initialize operating variables
a = np.random.rand(50000).astype(np.float16)
b = np.random.rand(50000).astype(np.float16)

# create an OpenCL context
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# create context buffers for a and b arrays
mf = cl.mem_flags
matrix = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
target = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

# OpenCL (C99) kernel code, compiled
prg = cl.Program(ctx, """
__kernel void ApplyTanh(__global const float *matrix, __global float *target) {
	int gid = get_global_id(0);
	target[gid] = tanh(matrix);
}
""").build()

# launch the kernel
event = prg.ApplyTanh(queue, a.shape, None, matrix, target)

# copy the output from the context to the Python process
cl.enqueue_copy(queue, b, target)

# check on CPU with Numpy:
print(b)
print(tanh(a))
