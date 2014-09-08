#!/usr/bin/env python
# -*- coding: utf-8 -*-
# gus

import numpy as np
import pyopencl as cl
import scipy as sci

a = np.random.rand(50000).astype(np.float32)
b = np.random.rand(50000).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
mat = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
target = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
#a = pycl_array.to_device(queue, np.random.rand(50000).astype(np.float32))


#__global__ void kApplySigmoid(float* mat, float* target, unsigned int len) {
#    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
#    const unsigned int numThreads = blockDim.x * gridDim.x;
#
#    for (unsigned int i = idx; i < len; i += numThreads) {
#        target[i] = 1 / (1 + __expf(-mat[i]));
#   }
#}
# a_g=mat, b_g=target, res_g=len)

prg = cl.Program(ctx, """
__kernel void ApplySigmoid(__global const float *mat, __global float *target) {
	int gid = get_global_id(0);
		target[gid] = 1 / (1 + exp(-mat[gid]));
}
""").build()

target = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)
prg.ApplySigmoid(queue, a.shape, None, mat, target)

target = np.empty_like(a)
#cl.enqueue_copy(queue, target, None)

# Check on CPU with Numpy:
# target[i] = 1 / (1 + __expf(-mat[i]))
print(target)
print(1 / (1 + np.exp(-a)))
