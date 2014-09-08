#!/usr/bin/env python
# -*- coding: utf-8 -*-
# dave

import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel

# Create arrays
n = 10
a_np = np.random.randn(n).astype(np.float32)
b_np = np.random.randn(n).astype(np.float32)

# Create context and queue

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# map arrays to the device using queue

a_g = cl.array.to_device(queue, a_np)
b_g = cl.array.to_device(queue, b_np)

# pass in ElementwiseKernel(context, arguments, operation, name="kernel")
lin_mult = ElementwiseKernel(ctx,
    "float *a_g, float *b_g, float *res_g",
    "res_g[i] = a_g[i] * b_g[i]",
    "lin_mult"
)

# create array for results
res_g = cl.array.empty_like(a_g)

# pass in mapped arrays to kernel method
lin_mult(a_g, b_g, res_g)

# Check on GPU with PyOpenCL Array: Do all the elements reconcile?
print((res_g - (a_g * b_g)).get())

# Check on CPU with Numpy: Do all the elements reconcile? 
res_np = res_g.get()
print(res_np - (a_np * b_np))
print(np.linalg.norm(res_np - (a_np * b_np)))

# Check arrays visually for results
print a_g
print b_g
print res_g
