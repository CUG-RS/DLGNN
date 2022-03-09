from collections import namedtuple
import cupy
import torch
from string import Template
import numpy as np

Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


@cupy.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    #print( 'type(code): ',type(code))#<class 'str'>
    #print( 'type(kernel_code): ',type(kernel_code))#<class 'cupy.cuda.function.Module'>
    #kernel_code = np.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)
