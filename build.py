import os
import torch
from torch.utils.ffi import create_extension

sources = []
headers = []
extra_objects = []

defines = []
with_cuda = False

if torch.cuda.is_available():
    print('including cuda code.')
    sources += ['src/deformable_convnet_gpu.c']
    headers += ['src/deformable_convnet_gpu.h']
    extra_objects += ['src/cuda/deformable_convnet_kernels.cu.o']

    defines += [('WITH_CUDA', None)]
    with_cuda = True
else:
    print('Only cuda code is implemented!')

this_file = os.path.dirname(os.path.abspath(__file__))
print('This file directory: ' + this_file)

sources = [os.path.join(this_file, fname) for fname in sources]
headers = [os.path.join(this_file, fname) for fname in headers]
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

#  extra_compile_arg = ['-std=c99']

ffi = create_extension(
    os.path.join(this_file, '_ext.deformable_convnet'),
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=this_file,
    with_cuda=with_cuda,
    #  extra_compile_arg=extra_compile_arg,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()




