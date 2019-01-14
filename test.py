import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from modules.deformable_convnet import DeformableConv
from torch.autograd import Variable
import time

batch_size = 4

in_channels = 1024
out_channels = 1024

in_h = 64
in_w = 64

k_h = 3
k_w = 3

deformable_conv = DeformableConv(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(k_h, k_w), padding=(1, 1),
                                 modulate=True).cuda()

input = Variable(torch.randn(batch_size, in_channels, in_h, in_w).cuda(), requires_grad=True)

print('warming up')
output = deformable_conv(input)

t1 = time.time()
output = deformable_conv(input)
t2 = time.time()
print('forward time: ', t2 - t1)

output.backward(output.data)
t3 = time.time()
print('backward time: ', t3 - t2)

print('output size: ', output.size())
print('input gradient size: ', input.grad.size())


