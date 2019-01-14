from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
import torch
import math
from functions.deformable_convnet_function import Conv2dOffsetFunction


class DeformableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, deformable_groups=1, bias=False, modulate=False):
        """
        :param modulate: bool, if True, initialize the v2 version of deformable convnets, otherwise, initialize the 
                    v1 one. 
        """
        super(DeformableConv, self).__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_chanels must be divisible by groups")
        if in_channels % deformable_groups != 0:
            raise ValueError("in_channels must be divisible by deformable_groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.bais = bias
        self.modulate = modulate

        self.conv2dOffsetFunction = Conv2dOffsetFunction(kernel_size=self.kernel_size, stride=self.stride,
                                                         padding=self.padding, dilation=self.dilation,
                                                         deformable_groups=self.deformable_groups,
                                                         modulate=self.modulate)

        self.weight_conv = Parameter(torch.Tensor(out_channels, in_channels // groups, * self.kernel_size))
        if bias:
            raise NotImplementedError('Bias are not supported currently.')
        else:
            self.register_parameter('bias', None)

        # Note that the spatial resolution of offset is the same with output feature map
        if self.modulate:
            out_channels = (2 + 1) * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
        else:
            out_channels = 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(in_channels=self.in_channels,
                                          out_channels=out_channels,
                                          kernel_size=self.kernel_size, stride=self.stride,
                                          padding=self.padding, dilation=self.dilation,
                                          groups=self.groups, bias=self.bias)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight_conv.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        self.conv_offset_mask.weight.data.fill_(0)
        if self.conv_offset_mask.bias:
            self.conv_offset_mask.bias.data.fill_(0)

    def forward(self, input_feat):
        offset_mask = self.conv_offset_mask(input_feat)
        # TODO: support bias
        return self.conv2dOffsetFunction(input_feat, offset_mask, self.weight_conv)

