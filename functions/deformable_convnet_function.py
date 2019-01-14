from torch.autograd import Function
from _ext import deformable_convnet
import torch.nn.functional as F
import torch


class Conv2dOffsetFunction(Function):
    def __init__(self, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1),
                 deformable_groups=1, modulate=False):
        super(Conv2dOffsetFunction, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        self.modulate = modulate

    def forward(self, feature_map, offset_mask, weight):

        # TODO: support bias
        self.save_for_backward(feature_map, offset_mask, weight)

        offset_mask_channels = offset_mask.size(1)
        if self.modulate:
            boundary_channel = int(2 / 3 * offset_mask_channels)
            offset = offset_mask[:, 0:boundary_channel, :, :].contiguous()
            mask = F.sigmoid(offset_mask[:, boundary_channel:, :, :]).contiguous()
        else:
            offset = offset_mask.contiguous()
            mask = offset_mask[:, 0:int(offset_mask_channels/2), :, :].clone().fill_(1).contiguous()

        output = feature_map.new()
        columns = feature_map.new()

        if feature_map.is_cuda:
            deformable_convnet.deformable_conv_forward_cuda(
                feature_map,
                weight,
                offset,
                mask,
                output,
                columns,
                self.kernel_size[0], # kernel height
                self.kernel_size[1], # kernel width
                self.stride[0],
                self.stride[1],
                self.padding[0],
                self.padding[1],
                self.dilation[0],
                self.dilation[1],
                self.deformable_groups)

            return output
        else:
            raise NotImplementedError('Only CUDA version of deformable convolution is implemented. '
                                      'CPU version is not implemented!')

    def backward(self, grad_outputs):
        # TODO: support bias
        feature_map, offset_mask, weight = self.saved_tensors

        offset_mask_channels = offset_mask.size(1)
        if self.modulate:
            boundary_channel = int(2 / 3 * offset_mask_channels)
            offset = offset_mask[:, 0:boundary_channel, :, :].contiguous()
            mask = F.sigmoid(offset_mask[:, boundary_channel:, :, :]).contiguous()
        else:
            offset = offset_mask.contiguous()
            mask = offset_mask[:, 0:int(offset_mask_channels/2), :, :].clone()
            mask = mask.fill_(1).contiguous()

        batch_size = feature_map.size(0)

        if feature_map.is_cuda:
            scale = 1 / batch_size

            grad_input_f = feature_map.new()
            grad_offset = feature_map.new()
            grad_mask = feature_map.new()
            grad_weight = feature_map.new()
            columns = feature_map.new()

            deformable_convnet.deformable_conv_backward_cuda(
                feature_map,
                offset,
                mask,
                weight,
                columns,
                grad_outputs,
                grad_input_f,
                grad_offset,
                grad_mask,
                grad_weight,
                self.kernel_size[0],
                self.kernel_size[1],
                self.stride[0],
                self.stride[1],
                self.padding[0],
                self.padding[1],
                self.dilation[0],
                self.dilation[1],
                self.deformable_groups,
                scale)

            if self.modulate:
                grad_mask = grad_mask * mask * (1 - mask)  # for sigmoid
                grad_offset_mask = torch.cat((grad_offset, grad_mask), dim=1).contiguous()
            else:
                grad_offset_mask = grad_offset.contiguous()

            return grad_input_f, grad_offset_mask, grad_weight
        else:
            raise NotImplementedError('Only CUDA version of deformable convolution is implemented. '
                                      'CPU version is not implemented!')


