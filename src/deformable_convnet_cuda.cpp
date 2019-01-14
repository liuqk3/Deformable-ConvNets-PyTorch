#include <torch/torch.h>
#include <vector>

// cuda function declarations

at::Tensor deformable_im2col(
    at::Tensor data_im, // [channel, height, width]
    at::Tensor data_offset, // [2*kh*kw, output_h, output_width]
    at::Tensor data_mask,
    const int ksize_h,
    const int ksize_w,
    const int pad_h,
    const int pad_w,
    const int strid_h,
    const int strid_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group // the feature maps are divided into deformable_group groups along
                                // channel dimension, the features in each group share the same offsets.
                                // And it is set to 1 in the deformable paper, that is all features share
                                // the same offsets.
);

at::Tensor deformable_col2im( // backward to get the grads w.r.t input feature map
    at::Tensor data_col, // the grads w.r.t data_col
    at::Tensor data_offset, // [2*k_h*kw, output_h, output_w]
    at::Tensor data_mask,
    const int ksize_h,
    const int ksize_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group // the feature maps are divided into deformable_group groups along
                                // channel dimension, the features in each group share the same offsets.
                                // And it is set to 1 in the deformable paper, that is all features share
                                // the same offsets.
);

at::Tensor deformable_col2im_offset( // backward to get the grads w.r.t the offset
    at::Tensor data_col, // the grad w.r.t the data_col
    at::Tensor data_im,
    at::Tensor data_offset,
    at::Tensor data_mask,
    const int ksize_h,
    const int ksize_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group
);

at::Tensor deformable_col2im_mask(
    at::Tensor data_col, // the grad w.r.t the data_col
    const int height,
    const int width,
    const int ksize_h,
    const int ksize_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group
);

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TENSOR(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> deformable_conv_forward_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor offset,
    at::Tensor mask,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int deformable_group
)
{
    CHECK_TENSOR(input);
    CHECK_TENSOR(weight);
    CHECK_TENSOR(offset);
    CHECK_TENSOR(mask);

    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int out_channels = weight.size(0);
    int kH = weight.size(2);
    int kW = weight.size(3);

    int height_out = (height + 2 * padH - (dilationH * (kH - 1) + 1)) / strideH + 1;
    int width_out = (width + 2 * padW - (dilationW * (kW - 1) + 1)) / strideW + 1;

    // change the input to a column image
    auto columns = at::zeros_like(mask).resize_({1, 1});
    //CHECK_TENSOR(columns);

    for (int b = 0; b < batch_size; b++)
    {
        auto input_b = at::select(input, 0, b);
        auto offset_b = at::select(offset, 0, b);
        auto mask_b = at::select(mask, 0, b);

        auto column_b = deformable_im2col(
            input_b,
            offset_b,
            mask_b,
            kH,
            kW,
            padH,
            padW,
            strideH,
            strideW,
            dilationH,
            dilationW,
            deformable_group);
        column_b = at::unsqueeze(column_b, 0);
        //CHECK_TENSOR(column_b);

        if (b == 0)
        {
            columns = column_b.clone();
        }
        else
        {
            columns = at::cat({columns, column_b}, 0); //[batch_size, channels * kh * kw, output_h, output_w]
        }
    }

    // now get the output
    columns = columns.permute({0, 2, 1}); // change from [b, channels * kH * kW, out_h * out_w] to [b, out_h * out_w, channels * kH * kW,]
    auto output = columns.matmul(weight.view({out_channels, channels * kH * kW}).permute({1, 0})); // [b, out_h * out_w, out_channels]
    output = output.permute({0, 2, 1}).view({batch_size, out_channels, height_out, width_out}).contiguous();

    return {output};
}


std::vector<at::Tensor> deformable_conv_backward_cuda(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor weight,
    at:: Tensor gradOutput,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int deformable_group,
    float scale)
{
    CHECK_TENSOR(input);
    CHECK_TENSOR(offset);
    CHECK_TENSOR(mask);
    CHECK_TENSOR(weight);
    CHECK_TENSOR(gradOutput);

    int batch_size = input.size(0);
    int height = input.size(2);
    int width = input.size(3);

    int out_channels = weight.size(0);
    int channels = weight.size(1);
    int kH = weight.size(2);
    int kW = weight.size(3);

    int height_out = (height + 2 * padH - (dilationH * (kH - 1) + 1)) / strideH + 1;
    int width_out = (width + 2 * padW - (dilationW * (kW - 1) + 1)) / strideW + 1;

    // the gradOut is w.r.t. the output, the size is [b, out_channels, out_h, out_w]
    // we need to get the grad. w.r.t. the columns input data, which has size [b, channels*kH*kW, out_h * out_w]
    auto grad_col =  gradOutput.view({batch_size, out_channels, height_out * width_out}).permute({0, 2, 1}).matmul(
                     weight.view({out_channels, channels * kH * kW})).permute(// [b, height_out * width_out, channels * kh*kW]
                     {0, 2, 1}).contiguous(); // [b, channels * kh*kW, height_out * width_out]


    auto gradInput = at::zeros_like(mask).resize_({1, 1});
    auto gradOffset = at::zeros_like(mask).resize_({1, 1});
    auto gradMask = at::zeros_like(mask).resize_({1, 1});
    auto gradWeight = at::zeros_like(weight).view({out_channels, channels * kH * kW}).contiguous();

    for (int b = 0; b < batch_size; b++)
    {
        auto grad_output_b = at::select(gradOutput, 0, b);
        auto grad_col_b = at::select(grad_col, 0, b);
        auto offset_b = at::select(offset, 0, b);
        auto mask_b = at::select(mask, 0, b);
        auto input_b = at::select(input, 0, b);

        // grad of input
        auto grad_input_b = deformable_col2im(
            grad_col_b,
            offset_b,
            mask_b,
            kH,
            kW,
            padH,
            padW,
            strideH,
            strideW,
            dilationH,
            dilationW,
            deformable_group);
        grad_input_b = at::unsqueeze(grad_input_b, 0);

        // grad of offset
        auto grad_offset_b = deformable_col2im_offset(
            grad_col_b,
            input_b,
            offset_b,
            mask_b,
            kH,
            kW,
            padH,
            padW,
            strideH,
            strideW,
            dilationH,
            dilationW,
            deformable_group);
        grad_offset_b = at::unsqueeze(grad_offset_b, 0);

        // grad of mask
        auto grad_mask_b = deformable_col2im_mask(
            grad_col_b,
            height,
            width,
            kH,
            kW,
            padH,
            padW,
            strideH,
            strideW,
            dilationH,
            dilationW,
            deformable_group);
        grad_mask_b = at::unsqueeze(grad_mask_b, 0);

        if (b == 0)
        {
            gradInput = grad_input_b.clone();
            gradOffset = grad_offset_b.clone();
            gradMask = grad_mask_b.clone();
        }
        else
        {
            gradInput = at::cat({gradInput, grad_input_b}, 0);
            gradOffset = at::cat({gradOffset, grad_offset_b}, 0);
            gradMask = at::cat({gradMask, grad_mask_b}, 0);
        }


        //grad of weight
        // in order to get the grad of weight,
        // we need to change the input to col data firstly
        auto column_b = deformable_im2col( // column_b has the size [channels*kH*kW, out_h*out_w]
            input_b,
            offset_b,
            mask_b,
            kH,
            kW,
            padH,
            padW,
            strideH,
            strideW,
            dilationH,
            dilationW,
            deformable_group);
        gradWeight = at::addmm(gradWeight,
                               grad_output_b.view({out_channels, height_out * width_out}),
                               column_b.permute({1, 0}),
                               1, scale);

    }

    gradWeight = gradWeight.view({out_channels, channels, kH, kW});

    return {gradInput, gradOffset, gradMask, gradWeight};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deformable_conv_forward_cuda", &deformable_conv_forward_cuda, "deformable convolution forward (CUDA)");
  m.def("deformable_conv_backward_cuda", &deformable_conv_backward_cuda, "deformable convolution backward (CUDA)");
}


