#include<THC/THC.h>
#include "cuda/deformable_convnet_kernels.h"

extern THCState * state;

void shape_check(
    THCState * state,
    THCudaTensor * input,
    THCudaTensor * offset,
    THCudaTensor * gradOutput,
    THCudaTensor * weight,
    int kH,
    int kW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int deformable_group
)
{
    if (!THCudaTensor_isContiguous(state, weight))
        THError("weight tensor has to be contiguous.");

    if (weight->nDimension != 4)
        THError("4D weight tensor (output_channel, input_channel, kernel_h, kernel_w) expected, but got : %d.", weight->nDimension);

    if (kW <= 0 || kH <= 0)
        THError("kernel size should be greater than zero, but got (%d, %d).", kH, kW);

    if (weight->size[2] != kH || weight->size[3] != kW)
        THError("Kernel size should be consistent with weight, but got kernel size (%d, %d), weight size (%d, %d, %d, %d).",
               kH, kW, weight->size[0], weight->size[1], weight->size[2], weight->size[3]);

    if (strideW <= 0 || strideH <= 0)
        THError("stride should be greater than zero, but got stride (%d, %d).", strideH, strideW);

    if (dilationW <= 0 || dilationH <= 0)
        THError("dilation should be greater than zero, but got (%d, %d).", dilationH, dilationW);

    long nInputPlane = weight->size[1];
    long inputHeight = input->size[2];
    long inputWidth = input->size[3];
    long nOutputPlane = weight->size[0];

    long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / strideH + 1;
    long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / strideW + 1;

    if (nInputPlane % deformable_group != 0)
        THError("input channels must divide deformable group size.");

    if (outputHeight < 1 || outputWidth < 1)
        THError("Given input size: (%ld, %ld, %ld), Calculated output size: (%ld, %ld, %ld).",
                nInputPlane, inputHeight, inputWidth, nOutputPlane, outputHeight, outputWidth);

    if (input->size[1] != nInputPlane)
        THError("invalid number of input planes, expected: %d, but got: %d.", nInputPlane, input->size[1]);

    if (inputHeight <= kH || inputWidth <= kW)
        THError("input image spatial size is smaller than kernel.");

    if (offset->size[2] != outputHeight || offset->size[3] != outputWidth)
        THError("invalid spatial size of offset, expected (%d, %d), but got (%d, %d).",
               outputHeight, outputWidth, offset->size[2], offset->size[3]);

    if (offset->size[1] != deformable_group * 2 * kH * kW)
        THError("invalid number of channels of offset.");

    if (gradOutput != NULL)
    {
        if (gradOutput->size[1] != nOutputPlane)
            THError("invalid number of gradOutput channels, expected: %d, but got : %d.",
                   nOutputPlane, gradOutput->size[1]);

        if (gradOutput->size[2] != outputHeight || gradOutput->size[3] != outputWidth)
            THError("invalid spatial size of gradOutput, expected (%d, %d), but got (%d, %d).",
                   outputHeight, outputWidth, gradOutput->size[2], gradOutput->size[3]);
    }
}

void deformable_conv_forward_cuda(
    THCudaTensor * input,
    THCudaTensor * weight,
    THCudaTensor * offset,
    THCudaTensor * mask,
    THCudaTensor * output,
    THCudaTensor * columns,
    int kH,
    int kW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int deformable_group
){

    //printf("deformable_conv_forward_cuda()\n");

    THCAssertSameGPU(THCudaTensor_checkGPU(state, 6, input, weight, offset, mask, output, columns));
    shape_check(state, input, offset, NULL, weight, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW, deformable_group);

    input = THCudaTensor_newContiguous(state, input);
    weight = THCudaTensor_newContiguous(state, weight);
    offset = THCudaTensor_newContiguous(state, offset);
    mask = THCudaTensor_newContiguous(state, mask);

    long batchSize = input->size[0];
    long nInputPlane = input->size[1];
    long inputHeight = input->size[2];
    long inputWidth = input->size[3];

    long nOutputPlane = weight->size[0];
    long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / strideH + 1;
    long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / strideW + 1;

    //THArgCheck(offset->size[0] == batchSize, 3, "invalud batch size of offset.");

    THCudaTensor_resize4d(state, output, batchSize, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize2d(state, columns, kH * kW * nInputPlane, outputHeight * outputWidth);

    THCudaTensor * input_n = THCudaTensor_new(state);
    THCudaTensor * offset_n = THCudaTensor_new(state);
    THCudaTensor * mask_n = THCudaTensor_new(state);
    THCudaTensor * output_n = THCudaTensor_new(state);

    for (int elt = 0; elt < batchSize; elt++)
    {
        THCudaTensor_select(state, input_n, input, 0, elt);
        THCudaTensor_select(state, offset_n, offset, 0, elt);
        THCudaTensor_select(state, mask_n, mask, 0, elt);
        THCudaTensor_select(state, output_n, output, 0, elt);

        THCudaTensor_zero(state, output_n);
        deformable_im2col(
            THCState_getCurrentStream(state),
            THCudaTensor_data(state, input_n),
            THCudaTensor_data(state, offset_n),
            THCudaTensor_data(state, mask_n),
            nInputPlane,
            inputHeight,
            inputWidth,
            kH,
            kW,
            padH,
            padW,
            strideH,
            strideW,
            dilationH,
            dilationW,
            deformable_group,
            THCudaTensor_data(state, columns));

        // THCudaBlas_Sgemm() refer to: https://blog.csdn.net/g11d111/article/details/83021651
        long m = nOutputPlane;
        long n = columns->size[1];
        long k = nInputPlane * kH * kW;
        THCudaBlas_Sgemm(state, 'n', 'n', n, m, k, 1.0f,
                         THCudaTensor_data(state, columns), n, // columns: o_h*o_w x i_c*k_h*k_w
                         THCudaTensor_data(state, weight), k, 0.0f, // weight: i_c*k_h*k_w x o_c
                         THCudaTensor_data(state, output_n), n); // outout_n: o_h*o_w x o_c
    }

    THCudaTensor_free(state, input_n);
    THCudaTensor_free(state, offset_n);
    THCudaTensor_free(state, output_n);
    THCudaTensor_free(state, mask_n);

    /*
    THCudaTensor_free(state, input);
    THCudaTensor_free(state, weight);
    THCudaTensor_free(state, offset);
    THCudaTensor_free(state, mask);
    THCudaTensor_free(state, output);
    THCudaTensor_free(state, columns);
    */
}

void deformable_conv_backward_cuda(
    THCudaTensor * input,
    THCudaTensor * offset,
    THCudaTensor * mask,
    THCudaTensor * weight,
    THCudaTensor * columns,
    THCudaTensor * gradOutput,
    THCudaTensor * gradInput,
    THCudaTensor * gradOffset,
    THCudaTensor * gradMask,
    THCudaTensor * gradWeight,
    int kH,
    int kW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilationH,
    int dilationW,
    int deformable_group,
    float scale)
{
    //printf("deformable_conv_backward_input_cuda()\n");

    THCAssertSameGPU(THCudaTensor_checkGPU(state, 9, input, offset, mask, weight, columns, gradOutput, gradInput, gradOffset, gradMask, gradWeight));
    shape_check(state, input, offset, gradOutput, weight, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW, deformable_group);

    input = THCudaTensor_newContiguous(state, input);
    offset = THCudaTensor_newContiguous(state, offset);
    mask = THCudaTensor_newContiguous(state, mask);
    gradOutput = THCudaTensor_newContiguous(state, gradOutput);
    weight = THCudaTensor_newContiguous(state, weight);

    long batchSize = input->size[0];
    long nInputPlane = input->size[1];
    long inputHeight = input->size[2];
    long inputWidth = input->size[3];

    long nOutputPlane = weight->size[0];
    long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / strideH + 1;
    long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / strideW + 1;

    // THArgCheck(offset->size[0] == batchSize && mask->size[0] == batchSize, 3, "invalid batch size of offset");
    THCudaTensor_resize4d(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);
    THCudaTensor_resize4d(state, gradOffset, offset->size[0], offset->size[1], offset->size[2], offset->size[3]);
    THCudaTensor_resize4d(state, gradMask, mask->size[0], mask->size[1], mask->size[2], mask->size[3]);
    THCudaTensor_resize4d(state, gradWeight, weight->size[0], weight->size[1], weight->size[2], weight->size[3]);
    THCudaTensor_resize2d(state, columns, nInputPlane * kH * kW, outputHeight * outputWidth);

    THCudaTensor * gradInput_n = THCudaTensor_new(state);
    THCudaTensor * gradOffset_n = THCudaTensor_new(state);
    THCudaTensor * mask_n = THCudaTensor_new(state);
    THCudaTensor * input_n = THCudaTensor_new(state);
    THCudaTensor * offset_n = THCudaTensor_new(state);
    THCudaTensor * gradOutput_n = THCudaTensor_new(state);
    THCudaTensor * gradMask_n = THCudaTensor_new(state);

    long m = 0; //nInputPlane * kH * kW;
    long n = 0; //outputHeight * outputWidth
    long k = 0; //nOutputPlane;

    for (int elt = 0; elt < batchSize; elt++)
    {
        THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);
        THCudaTensor_select(state, gradOffset_n, gradOffset, 0, elt);
        THCudaTensor_select(state, input_n, input, 0, elt);
        THCudaTensor_select(state, offset_n, offset, 0, elt);
        THCudaTensor_select(state, mask_n, mask, 0, elt);
        THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);
        THCudaTensor_select(state, gradMask_n, gradMask, 0, elt);

        m = nInputPlane * kH * kW;
        n = outputHeight * outputWidth;
        k = nOutputPlane;

        THCudaBlas_Sgemm(state, 'n', 't', n, m, k, 1.0f,
            THCudaTensor_data(state, gradOutput_n), n, //gradOutput_n: o_h*o_w x o_c
            THCudaTensor_data(state, weight), m, 0.0f, // weight: i_c*k_h*k_w x o_c
            THCudaTensor_data(state, columns), n); // columns: o_h*o_w x i_c*k_h*k_w

        // grad of input feature
        deformable_col2im(
            THCState_getCurrentStream(state),
            THCudaTensor_data(state, columns),
            THCudaTensor_data(state, offset_n),
            THCudaTensor_data(state, mask_n),
            nInputPlane,
            inputHeight,
            inputWidth,
            kH,
            kW,
            padH,
            padW,
            strideH,
            strideW,
            dilationH,
            dilationW,
            deformable_group,
            THCudaTensor_data(state, gradInput_n));

        // grad of offset
        deformable_col2im_offset(
            THCState_getCurrentStream(state),
            THCudaTensor_data(state, columns),
            THCudaTensor_data(state, input_n),
            THCudaTensor_data(state, offset_n),
            THCudaTensor_data(state, mask_n),
            nInputPlane,
            inputHeight,
            inputWidth,
            kH,
            kW,
            padH,
            padW,
            strideH,
            strideW,
            dilationH,
            dilationW,
            deformable_group,
            THCudaTensor_data(state, gradOffset_n));

        // grad of mask
        deformable_col2im_mask(
            THCState_getCurrentStream(state),
            THCudaTensor_data(state, columns),
            nInputPlane,
            inputHeight,
            inputWidth,
            kH,
            kW,
            padH,
            padW,
            strideH,
            strideW,
            dilationH,
            dilationW,
            deformable_group,
            THCudaTensor_data(state, gradMask_n));

        // gard of weight
        m = nOutputPlane;
        n = nInputPlane * kH * kW;
        k = outputHeight * outputWidth;

        deformable_im2col(
            THCState_getCurrentStream(state),
            THCudaTensor_data(state, input_n),
            THCudaTensor_data(state, offset_n),
            THCudaTensor_data(state, mask_n),
            nInputPlane,
            inputHeight,
            inputWidth,
            kH,
            kW,
            padH,
            padW,
            strideH,
            strideW,
            dilationH,
            dilationW,
            deformable_group,
            THCudaTensor_data(state, columns));

        THCudaBlas_Sgemm(state, 't', 'n', n, m, k, scale,
            THCudaTensor_data(state, columns), k, //output_h*output_w x input_c*kH*kW
            THCudaTensor_data(state, gradOutput_n), k, 1.0f, // output_h*output_w x output_c
            THCudaTensor_data(state, gradWeight), n); // input_c*kH*kW x output_c
    }

    THCudaTensor_free(state, gradInput_n);
    THCudaTensor_free(state, gradOffset_n);
    THCudaTensor_free(state, input_n);
    THCudaTensor_free(state, offset_n);
    THCudaTensor_free(state, gradOutput_n);
    THCudaTensor_free(state, mask_n);
    THCudaTensor_free(state, gradMask_n);
    /*
    THCudaTensor_free(state, input);
    THCudaTensor_free(state, offset);
    THCudaTensor_free(state, mask);
    THCudaTensor_free(state, weight);
    THCudaTensor_free(state, columns);
    THCudaTensor_free(state, gradOutput);
    THCudaTensor_free(state, gradOffset);
    THCudaTensor_free(state, gradMask);
    THCudaTensor_free(state, gradWeight);
    */
}
