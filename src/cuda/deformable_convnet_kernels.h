
void deformable_im2col( // forward convolution
    cudaStream_t stream,
    const float * data_im, // the input data, [batch_size, c, h, w]
    const float * data_offset,
    const float * data_mask,
    const int channels,
    const int height,
    const int width,
    const int ksize_h,
    const int ksize_w,
    const int pad_h,
    const int pad_w,
    const int strid_h,
    const int strid_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    float * data_col // the data changed to column
);

void deformable_col2im( // backward to get the grads w.r.t input feature map
    cudaStream_t stream,
    const float * data_col, // the column data need to be changed
    const float * data_offset,
    const float * data_mask,
    const int channels,
    const int height,
    const int width,
    const int ksize_h,
    const int ksize_w,
    const int pad_h,
    const int pad_w,
    const int strid_h,
    const int strid_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    float * grad_im
);

void deformable_col2im_offset( // backward to get the grads w.r.t the offset
    cudaStream_t stream,
    const float * data_col,
    const float * data_im,
    const float * data_offset,
    const float * data_mask,
    const int channels,
    const int height,
    const int width,
    const int ksize_h,
    const int ksize_w,
    const int pad_h,
    const int pad_w,
    const int strid_h,
    const int strid_w,
    const int dilation_h,
    const int dilation_w,
    const int deformable_group,
    float * grad_offset
);

void deformable_col2im_mask(
    cudaStream_t stream,
    const float * data_col, // the grad w.r.t the data_col
    const int channels,
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
    const int deformable_group,
    float * grad_mask
);