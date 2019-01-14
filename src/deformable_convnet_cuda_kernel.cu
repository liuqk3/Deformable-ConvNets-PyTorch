#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>



namespace {
#define CUDA_1D_KERNEL_LOOP(i, n) \
for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024
//const int THREADS_PER_BLOCK = 1024; // the number of threads per block, i.e blockDim.x = 1024
inline int GET_NUM_BLOCKS(const int n)
{
    return (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t deformable_im2col_bilinear(
    const scalar_t * bottom_data, // the input image starting from the top-left of filter
    const int data_width, // the width of input image
    const int height, // the rest height of image
    const int width, // the rest width of image
    scalar_t h, // the fraction h w.r.t. the top-left of filter
    scalar_t w // the fraction w w.r.t. the top-left of filter
){
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high;
    int w_high;

    if (h_low >= height - 1)
    {
        h_high = h_low = height - 1;
        h = (scalar_t)h_low;
    }
    else
    {
        h_high = h_low + 1;
    }

    if (w_low >= width - 1)
    {
        w_high = w_low = width - 1;
        w = (scalar_t)w_low;
    }
    else
    {
        w_high = w_low + 1;
    }

    scalar_t top_left = bottom_data[h_low * data_width + w_low];
    scalar_t top_right = bottom_data[h_low * data_width + w_high];
    scalar_t bottom_left = bottom_data[h_high * data_width + w_low];
    scalar_t bottom_right = bottom_data[h_high * data_width + w_high];

    scalar_t lh = h - h_low;
    scalar_t lw = w - w_low;
    scalar_t hh = 1 - lh;
    scalar_t hw = 1 - lw;

    scalar_t w_tl = hw * hh, w_tr = lw * hh, w_bl = lh * hw, w_br = lh * lw;

    scalar_t val = w_tl * top_left + w_tr * top_right + w_bl * bottom_left + w_br * bottom_right;
    return val;
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t get_im_weight( // get the weight of gradient w.r.t the data_im
    scalar_t argument_h, // fractional h
    scalar_t argument_w, // fractional w
    const int h, // the real h location in the image
    const int w, // the real w location in the image
    const int height, // the height of input image
    const int width // the width of input image
){
    if (argument_h < 0 || argument_h > height || argument_w < 0 || argument_w > width)
    {
        // empty
        return 0;
    }

    argument_h = max(argument_h, (scalar_t)0.0f);
    argument_w = max(argument_w, (scalar_t)0.0f);

    int argument_h_low = (int)argument_h;
    int argument_w_low = (int)argument_w;
    int argument_h_high;
    int argument_w_high;
    if (argument_h_low >= height - 1)
    {
        argument_h_high = argument_h_low = height - 1;
        argument_h = (scalar_t)argument_h_low;
    }
    else
    {
        argument_h_high = argument_h_low + 1;
    }

    if (argument_w_low >= width - 1)
    {
        argument_w_high = argument_w_low = width - 1;
        argument_w = (scalar_t)argument_w_low;
    }
    else
    {
        argument_w_high = argument_w_low + 1;
    }

    scalar_t weight = 0;
    if (h == argument_h_low)
    {
        if (w == argument_w_low)
        {
            weight = (h + 1 - argument_h) * (w + 1 - argument_w); // top_left
        }
        else if (w == argument_w_high)
        {
            weight = (h + 1 - argument_h) * (argument_w + 1 - w); // top_right
        }
    }
    else if (h == argument_h_high)
    {
        if (w == argument_w_low)
        {
            weight = (argument_h + 1 - h) * (w + 1 - argument_w); // bottom_left
        }
        else if (w == argument_w_high)
        {
            weight = (argument_h + 1 - h) * (argument_w + 1 - w); // bottom_right
        }
    }
    return weight;
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t get_offset_weight( // get the weight of gradient w.r.t to the offset
    scalar_t argument_h, // the real fractional h in the image
    scalar_t argument_w, // the real fractional w in the image
    const int height, // the height of input image
    const int width, // the width of input image
    const scalar_t * im_data, // the input image data, starting current channel
    const int data_width, // the width of input image data
    const int bp_dir // denoting x or y
){
    if (argument_h < 0 || argument_h > height || argument_w < 0 || argument_w > width)
    {
        // empty
        return 0;
    }

    if (argument_h < 0)
        argument_h = 0;
    if (argument_w < 0)
        argument_w = 0;

    int argument_h_low = (int)argument_h;
    int argument_w_low = (int)argument_w;
    int argument_h_high;
    int argument_w_high;

    if (argument_h_low >= height - 1)
    {
        argument_h_high = argument_h_low = height - 1;
        argument_h = (scalar_t)argument_h_low;
    }
    else
    {
        argument_h_high = argument_h_low + 1;
    }
    if (argument_w_low >= width - 1)
    {
        argument_w_high = argument_w_low = width - 1;
        argument_w = (scalar_t)argument_w_low;
    }
    else
    {
        argument_w_high = argument_w_low + 1;
    }

    scalar_t weight = 0;
    if (bp_dir == 0) // coordinate h, or y
    {
        weight += -1 * (argument_w_low + 1 - argument_w) * im_data[argument_h_low * data_width + argument_w_low]; // top_left
        weight += -1 * (argument_w - argument_w_low) * im_data[argument_h_low * data_width + argument_w_high]; // top_right
        weight += (argument_w_low + 1 - argument_w) * im_data[argument_h_high * data_width + argument_w_low]; // bottom_left
        weight += (argument_w - argument_w_low) * im_data[argument_h_high * data_width + argument_w_high]; // bottom_right
    }
    else if (bp_dir == 1) // coordinate w, or x
    {
        weight += -1 * (argument_h_low + 1 - argument_h) * im_data[argument_h_low * data_width + argument_w_low]; // top_left
        weight += (argument_h_low + 1 - argument_h) * im_data[argument_h_low * data_width + argument_w_high]; // top_right
        weight += -1 * (argument_h - argument_h_low) * im_data[argument_h_high * data_width + argument_w_low]; // bottom_left
        weight += (argument_h - argument_h_low) * im_data[argument_h_high * data_width + argument_w_high]; // bottom_right
    }

    return weight;
}


template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel(
    const int n,
    const scalar_t * data_im,
    const scalar_t * data_offset,
    const scalar_t * data_mask,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int strid_h,
    const int strid_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int height_col,
    const int width_col,
    scalar_t * data_col
){
    CUDA_1D_KERNEL_LOOP(index, n)
    {
        // index is the index of output matrix.
        // Note that the data_col is a 2D matrix,
        // but we can treat it as a 3D tensor with size
        // [input_channel * kernel_h * kernel_w, output_h, output_w]
        // then it can be indexed with the same mechanism that applied to
        // 3D tensors.
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int c_im = (index / width_col) / height_col;
        const int c_col = c_im * kernel_h * kernel_w;

        // compute deformable group index
        const int deformable_group_index = c_im / channel_per_deformable_group;

        const int h_in = h_col * strid_h - pad_h;
        const int w_in = w_col * strid_w - pad_w; // (h_in, w_in) is the top_left coordinate of kernel in the data_im
        scalar_t * data_col_ptr = data_col + (c_col * height_col + h_col) * width_col + w_col;
        const scalar_t * data_im_ptr = data_im + (c_im * height + h_in) * width + w_in;
        const scalar_t * data_offset_ptr = // 2 is the x, y dimension
                data_offset + deformable_group_index * 2 * (kernel_h * kernel_w) * height_col * width_col;
        const scalar_t * data_mask_ptr = data_mask + deformable_group_index * (kernel_h * kernel_w) * height_col * width_col;

        for (int i = 0; i < kernel_h; ++i)
        {
            for (int j = 0; j < kernel_w; ++j)
            {
                const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
                const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
                const int data_mask_magnitude_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
                const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
                const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
                const scalar_t mask_magnitude = data_mask_ptr[data_mask_magnitude_ptr];

                scalar_t val = 0;

                // get the sampled coordinates
                const scalar_t h_im = h_in + i * dilation_h + offset_h;
                const scalar_t w_im = w_in + j * dilation_w + offset_w;
                if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                {
                    const scalar_t map_h = i * dilation_h + offset_h;
                    const scalar_t map_w = j * dilation_w + offset_w;
                    const int cur_height = height - h_in; // the rest height
                    const int cur_width = width - w_in; // the rest width

                    val = deformable_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
                }

                * data_col_ptr = val * mask_magnitude;
                data_col_ptr += height_col * width_col; // change to the next row in 2D matrix data_col
            }
        }
    }
}


template <typename scalar_t>
__global__ void deformable_col2im_gpu_kernel(
    const int n,
    const scalar_t * data_col,
    const scalar_t * data_offset,
    const scalar_t * data_mask,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int height_col,
    const int width_col,
    scalar_t * grad_im
){
    CUDA_1D_KERNEL_LOOP(index, n) // index is the index of data_col
    {
        // i, j is the index in the filter (weight of this conv)
        const int j = (index / width_col / height_col) % kernel_w;
        const int i = (index / width_col / height_col / kernel_w) % kernel_h;

        const int c_im = index / width_col / height_col / kernel_w / kernel_h; // the channel of input data_im
        // compute the start and end of the output

        const int deformable_group_index = c_im / channel_per_deformable_group;

        int w_col = index % width_col;
        int h_col = (index / width_col) % height_col;
        int w_in = w_col * stride_w - pad_w;
        int h_in = h_col * stride_h - pad_h; // w_in, h_in is the coordinate of top_left of this kernel in the input data

        const scalar_t * data_offset_ptr = data_offset + deformable_group_index * 2 * kernel_h * kernel_w * height_col * width_col;
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        const scalar_t cur_inv_h_data = h_in + i * dilation_h + offset_h;
        const scalar_t cur_inv_w_data = w_in + j * dilation_w + offset_w;

        const scalar_t * data_mask_ptr = data_mask + deformable_group_index * kernel_h * kernel_w * height_col * width_col;
        const int data_mask_magnitude_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const scalar_t mask_magnitude = data_mask_ptr[data_mask_magnitude_ptr];

        const scalar_t cur_top_grad = data_col[index];
        const int cur_h = (int)cur_inv_h_data;
        const int cur_w = (int)cur_inv_w_data;

        // TODO: simplify following back propagation
        for (int dy = -2; dy <= 2; dy++)
        {
            if (cur_h + dy >= 0 && cur_h + dy < height && abs(cur_inv_h_data - (cur_h + dy)) < 1)
            {
                for (int dx = -2; dx <= 2; dx++)
                {
                    if (cur_w + dx >= 0 && cur_w + dx < width && abs(cur_inv_w_data - (cur_w + dx)) < 1)
                    {
                        int cur_bottom_grad_pos = (c_im * height + cur_h + dy) * width + cur_w + dx;
                        scalar_t weight = get_im_weight(
                            cur_inv_h_data,
                            cur_inv_w_data,
                            cur_h + dy,
                            cur_w + dx,
                            height,
                            width);
                        atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad * mask_magnitude);
                    }
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void deformable_col2im_offset_gpu_kernel(
    const int n, // number of threads
    const scalar_t * data_col,
    const scalar_t * data_im,
    const scalar_t * data_offset,
    const scalar_t * data_mask,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int height_col,
    const int width_col,
    scalar_t * grad_offset)
{
    CUDA_1D_KERNEL_LOOP(index, n) // index is the index of grad_offset
    {
        scalar_t val = 0;

        // compute the start of the output
        // w and h are the index of data_col, while c is the index of data_offset
        int w_col = index % width_col;
        int h_col = (index / width_col) % height_col;
        int c = index / width_col / height_col;

        const int deformable_group_index = c / (2 * kernel_h * kernel_w); // if deformable_group is 1, this value is 0
                                                                          // deformable_group = c / width_col / height_col
                                                                          // c is the number of channels of offset
        const int col_step = kernel_h * kernel_w;
        int cnt = 0;
        const scalar_t * data_col_ptr = // treat data_col as a 3D tensor with size [c, h, w], where c = input_c * k_h * k_w
            data_col + deformable_group_index * channel_per_deformable_group * width_col * height_col;
        const scalar_t * data_im_ptr =
            data_im + deformable_group_index * channel_per_deformable_group / kernel_h / kernel_w * height * width;
        const scalar_t * data_offset_ptr =
            data_offset + deformable_group_index * 2 * kernel_h * kernel_w * height_col * width_col;
        const scalar_t * data_mask_ptr =
            data_mask + deformable_group_index * kernel_h * kernel_w * height_col * width_col;
        int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

        for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step)
        {   // each group, the first k_h * k_w channels are y offsets, and the rest k_h * k_w channels are x offsets
            const int col_pos = ((col_c * height_col) + h_col) * width_col + w_col;
            const int bp_dir = offset_c % 2; // compute y (0) or w (1)

            int j = (col_pos / width_col / height_col) % kernel_w;
            int i = ((col_pos / width_col / height_col) / kernel_w) % kernel_h;
            int w_in = w_col * stride_w - pad_w;
            int h_in = h_col * stride_h - pad_h;

            const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col);
            const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col);
            const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
            const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
            scalar_t inv_h = h_in + i * dilation_h + offset_h;
            scalar_t inv_w = w_in + i * dilation_w + offset_w;

            const int data_mask_magnitude_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
            const scalar_t mask_magnitude = data_mask_ptr[data_mask_magnitude_ptr];

            if (inv_h < 0 || inv_w < 0 || inv_h >= height || inv_w >= width)
            {
                inv_h = inv_w = -1;
            }

            const scalar_t weight = get_offset_weight(
                inv_h,
                inv_w,
                height,
                width,
                data_im_ptr + cnt * height * width,
                width,
                bp_dir);
            val += weight * data_col_ptr[col_pos] * mask_magnitude;
            cnt += 1;
        }

        grad_offset[index] = val;
    }
}

template <typename scalar_t>
__global__ void deformable_col2im_mask_gpu_kernel(
    const int n, // number of threads
    const scalar_t * data_col,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int deformable_group,
    const int height_col,
    const int width_col,
    scalar_t * grad_mask)
{
    CUDA_1D_KERNEL_LOOP(index, n) // index is the index of grad_mask
    {
        scalar_t val = 0;

        // note that:
        // num_threads = deformable_group * (ksize_h * ksize_w) * height_col * width_col;
        // so, index = (((group_index * (ksize_h + i) * ksize_w + j) * height_col + h_col) * width_col + w_col;

        // compute the start of the output
        // w_col and h_col are the index of data_col, while c is the index of data_mask
        int w_col = index % width_col;
        int h_col = (index / width_col) % height_col;
        int c = (index / width_col / height_col);

        const int deformable_group_index = c / (kernel_h * kernel_w);

        const int col_step = kernel_h * kernel_w;
        const int channel_per_deformable_group = channels * kernel_h * kernel_w / deformable_group;

        const scalar_t * data_col_ptr =
            data_col + deformable_group_index * channel_per_deformable_group * width_col * height_col;
        int mask_c = c - deformable_group_index * (kernel_h * kernel_w);
        for (int col_c = mask_c; col_c < channel_per_deformable_group; col_c += col_step)
        {
            int col_pos = (col_c * height_col + h_col) * width_col + w_col;
            val += data_col_ptr[col_pos];
        }

        grad_mask[index] = val;

    }
}

} // namespace


// https://blog.csdn.net/mrhiuser/article/details/52672824?tdsourcetag=s_pctim_aiomsg
//std::vector<at::Tensor> deformable_im2col(
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
){

    const auto channels = data_im.size(0);
    const auto height = data_im.size(1);
    const auto width = data_im.size(2);

    // we are going to launch channels * height_col * width_col threads,
    // each thread responsible for copying a sub-column with length ksize_h * ksize_w
    // in 2D matrix data_col.
    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / strid_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / strid_w + 1;

    auto data_col = at::zeros_like(data_im).resize_({channels * ksize_h * ksize_w, height_col * width_col});

    int num_threads = channels * height_col * width_col;
    int channel_per_deformable_group = channels / deformable_group;
    int num_blocks = GET_NUM_BLOCKS(num_threads);

    // launch
    AT_DISPATCH_FLOATING_TYPES(data_im.type(), "deformable_im2col", ([&] {
        deformable_im2col_gpu_kernel<scalar_t><<<num_blocks, THREADS_PER_BLOCK>>>(
        num_threads,
        data_im.data<scalar_t>(),
        data_offset.data<scalar_t>(),
        data_mask.data<scalar_t>(),
        height,
        width,
        ksize_h,
        ksize_w,
        pad_h,
        pad_w,
        strid_h,
        strid_w,
        dilation_h,
        dilation_w,
        channel_per_deformable_group,
        height_col,
        width_col,
        data_col.data<scalar_t>());
    }));

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("\n error in deformable_im2col: %s\n", cudaGetErrorString(err));
    }

    return data_col;
}


//std::vector<at::Tensor> deformable_col2im( // backward to get the grads w.r.t input feature map
at::Tensor deformable_col2im(
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
){

    int channels = data_col.size(0) / ksize_h / ksize_w;

    int height_col = data_offset.size(1);
    int width_col = data_offset.size(2);

    int height = (height_col - 1) * stride_h + (dilation_h * (ksize_h - 1) + 1) - 2 * pad_h;
    int width = (width_col - 1) * stride_w + (dilation_w * (ksize_w - 1) + 1) - 2 * pad_w;

    auto grad_im = at::zeros_like(data_mask).resize_({channels, height, width});

    int num_threads = channels * ksize_h * ksize_w * height_col * width_col;
    int channel_per_deformable_group = channels / deformable_group;

    // to avoid involving atomic operations, we will lauch one thread per bottom dimension,
    // and then in the kernel add up the top dimensions.
    int num_blocks = GET_NUM_BLOCKS(num_threads);

    AT_DISPATCH_FLOATING_TYPES(data_col.type(), "deformable_col2im", ([&] {
        deformable_col2im_gpu_kernel<scalar_t><<<num_blocks, THREADS_PER_BLOCK>>>(
            num_threads,
            data_col.data<scalar_t>(),
            data_offset.data<scalar_t>(),
            data_mask.data<scalar_t>(),
            channels,
            height,
            width,
            ksize_h,
            ksize_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            channel_per_deformable_group,
            height_col,
            width_col,
            grad_im.data<scalar_t>());
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("\n error in deformable_col2im: %s", cudaGetErrorString(err));
    }

    return grad_im;
}


//std::vector<at::Tensor> deformable_col2im_offset( // backward to get the grads w.r.t the offset
at::Tensor deformable_col2im_offset(
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
){
    int channels = data_col.size(0) / ksize_h / ksize_w;

    int height_col = data_offset.size(1);
    int width_col = data_offset.size(2);

    int height = (height_col - 1) * stride_h + (dilation_h * (ksize_h - 1) + 1) - 2 * pad_h;
    int width = (width_col - 1) * stride_w + (dilation_w * (ksize_w - 1) + 1) - 2 * pad_w;

    auto grad_offset = at::zeros_like(data_offset);

    // to avoid involving atomic operations, we will launch one thread per
    // bottom dimension, and then in the thread add up the top dimensions.
    int num_threads = height_col * width_col * 2 * ksize_h * ksize_w * deformable_group;

    int channel_per_deformable_group = channels * ksize_h * ksize_w / deformable_group;

    int num_blocks = GET_NUM_BLOCKS(num_threads);

    AT_DISPATCH_FLOATING_TYPES(data_col.type(), "defomable_col2im_offset", ([&] {
        deformable_col2im_offset_gpu_kernel<scalar_t><<<num_blocks, THREADS_PER_BLOCK>>>(
            num_threads,
            data_col.data<scalar_t>(),
            data_im.data<scalar_t>(),
            data_offset.data<scalar_t>(),
            data_mask.data<scalar_t>(),
            height,
            width,
            ksize_h,
            ksize_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            channel_per_deformable_group,
            height_col,
            width_col,
            grad_offset.data<scalar_t>());
    }));

     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess)
     {
        printf("\n error in deformable_col2im_offset: %s\n ", cudaGetErrorString(err));
     }

     return grad_offset;

}

//std::vector<at::Tensor> deformable_col2im_mask(
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
){

    int channels = data_col.size(0) / ksize_h / ksize_w;

    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;

    auto grad_mask = at::zeros_like(data_col).resize_({ksize_h * ksize_w * deformable_group, height_col, width_col});

    // to avoid involving atomic operations, we will launch one thread per
    // bottom dimension, and then in the thread add up the top dimensions.
    int num_threads = deformable_group * (ksize_h * ksize_w) * height_col * width_col;

    // printf("deformable_col2im_mask, num_threads: %d \n", num_threads);
    int num_blocks = GET_NUM_BLOCKS(num_threads);

    AT_DISPATCH_FLOATING_TYPES(data_col.type(), "deformable_col2im_mask", ([&] {
        deformable_col2im_mask_gpu_kernel<scalar_t><<<num_blocks, THREADS_PER_BLOCK>>>(
            num_threads,
            data_col.data<scalar_t>(),
            channels,
            height,
            width,
            ksize_h,
            ksize_w,
            deformable_group,
            height_col,
            width_col,
            grad_mask.data<scalar_t>());
    }));

    //printf("deformable_col2im_mask, num_threads: %d \n", num_threads);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
       printf("\n error in deformable_col2im_mask: %s\n", cudaGetErrorString(err));
    }

    return grad_mask;
}























