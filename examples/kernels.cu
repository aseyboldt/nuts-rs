#include <cub/block/block_reduce.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>


const int ITEMS_PER_THREAD = 1;
const int BLOCK_THREADS = 256;

extern "C" __global__ void sum(const size_t n, const float *src, float *dst) {
    using BlockReduceT = cub::BlockReduce<float, BLOCK_THREADS>;
    using BlockLoadT = cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>;

    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockReduceT::TempStorage reduce;
    } temp_storage;

    float thread_data[ITEMS_PER_THREAD];

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    BlockLoadT(temp_storage.load).Load(src + block_offset, thread_data, n - block_offset, 0.0);

    __syncthreads();

    float sum = BlockReduceT(temp_storage.reduce).Sum(thread_data);

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(dst, sum);
    }
}

extern "C" __global__ void normal_logp(
    const size_t n,
    const float *x,
    float *logp_grad,
    float *logp)
{
    using BlockReduceT = cub::BlockReduce<float, BLOCK_THREADS, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
    using BlockLoadT = cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockStoreT = cub::BlockStore<float, BLOCK_THREADS, ITEMS_PER_THREAD>;

    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockReduceT::TempStorage reduce;
        typename BlockStoreT::TempStorage store;
    } temp_storage;

    float thread_x[ITEMS_PER_THREAD];
    float thread_logp[ITEMS_PER_THREAD];

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    BlockLoadT(temp_storage.load).Load(x + block_offset, thread_x, n - block_offset, 0.0);

    __syncthreads();

    // Write element-wise logp value into temporary storage
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        float val = thread_x[i];
        thread_logp[i] = - 0.5 * (val * val);
    }

    // Compute the block-wise sum of logp values
    float logp_sum = BlockReduceT(temp_storage.reduce).Sum(thread_logp);

    // Write the gradient values into temporary storage
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        thread_x[i] = -thread_x[i];
    }

    __syncthreads();

    BlockStoreT(temp_storage.store).Store(logp_grad + block_offset, thread_x, n - block_offset);

    if (threadIdx.x == 0) {
        atomicAdd(logp, logp_sum);
    }
}


extern "C" __global__ void scalar_prods3(
    const int n,
    const float *positive1,
    const float *negative1,
    const float *positive2,
    const float *x,
    const float *y,
    float *prods
) {
    using BlockLoadT = cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockReduceT = cub::BlockReduce<float, BLOCK_THREADS, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockReduceT::TempStorage reduce;
    } temp_storage;

    float thread_p1[ITEMS_PER_THREAD];
    float thread_n1[ITEMS_PER_THREAD];
    float thread_p2[ITEMS_PER_THREAD];
    float thread_other[ITEMS_PER_THREAD];

    float prod[ITEMS_PER_THREAD];

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int n_remaining = n - block_offset;
    BlockLoadT(temp_storage.load).Load(positive1 + block_offset, thread_p1, n_remaining, 0.0);
    __syncthreads();
    BlockLoadT(temp_storage.load).Load(negative1 + block_offset, thread_n1, n_remaining, 0.0);
    __syncthreads();
    BlockLoadT(temp_storage.load).Load(positive2 + block_offset, thread_p2, n_remaining, 0.0);
    __syncthreads();
    BlockLoadT(temp_storage.load).Load(x + block_offset, thread_other, n_remaining, 0.0);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        prod[i] = (thread_p1[i] - thread_n1[i] + thread_p2[i]) * thread_other[i];
    }

    __syncthreads();
    float prod1_sum = BlockReduceT(temp_storage.reduce).Sum(prod);

    __syncthreads();
    BlockLoadT(temp_storage.load).Load(y + block_offset, thread_other, n_remaining, 0.0);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        prod[i] = (thread_p1[i] - thread_n1[i] + thread_p2[i]) * thread_other[i];
    }

    __syncthreads();
    float prod2_sum = BlockReduceT(temp_storage.reduce).Sum(prod);

    if (threadIdx.x == 0) {
        atomicAdd(prods, prod1_sum);
        atomicAdd(prods + 1, prod2_sum);
    }
}


extern "C" __global__ void scalar_prods2(
    const int n,
    const float *positive1,
    const float *positive2,
    const float *x,
    const float *y,
    float *prods
) {
    using BlockLoadT = cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockReduceT = cub::BlockReduce<float, BLOCK_THREADS, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockReduceT::TempStorage reduce;
    } temp_storage;

    float thread_p1[ITEMS_PER_THREAD];
    float thread_p2[ITEMS_PER_THREAD];
    float thread_other[ITEMS_PER_THREAD];

    float prod[ITEMS_PER_THREAD];

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int n_remaining = n - block_offset;
    BlockLoadT(temp_storage.load).Load(positive1 + block_offset, thread_p1, n_remaining, 0.0);
    __syncthreads();
    BlockLoadT(temp_storage.load).Load(positive2 + block_offset, thread_p2, n_remaining, 0.0);
    __syncthreads();
    BlockLoadT(temp_storage.load).Load(x + block_offset, thread_other, n_remaining, 0.0);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        prod[i] = (thread_p1[i] + thread_p2[i]) * thread_other[i];
    }

    __syncthreads();
    float prod1_sum = BlockReduceT(temp_storage.reduce).Sum(prod);

    __syncthreads();
    BlockLoadT(temp_storage.load).Load(y + block_offset, thread_other, n_remaining, 0.0);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        prod[i] = (thread_p1[i] + thread_p2[i]) * thread_other[i];
    }

    __syncthreads();
    float prod2_sum = BlockReduceT(temp_storage.reduce).Sum(prod);

    if (threadIdx.x == 0) {
        atomicAdd(prods, prod1_sum);
        atomicAdd(prods + 1, prod2_sum);
    }
}


extern "C" __global__ void sq_norm_sum(
    const int n,
    const float *x,
    const float *y,
    float *out
) {
    using BlockLoadT = cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockReduceT = cub::BlockReduce<float, BLOCK_THREADS, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockReduceT::TempStorage reduce;
    } temp_storage;

    float thread_x[ITEMS_PER_THREAD];
    float thread_y[ITEMS_PER_THREAD];
    float thread_out[ITEMS_PER_THREAD];

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int n_remaining = n - block_offset;
    BlockLoadT(temp_storage.load).Load(x + block_offset, thread_x, n_remaining, 0.0);
    __syncthreads();
    BlockLoadT(temp_storage.load).Load(y + block_offset, thread_y, n_remaining, 0.0);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        float val = thread_x[i] + thread_y[i];
        thread_out[i] = val * val;
    }

    __syncthreads();
    float thread_sum = BlockReduceT(temp_storage.reduce).Sum(thread_out);

    if (threadIdx.x == 0) {
        atomicAdd(out, thread_sum);
    }
}

extern "C" __global__ void axpy_out(
    const size_t n,
    const float *x,
    const float *y,
    const float a,
    float *out
) {
    using BlockLoadT = cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockStoreT = cub::BlockStore<float, BLOCK_THREADS, ITEMS_PER_THREAD>;

    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockStoreT::TempStorage store;
    } temp_storage;

    float thread_x[ITEMS_PER_THREAD];
    float thread_y[ITEMS_PER_THREAD];
    float thread_out[ITEMS_PER_THREAD];

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int n_remaining = n - block_offset;
    BlockLoadT(temp_storage.load).Load(x + block_offset, thread_x, n_remaining, 0.0);

    __syncthreads();
    BlockLoadT(temp_storage.load).Load(y + block_offset, thread_y, n_remaining, 0.0);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        thread_out[i] = a * thread_x[i] + thread_y[i];
    }

    __syncthreads();
    BlockStoreT(temp_storage.store).Store(out + block_offset, thread_out, n_remaining);
}

extern "C" __global__ void axpy(
    const size_t n,
    const float *x,
    float *y,
    const float a
) {
    using BlockLoadT = cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockStoreT = cub::BlockStore<float, BLOCK_THREADS, ITEMS_PER_THREAD>;

    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockStoreT::TempStorage store;
    } temp_storage;

    float thread_x[ITEMS_PER_THREAD];
    float thread_y[ITEMS_PER_THREAD];
    float thread_out[ITEMS_PER_THREAD];

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int n_remaining = n - block_offset;
    BlockLoadT(temp_storage.load).Load(x + block_offset, thread_x, n_remaining, 0.0);

    __syncthreads();
    BlockLoadT(temp_storage.load).Load(y + block_offset, thread_y, n_remaining, 0.0);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        thread_out[i] = a * thread_x[i] + thread_y[i];
    }

    __syncthreads();
    BlockStoreT(temp_storage.store).Store(y + block_offset, thread_out, n_remaining);
}


extern "C" __global__ void fill_array(
    const size_t n,
    float *array,
    const float val
) {
    using BlockStoreT = cub::BlockStore<float, BLOCK_THREADS, ITEMS_PER_THREAD>;

    __shared__ union {
        typename BlockStoreT::TempStorage store;
    } temp_storage;

    float thread_out[ITEMS_PER_THREAD];

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        thread_out[i] = val;
    }

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int n_remaining = n - block_offset;
    BlockStoreT(temp_storage.store).Store(array + block_offset, thread_out, n_remaining);
}


extern "C" __global__ void array_mult(
    const size_t n,
    const float *x,
    const float *y,
    float *out
) {
    using BlockLoadT = cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockStoreT = cub::BlockStore<float, BLOCK_THREADS, ITEMS_PER_THREAD>;

    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockStoreT::TempStorage store;
    } temp_storage;

    float thread_x[ITEMS_PER_THREAD];
    float thread_y[ITEMS_PER_THREAD];
    float thread_out[ITEMS_PER_THREAD];

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int n_remaining = n - block_offset;
    BlockLoadT(temp_storage.load).Load(x + block_offset, thread_x, n_remaining, 0.0);

    __syncthreads();
    BlockLoadT(temp_storage.load).Load(y + block_offset, thread_y, n_remaining, 0.0);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        thread_out[i] = thread_x[i] * thread_y[i];
    }

    __syncthreads();
    BlockStoreT(temp_storage.store).Store(out + block_offset, thread_out, n_remaining);
}


extern "C" __global__ void array_vector_dot(
    const int n,
    const float *x,
    const float *y,
    float *prods
) {
    using BlockLoadT = cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockReduceT = cub::BlockReduce<float, BLOCK_THREADS, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockReduceT::TempStorage reduce;
    } temp_storage;

    float thread_x[ITEMS_PER_THREAD];
    float thread_y[ITEMS_PER_THREAD];
    float prod[ITEMS_PER_THREAD];

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int n_remaining = n - block_offset;
    BlockLoadT(temp_storage.load).Load(x + block_offset, thread_x, n_remaining, 0.0);
    __syncthreads();
    BlockLoadT(temp_storage.load).Load(y + block_offset, thread_y, n_remaining, 0.0);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        prod[i] = thread_x[i] * thread_y[i];
    }

    __syncthreads();
    float sum = BlockReduceT(temp_storage.reduce).Sum(prod);

    if (threadIdx.x == 0) {
        atomicAdd(prods, sum);
    }
}

extern "C" __global__ void array_update_variance(
    const int n,
    float *mean,
    float *variance,
    const float *value,
    float diff_scale
) {
    using BlockLoadT = cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockStoreT = cub::BlockStore<float, BLOCK_THREADS, ITEMS_PER_THREAD>;

    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockStoreT::TempStorage store;
    } temp_storage;

    float thread_mean[ITEMS_PER_THREAD];
    float thread_var[ITEMS_PER_THREAD];
    float thread_val[ITEMS_PER_THREAD];

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int n_remaining = n - block_offset;
    BlockLoadT(temp_storage.load).Load(mean + block_offset, thread_mean, n_remaining, 0.0);
    __syncthreads();
    BlockLoadT(temp_storage.load).Load(variance + block_offset, thread_var, n_remaining, 0.0);
    __syncthreads();
    BlockLoadT(temp_storage.load).Load(value + block_offset, thread_val, n_remaining, 0.0);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        float diff = thread_mean[i] - thread_val[i];
        float diff_scaled = diff_scale * diff;
        thread_mean[i] += diff_scaled;
        thread_var[i] += diff * diff;
    }

    __syncthreads();
    BlockStoreT(temp_storage.store).Store(mean + block_offset, thread_mean, n_remaining);
    __syncthreads();
    BlockStoreT(temp_storage.store).Store(variance + block_offset, thread_var, n_remaining);
}


extern "C" __global__ void array_update_var_inv_std_draw_grad(
    const int n,
    float *variance_out,
    float *inv_std,
    const float *draw_var,
    const float *grad_var,
    float fill_invalid, /* ignore if == -1 */
    float clamp_min,
    float clamp_max
) {
    using BlockLoadT = cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockStoreT = cub::BlockStore<float, BLOCK_THREADS, ITEMS_PER_THREAD>;

    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockStoreT::TempStorage store;
    } temp_storage;

    float thread_var[ITEMS_PER_THREAD];
    float thread_inv_std[ITEMS_PER_THREAD];
    float thread_grad_var[ITEMS_PER_THREAD];
    float thread_draw_var[ITEMS_PER_THREAD];

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int n_remaining = n - block_offset;
    BlockLoadT(temp_storage.load).Load(draw_var + block_offset, thread_draw_var, n_remaining, 0.0);
    __syncthreads();
    BlockLoadT(temp_storage.load).Load(grad_var + block_offset, thread_grad_var, n_remaining, 0.0);
    __syncthreads();
    BlockLoadT(temp_storage.load).Load(variance_out + block_offset, thread_var, n_remaining, 0.0);
    __syncthreads();
    BlockLoadT(temp_storage.load).Load(inv_std + block_offset, thread_inv_std, n_remaining, 0.0);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        float val = sqrt(thread_draw_var[i] / thread_grad_var[i]);
        if (!isfinite(val) || val == 0) {
            if (fill_invalid != -1) {
                thread_var[i] = fill_invalid;
                thread_inv_std[i] = 1 / sqrt(fill_invalid);
            }
        } else {
            float clipped = fmaxf(clamp_min, fminf(clamp_max, val));
            thread_var[i] = clipped;
            thread_inv_std[i] = 1 / sqrt(clipped);
        }
    }

    __syncthreads();
    BlockStoreT(temp_storage.store).Store(variance_out + block_offset, thread_var, n_remaining);
    __syncthreads();
    BlockStoreT(temp_storage.store).Store(inv_std + block_offset, thread_inv_std, n_remaining);
}

extern "C" __global__ void array_update_var_inv_std_grad(
    const int n,
    float *variance_out,
    float *inv_std,
    const float *gradient,
    float fill_invalid,
    float clamp_min,
    float clamp_max
) {
    using BlockLoadT = cub::BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>;
    using BlockStoreT = cub::BlockStore<float, BLOCK_THREADS, ITEMS_PER_THREAD>;

    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockStoreT::TempStorage store;
    } temp_storage;

    float thread_var[ITEMS_PER_THREAD];
    float thread_inv_std[ITEMS_PER_THREAD];
    float thread_grad[ITEMS_PER_THREAD];

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
    int n_remaining = n - block_offset;
    BlockLoadT(temp_storage.load).Load(gradient + block_offset, thread_grad, n_remaining, 0.0);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        float val = abs(thread_grad[i]);
        float clipped = fmaxf(clamp_min, fminf(clamp_max, val));
        float clipped_inv = 1.0 / clipped;
        float filled = isfinite(clipped_inv) ? clipped_inv : fill_invalid;
        thread_var[i] = filled;
        thread_inv_std[i] = 1 / sqrt(clipped);
    }

    __syncthreads();
    BlockStoreT(temp_storage.store).Store(variance_out + block_offset, thread_var, n_remaining);
    __syncthreads();
    BlockStoreT(temp_storage.store).Store(inv_std + block_offset, thread_inv_std, n_remaining);
}
