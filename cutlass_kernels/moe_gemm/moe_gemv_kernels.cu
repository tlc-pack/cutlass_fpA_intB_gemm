#include "./moe_gemv_kernels.h"

namespace tensorrt_llm
{
namespace kernels
{

template <template <typename T> class ActOp, bool Bias, int NPerBlock, int Batch, int BlockSize>
__device__ void half_batched_gemv_impl(const half* weight, const half* in, const half* bias, half* out, const int n,
    const int k, int bias_stride, uint8_t* shmem)
{
    static_assert(NPerBlock == 1 || (NPerBlock % 2 == 0));
    // using 128 bit global access
    static constexpr int kAccessSize = 128;
    static constexpr int kElemsPerThread = kAccessSize / (sizeof(half) * 8);
    using AccessType = uint4;

    constexpr int WarpSize = 32;
    constexpr int Num = Batch * NPerBlock;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n_start_id = bid * NPerBlock;

    // Weight: Column major (n, k)
    weight += n_start_id * k;

    float(*sm)[Num] = reinterpret_cast<float(*)[Num]>(shmem);

    // In order to take advantage of hfma2, we use fp16 for accumulation within threads and fp32 for accumulation
    // between threads.
    half accumulator[Num];
    for (int i = 0; i < Num; ++i)
    {
        accumulator[i] = __float2half_rn(0.f);
    }

    // Iteration in k dimensions
    for (int local_k = tid * kElemsPerThread; local_k < k; local_k += BlockSize * kElemsPerThread)
    {
        half weights_v[kElemsPerThread * NPerBlock];

        if constexpr (NPerBlock == 1)
        {
            load<AccessType>(weights_v, weight + local_k);
        }
        else
        {
            half weights_vec_k[kElemsPerThread];
#pragma unroll
            for (int x = 0; x < NPerBlock; ++x)
            {
                load<AccessType>(weights_vec_k, weight + x * k + local_k);
#pragma unroll
                for (int i = 0; i < kElemsPerThread; ++i)
                {
                    weights_v[i * NPerBlock + x] = weights_vec_k[i];
                }
            }
        }

#pragma unroll
        for (int b = 0; b < Batch; ++b)
        {
            half in_v[kElemsPerThread];
            // load activation elements
            load<AccessType>(in_v, in + b * k + local_k);
            // Perform vector inner product and accumulate
            if constexpr (NPerBlock == 1)
            {
                half2 v = __float2half2_rn(0.f);
#pragma unroll
                for (int y = 0; y < kElemsPerThread; y += 2)
                {
                    v = __hfma2(*reinterpret_cast<half2*>(weights_v + y), *reinterpret_cast<half2*>(in_v + y), v);
                }
                accumulator[b] += __hadd(v.x, v.y);
            }
            else
            {
#pragma unroll
                for (int x = 0; x < NPerBlock / 2; ++x)
                {
#pragma unroll
                    for (int y = 0; y < kElemsPerThread; ++y)
                    {
                        *reinterpret_cast<half2*>(accumulator + b * NPerBlock + x * 2)
                            = __hfma2(*reinterpret_cast<half2*>(weights_v + y * NPerBlock + x * 2),
                                __half2half2(in_v[y]), *reinterpret_cast<half2*>(accumulator + b * NPerBlock + x * 2));
                    }
                }
            }
        }
    }
    float reses[Num];
#pragma unroll
    for (int i = 0; i < Num; ++i)
    {
        reses[i] = __half2float(accumulator[i]);
    }

    // Each warp completes the internal reduce and writes the [Batch * NPerBlock * Interleave] results to the
    // corresponding address in shared memory
    __syncwarp();
#pragma unroll
    for (int i = 0; i < Num; ++i)
    {
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            reses[i] += __shfl_xor_sync(~0, reses[i], offset);
        }
    }
    if (tid % WarpSize == 0)
    {
#pragma unroll
        for (int i = 0; i < Num; ++i)
        {
            sm[tid / WarpSize][i] = reses[i];
        }
    }
    __syncthreads();

    // Each thread is responsible for the accumulation and store to global memory of one element
    for (int i = tid; i < Num; i += BlockSize)
    {
        int nid = i % NPerBlock;
        float v = 0.f;
        for (int j = 0; j < BlockSize / WarpSize; ++j)
        {
            v += sm[j][i];
        }
        float bias_v = 0.f;
        int b = i / NPerBlock;
        if constexpr (Bias)
        {
            bias_v = __half2float(bias[b * bias_stride + n_start_id + nid]);
        }
        out[b * n + n_start_id + nid] = __float2half_rn(ActOp<float>::apply(v + bias_v));
    }
}

template <int NPerBlock, int Batch, int BlockSize>
__global__ void moe_gemv_f16_kernel(const half* weight, const half* in, const half* bias, half* out,
    int64_t* total_rows_before_expert, int64_t total_rows, int64_t n, int64_t k, int num_experts)
{
    int bias_stride = 0;
    extern __shared__ uint8_t shmem[];

    const int gid = blockIdx.y;
    for (int i = 0; i < num_experts; i++)
    {
        if (total_rows_before_expert[i] >= (gid + 1))
        {
            weight += i * n * k;
            in += gid * k;
            out += gid * n;
            break;
        }
    }

    half_batched_gemv_impl<IdentityActivation, false, NPerBlock, Batch, BlockSize>(
        weight, in, bias, out, n, k, bias_stride, shmem);
}

void moe_gemv(const half* A, const half* B, half* C, int64_t* total_rows_before_expert, int64_t total_rows,
    int64_t gemm_n, int64_t gemm_k, int num_experts, cudaStream_t stream)
{
    constexpr int Batch = 1;
    constexpr int BlockSize = 128;
    constexpr int NPerBlock = 1;
    dim3 grid(gemm_n / NPerBlock, total_rows);
    dim3 block(BlockSize);
    int size = sizeof(float) * BlockSize / 32 * Batch * NPerBlock;
    moe_gemv_f16_kernel<NPerBlock, Batch, BlockSize><<<grid, block, size, stream>>>(
        B, A, nullptr, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts);
}

} // namespace kernels
} // namespace tensorrt_llm
