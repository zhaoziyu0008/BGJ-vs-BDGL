/// @note complie with:
/// nvcc -ccbin g++ -Xcompiler "-O3 -g -march=native -pthread" -gencode arch=compute_89,code=sm_89 -O3 test_la.cu -lcudart -o test_la

#define private public
#define protected public

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <thread>
#include <random>
#include <pthread.h>

#include <immintrin.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mma.h>

using namespace nvcuda;

typedef wmma::fragment<wmma::matrix_a, 8, 8, 128, wmma::experimental::precision::b1, wmma::row_major> wma_t;
typedef wmma::fragment<wmma::matrix_b, 8, 8, 128, wmma::experimental::precision::b1, wmma::col_major> wmb_t;
typedef wmma::fragment<wmma::accumulator, 8, 8, 128, int> wacc_t;

#define CHECK_CUDA_ERR(val) do {                                        \
    if (val) {                                                          \
        fprintf(stderr, "CUDA error at %s:%d \"%s\", code = %d(%s)\n",  \
        __FILE__, __LINE__, #val, val, cudaGetErrorString(val));        \
        cudaGetLastError();                                             \
        sleep(100000);                                                  \
    }                                                                   \
} while (0)

#define CHECK_LAST_ERR do {                                             \
    cudaError_t __err = cudaGetLastError();                             \
    if (__err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error found at %s:%d, code = %d(%s)\n",   \
        __FILE__, __LINE__, __err, cudaGetErrorString(__err));          \
        sleep(100000);                                                  \
    }                                                                   \
} while (0)

constexpr long db_gen_blocks  = 16;
constexpr long db_gen_threads = 256;
constexpr long db_gen_shmem   = 65792;

__global__ void kernel_db_gen(float *dst, int CSD, int to_gen, float db_scale, long seed) {
    extern __shared__ float sh_vec[][257];

    curandState state;
    curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &state);

    for (int ind = blockIdx.x * 64; ind < to_gen; ind += gridDim.x * 64) {
        int cnt = 0;
        int max_cnt = ind + 64 >= to_gen ? to_gen - ind : 64;
        for (;;) {
            for (int i = 0; i < 64; i++) {
                sh_vec[i][threadIdx.x] = threadIdx.x < CSD ? curand_normal(&state) : 0.0f;
            }

            if (threadIdx.x < 64) sh_vec[threadIdx.x][256] = 0.0f;

            __syncthreads();

            float norm = 0.0f;
            for (int i = 0; i < 64; i++) {
                norm += sh_vec[threadIdx.x / 4][i + (threadIdx.x % 4) * 64] * sh_vec[threadIdx.x / 4][i + (threadIdx.x % 4) * 64];
            }
            atomicAdd(&sh_vec[threadIdx.x / 4][256], norm);

            __syncthreads();

            for (int i = 0; i < 64; i++) {
                float norm = sh_vec[i][256];
                if (norm < CSD) {
                    float scale = sqrtf(CSD / norm) * 0.7f + 0.3f;
                    dst[256L * (ind + cnt) + threadIdx.x] = sh_vec[i][threadIdx.x] * db_scale * scale;
                    if (++cnt >= max_cnt) break;
                }
            }

            __syncthreads();

            if (cnt >= max_cnt) break;
        }
    }
}

constexpr long normal_blocks  = 16;
constexpr long normal_threads = 256;
constexpr long normal_shmem   = 65792;

__global__ void kernel_normal(float *dst, int ind_l, int ind_r, int to_gen, float target_len, long seed) {
    extern __shared__ float sh_vec[][257];

    curandState state;
    curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &state);

    for (int ind = blockIdx.x * 64; ind < to_gen; ind += gridDim.x * 64) {
        int max_cnt = ind + 64 >= to_gen ? to_gen - ind : 64;

        for (int i = 0; i < 64; i++) {
            sh_vec[i][threadIdx.x] = (threadIdx.x < ind_r && threadIdx.x >= ind_l) ? curand_normal(&state) : 0.0f;
        }

        if (threadIdx.x < 64) sh_vec[threadIdx.x][256] = 0.0f;

        __syncthreads();

        float norm = 0.0f;
        for (int i = 0; i < 64; i++) {
            norm += sh_vec[threadIdx.x / 4][i + (threadIdx.x % 4) * 64] * sh_vec[threadIdx.x / 4][i + (threadIdx.x % 4) * 64];
        }
        atomicAdd(&sh_vec[threadIdx.x / 4][256], norm);

        __syncthreads();

        for (int i = 0; i < 64; i++) {
            if (i >= max_cnt) break;
            float norm = sh_vec[i][256];
            dst[256L * (ind + i) + threadIdx.x] = sh_vec[i][threadIdx.x] * target_len * sqrtf(1.0f / norm);
        }

        __syncthreads();
    }
}

constexpr long buc_blocks = 128;
constexpr long buc_threads = 256;
constexpr long buc_shmem = 65792;

template <int type>
__global__ void kernel_buc(int *d_buc, int *d_num, float *d_src, int nvec, float *d_ct0, 
                           float *d_ct1, float *d_ct2, int nctr, float alpha0, float alpha1, float alpha2, unsigned long long *d_cost) {
    extern __shared__ float sh_vec[][257];

    const int tid = threadIdx.x;

    __shared__ float dp0[16][16];
    __shared__ float dp1[16][16];
    __shared__ float dp2[16][16];

    unsigned long long t_cost0 = 0;
    unsigned long long t_cost1 = 0;
    unsigned long long t_cost2 = 0;

    for (int cnd = blockIdx.x * 16; cnd < nctr; cnd += gridDim.x * 16) {
        int max_cnt = cnd + 16 >= nctr ? nctr - cnd : 16;
        for (int i = 0; i < 16; i++) {
            if (type >= 1) sh_vec[i][tid] = d_ct0[(cnd + i) * 256L + tid];
            if (type >= 2) sh_vec[i + 16][tid] = d_ct1[(cnd + i) * 256L + tid];
            if (type >= 3) sh_vec[i + 32][tid] = d_ct2[(cnd + i) * 256L + tid]; 
        }

        __syncthreads();
        for (int ind = 0; ind < nvec; ind += 16) {
            for (int i = 0; i < 16; i++) {
                sh_vec[i + 48][tid] = d_src[(ind + i) * 256L + tid];
            }

            dp0[tid / 16][tid % 16] = 0.0f;
            dp1[tid / 16][tid % 16] = 0.0f;
            dp2[tid / 16][tid % 16] = 0.0f;
            if (tid < 64) sh_vec[tid][256] = 0.0f; 
            __syncthreads();

            float t_norm = 0.0f;
            for (int i = 0; i < 16; i++) {
                float val = sh_vec[tid / 16 + 48][((tid % 16) + i * 16 + ((tid & 16) ? 15 : 0)) % 256];
                t_norm += val * val;
            }

            {
                t_norm += __shfl_xor_sync(0xffffffff, t_norm, 1);
                t_norm += __shfl_xor_sync(0xffffffff, t_norm, 2);
                t_norm += __shfl_xor_sync(0xffffffff, t_norm, 4);
                t_norm += __shfl_xor_sync(0xffffffff, t_norm, 8);
                if ((tid & 15) == 0) sh_vec[tid / 16 + 48][256] = t_norm;
            }

            if (type >= 1) {
                float thread_ct[4][16];
                float thread_vt[4][16];
                float dp[16] = {};
                int ci = ((tid / 16) / 4) * 4;
                int vi = ((tid / 16) % 4) * 4;

                for (int l = 0; l < 16; l++) {
                    for (int i = 0; i < 4; i++) {
                        thread_ct[i][l] = sh_vec[ci + i][(tid % 16) + l * 16];
                        thread_vt[i][l] = sh_vec[vi + i + 48][(tid % 16) + l * 16];
                    }

                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            dp[i * 4 + j] += thread_ct[i][l] * thread_vt[j][l];
                        }
                    }
                }

                {
                    float acc3[8], acc2[4], acc1[2], acc0[1];
                    for (int l = 0; l < 8; l++) {
                        float send = (tid & 1) ? dp[2 * l] : dp[2 * l + 1];
                        float recv = __shfl_xor_sync(0xffffffff, send, 1);
                        acc3[l] = (tid & 1) ? dp[2 * l + 1] + recv : dp[2 * l] + recv;
                    }

                    for (int l = 0; l < 4; l++) {
                        float send = (tid & 2) ? acc3[2 * l] : acc3[2 * l + 1];
                        float recv = __shfl_xor_sync(0xffffffff, send, 2);
                        acc2[l] = (tid & 2) ? acc3[2 * l + 1] + recv : acc3[2 * l] + recv;
                    }

                    for (int l = 0; l < 2; l++) {
                        float send = (tid & 4) ? acc2[2 * l] : acc2[2 * l + 1];
                        float recv = __shfl_xor_sync(0xffffffff, send, 4);
                        acc1[l] = (tid & 4) ? acc2[2 * l + 1] + recv : acc2[2 * l] + recv;
                    }

                    {
                        float send = (tid & 8) ? acc1[0] : acc1[1];
                        float recv = __shfl_xor_sync(0xffffffff, send, 8);
                        acc0[0] = (tid & 8) ? acc1[1] + recv : acc1[0] + recv;
                    }

                    dp0[ci + (tid % 16) / 4][vi + ((tid % 16) % 4)] = acc0[0];
                }
            }

            if (type >= 2) {
                float thread_ct[4][16];
                float thread_vt[4][16];
                float dp[16] = {};
                int ci = ((tid / 16) / 4) * 4;
                int vi = ((tid / 16) % 4) * 4;

                for (int l = 0; l < 16; l++) {
                    for (int i = 0; i < 4; i++) {
                        thread_ct[i][l] = sh_vec[ci + i + 16][(tid % 16) + l * 16];
                        thread_vt[i][l] = sh_vec[vi + i + 48][(tid % 16) + l * 16];
                    }

                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            dp[i * 4 + j] += thread_ct[i][l] * thread_vt[j][l];
                        }
                    }
                }

                {
                    float acc3[8], acc2[4], acc1[2], acc0[1];
                    for (int l = 0; l < 8; l++) {
                        float send = (tid & 1) ? dp[2 * l] : dp[2 * l + 1];
                        float recv = __shfl_xor_sync(0xffffffff, send, 1);
                        acc3[l] = (tid & 1) ? dp[2 * l + 1] + recv : dp[2 * l] + recv;
                    }

                    for (int l = 0; l < 4; l++) {
                        float send = (tid & 2) ? acc3[2 * l] : acc3[2 * l + 1];
                        float recv = __shfl_xor_sync(0xffffffff, send, 2);
                        acc2[l] = (tid & 2) ? acc3[2 * l + 1] + recv : acc3[2 * l] + recv;
                    }

                    for (int l = 0; l < 2; l++) {
                        float send = (tid & 4) ? acc2[2 * l] : acc2[2 * l + 1];
                        float recv = __shfl_xor_sync(0xffffffff, send, 4);
                        acc1[l] = (tid & 4) ? acc2[2 * l + 1] + recv : acc2[2 * l] + recv;
                    }

                    {
                        float send = (tid & 8) ? acc1[0] : acc1[1];
                        float recv = __shfl_xor_sync(0xffffffff, send, 8);
                        acc0[0] = (tid & 8) ? acc1[1] + recv : acc1[0] + recv;
                    }

                    dp1[ci + (tid % 16) / 4][vi + ((tid % 16) % 4)] = acc0[0];
                }
            }

            if (type >= 3) {
                float thread_ct[4][16];
                float thread_vt[4][16];
                float dp[16] = {};
                int ci = ((tid / 16) / 4) * 4;
                int vi = ((tid / 16) % 4) * 4;

                for (int l = 0; l < 16; l++) {
                    for (int i = 0; i < 4; i++) {
                        thread_ct[i][l] = sh_vec[ci + i + 32][(tid % 16) + l * 16];
                        thread_vt[i][l] = sh_vec[vi + i + 48][(tid % 16) + l * 16];
                    }

                    for (int i = 0; i < 4; i++) {
                        for (int j = 0; j < 4; j++) {
                            dp[i * 4 + j] += thread_ct[i][l] * thread_vt[j][l];
                        }
                    }
                }

                {
                    float acc3[8], acc2[4], acc1[2], acc0[1];
                    for (int l = 0; l < 8; l++) {
                        float send = (tid & 1) ? dp[2 * l] : dp[2 * l + 1];
                        float recv = __shfl_xor_sync(0xffffffff, send, 1);
                        acc3[l] = (tid & 1) ? dp[2 * l + 1] + recv : dp[2 * l] + recv;
                    }

                    for (int l = 0; l < 4; l++) {
                        float send = (tid & 2) ? acc3[2 * l] : acc3[2 * l + 1];
                        float recv = __shfl_xor_sync(0xffffffff, send, 2);
                        acc2[l] = (tid & 2) ? acc3[2 * l + 1] + recv : acc3[2 * l] + recv;
                    }

                    for (int l = 0; l < 2; l++) {
                        float send = (tid & 4) ? acc2[2 * l] : acc2[2 * l + 1];
                        float recv = __shfl_xor_sync(0xffffffff, send, 4);
                        acc1[l] = (tid & 4) ? acc2[2 * l + 1] + recv : acc2[2 * l] + recv;
                    }

                    {
                        float send = (tid & 8) ? acc1[0] : acc1[1];
                        float recv = __shfl_xor_sync(0xffffffff, send, 8);
                        acc0[0] = (tid & 8) ? acc1[1] + recv : acc1[0] + recv;
                    }

                    dp2[ci + (tid % 16) / 4][vi + ((tid % 16) % 4)] = acc0[0];
                }
            }
            
            __syncthreads();

            if (type == 1) {
                int ci = tid / 16;
                int vi = tid % 16;
                float _dp0 = dp0[ci][vi];
                float _n = sh_vec[48 + vi][256];
                if (_dp0 > _n * alpha0 && ci < max_cnt) {
                    int pos = atomicAdd(&d_num[cnd + ci], 1);
                    if (pos + 1 < 512) {
                        d_buc[512 * (cnd + ci) + pos] = ind + vi;
                    } else {
                        d_num[cnd + ci] = 512;
                    }
                }

                if (-_dp0 > _n * alpha0 && ci < max_cnt) {
                    int pos = atomicAdd(&d_num[cnd + ci], 1);
                    if (pos + 1 < 512) {
                        d_buc[512 * (cnd + ci) + pos] = (ind + vi) | 0x80000000;
                    } else {
                        d_num[cnd + ci] = 512;
                    }
                }

                t_cost0 += 1;
            }

            if (type == 2) {
                int ci = tid / 16;
                int vi = tid % 16;
                float _dp0 = dp0[ci][vi];
                float _dp1 = dp1[ci][vi];
                float _n = sh_vec[48 + vi][256];
                if (_dp0 > _n * alpha0 && _dp1 > _n * alpha1 && ci < max_cnt) {
                    int pos = atomicAdd(&d_num[cnd + ci], 1);
                    if (pos + 1 < 512) {
                        d_buc[512 * (cnd + ci) + pos] = ind + vi;
                    } else {
                        d_num[cnd + ci] = 512;
                    }
                }

                if (-_dp0 > _n * alpha0 && -_dp1 > _n * alpha1 && ci < max_cnt) {
                    int pos = atomicAdd(&d_num[cnd + ci], 1);
                    if (pos + 1 < 512) {
                        d_buc[512 * (cnd + ci) + pos] = (ind + vi) | 0x80000000;
                    } else {
                        d_num[cnd + ci] = 512;
                    }
                }

                if (fabs(_dp0) > _n * alpha0 && ci < max_cnt) t_cost1 += 1;
                t_cost0 += 1;
            }

            if (type == 3) {
                int ci = tid / 16;
                int vi = tid % 16;
                float _dp0 = dp0[ci][vi];
                float _dp1 = dp1[ci][vi];
                float _dp2 = dp2[ci][vi];
                float _n = sh_vec[48 + vi][256];
                if (_dp0 > _n * alpha0 && _dp1 > _n * alpha1 && _dp2 > _n * alpha2 && ci < max_cnt) {
                    int pos = atomicAdd(&d_num[cnd + ci], 1);
                    if (pos + 1 < 512) {
                        d_buc[512 * (cnd + ci) + pos] = ind + vi;
                    } else {
                        d_num[cnd + ci] = 512;
                    }
                }

                if (-_dp0 > _n * alpha0 && -_dp1 > _n * alpha1 && -_dp2 > _n * alpha2 && ci < max_cnt) {
                    int pos = atomicAdd(&d_num[cnd + ci], 1);
                    if (pos + 1 < 512) {
                        d_buc[512 * (cnd + ci) + pos] = (ind + vi) | 0x80000000;
                    } else {
                        d_num[cnd + ci] = 512;
                    }
                }

                if (_dp0 > _n * alpha0 && _dp1 > _n * alpha1 && ci < max_cnt) t_cost2 += 1;
                if (-_dp0 > _n * alpha0 && -_dp1 > _n * alpha1 && ci < max_cnt) t_cost2 += 1;
                if (fabs(_dp0) > _n * alpha0) t_cost1 += 1;
                t_cost0 += 1;
            }

            __syncthreads();
        }
    }

    atomicAdd(d_cost, t_cost0);
    atomicAdd(d_cost + 1, t_cost1);
    atomicAdd(d_cost + 2, t_cost2);
}

constexpr long red_blocks  = 128;
constexpr long red_threads = 256;
constexpr long red_shmem   = 65792;

__global__ void kernel_red(int *d_out, int *d_num, int max_out, float *d_buc, int buc_size, float th) {
    extern __shared__ float sh_vec[][257];
    
    const int tid = threadIdx.x;

    __shared__ float dp0[32][32];

    for (int cnd = blockIdx.x * 32; cnd < buc_size - 31; cnd += gridDim.x * 32) {
        for (int i = 0; i < 32; i++) {
            sh_vec[i][tid] = d_buc[(cnd + i) * 256L + tid];
        }

        if (tid < 32) sh_vec[tid][256] = 0.0f;
        __syncthreads();

        float t_norm = 0.0f;
        for (int i = 0; i < 32; i++) {
            t_norm += sh_vec[tid / 8][(tid % 8) + i * 8] * sh_vec[tid / 8][(tid % 8) + i * 8];
        }
        
        atomicAdd(&sh_vec[tid / 8][256], t_norm);

        __syncthreads();

        for (int ind = cnd + 32; ind < buc_size - 31; ind += 32) {
            for (int i = 0; i < 32; i++) {
                sh_vec[i + 32][tid] = d_buc[(ind + i) * 256L + tid];
            }

            dp0[tid / 32][tid % 32] = 0.0f;
            dp0[tid / 32 + 8][tid % 32] = 0.0f;
            dp0[tid / 32 + 16][tid % 32] = 0.0f;
            dp0[tid / 32 + 24][tid % 32] = 0.0f;
            if (tid < 32) sh_vec[tid + 32][256] = 0.0f;
            __syncthreads();

            float t_norm = 0.0f;
            for (int i = 0; i < 32; i++) {
                t_norm += sh_vec[tid / 8 + 32][(tid % 8) + i * 8] * sh_vec[tid / 8  + 32][(tid % 8) + i * 8];
            }

            {
                t_norm += __shfl_xor_sync(0xffffffff, t_norm, 1);
                t_norm += __shfl_xor_sync(0xffffffff, t_norm, 2);
                t_norm += __shfl_xor_sync(0xffffffff, t_norm, 4);
                if ((tid & 7) == 0) sh_vec[tid / 8 + 32][256] = t_norm;
            }

            
            float dp[64] = {};
            int ci = ((tid / 16) / 4) * 8;
            int vi = ((tid / 16) % 4) * 8;
            int li = (tid % 16) * 16;

            #pragma unroll
            for (int l = 0; l < 16; l++) {
                float thread_ct[8], thread_vt[8];

                for (int i = 0; i < 8; i++) {
                    thread_ct[i] = sh_vec[ci + i][li + l];
                    thread_vt[i] = sh_vec[vi + i + 32][li + l];
                }

                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 8; j++) {
                        dp[i * 8 + j] += thread_ct[i] * thread_vt[j];
                    }
                }
            }
            
            {
                float acc3[32], acc2[16], acc1[8], acc0[4];
                for (int l = 0; l < 32; l++) {
                    float send = (tid & 1) ? dp[2 * l] : dp[2 * l + 1];
                    float recv = __shfl_xor_sync(0xffffffff, send, 1);
                    acc3[l] = (tid & 1) ? dp[2 * l + 1] + recv : dp[2 * l] + recv;
                }

                for (int l = 0; l < 16; l++) {
                    float send = (tid & 2) ? acc3[2 * l] : acc3[2 * l + 1];
                    float recv = __shfl_xor_sync(0xffffffff, send, 2);
                    acc2[l] = (tid & 2) ? acc3[2 * l + 1] + recv : acc3[2 * l] + recv;
                }

                for (int l = 0; l < 8; l++) {
                    float send = (tid & 4) ? acc2[2 * l] : acc2[2 * l + 1];
                    float recv = __shfl_xor_sync(0xffffffff, send, 4);
                    acc1[l] = (tid & 4) ? acc2[2 * l + 1] + recv : acc2[2 * l] + recv;
                }

                for (int l = 0; l < 4; l++) {
                    float send = (tid & 8) ? acc1[2 * l] : acc1[2 * l + 1];
                    float recv = __shfl_xor_sync(0xffffffff, send, 8);
                    acc0[l] = (tid & 8) ? acc1[2 * l + 1] + recv : acc1[2 * l] + recv;
                }

                dp0[ci + 0 + (tid % 16) / 8][vi + (tid % 8)] = acc0[0];
                dp0[ci + 2 + (tid % 16) / 8][vi + (tid % 8)] = acc0[1];
                dp0[ci + 4 + (tid % 16) / 8][vi + (tid % 8)] = acc0[2];
                dp0[ci + 6 + (tid % 16) / 8][vi + (tid % 8)] = acc0[3];
            }

            __syncthreads();

            ci = tid / 8;
            vi = (tid % 8) * 4;
            float _dp0[4] = {dp0[ci][vi], dp0[ci][vi + 1], dp0[ci][vi + 2], dp0[ci][vi + 3]};
            float _vn[4] = {sh_vec[32 + vi][256], sh_vec[33 + vi][256], sh_vec[34 + vi][256], sh_vec[35 + vi][256]};
            float _cn = sh_vec[ci][256];
            for (int i = 0; i < 4; i++) {
                if (_cn + _vn[i] - 2 * _dp0[i] < th) {
                    int pos = atomicAdd(d_num, 1);
                    if (pos + 1 < max_out) {
                        d_out[2 * pos + 0] = cnd + ci;
                        d_out[2 * pos + 1] = ind + vi + i;
                    } else {
                        d_num[0] = max_out;
                    }
                }
            }

            __syncthreads();
        }
    }
}

constexpr long sim_blocks  = 128;
constexpr long sim_threads = 256;
constexpr long sim_shmem   = 65536;

__global__ void kernel_sim(int *d_out, int *d_num, int max_out, uint32_t *d_sim, int buc_size, int th) {
    extern __shared__ uint32_t sh_sim[][32];
    
    const int lid = threadIdx.x % 32;
    const int wid = threadIdx.x / 32;

    __shared__ int dp0[256][32];

    uint32_t (*wvec)[32] = &sh_sim[wid * 64];
    int (*wdp)[32] = &dp0[wid * 32];

    for (int cnd = (blockIdx.x * 8 + wid) * 32; cnd < buc_size - 31; cnd += gridDim.x * 8 * 32) {
        for (int i = 0; i < 32; i++) {
            wvec[i][lid] = d_sim[(cnd + i) * 32L + lid];
        }

        __syncthreads();

        for (int ind = cnd + 32; ind < buc_size - 31; ind += 32) {
            for (int i = 0; i < 32; i++) {
                wvec[i + 32][lid] = d_sim[(ind + i) * 32L + lid];
            }

            __syncthreads();

            for (int ci = 0; ci < 32; ci += 8) {
                wma_t center_frag[8];
                for (int l = 0; l < 8; l++) {
                    wmma::load_matrix_sync(center_frag[l], &wvec[ci][l * 4], 1024);
                }
                for (int vi = 0; vi < 32; vi += 8) {
                    wacc_t acc;
                    wmma::fill_fragment(acc, 0);
                    for (int l = 0; l < 8; l++) {
                        wmb_t vec_frag;
                        wmma::load_matrix_sync(vec_frag, &wvec[32 + vi][l * 4], 1024);
                        wmma::bmma_sync(acc, center_frag[l], vec_frag, acc);
                    }

                    wmma::store_matrix_sync(&wdp[ci][vi], acc, 32, wmma::mem_row_major);
                }
            }

            __syncthreads();

            int rdp[32], count = 0;
            for (int i = 0; i < 32; i++) {
                rdp[i] = wdp[i][lid];
                if (rdp[i] < th) count++;
            }

            int pos = atomicAdd(d_num, count);
            if (pos + count >= max_out) {
                d_num[0] = max_out;
                count = max_out - pos;
            }

            for (int i = 0; i < 32; i++) {
                if (rdp[i] < th && count > 0) {
                    d_out[2 * pos + 0] = cnd + i;
                    d_out[2 * pos + 1] = ind + lid;
                    pos++;
                    count--;
                }
            }

            __syncthreads();
        }
    }
}

constexpr long int8_blocks  = 128;
constexpr long int8_threads = 256;
constexpr long int8_shmem   = 65792;

__global__ void kernel_int8(float *d_buc, int buc_size) {
    extern __shared__ float sh_vec[][257];

    const int tid = threadIdx.x;

    for (int ind = blockIdx.x * 64; ind < buc_size; ind += gridDim.x * 64) {
        for (int i = 0; i < 64; i++) {
            sh_vec[i][tid] = d_buc[(ind + i) * 256L + tid];
        }
        __syncthreads();

        for (int i = 0; i < 64; i++) {
            float val = sh_vec[i][tid];
            int val_int = __float2int_rn(val * 288.f);
            if (val_int >= 128) val_int = 127;
            if (val_int <= -128) val_int = -127;
            sh_vec[i][tid] = (float)val_int / 288.f;
        }

        __syncthreads();

        for (int i = 0; i < 64; i++) {
            if (ind + i >= buc_size) break; 
            d_buc[(ind + i) * 256L + tid] = sh_vec[i][tid];
        }

        __syncthreads();
    }
}

constexpr long int4_blocks  = 128;
constexpr long int4_threads = 256;
constexpr long int4_shmem   = 65792;

__global__ void kernel_int4(float *d_buc, int buc_size) {
    extern __shared__ float sh_vec[][257];

    const int tid = threadIdx.x;

    for (int ind = blockIdx.x * 64; ind < buc_size; ind += gridDim.x * 64) {
        for (int i = 0; i < 64; i++) {
            sh_vec[i][tid] = d_buc[(ind + i) * 256L + tid];
        }
        __syncthreads();

        for (int i = 0; i < 64; i++) {
            float val = sh_vec[i][tid];
            int val_int = __float2int_rn(val * 36.f);
            if (val_int >= 8) val_int = 7;
            if (val_int <= -8) val_int = -7;
            sh_vec[i][tid] = (float)val_int / 36.f;
        }

        __syncthreads();

        for (int i = 0; i < 64; i++) {
            if (ind + i >= buc_size) break; 
            d_buc[(ind + i) * 256L + tid] = sh_vec[i][tid];
        }

        __syncthreads();
    }
}

constexpr long int512_blocks  = 128;
constexpr long int512_threads = 256;
constexpr long int512_shmem   = 65792;

__global__ void kernel_int512(uint32_t *d_sim, float *d_buc, int buc_size) {
    extern __shared__ float sh_vec[][257];

    __shared__ uint32_t sh_sim[64][33];

    const int tid = threadIdx.x;

    for (int ind = blockIdx.x * 64; ind < buc_size; ind += gridDim.x * 64) {
        for (int i = 0; i < 64; i++) {
            sh_vec[i][tid] = d_buc[(ind + i) * 256L + tid];
        }
        __syncthreads();

        int vid = tid / 4;

        curandState state;
        curand_init(0, tid % 4, 0, &state);

        uint32_t hash[4] = {};

        for (int l = 0; l < 128; l++) {
            float val = 0.0f;
            for (int i = 0; i < 6; i++) {
                int p = curand(&state) % 256;
                int n = curand(&state) % 256;
                val += sh_vec[vid][p] - sh_vec[vid][n];
            }

            hash[l / 32] |= ((*(uint32_t *)&val) & 0x80000000) >> (l % 32);
        }

        sh_sim[vid][(tid % 4) + 0] = hash[0];
        sh_sim[vid][(tid % 4) + 4] = hash[1];
        sh_sim[vid][(tid % 4) + 8] = hash[2];
        sh_sim[vid][(tid % 4) + 12] = hash[3];
        sh_sim[vid][(tid % 4) + 16] = 0;
        sh_sim[vid][(tid % 4) + 20] = 0;
        sh_sim[vid][(tid % 4) + 24] = 0;
        sh_sim[vid][(tid % 4) + 28] = 0;

        __syncthreads();

        for (int i = 0; i < 64; i++) {
            if (ind + i >= buc_size) break; 
            if (tid / 32 == (i % 8)) d_sim[(ind + i) * 32L + (tid % 32)] = sh_sim[i][(tid % 32)];
        }

        __syncthreads();
    } 
}

constexpr long int768_blocks  = 128;
constexpr long int768_threads = 256;
constexpr long int768_shmem   = 65792;

__global__ void kernel_int768(uint32_t *d_sim, float *d_buc, int buc_size) {
    extern __shared__ float sh_vec[][257];

    __shared__ uint32_t sh_sim[64][33];

    const int tid = threadIdx.x;

    for (int ind = blockIdx.x * 64; ind < buc_size; ind += gridDim.x * 64) {
        for (int i = 0; i < 64; i++) {
            sh_vec[i][tid] = d_buc[(ind + i) * 256L + tid];
        }
        __syncthreads();

        int vid = tid / 4;

        curandState state;
        curand_init(0, tid % 4, 0, &state);

        uint32_t hash[6] = {};

        for (int l = 0; l < 192; l++) {
            float val = 0.0f;
            for (int i = 0; i < 6; i++) {
                int p = curand(&state) % 256;
                int n = curand(&state) % 256;
                val += sh_vec[vid][p] - sh_vec[vid][n];
            }

            hash[l / 32] |= ((*(uint32_t *)&val) & 0x80000000) >> (l % 32);
        }

        sh_sim[vid][(tid % 4) + 0] = hash[0];
        sh_sim[vid][(tid % 4) + 4] = hash[1];
        sh_sim[vid][(tid % 4) + 8] = hash[2];
        sh_sim[vid][(tid % 4) + 12] = hash[3];
        sh_sim[vid][(tid % 4) + 16] = hash[4];
        sh_sim[vid][(tid % 4) + 20] = hash[5];
        sh_sim[vid][(tid % 4) + 24] = 0;
        sh_sim[vid][(tid % 4) + 28] = 0;

        __syncthreads();

        for (int i = 0; i < 64; i++) {
            if (ind + i >= buc_size) break; 
            if (tid / 32 == (i % 8)) d_sim[(ind + i) * 32L + (tid % 32)] = sh_sim[i][(tid % 32)];
        }

        __syncthreads();
    }
}

int num_device = 0;
double buc_total_dp = 0.0;

/// DB file format:
/// First 8 bytes: CSD (int64_t, vector dimension)
/// Next 8 bytes: N (int64_t, total number of vectors)
/// Next 4 bytes: db_scale (float, scaling factor)
/// Remaining N * CSD * sizeof(float) bytes: N vectors, each of length CSD
/// In case that DB is not avaible, 
/// If no database file is specified, the program will automatically use 
/// the internally deterministically generated vector database.

struct random_interval_iter_t {
    random_interval_iter_t(int32_t end) {
        num = end;
        entries = new int[end];
        for (int i = 0; i < end; i++) entries[i] = i;
        std::random_device rd;
        std::mt19937 g(rd());
        for (int i = 0; i < end; i++) {
            int src = i;
            int dst = std::uniform_int_distribution<int>(0, end - 1)(g);
            int tmp = entries[src];
            entries[src] = entries[dst];
            entries[dst] = tmp;
        }
    }

    ~random_interval_iter_t() {
        delete[] entries;
    }

    inline int pop() {
        if (num <= 0) return -1;
        return entries[--num];
    }
    int num;
    int *entries = NULL;
};

struct data_stream_t {
    static constexpr long batch_size = 131072;
    
    data_stream_t(long CSD, char *db_path = NULL, long seed = 0) {
        this->_CSD  = CSD;
        this->_seed = seed;
        this->_ptr  = 0;
        pthread_spin_init(&_lock, PTHREAD_PROCESS_PRIVATE);
        if (db_path) {
            this->_db_path = strdup(db_path);
            FILE *temp_file = fopen(db_path, "rb");
            if (!temp_file) {
                printf("[Error] db file not found: %s\n", db_path);
                abort();
            }

            if (fread(&this->_CSD, sizeof(long), 1, temp_file) != 1) {
                printf("[Error] Failed to read CSD from db file\n");
                abort();
            }

            if (this->_CSD != CSD) {
                printf("[Error] input CSD(%ld) mismatch with db(%ld)\n", CSD, this->_CSD);
                abort();
            }

            if (fread(&this->_N, sizeof(long), 1, temp_file) != 1) {
                printf("[Error] failed to read N from db file\n");
                abort();
            }

            if (fread(&this->_db_scale, sizeof(float), 1, temp_file) != 1) {
                printf("[Error] failed to read db_scale from db file\n");
                abort();
            }

            fclose(temp_file);
            printf("[Info] using external db: CSD %ld, N %ld, db_scale 1 / %.2f\n", CSD, _N, 1.0 / _db_scale);
        } else {
            this->_db_path = NULL;
            this->_N       = long(2.7 * pow(4.0 / 3.0, CSD * 0.5));
            this->_db_scale = float(pow(0.77, -1.0 / (CSD + 1.0)) / sqrt(CSD));
            printf("[Info] using random db: CSD %ld, N %ld, db_scale 1 / %.2f\n", CSD, _N, 1.0 / _db_scale);
        }
    }

    ~data_stream_t() {
        if (this->_db_path) free(this->_db_path);
        pthread_spin_destroy(&_lock);
    }

    long pop(float *dst, float *buf, long &src_bias) {
        pthread_spin_lock(&_lock);
        long pos = _ptr;
        long to_gen = pos + batch_size <= _N ? batch_size : _N - pos;
        if (_CSD < 100 && to_gen > 32768) to_gen = 32768;
        _ptr += to_gen;
        pthread_spin_unlock(&_lock);
        src_bias = pos;

        if (to_gen == 0) return 0;

        if (_db_path) {
            FILE *db_file = fopen(_db_path, "rb");
            if (!db_file) {
                printf("[Error] db file not found: %s\n", _db_path);
                abort();
            }

            fseek(db_file, 20 + pos * _CSD * 4L, SEEK_SET);
            size_t vec_nbytes = to_gen * _CSD * 4L;
            size_t read_bytes = fread(buf, 1, vec_nbytes, db_file);
            if (read_bytes < vec_nbytes) {
                printf("[Error] failed to read enough data (%zu < %zu) from db file\n", read_bytes, vec_nbytes);
                abort();
            }

            fclose(db_file);
            for (long i = 0; i < to_gen; i++) {
                memcpy(&dst[i * 256], &buf[i * _CSD], _CSD * sizeof(float));
                for (long j = 0; j < _CSD; j++) {
                    dst[i * 256 + j] *= _db_scale;
                }
            }
        } else {
            kernel_db_gen<<<db_gen_blocks, db_gen_threads, db_gen_shmem>>>(
                buf, _CSD, to_gen, _db_scale, _seed + pos
            );
            CHECK_LAST_ERR;
            CHECK_CUDA_ERR(cudaMemcpy(dst, buf, to_gen * 256 * sizeof(float), cudaMemcpyDeviceToHost));
        }

        return to_gen;
    }

    private:
    long _CSD, _seed, _ptr, _N;
    float _db_scale;
    char *_db_path;
    pthread_spinlock_t _lock;
};    

struct filter_t;

inline int compare(const void *p1, const void *p2) {
    const uint64_t *pair1 = (const uint64_t *)p1;
    const uint64_t *pair2 = (const uint64_t *)p2;
    if (pair1[0] != pair2[0]) return pair1[0] < pair2[0] ? -1 : 1;
    return 0;
}

uint64_t hash_pair(uint64_t a, uint64_t b) {
    uint64_t seed = (a & 0xffffffffULL) | (b & 0xffffffffULL) << 32;
    return seed;
}

struct bucket_list_t {
    static constexpr long red_none     = 0;
    static constexpr long red_default  = 1;
    static constexpr long red_int8     = 2;
    static constexpr long red_int4     = 3;
    static constexpr long red_int1_512 = 4;
    static constexpr long red_int1_768 = 5;

    bucket_list_t(long CSD, long B) {
        this->_CSD    = CSD;
        this->_B      = B;
        this->_bucket = (float **) malloc(sizeof(float *) * B);
        this->_bucket_ptr = (uint64_t **) malloc(sizeof(uint64_t *) * B);
        this->_size   = (long *) malloc(sizeof(long) * B);
        this->_cap    = (long *) malloc(sizeof(long) * B);
        this->_lock   = (pthread_spinlock_t *) malloc(sizeof(pthread_spinlock_t) * B);
        for (long i = 0; i < B; i++) {
            this->_bucket[i] = (float *) malloc(8192 * 256L * sizeof(float));
            this->_bucket_ptr[i] = (uint64_t *) malloc(8192 * sizeof(uint64_t));
            this->_size[i]   = 0;
            this->_cap[i]    = 8192;
            pthread_spin_init(&this->_lock[i], PTHREAD_PROCESS_PRIVATE);
        }
    }

    ~bucket_list_t() {
        for (long i = 0; i < this->_B; i++) {
            free(this->_bucket[i]);
            free(this->_bucket_ptr[i]);
            pthread_spin_destroy(&this->_lock[i]);
        }
        free(this->_bucket);
        free(this->_bucket_ptr);
        free(this->_size);
        free(this->_cap);
        free((void *)this->_lock);
    }

    void append(long id, float *src, int *idx, long num_vecs, long bias) {
        pthread_spin_lock(&_lock[id]);
        if (_size[id] + num_vecs > _cap[id]) {
            _cap[id] = _cap[id] * 2 > _size[id] + num_vecs ? 
                       _cap[id] * 2 : _size[id] + num_vecs;
            _bucket[id] = (float *) realloc(_bucket[id], _cap[id] * 256L * sizeof(float));
            _bucket_ptr[id] = (uint64_t *)realloc(_bucket_ptr[id], _cap[id] * sizeof(uint64_t));
        }
        for (int i = 0; i < num_vecs; i++) {
            memcpy(&_bucket[id][(_size[id] + i) * 256L], &src[(idx[i] & 0x7fffffff) * 256L], 256L * sizeof(float));
            if (idx[i] & 0x80000000) for (int j = 0; j < 256; j++) _bucket[id][(_size[id] + i) * 256L + j] *= -1.0;
            _bucket_ptr[id][_size[id] + i] = (idx[i] & 0x7fffffff) + bias;
        }
        _size[id] += num_vecs;
        pthread_spin_unlock(&_lock[id]);
    }

    void collect(data_stream_t *data_stream, filter_t *filter, long num_threads = 1);

    void reduce(long type, float run_th, long num_threads = 1, float goal_th = 1.0) {
        if (type == red_int1_512 || type == red_int1_768) {
            printf("[Info] Running %s reducing with %ld threads, run_th = %d, goal_th = %.3f\n", this->name(type), num_threads, (int)run_th, goal_th);
        } else {
            printf("[Info] Running %s reducing with %ld threads, run_th = %.3f, goal_th = %.3f\n", this->name(type), num_threads, run_th, goal_th);
        }
        struct timeval start, end;
        gettimeofday(&start, NULL);

        constexpr long alloc_lim = 8388608;
        constexpr long loc_sol_lim = 1 << 25;
        constexpr long sol_lim = 1UL << 32;

        uint64_t *sol = (uint64_t *) malloc(sizeof(uint64_t) * sol_lim * 2L);
        uint64_t *hash = (uint64_t *) malloc(sizeof(uint64_t) * sol_lim);
        uint64_t num_sol = 0;
        uint64_t pre_sol = 0;
        pthread_spinlock_t lock;
        pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE);
        long remain = _B;

        double total_dp = 0.0;
        
        long alloc_num = 0;
        for (long i = 0; i < _B; i++) {
            total_dp += _size[i] * (_size[i] - 1) * 0.5;
            if (_size[i] > alloc_num) alloc_num = _size[i];
        }

        if (alloc_num > alloc_lim) {
            printf("[Warning] max bucket size(%ld) larger than alloc_lim (%ld), truncated\n", alloc_num, alloc_lim);
            alloc_num = alloc_lim;
        }

        alloc_num = (alloc_num + 511) / 512 * 512;

        std::vector<std::thread> threads;
        for (long thread = 0; thread < num_threads; thread++) {
            threads.emplace_back([&, thread]() {
                CHECK_CUDA_ERR(cudaSetDevice(thread % num_device));
                CHECK_CUDA_ERR(cudaFuncSetAttribute(kernel_red, cudaFuncAttributeMaxDynamicSharedMemorySize, red_shmem));
                CHECK_CUDA_ERR(cudaFuncSetAttribute(kernel_sim, cudaFuncAttributeMaxDynamicSharedMemorySize, sim_shmem));
                CHECK_CUDA_ERR(cudaFuncSetAttribute(kernel_int8, cudaFuncAttributeMaxDynamicSharedMemorySize, int8_shmem));
                CHECK_CUDA_ERR(cudaFuncSetAttribute(kernel_int4, cudaFuncAttributeMaxDynamicSharedMemorySize, int4_shmem));
                CHECK_CUDA_ERR(cudaFuncSetAttribute(kernel_int512, cudaFuncAttributeMaxDynamicSharedMemorySize, int512_shmem));
                CHECK_CUDA_ERR(cudaFuncSetAttribute(kernel_int768, cudaFuncAttributeMaxDynamicSharedMemorySize, int768_shmem));
                
                int *local_sol = (int *) malloc(sizeof(int) * loc_sol_lim * 2L);
                int local_num_sol = 0;

                float *d_buc;
                uint32_t *d_sim;
                CHECK_CUDA_ERR(cudaMalloc(&d_buc, alloc_num * 256L * sizeof(float)));
                CHECK_CUDA_ERR(cudaMalloc(&d_sim, alloc_num * 128L));
                int *d_out, *d_num;
                CHECK_CUDA_ERR(cudaMalloc(&d_num, sizeof(int)));
                CHECK_CUDA_ERR(cudaMalloc(&d_out, sizeof(int) * loc_sol_lim * 2));

                for (;;) {
                    long id = -1;
                    pthread_spin_lock(&lock);
                    if (remain > 0) id = --remain;
                    pthread_spin_unlock(&lock);

                    if (id == -1) break;
                    long buc_size = _size[id] > alloc_num ? alloc_num : _size[id];
                    CHECK_CUDA_ERR(cudaMemcpy(d_buc, _bucket[id], buc_size * 256 * sizeof(int), cudaMemcpyHostToDevice));
                    CHECK_CUDA_ERR(cudaMemset(d_num, 0, sizeof(int)));
                    if (type == red_default || type == red_int4 || type == red_int8) {
                        if (type == red_int8) {
                            kernel_int8<<<int8_blocks, int8_threads, int8_shmem>>>(
                                d_buc, buc_size
                            );
                        }
                        if (type == red_int4) {
                            kernel_int4<<<int4_blocks, int4_threads, int4_shmem>>>(
                                d_buc, buc_size
                            );
                        }
                        kernel_red<<<red_blocks, red_threads, red_shmem>>>(
                            d_out, d_num, loc_sol_lim, d_buc, buc_size, run_th
                        );
                    } else {
                        if (type == red_int1_512) {
                            kernel_int512<<<int512_blocks, int512_threads, int512_shmem>>>(
                                d_sim, d_buc, buc_size
                            );
                        }
                        if (type == red_int1_768) {
                            kernel_int768<<<int768_blocks, int768_threads, int768_shmem>>>(
                                d_sim, d_buc, buc_size
                            );
                        }
                        kernel_sim<<<sim_blocks, sim_threads, sim_shmem>>>(
                            d_out, d_num, loc_sol_lim, d_sim, buc_size, (int)run_th
                        );
                    }
                    
                    CHECK_LAST_ERR;
                    CHECK_CUDA_ERR(cudaMemcpy(&local_num_sol, d_num, sizeof(int), cudaMemcpyDeviceToHost));
                    if (local_num_sol > loc_sol_lim) {
                        printf("[Warning] local sol buffer overflow: %d / %ld, some sol ignored\n", local_num_sol, loc_sol_lim);
                        local_num_sol = loc_sol_lim;
                    }
                    CHECK_CUDA_ERR(cudaMemcpy(local_sol, d_out, 8 * local_num_sol, cudaMemcpyDeviceToHost));
                    pthread_spin_lock(&lock);
                    pre_sol += local_num_sol;
                    pthread_spin_unlock(&lock);

                    long local_real = local_num_sol;
                    #pragma unroll
                    for (long i = 0; i < local_real; i++) {
                        float *vv = &_bucket[id][local_sol[2 * i] * 256L];
                        float *uv = &_bucket[id][local_sol[2 * i + 1] * 256L];
                        __m512 acc = _mm512_setzero_ps();
                        for (long j = 0; j < 256; j += 16) {
                            __m512 v1 = _mm512_loadu_ps(&vv[j]);
                            __m512 v2 = _mm512_loadu_ps(&uv[j]);
                            __m512 sub = _mm512_sub_ps(v1, v2);
                            acc = _mm512_fmadd_ps(sub, sub, acc);
                        }
                        float n = _mm512_reduce_add_ps(acc);
                        if (n >= goal_th) {
                            local_sol[2 * i] = local_sol[2 * (local_real - 1)];
                            local_sol[2 * i + 1] = local_sol[2 * (local_real - 1) + 1];
                            local_real--;
                            i--;
                            continue;
                        }
                    }

                    pthread_spin_lock(&lock);
                    long pos = num_sol;
                    long to_add = pos + local_real < sol_lim ? local_real : sol_lim - pos;
                    num_sol += to_add;
                    pthread_spin_unlock(&lock);
                    for (long i = 0; i < to_add; i++) {
                        sol[2L * (pos + i)] = _bucket_ptr[id][local_sol[2 * i]];
                        sol[2L * (pos + i) + 1] = _bucket_ptr[id][local_sol[2 * i + 1]];
                    }
                }

                CHECK_CUDA_ERR(cudaFree(d_out));
                CHECK_CUDA_ERR(cudaFree(d_num));
                CHECK_CUDA_ERR(cudaFree(d_buc));
                free(local_sol);
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        gettimeofday(&end, NULL);
        printf("[Info] Completed, %.2f s elapsed\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6);

        for (long i = 0; i < num_sol; i++) {
            if (sol[2 * i] > sol[2 * i + 1]) {
                uint64_t tmp = sol[2 * i];
                sol[2 * i] = sol[2 * i + 1];
                sol[2 * i + 1] = tmp;
            }
            hash[i] = hash_pair(sol[2 * i], sol[2 * i + 1]);
        }

        free(sol);
        
        qsort(hash, num_sol, sizeof(uint64_t), compare);
        uint64_t unique_count = num_sol ? 1 : 0;
        for (uint64_t i = 1; i < num_sol; i++) {
            if (hash[i] != hash[i - 1]) unique_count++;
        }

        printf("[Info] Total dp %.2fG, 1 / %.2f pre sol pass, Total find: %lu(%.2f), Unique sol: %lu(%.2f)\n", total_dp / 1e9, 
            pre_sol / (num_sol + 1e-9), num_sol, total_dp / (num_sol + 1e-9), unique_count, total_dp / (unique_count + 1e-9));
        printf("[Info] metrics: {%.2f, %.2f, %.3f}\n", (buc_total_dp + total_dp) / (unique_count + 1e-9), 
                total_dp / (unique_count + 1e-9), num_sol / (unique_count + 1e-9));
        free(hash);
        pthread_spin_destroy(&lock);
    }

    const char *name(long type) {
        if (type == red_none) return "none";
        if (type == red_default) return "default";
        if (type == red_int8) return "int8";
        if (type == red_int4) return "int4";
        if (type == red_int1_512) return "int1-512";
        if (type == red_int1_768) return "int1-768";
        return "unknown";
    } 

    private:
    long _CSD, _B;
    float **_bucket;
    uint64_t **_bucket_ptr;
    long *_size, *_cap;
    pthread_spinlock_t *_lock;
};

struct filter_t {
    static constexpr long type_bgj1 = 1;
    static constexpr long type_bgj2 = 2;
    static constexpr long type_bgj3 = 3;
    static constexpr long type_bdgl2 = 5;
    static constexpr long type_bdgl3 = 6;

    filter_t(long CSD, const char *filter_str) {
        this->_CSD = CSD;
        _parse_filter(filter_str);
        _gen_centers();
    }
    
    ~filter_t() {
        if (_center0) CHECK_CUDA_ERR(cudaFreeHost(_center0));
        if (_center1) CHECK_CUDA_ERR(cudaFreeHost(_center1));
        if (_center2) CHECK_CUDA_ERR(cudaFreeHost(_center2));
    }

    double apply(bucket_list_t *bucket_list, float *src, long num_vecs, float *d_buf, float *h_buf, long src_bias) {
        float *d_src = d_buf + 131072 * 256L;
        float *d_ct0 = d_buf + 131072 * 512L;
        float *d_ct1 = d_buf + 131072 * 768L;
        float *d_ct2 = d_buf + 131072 * 1024L;
        int *d_buc = (int *)d_buf + 131072 * 1280L;
        int *d_num = (int *)d_buf + 131072 * 1792L;
        int *h_buc = (int *)h_buf;
        int *h_num = (int *)h_buf + 131072 * 512L;

        double ret = 0.0;
        unsigned long long *d_cost = (unsigned long long *)(d_buf + 131072 * 1798L);
        unsigned long long h_cost[3] = {};

        CHECK_CUDA_ERR(cudaMemcpy(d_src, src, num_vecs * 256L * sizeof(float), cudaMemcpyHostToDevice));

        for (long bias = 0; bias < _B; bias += 131072) {
            long nctr = _B - bias < 131072 ? _B - bias : 131072;
            long nvec = num_vecs;
            CHECK_CUDA_ERR(cudaMemset(d_num, 0, 131072 * sizeof(int)));
            CHECK_CUDA_ERR(cudaMemset(d_cost, 0, 3 * sizeof(unsigned long long)));
            if (_center0) CHECK_CUDA_ERR(cudaMemcpy(d_ct0, _center0 + bias * 256L, nctr * 256L * sizeof(float), cudaMemcpyHostToDevice));
            if (_center1) CHECK_CUDA_ERR(cudaMemcpy(d_ct1, _center1 + bias * 256L, nctr * 256L * sizeof(float), cudaMemcpyHostToDevice));
            if (_center2) CHECK_CUDA_ERR(cudaMemcpy(d_ct2, _center2 + bias * 256L, nctr * 256L * sizeof(float), cudaMemcpyHostToDevice));

            if (_type == type_bgj1 || _type == type_bdgl2 || _type == type_bdgl3) {
                kernel_buc<1><<<buc_blocks, buc_threads, buc_shmem>>>(
                    d_buc, d_num, d_src, nvec, d_ct0, NULL, NULL, nctr, _alpha0, 0.f, 0.f, d_cost
                );
            } else if (_type == type_bgj2) {
                kernel_buc<2><<<buc_blocks, buc_threads, buc_shmem>>>(
                    d_buc, d_num, d_src, nvec, d_ct0, d_ct1, NULL, nctr, _alpha0, _alpha1, 0.f, d_cost
                );
            } else if (_type == type_bgj3) {
                kernel_buc<3><<<buc_blocks, buc_threads, buc_shmem>>>(
                    d_buc, d_num, d_src, nvec, d_ct0, d_ct1, d_ct2, nctr, _alpha0, _alpha1, _alpha2, d_cost
                );
            } else {
                printf("[Error] Unknown filter type: %ld\n", _type);
                abort();
            }

            CHECK_LAST_ERR;
            CHECK_CUDA_ERR(cudaMemcpy(h_num, d_num, 131072 * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERR(cudaMemcpy(h_buc, d_buc, 131072 * 512L * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERR(cudaMemcpy(h_cost, d_cost, 3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            if (_type == type_bgj1) ret += h_cost[0];
            if (_type == type_bdgl2) ret += h_cost[0] * 2.0 / _batch1;
            if (_type == type_bdgl3) ret += h_cost[0] * 3.0 / _batch1 / _batch1;
            if (_type == type_bgj2) ret += h_cost[0] * 1.0 / _batch1 + h_cost[1];
            if (_type == type_bgj3) ret += h_cost[0] * 1.0 / _batch1 / _batch2 + h_cost[1] * 1.0 / _batch2 + h_cost[2];
            random_interval_iter_t iter(nctr);
            for (long i = 0; i < nctr; i++) {
                int buc_id = iter.pop();
                if (h_num[buc_id] > 0) {
                    bucket_list->append(buc_id + bias, src, &h_buc[buc_id * 512L], h_num[buc_id], src_bias);
                }
            }
        }

        return ret;
    }

    const char *name() {
        if (_type == type_bgj1) return "bgj1";
        if (_type == type_bgj2) return "bgj2";
        if (_type == type_bgj3) return "bgj3";
        if (_type == type_bdgl2) return "bdgl2";
        if (_type == type_bdgl3) return "bdgl3";
        return "unknown";
    }

    private:
    long _type;
    float _alpha0, _alpha1, _alpha2, _beta0, _gamma0, _gamma1;
    long _batch0, _batch1, _batch2, _CSD, _B;
    float *_center0 = NULL, *_center1 = NULL, *_center2 = NULL;

    void _parse_filter(const char *filter_str) {
        char *colon_pos = strchr((char *)filter_str, ':');
        if (colon_pos) {
            *colon_pos = '\0';
        } else {
            printf("[Error] Invalid filter format: %s\n", filter_str);
            abort();
        }
        
        const char *params_str = colon_pos + 1;
        char filter_type[32];
        size_t type_len = colon_pos - filter_str;
        if (type_len >= sizeof(filter_type)) {
            printf("[Error] Filter type too long: %s\n", filter_str);
            abort();
        }
        strncpy(filter_type, filter_str, type_len);
        filter_type[type_len] = '\0';
        if (strcmp(filter_type, "bgj1") == 0) {
            this->_type = type_bgj1;
            if (sscanf(params_str, "%f,%ld", &_alpha0, &_batch0) != 2) {
                printf("[Error] bgj1 requires 2 params: alpha0, batch0\n");
                abort();
            }
            this->_B = _batch0;
        } else if (strcmp(filter_type, "bgj2") == 0) {
            this->_type = type_bgj2;
            if (sscanf(params_str, "%f,%f,%f,%ld,%ld", &_beta0, &_alpha0, &_alpha1, &_batch0, &_batch1) != 5) {
                printf("[Error] bgj2 requires 5 params: beta0, alpha0, alpha1, batch0, batch1\n");
                abort();
            }
            this->_B = _batch0 * _batch1;
        } else if (strcmp(filter_type, "bgj3") == 0) {
            this->_type = type_bgj3;
            if (sscanf(params_str, "%f,%f,%f,%f,%f,%f,%ld,%ld,%ld", &_beta0, &_gamma0, &_gamma1, &_alpha0, &_alpha1, &_alpha2, &_batch0, &_batch1, &_batch2) != 9) {
                printf("[Error] bgj3 requires 9 params: beta0, gamma0, gamma1, alpha0, alpha1, alpha2, batch0, batch1, batch2\n");
                abort();
            }
            this->_B = _batch0 * _batch1 * _batch2;
        } else if (strcmp(filter_type, "bdgl2") == 0) {
            this->_type = type_bdgl2;
            if (sscanf(params_str, "%f,%ld,%ld", &this->_alpha0, &this->_batch1, &this->_batch0) != 3) {
                printf("[Error] bdgl2 requires 3 params: alpha, batch, repeat\n");
                abort();
            }
            this->_B = this->_batch1 * this->_batch1 * this->_batch0;
        } else if (strcmp(filter_type, "bdgl3") == 0) {
            this->_type = type_bdgl3;
            if (sscanf(params_str, "%f,%ld,%ld", &this->_alpha0, &this->_batch1, &this->_batch0) != 3) {
                printf("[Error] bdgl3 requires 3 params: alpha, batch, repeat\n");
                abort();
            }
            this->_B = this->_batch1 * this->_batch1 * this->_batch1 * this->_batch0;
        } else {
            printf("[Error] Unknown filter type: %s\n", filter_type);
            abort();
        }
    }

    void _gen_centers() {
        float *d_buf;
        long seed = 0x79660588;
        CHECK_CUDA_ERR(cudaMalloc(&d_buf, 131072 * 256L * sizeof(float)));
        CHECK_CUDA_ERR(cudaFuncSetAttribute(kernel_normal, cudaFuncAttributeMaxDynamicSharedMemorySize, normal_shmem));
        if (this->_type == type_bgj1) {
            CHECK_CUDA_ERR(cudaMallocHost(&_center0, _B * 256L * sizeof(float)));
            _fill_normal(_center0, 0, _CSD, _B, 1.0, seed, d_buf);
        } else if (this->_type == type_bgj2) {
            CHECK_CUDA_ERR(cudaMallocHost(&_center0, _B * 256L * sizeof(float)));
            CHECK_CUDA_ERR(cudaMallocHost(&_center1, _B * 256L * sizeof(float)));
            float *v0 = (float *) malloc(_batch0 * 256L * sizeof(float));
            float *v1 = (float *) malloc(_batch0 * _batch1 * 256L * sizeof(float));
            _fill_normal(v0, 0, _CSD, _batch0, 1.0, seed, d_buf);
            _fill_normal(v1, 0, _CSD, _batch0 * _batch1, 1.0, seed, d_buf);

            for (int i = 0; i < _batch0; i++) {
                for (int j = 0; j < _batch1; j++) {
                    float *_v0 = v0 + i * 256L;
                    float *_v1 = v1 + (i * _batch1 + j) * 256L;
                    float dp = _dot(_v0, _v1);
                    float X = sqrt((1 - _beta0 * _beta0) / (1 - dp * dp));
                    for (long l = 0; l < _CSD; l++) _v1[l] = (_v1[l] - dp * _v0[l]) * X + _v0[l] * _beta0;
                    memcpy(_center0 + (i * _batch1 + j) * 256L, _v0, 256 * sizeof(float));
                    memcpy(_center1 + (i * _batch1 + j) * 256L, _v1, 256 * sizeof(float));
                }
            }

            free(v0);
            free(v1);
        } else if (this->_type == type_bgj3) {
            CHECK_CUDA_ERR(cudaMallocHost(&_center0, _B * 256L * sizeof(float)));
            CHECK_CUDA_ERR(cudaMallocHost(&_center1, _B * 256L * sizeof(float)));
            CHECK_CUDA_ERR(cudaMallocHost(&_center2, _B * 256L * sizeof(float)));
            float *v0 = (float *) malloc(_batch0 * 256L * sizeof(float));
            float *v1 = (float *) malloc(_batch0 * _batch1 * 256L * sizeof(float));
            float *v2 = (float *) malloc(_batch0 * _batch1 * _batch2 * 256L * sizeof(float));
            _fill_normal(v0, 0, _CSD, _batch0, 1.0, seed, d_buf);
            _fill_normal(v1, 0, _CSD, _batch0 * _batch1, 1.0, seed, d_buf);
            _fill_normal(v2, 0, _CSD, _batch0 * _batch1 * _batch2, 1.0, seed, d_buf);
            
            for (int i = 0; i < _batch0; i++) {
                for (int j = 0; j < _batch1; j++) {
                    for (int k = 0; k < _batch2; k++) {
                        float *_v0 = v0 + i * 256L;
                        float *_v1 = v1 + (i * _batch1 + j) * 256L;
                        float *_v2 = v2 + (i * _batch1 * _batch2 + j * _batch2 + k) * 256L;
                        float v1_tmp[256] = {};
                        float v2_tmp[256] = {};
                        float dp01 = _dot(_v0, _v1);
                        for (long l = 0; l < _CSD; l++) v1_tmp[l] = (_v1[l] - dp01 * _v0[l]) / sqrt(1 - dp01 * dp01);
                        float dp02 = _dot(_v0, _v2);
                        float dp12 = _dot(v1_tmp, _v2);
                        for (long l = 0; l < _CSD; l++) v2_tmp[l] = (_v2[l] - dp02 * _v0[l] - dp12 * v1_tmp[l]) / sqrt(1 - dp02 * dp02 - dp12 * dp12);
                        float X = (_gamma1 - _gamma0 * _beta0) / sqrt(1 - _beta0 * _beta0);
                        for (long l = 0; l < _CSD; l++) v2_tmp[l] = v2_tmp[l] * sqrt(1 - X * X - _gamma0 * _gamma0) + X * v1_tmp[l] + _gamma0 * _v0[l];
                        for (long l = 0; l < _CSD; l++) v1_tmp[l] = _v0[l] * _beta0 + v1_tmp[l] * sqrt(1 - _beta0 * _beta0);
                        memcpy(_center0 + (i * _batch1 * _batch2 + j * _batch2 + k) * 256L, _v0, 256 * sizeof(float));
                        memcpy(_center1 + (i * _batch1 * _batch2 + j * _batch2 + k) * 256L, v1_tmp, 256 * sizeof(float));
                        memcpy(_center2 + (i * _batch1 * _batch2 + j * _batch2 + k) * 256L, v2_tmp, 256 * sizeof(float));
                    }
                }
            }

            free(v0);
            free(v1);
            free(v2);
        } else if (this->_type == type_bdgl2) {
            const long d0 = _CSD / 2;
            const long d1 = (_CSD + 1) / 2;

            CHECK_CUDA_ERR(cudaMallocHost(&_center0, _B * 256L * sizeof(float)));
            float *v0 = (float *) malloc(_batch0 * _batch1 * 256L * sizeof(float));
            float *v1 = (float *) malloc(_batch0 * _batch1 * 256L * sizeof(float));
            _fill_normal(v0, 0, d0, _batch0 * _batch1, sqrt((float)d0 / _CSD), seed, d_buf);
            _fill_normal(v1, d0, d0 + d1, _batch0 * _batch1, sqrt((float)d1 / _CSD), seed, d_buf);
            for (int i = 0; i < _batch0; i++) {
                for (int j = 0; j < _batch1 * _batch1; j++) {
                    float *_dst = _center0 + (i * _batch1 * _batch1 + j) * 256L;
                    float *_src0 = v0 + (i * _batch1 + j / _batch1) * 256L;
                    float *_src1 = v1 + (i * _batch1 + j % _batch1) * 256L;
                    for (int l = 0; l < 256; l++) _dst[l] = _src0[l] + _src1[l];
                }
            }

            free(v0);
            free(v1);
        } else if (this->_type == type_bdgl3) {
            const long d0 = _CSD / 3;
            const long d1 = (_CSD + 1) / 3;
            const long d2 = (_CSD + 2) / 3;
            CHECK_CUDA_ERR(cudaMallocHost(&_center0, _B * 256L * sizeof(float)));
            float *v0 = (float *) malloc(_batch0 * _batch1 * 256L * sizeof(float));
            float *v1 = (float *) malloc(_batch0 * _batch1 * 256L * sizeof(float));
            float *v2 = (float *) malloc(_batch0 * _batch1 * 256L * sizeof(float));
            _fill_normal(v0, 0, d0, _batch0 * _batch1, sqrt((float)d0 / _CSD), seed, d_buf);
            _fill_normal(v1, d0, d0 + d1, _batch0 * _batch1, sqrt((float)d1 / _CSD), seed, d_buf);
            _fill_normal(v2, d0 + d1, _CSD, _batch0 * _batch1, sqrt((float)d2 / _CSD), seed, d_buf);
            for (int i = 0; i < _batch0; i++) {
                for (int j = 0; j < _batch1 * _batch1 * _batch1; j++) {
                    float *_dst = _center0 + (i * _batch1 * _batch1 * _batch1 + j) * 256L;
                    float *_src0 = v0 + (i * _batch1 + j / _batch1 / _batch1) * 256L;
                    float *_src1 = v1 + (i * _batch1 + (j / _batch1) % _batch1) * 256L;
                    float *_src2 = v2 + (i * _batch1 + j % _batch1) * 256L;
                    for (int l = 0; l < 256; l++) _dst[l] = _src0[l] + _src1[l] + _src2[l];
                }
            }

            free(v0);
            free(v1);
            free(v2);
        }

        CHECK_CUDA_ERR(cudaFree(d_buf));
    }

    inline float _dot(float *x, float *y) {
        float ret = 0.f;
        for (int i = 0; i < 256; i++) {
            ret += x[i] * y[i];
        }
        return ret;
    }

    void _fill_normal(float *dst, int ind_l, int ind_r, long to_gen, float target_len, long &seed, float *d_buf) {
        for (long pos = 0; pos < to_gen; pos += 131072) {
            long part = pos + 131072 < to_gen ? 131072 : to_gen - pos;
            kernel_normal<<<normal_blocks, normal_threads, normal_shmem>>>(
                d_buf, ind_l, ind_r, part, target_len, seed++
            );
            CHECK_LAST_ERR;
            CHECK_CUDA_ERR(cudaMemcpy(dst + pos * 256L, d_buf, part * 256 * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
};

void bucket_list_t::collect(data_stream_t *data_stream, filter_t *filter, long num_threads) {
    printf("[Info] Running %s bucketing with %ld threads\n", filter->name(), num_threads);
    struct timeval start, end;
    gettimeofday(&start, NULL);

    double total_dp = 0.0;
    pthread_spinlock_t lock;
    pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE);

    std::vector<std::thread> threads;
    for (long thread = 0; thread < num_threads; thread++) {
        threads.emplace_back([thread, this, data_stream, filter, &total_dp, &lock]() {
            CHECK_CUDA_ERR(cudaSetDevice(thread % num_device));
            CHECK_CUDA_ERR(cudaFuncSetAttribute(kernel_db_gen, cudaFuncAttributeMaxDynamicSharedMemorySize, db_gen_shmem));
            CHECK_CUDA_ERR(cudaFuncSetAttribute(kernel_buc<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, buc_shmem));
            CHECK_CUDA_ERR(cudaFuncSetAttribute(kernel_buc<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, buc_shmem));
            CHECK_CUDA_ERR(cudaFuncSetAttribute(kernel_buc<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, buc_shmem));
            float *dst, *buf;
            CHECK_CUDA_ERR(cudaMallocHost(&dst, data_stream->batch_size * 256L * sizeof(float)));
            if (!data_stream->_db_path) {
                CHECK_CUDA_ERR(cudaMalloc(&buf, data_stream->batch_size * 256L * sizeof(float)));
            } else {
                buf = (float *) malloc(data_stream->batch_size * 256L * sizeof(float));
            }

            float *d_buf, *h_buf;
            long d_buf_size = 131072 * 1800L * sizeof(int);
            long h_buf_size = 131072 * 515L * sizeof(int);
            CHECK_CUDA_ERR(cudaMalloc(&d_buf, d_buf_size));
            CHECK_CUDA_ERR(cudaMallocHost(&h_buf, h_buf_size));

            for (;;) {
                long src_bias;
                long num = data_stream->pop(dst, buf, src_bias);
                double fil_dp = 0.0;
                if (num) fil_dp += filter->apply(this, dst, num, d_buf, h_buf, src_bias);
                pthread_spin_lock(&lock);
                total_dp += fil_dp;
                pthread_spin_unlock(&lock);
                if (num < data_stream->batch_size && _CSD >= 100) break;
                if (num < 32768 && _CSD < 100) break;
            }

            CHECK_CUDA_ERR(cudaFree(d_buf));
            CHECK_CUDA_ERR(cudaFreeHost(h_buf));
            CHECK_CUDA_ERR(cudaFreeHost(dst));
            if (!data_stream->_db_path) {
                CHECK_CUDA_ERR(cudaFree(buf));
            } else {
                free(buf);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
    gettimeofday(&end, NULL);
    printf("[Info] Completed, %.2f s elapsed\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6);

    float avg_size = 0., std_size = 0.;
    for (long i = 0; i < _B; i++) avg_size += _size[i];
    avg_size /= _B;
    for (long i = 0; i < _B; i++) std_size += (_size[i] - avg_size) * (_size[i] - avg_size);
    std_size /= _B;
    printf("[Info] Total dp %.2fG, Avg bucket size: %.2f, std: %.2f\n", total_dp / 1e9, avg_size, sqrt(std_size));
    buc_total_dp = total_dp;

    pthread_spin_destroy(&lock);
}

void print_help(char *prog_name) {
    printf("Usage: %s [-h] --CSD CSD [--thread THREAD] --filter FILTER [--red [RED ...]] [--db DB]\n", prog_name);
    printf("\n");
    printf("Options:\n");
    printf("  -h, --help   : Show this help message and exit\n");
    printf("  -t, --thread : Number of threads to use (default: 1)\n");
    printf("  --CSD        : Vector dimension (required)\n");
    printf("  --filter     : Filter type and params, e.g. \"bgj1:0.31,8192\" (required)\n");
    printf("  --red        : Reducer type, e.g. \"int8:1.0 int4:1.05 int1-512:200\" (optional)\n");
    printf("  --db         : Path to db file (optional)\n");
}

int main(int argc, char *argv[]) {
    long CSD         = 0;
    long num_thread  = 1;
    char *filter_str = NULL;
    char *red_str    = NULL;
    char *db_path    = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcasecmp(argv[i], "-h") == 0 || strcasecmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcasecmp(argv[i], "-t") == 0 || strcasecmp(argv[i], "--thread") == 0) {
            if (i + 1 < argc) {
                if (argv[i + 1][0] != '-') {
                    num_thread = atoi(argv[++i]);
                    continue;
                }
            }
            printf("[Error] Missing value for --thread\n");
        } else if (strcasecmp(argv[i], "--CSD") == 0 && !CSD) {
            if (i + 1 < argc) {
                if (argv[i + 1][0] != '-') {
                    CSD = atoi(argv[++i]);
                    continue;
                }
            }
            printf("[Error] Missing value for --CSD\n");
        } else if (strcasecmp(argv[i], "--filter") == 0 && !filter_str) {
            if (i + 1 < argc) {
                if (argv[i + 1][0] != '-') {
                    filter_str = strdup(argv[++i]);
                    continue;
                }
            }
            printf("[Error] Missing value for --filter\n");
        } else if (strcasecmp(argv[i], "--red") == 0 && !red_str) {
            int task_cnt = 0;
            int char_len = 1;
            for (int j = i + 1; j < argc; j++) {
                if (argv[j][0] != '-') {
                    task_cnt++;
                    char_len += strlen(argv[j]) + 1;
                } else break;
            }

            red_str = (char *) calloc(char_len, 1);

            for (int j = 0; j < task_cnt; j++) {
                strcat(red_str, argv[i + 1 + j]);
                if (j < task_cnt - 1) strcat(red_str, " ");
            }

            i += task_cnt;
            continue;
        } else if (strcasecmp(argv[i], "--db") == 0 && !db_path) {
            if (i + 1 < argc) {
                if (argv[i + 1][0] != '-') {
                    db_path = strdup(argv[++i]);
                    continue;
                }
            }
            printf("[Error] Missing value for --db\n");
        } else {
            printf("[Error] Unknown option: %s\n", argv[i]);
            print_help(argv[0]);
        }

        return -1;
    }

    if (!CSD || !filter_str) {
        printf("[Error] Missing value for --%s\n", !CSD ? "CSD" : "filter");
        print_help(argv[0]);
        return -1;
    }
    
    CHECK_CUDA_ERR(cudaGetDeviceCount(&num_device));
    if (num_device <= 0) {
        printf("[Error] No CUDA device found\n");
        return -1;
    }

    for (long i = 0; i < num_device && i < num_thread; i++) {
        CHECK_CUDA_ERR(cudaSetDevice(i));
    }

    CHECK_CUDA_ERR(cudaSetDevice(0));

    data_stream_t data_stream(CSD, db_path);

    filter_t filter(CSD, filter_str);

    bucket_list_t bucket_list(CSD, filter._B);

    bucket_list.collect(&data_stream, &filter, num_thread);    

    if (red_str) {
        bucket_list.reduce(bucket_list_t::red_default, 1.0, num_thread, 1.0);

        char *red_str_copy = strdup(red_str);
        char *token = strtok(red_str_copy, " ");
        while (token != NULL) {
            char *colon_pos = strchr(token, ':');
            if (colon_pos) {
                *colon_pos = '\0';
                char *red_type_str = token;
                char *red_th_str = colon_pos + 1;
                long red_type = bucket_list_t::red_none;
                if (strcmp(red_type_str, "default") == 0) red_type = bucket_list_t::red_default;
                else if (strcmp(red_type_str, "int8") == 0) red_type = bucket_list_t::red_int8;
                else if (strcmp(red_type_str, "int4") == 0) red_type = bucket_list_t::red_int4;
                else if (strcmp(red_type_str, "int1-512") == 0) red_type = bucket_list_t::red_int1_512;
                else if (strcmp(red_type_str, "int1-768") == 0) red_type = bucket_list_t::red_int1_768;
                else {
                    printf("[Error] Unknown red_type: %s\n", red_type_str);
                    token = strtok(NULL, " ");
                    continue;
                }
                if (red_type == bucket_list_t::red_int1_512 || red_type == bucket_list_t::red_int1_768) {
                    int run_th = 200;
                    float goal_th = 1.0f;
                    
                    if (sscanf(red_th_str, "%d,%f", &run_th, &goal_th) == 2) {

                    } else if (sscanf(red_th_str, "%d", &run_th) == 1) {
                        goal_th = 1.0f;
                    } else {
                        printf("Error: Invalid red_th_str format: %s\n", red_th_str);
                        continue;
                    }
                    bucket_list.reduce(red_type, run_th, num_thread, goal_th);
                } else {
                    float run_th = 1.0f;
                    float goal_th = 1.0f;
                    if (sscanf(red_th_str, "%f,%f", &run_th, &goal_th) == 2) {

                    } else if (sscanf(red_th_str, "%f", &run_th) == 1) {
                        goal_th = 1.0f;
                    } else {
                        printf("Error: Invalid red_th_str format: %s\n", red_th_str);
                        continue;
                    }
                    bucket_list.reduce(red_type, run_th, num_thread, goal_th);
                }
            }
            token = strtok(NULL, " ");
        }
        free(red_str_copy);
    }

    if (filter_str) free(filter_str);
    if (red_str) free(red_str);
    if (db_path) free(db_path);

    return 0;
}