#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// ----------------------------------------------------------------------------
// 128-bit 向量化访存工具
__device__ inline float4 ld_vec(const float* address) {
    return *reinterpret_cast<const float4*>(address);
}

__device__ inline void st_vec(float* address, float4 val) {
    *reinterpret_cast<float4*>(address) = val;
}

// ----------------------------------------------------------------------------
// 核函数：结合了 128-bit 向量化访存、Shared Memory Bank 冲突打散与 Bias 融合
// 限制单线程寄存器用量以保证 SM 至少调度 2 个活跃 Block (Occupancy 提升)
// QKV 投影矩阵乘法核函数：输入 (B_T, C)，输出 (B_T, 3*C)
__global__ void __launch_bounds__(16 * 16, 2) matmul_vectorized_kernel2(
    float* out,
    const float* inp,
    const float* weight,
    const float* bias,
    int B_T, int C, int OC) {
    
    // out:    [B_T, OC] (此处 OC = 3 * C)
    // inp:    [B_T, C]
    // weight: [OC, C]   (行优先存储，每一行是一个输出通道的权重向量)
    // bias:   [OC]
    int oc = 8 * (blockIdx.y * blockDim.y + threadIdx.y);

    // 申请双侧共享内存缓存 (形状一致，均为 128 行 x 32 列)
    __shared__ float lhs_s[128][32];
    __shared__ float rhs_s[128][32];

    // 指针定位到当前 Block 的起点
    inp    += 128 * blockIdx.x * C;
    weight += 128 * blockIdx.y * C;
    out    += 128 * blockIdx.x * OC + 128 * blockIdx.y;

    // 1. 寄存器初始化：若有 bias 直接向量化加载到累加器
    float vals[8][8] = {0.0f};
    if (bias != NULL) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            #pragma unroll
            for (int j = 0; j < 8; j += 4) {
                float4 b = ld_vec(bias + oc + j);
                vals[i][j + 0] = b.x;
                vals[i][j + 1] = b.y;
                vals[i][j + 2] = b.z;
                vals[i][j + 3] = b.w;
            }
        }
    }

    // 打散同一 Warp 内线程访问 Shared Memory 的起始相位
    int si_start = 4 * (16 * threadIdx.y + threadIdx.x);

    // 2. 沿 C 轴以 32 为步长推进
    for (int so = 0; so < C; so += 32) {
        __syncthreads();

        // 向量化搬运全局显存到共享内存 (Global -> Shared Memory)
        int xmod8 = threadIdx.x % 8;
        int xby8  = threadIdx.x / 8;
        int xo    = 4 * xmod8;

        #pragma unroll
        for (int y = 2 * threadIdx.y + xby8; y < 128; y += 32) {
            st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
            st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
        }

        __syncthreads();

        // 计算：从共享内存取到寄存器并做外积累加
        #pragma unroll
        for (int si = si_start; si < si_start + 32; si += 4) {
            int k_sub = si % 32;

            float4 rhs[8];
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                rhs[u] = ld_vec(&rhs_s[u + 8 * threadIdx.y][k_sub]);
            }

            #pragma unroll
            for (int ii = 0; ii < 8; ++ii) {
                float4 lhs = ld_vec(&lhs_s[ii + 8 * threadIdx.x][k_sub]);
                #pragma unroll
                for (int ji = 0; ji < 8; ++ji) {
                    vals[ii][ji] += lhs.x * rhs[ji].x;
                    vals[ii][ji] += lhs.y * rhs[ji].y;
                    vals[ii][ji] += lhs.z * rhs[ji].z;
                    vals[ii][ji] += lhs.w * rhs[ji].w;
                }
            }
        }
    }

    // 3. 向量化写回全局显存 (Register -> Global Memory)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; j += 4) {
            float4 result;
            result.x = vals[i][j + 0];
            result.y = vals[i][j + 1];
            result.z = vals[i][j + 2];
            result.w = vals[i][j + 3];
            st_vec(out + (8 * threadIdx.x + i) * OC + 8 * threadIdx.y + j, result);
        }
    }
}

// ----------------------------------------------------------------------------
// CPU 验证基准 (Y = X @ W.T + b)
void matmul_cpu(const float* inp, const float* weight, const float* bias, float* out, int B_T, int C, int OC) {
    for (int i = 0; i < B_T; i++) {
        for (int j = 0; j < OC; j++) {
            float sum = (bias != NULL) ? bias[j] : 0.0f;
            for (int k = 0; k < C; k++) {
                sum += inp[i * C + k] * weight[j * C + k];
            }
            out[i * OC + j] = sum;
        }
    }
}

// ----------------------------------------------------------------------------
int main() {
    // GPT-2 Base 配置
    int B   = 4;     // Batch Size
    int T   = 1024;  // Sequence Length
    int B_T = B * T; // 4096 (必须是 128 的整数倍以满足当前 kernel 分块)
    int C   = 768;   // 通道数 / 嵌入维度
    int OC  = 3 * C; // 768 * 3 = 2304 (QKV 联合输出通道数)

    size_t bytes_inp    = (size_t)B_T * C * sizeof(float);
    size_t bytes_weight = (size_t)OC * C * sizeof(float);
    size_t bytes_bias   = (size_t)OC * sizeof(float);
    size_t bytes_out    = (size_t)B_T * OC * sizeof(float);

    printf("矩阵规模配置:\n");
    printf("  Batch * SeqLen (B_T) : %d\n", B_T);
    printf("  Channels (C)         : %d\n", C);
    printf("  Output Channels (OC) : %d (3 * C = %d，融合 QKV)\n", OC, OC);
    printf("  QKV 权重矩阵显存占用 : %.2f MB\n", (float)bytes_weight / (1024 * 1024));

    // Host 分配
    float* h_inp     = (float*)malloc(bytes_inp);
    float* h_weight  = (float*)malloc(bytes_weight);
    float* h_bias    = (float*)malloc(bytes_bias);
    float* h_out_gpu = (float*)malloc(bytes_out);
    float* h_out_cpu = (float*)malloc(bytes_out);

    // 随机初始化
    for (size_t i = 0; i < (size_t)B_T * C; i++)   h_inp[i]    = (float)(rand() % 10) / 10.0f;
    for (size_t i = 0; i < (size_t)OC * C; i++)    h_weight[i] = (float)(rand() % 10) / 10.0f;
    for (size_t i = 0; i < (size_t)OC; i++)        h_bias[i]   = (float)(rand() % 10) / 10.0f;

    // Device 分配
    float *d_inp, *d_weight, *d_bias, *d_out;
    cudaMalloc((void**)&d_inp, bytes_inp);
    cudaMalloc((void**)&d_weight, bytes_weight);
    cudaMalloc((void**)&d_bias, bytes_bias);
    cudaMalloc((void**)&d_out, bytes_out);

    cudaMemcpy(d_inp, h_inp, bytes_inp, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, bytes_weight, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bytes_bias, cudaMemcpyHostToDevice);

    // 配置 Grid：X 轴切分 B_T，Y 轴切分 OC (3*C)
    dim3 blockDim(16, 16);
    dim3 gridDim(CEIL_DIV(B_T, 128), CEIL_DIV(OC, 128));

    printf("启动 QKV 融合投影 Kernel...\n");
    printf("  Grid 维度: (%d, %d)\n", gridDim.x, gridDim.y);

    matmul_vectorized_kernel2<<<gridDim, blockDim>>>(d_out, d_inp, d_weight, d_bias, B_T, C, OC);
    cudaDeviceSynchronize();

    // 拷贝回 CPU 比对
    cudaMemcpy(h_out_gpu, d_out, bytes_out, cudaMemcpyDeviceToHost);
    matmul_cpu(h_inp, h_weight, h_bias, h_out_cpu, B_T, C, OC);

    float max_diff = 0.0f;
    for (size_t i = 0; i < (size_t)B_T * OC; i++) {
        float diff = fabsf(h_out_gpu[i] - h_out_cpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("最大绝对误差: %e\n", max_diff);
    if (max_diff < 1e-3f) {
        printf("验证通过！QKV 融合投影计算结果与 CPU 完全一致。\n");
    } else {
        printf("验证失败！存在精度误差。\n");
    }

    // 演示说明：此时 d_out 中的数据在逻辑上是 (B, T, 3, C)
    // 每一个 token 在显存中连续存放 [Q (0~C-1), K (C~2C-1), V (2C~3C-1)]
    // 下一步便可直接对接 permute_kernel 将其切分并重排为多头形状 (B, NH, T, HS)

    cudaFree(d_inp); cudaFree(d_weight); cudaFree(d_bias); cudaFree(d_out);
    free(h_inp); free(h_weight); free(h_bias); free(h_out_gpu); free(h_out_cpu);

    return 0;
}