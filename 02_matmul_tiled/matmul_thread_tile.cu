#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// 分块参数定义
#define BLOCK_SIZE_X 16       // 线程块 X 方向线程数 (列方向线程数)
#define BLOCK_SIZE_Y 16       // 线程块 Y 方向线程数 (行方向线程数)
#define THREAD_TILE_X 8       // 每个线程处理 8 列
#define THREAD_TILE_Y 8       // 每个线程处理 8 行

#define BLOCK_TILE_X (BLOCK_SIZE_X * THREAD_TILE_X) // 16 * 8 = 128
#define BLOCK_TILE_Y (BLOCK_SIZE_Y * THREAD_TILE_Y) // 16 * 8 = 128
#define TILE_K 32                                   // 沿 K 轴每次前进的步长，取 32

// 矩阵乘法核函数：2D Thread Tiling (8x8)，K 步长为 32
__global__ void matmulThreadTiledKernelK32(const float* M, const float* N, float* Out, int r, int w) {
    // 1. 申请共享内存 (各自 128x32，共 32KB)
    __shared__ float s_M[BLOCK_TILE_Y][TILE_K]; // 128 x 32
    __shared__ float s_N[TILE_K][BLOCK_TILE_X]; // 32 x 128

    int tx = threadIdx.x; // 0 ~ 15
    int ty = threadIdx.y; // 0 ~ 15
    int tid = ty * BLOCK_SIZE_X + tx; // 0 ~ 255 (线程一维 ID)

    // 本 Block 负责计算的输出矩阵区域的左上角全局行列索引
    int block_row = blockIdx.y * BLOCK_TILE_Y;
    int block_col = blockIdx.x * BLOCK_TILE_X;   

    // 2. 线程私有的寄存器数组，用于累加 8x8 的结果
    float sum[THREAD_TILE_Y][THREAD_TILE_X] = {0.0f};

    // 沿 K 维度分步滑动推进 (步长 32)
    int numTiles = (w + TILE_K - 1) / TILE_K;
    for (int t = 0; t < numTiles; t++) {
        int k_offset = t * TILE_K;

        // ---------- 协同搬运 M 到 s_M (128 x 32 = 4096 个元素) ----------
        // 256 个线程平分 4096 个元素，每个线程搬运 4096 / 256 = 16 个元素
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int sm_index = tid + i * 256;      // 0 ~ 4095
            int sm_row = sm_index / TILE_K;    // 0 ~ 127
            int sm_col = sm_index % TILE_K;    // 0 ~ 31

            int global_row = block_row + sm_row;
            int global_col = k_offset + sm_col;

            if (global_row < r && global_col < w) {
                s_M[sm_row][sm_col] = M[global_row * w + global_col];
            } else {
                s_M[sm_row][sm_col] = 0.0f;
            }
        }

        // ---------- 协同搬运 N 到 s_N (32 x 128 = 4096 个元素) ----------
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int sn_index = tid + i * 256;          // 0 ~ 4095
            int sn_row = sn_index / BLOCK_TILE_X;  // 0 ~ 31
            int sn_col = sn_index % BLOCK_TILE_X;  // 0 ~ 127

            int global_row = k_offset + sn_row;
            int global_col = block_col + sn_col;

            if (global_row < w && global_col < w) {
                s_N[sn_row][sn_col] = N[global_row * w + global_col];
            } else {
                s_N[sn_row][sn_col] = 0.0f;
            }
        }

        // 栅栏同步：确保 4096 个元素全部搬运完成
        __syncthreads();

        // ---------- 计算：沿 K=32 展开，从共享内存读取到寄存器并做外积累加 ----------
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            // 寄存器缓存：每个线程读取它需要的 8 个 M 元素与 8 个 N 元素
            float reg_M[THREAD_TILE_Y];
            float reg_N[THREAD_TILE_X];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_Y; i++) {
                reg_M[i] = s_M[ty * THREAD_TILE_Y + i][k];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_TILE_X; j++) {
                reg_N[j] = s_N[k][tx * THREAD_TILE_X + j];
            }

            // 在寄存器内完成 8x8=64 次 FMA 乘加
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_Y; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_X; j++) {
                    sum[i][j] += reg_M[i] * reg_N[j];
                }
            }
        }

        // 栅栏同步：确保计算完成，再进入下一轮 Tile 加载
        __syncthreads();
    }

    // 3. 将 8x8 的计算结果写回全局显存 Out
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_Y; i++) {
        int global_row = block_row + ty * THREAD_TILE_Y + i;
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_X; j++) {
            int global_col = block_col + tx * THREAD_TILE_X + j;
            if (global_row < r && global_col < w) {
                Out[global_row * w + global_col] = sum[i][j];
            }
        }
    }
}

// CPU 矩阵乘法对比函数
void matmul_cpu(const float* M, const float* N, float* Out, int r, int w) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < w; j++) {
            float sum = 0.0f;
            for (int k = 0; k < w; k++) {
                sum += M[i * w + k] * N[k * w + j];
            }
            Out[i * w + j] = sum;
        }
    }
}

int main() {
    int r = 512;
    int w = 512;
    int dim = r * w;
    size_t bytes = dim * sizeof(float);

    float* M_h = (float*)malloc(bytes);
    float* N_h = (float*)malloc(bytes);
    float* Out_gpu = (float*)malloc(bytes);
    float* Out_cpu = (float*)malloc(bytes);

    for (int i = 0; i < dim; i++) {
        M_h[i] = (float)(rand() % 10) / 10.0f;
        N_h[i] = (float)(rand() % 10) / 10.0f;
    }

    float *M_d = NULL, *N_d = NULL, *Out_d = NULL;
    cudaMalloc((void**)&M_d, bytes);
    cudaMalloc((void**)&N_d, bytes);
    cudaMalloc((void**)&Out_d, bytes);

    cudaMemcpy(M_d, M_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, bytes, cudaMemcpyHostToDevice);

    // Block 形状 (16, 16) = 256 线程
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    // Grid 尺寸：每个 Block 算 128x128
    dim3 blocksPerGrid((w + BLOCK_TILE_X - 1) / BLOCK_TILE_X,
                       (r + BLOCK_TILE_Y - 1) / BLOCK_TILE_Y);

    matmulThreadTiledKernelK32<<<blocksPerGrid, threadsPerBlock>>>(M_d, N_d, Out_d, r, w);

    cudaMemcpy(Out_gpu, Out_d, bytes, cudaMemcpyDeviceToHost);

    matmul_cpu(M_h, N_h, Out_cpu, r, w);

    const float eps = 1e-4f;
    for (int i = 0; i < dim; i++) {
        if (fabsf(Out_gpu[i] - Out_cpu[i]) > eps) {
            fprintf(stderr, "验证失败！索引 %d 处值不匹配: GPU=%f, CPU=%f\n", i, Out_gpu[i], Out_cpu[i]);
            cudaFree(M_d); cudaFree(N_d); cudaFree(Out_d);
            free(M_h); free(N_h); free(Out_gpu); free(Out_cpu);
            return -1;
        }
    }

    printf("矩阵大小: %d x %d\n", r, w);
    printf("Block 负责区域: %d x %d\n", BLOCK_TILE_X, BLOCK_TILE_Y);
    printf("Thread 负责区域: %d x %d\n", THREAD_TILE_X, THREAD_TILE_Y);
    printf("K 轴滑动步长: %d\n", TILE_K);
    printf("验证通过：K=32 时的 Thread Tiling 计算结果与 CPU 完全一致！\n");

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(Out_d);
    free(M_h);
    free(N_h);
    free(Out_gpu);
    free(Out_cpu);

    return 0;
}