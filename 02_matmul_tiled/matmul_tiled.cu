//%%writefile matmul_tiled.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

// 共享内存分块矩阵乘法核函数
__global__ void matmulTiledKernel(const float* M, const float* N, float* Out, int r, int w) {
    // 申请线程块内共享的高速片上内存
    __shared__ float s_M[TILE_DIM][TILE_DIM];
    __shared__ float s_N[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 输出矩阵 Out 的全局行列索引
    int row = blockDim.y * blockIdx.y + ty;
    int col = blockDim.x * blockIdx.x + tx;

    float sum = 0.0f;

    // 按 TILE_DIM 沿 K 维度分步滑动推进
    int numTiles = (w + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; t++) {
        // 1. 协同协作：每个线程负责将全局显存的一个元素搬运到共享内存
        // M 矩阵边界判断
        int m_col = t * TILE_DIM + tx;
        if (row < r && m_col < w) {
            s_M[ty][tx] = M[row * w + m_col];
        } else {
            s_M[ty][tx] = 0.0f; // 越界填 0，避免影响乘加计算
        }

        // N 矩阵边界判断
        int n_row = t * TILE_DIM + ty;
        if (n_row < w && col < w) {
            s_N[ty][tx] = N[n_row * w + col];
        } else {
            s_N[ty][tx] = 0.0f;
        }

        // 栅栏同步：确保该 Block 内所有线程都已将数据写入 Shared Memory
        __syncthreads();

        // 2. 从共享内存高速读取数据并进行点积累加
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            sum += s_M[ty][k] * s_N[k][tx];
        }

        // 栅栏同步：确保所有线程都计算完毕，再开始下一轮 Tile 写入，防止数据覆盖冲突（RAW 冒险）
        __syncthreads();
    }

    // 3. 将最终累加和写回全局显存
    if (row < r && col < w) {
        Out[row * w + col] = sum;
    }
}

// CPU 矩阵乘法（用于结果验证）
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

    // 初始化随机数据
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

    // 配置线程块与网格
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 blocksPerGrid((w + TILE_DIM - 1) / TILE_DIM,
                       (r + TILE_DIM - 1) / TILE_DIM);

    // 运行优化后的核函数
    matmulTiledKernel<<<blocksPerGrid, threadsPerBlock>>>(M_d, N_d, Out_d, r, w);

    cudaMemcpy(Out_gpu, Out_d, bytes, cudaMemcpyDeviceToHost);

    // 计算 CPU 结果并比对
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
    printf("分块大小 (Tile): %d x %d\n", TILE_DIM, TILE_DIM);
    printf("验证通过：Shared Memory Tiling 计算结果与 CPU 完全一致！\n");

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(Out_d);
    free(M_h);
    free(N_h);
    free(Out_gpu);
    free(Out_cpu);

    return 0;
}
