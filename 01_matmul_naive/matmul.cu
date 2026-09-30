
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CPU 矩阵乘法
float* matmul2(void) {
    int r = 2;
    int w = r;
    int dim = r * w;

    float* M = (float*)malloc(dim * sizeof(float));
    float* N = (float*)malloc(dim * sizeof(float));
    float* Outp = (float*)malloc(dim * sizeof(float));

    if (!M || !N || !Outp) {
        fprintf(stderr, "CPU 内存分配失败\n");
        free(M); free(N); free(Outp);
        exit(EXIT_FAILURE);
    }

    // 数据初始化
    for (int i = 0; i < r; i++) {
        for (int j = 0; j< w; j++) {
            M[i * w + j] = (float) (i * w + j +1);
            N[i * w + j] = (float) (i * w + j + 5);
        }
    }


    // 矩阵乘法
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < w; j++) {
            int inx = i * w + j;
            Outp[inx] = 0.0f;
            for (int k = 0; k < w; k++) {
                Outp[inx] += M[i*w + k] * N[k * w + j];
            }
        }
    }
    // 释放内存
    free(M);
    free(N);
    // free(Outp);

    return Outp;
}

// 矩阵乘法核函数：Out = M * N
// 假设 M 为 r x w, N 为 w x c, Out 为 r x c (此处以方阵 r x w 且 r == w 为例)
__global__ void matmulKernel(const float* M, const float* N, float* Out, int r, int w) {
    // 线程映射到目标矩阵的行和列
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // 边界保护
    if (row < r && col < w) {
        float sum = 0.0f;
        // 计算 M 的第 row 行与 N 的第 col 列的点积
        for (int k = 0; k < w; k++) {
            sum += M[row * w + k] * N[k * w + col];
        }
        Out[row * w + col] = sum;
    }
}

float* matmul() {
    int r = 2;
    int w = r;
    int dim = r * w;
    size_t bytes = dim * sizeof(float);

    // 1. Host 端内存分配
    float* M_h = (float*)malloc(bytes);
    float* N_h = (float*)malloc(bytes);
    float* Out_h = (float*)malloc(bytes);

    if (M_h == NULL || N_h == NULL || Out_h == NULL) {
        fprintf(stderr, "Host 内存分配失败\n");
        free(M_h); free(N_h); free(Out_h);
        exit(EXIT_FAILURE);
    }

    // 2. 数据初始化
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < w; j++) {
            M_h[i * w + j] = (float)(i * w + j + 1);
            N_h[i * w + j] = (float)(i * w + j + 5);
        }
    }

    // 3. Device 端内存分配
    float *M_d = NULL, *N_d = NULL, *Out_d = NULL;
    cudaMalloc((void**)&M_d, bytes);
    cudaMalloc((void**)&N_d, bytes);
    cudaMalloc((void**)&Out_d, bytes);

    // 4. Host -> Device 数据拷贝
    cudaMemcpy(M_d, M_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, bytes, cudaMemcpyHostToDevice);

    // 5. 配置线程块和网格
    // 针对 2D 矩阵计算，使用 2D 线程排布
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((w + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (r + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 6. 启动核函数
    matmulKernel<<<blocksPerGrid, threadsPerBlock>>>(M_d, N_d, Out_d, r, w);

    // 7. Device -> Host 数据拷贝回传（隐式包含 CPU/GPU 同步）
    cudaMemcpy(Out_h, Out_d, bytes, cudaMemcpyDeviceToHost);

    // 8. 打印结果验证
    printf("Result Out:\n");
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < w; j++) {
            printf("%6.1f ", Out_h[i * w + j]);
        }
        printf("\n");
    }

    // 9. 释放 Device 内存
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(Out_d);

    // 10. 释放 Host 内存
    free(M_h);
    free(N_h);
    // free(Out_h);
    return Out_h;
}

#include <math.h> // 需要引入 math.h 用于 fabsf

int main() {
    int r = 2;
    int w = r;
    float* data1 = matmul();   // GPU 计算结果
    float* data2 = matmul2();  // CPU 计算结果

    // 对比两组数据是否相同
    const float eps = 1e-5f; // 浮点容差阈值
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < w; j++) {
            int idx = i * w + j;
            float val_gpu = data1[idx];
            float val_cpu = data2[idx];

            if (fabsf(val_gpu - val_cpu) > eps) {
                fprintf(stderr, "数值不相等, i = %d, j = %d, GPU = %f, CPU = %f\n", 
                        i, j, val_gpu, val_cpu);
                free(data1);
                free(data2);
                exit(EXIT_FAILURE);
            }
        }
    }

    printf("验证通过：GPU 与 CPU 计算结果一致！\n");

    free(data1);
    free(data2);

    return 0;
}
