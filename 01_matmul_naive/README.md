# 01_matmul_naive: 基础原生矩阵乘法（Naive GEMM）

本目录包含 GPU 矩阵乘法的最基础实现（Naive 实现），作为整个 CUDA 矩阵乘法算子优化系列的学习起点与基准对照组（Baseline）。

## 目录结构

```
01_matmul_naive/
├── matmul.cu        # 基础原生矩阵乘法实现与 CPU 结果比对验证
└── README.md        # 本说明文档
```

## 核心设计与实现逻辑

### 1. 经典二维直角坐标映射
在 Naive 实现中，采用初学者最直观的笛卡尔直角坐标映射方式：
* **`threadIdx.x` / `blockIdx.x`**：映射到输出矩阵的横向（列，Column）
* **`threadIdx.y` / `blockIdx.y`**：映射到输出矩阵的纵向（行，Row）

每个线程负责计算输出矩阵 $Out$ 中的**唯一一个元素**：
```cpp
int row = blockDim.y * blockIdx.y + threadIdx.y;
int col = blockDim.x * blockIdx.x + threadIdx.x;
```

### 2. 内积点积计算（Dot Product）
核函数直接按照线性代数的标准教科书定义，使用 $M$ 的第 `row` 行与 $N$ 的第 `col` 列做向量内积：
```cpp
float sum = 0.0f;
for (int k = 0; k < w; k++) {
    sum += M[row * w + k] * N[k * w + col];
}
Out[row * w + col] = sum;
```

### 3. 端到端流程闭环
`matmul.cu` 完整展示了标准 CUDA 应用程序生命周期：
1. **Host 内存分配与数据初始化**
2. **Device 显存分配 (`cudaMalloc`)**
3. **数据由 Host 传输到 Device (`cudaMemcpyHostToDevice`)**
4. **2D 线程网格配置与 Kernel 启动**
5. **计算结果拷回 Host (`cudaMemcpyDeviceToHost`)**
6. **CPU 串行参考实现与浮点容差验证 (`fabsf(val_gpu - val_cpu) < eps`)**
7. **资源释放与清理**

---

## 性能瓶颈分析（为什么需要优化？）

虽然 Naive 版本逻辑清晰且易于理解，但其硬件执行效率极低，主要存在以下致命缺陷：

1. **严重的全局显存冗余访存（Memory Bound 严重）**：
   * 在计算 $Out$ 时，相邻的线程需要重复从高延迟的全局显存（DRAM/HBM）中读取相同的矩阵元素。
   * 缺乏任何片上缓存（Shared Memory / Register）复用机制，计算访存比极低，硬件算力大部分时间处于空转等待显存回包状态。
2. **非合并访存（Uncoalesced Memory Access）**：
   * 矩阵 $N$ 是按列读取的（步长为 $w$），跨行步进导致读取 $N$ 时无法合并为高效的显存事务。
3. **指令吞吐低**：
   * 每次读取仅为单精度 `float`（32-bit），未利用 128-bit 向量化访存能力。

---

## 优化演进关系

本目录是后续所有优化版本的出发点：

```
01_matmul_naive (当前目录)
  └── 单线程负责 1 个元素，纯全局显存直接读取，计算访存比低
       │
       ▼
02_matmul_tiled (进阶优化)
  ├── Step 1 (matmul_tiled): 引入共享内存分块（Shared Memory Tiling），消除全局显存冗余读取
  ├── Step 2 (matmul_thread_tiled): 引入线程级寄存器平铺（Thread Tiling），外积累加榨干寄存器带宽
  └── Step 3 (matmul_vectorized_qkv): 引入 128-bit 向量化访存、Bank Conflict 消除与 QKV 投影融合
```

---

## 编译与运行

使用 `nvcc` 直接编译并执行：

```bash
nvcc -O3 -arch=native matmul.cu -o matmul
./matmul
```

终端将打印计算结果矩阵，并输出与 CPU 基准比对的通过确认信息。
