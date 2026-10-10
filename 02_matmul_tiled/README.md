# 02_matmul_tiled: 高性能矩阵乘法（GEMM）进阶实战

本目录专注于从零构建并剖析高性能 GPU 矩阵乘法（GEMM）算子，以大模型（LLM）核心的 **QKV 融合投影线性层（$Y = XW^T + b$）** 为最终落地目标。

目录内的源码与解析文档呈现**阶梯式递进（Progressive Complexity）**设计：从最基础的共享内存分块开始，逐步演进至线程级寄存器细粒度切块（Thread Tiling），最终达到工业级向量化访存（`float4`）与 Bank Conflict 优化的全融合实现。

---

## 目录结构一览

```text
02_matmul_tiled/
├── assets/                       # 架构几何映射示意图、坐标图等资源
├── bank-conflict.md              # 专题解析：Shared Memory Bank Conflict 原理与规避
├── outer_product.md              # 专题解析：外积（Outer Product）与内积的数学及硬件对应
│
├── matmul_tiled.cu               # Step 1: 基础共享内存分块实现
├── matmul_tiled.md               # Step 1: 源码深度解析文档
│
├── matmul_thread_tile.cu         # Step 2: 线程级寄存器平铺（Thread Tiling）与外积优化
├── matmul_thread_tiled.md        # Step 2: 源码深度解析文档
│
├── matmul_vectorized_qkv.cu      # Step 3: 128-bit 向量化访存 + QKV 融合投影实现
├── matmul_vectorized_qkv.md      # Step 3: 源码深度解析文档
└── README.md                     # 本说明文档
```

---

## 优化演进路线（复杂度循序渐进）

本目录下的三个主要算子文件展示了现代 GPU 算子性能优化的标准技术跃迁路径：

### 1. 基础分块：共享内存协同搬运 (`matmul_tiled`)
* **核心思想**：利用 GPU 快速片上存储（Shared Memory），将大矩阵切分为适合 Block 尺寸的子矩阵 Tile。
* **解决痛点**：单线程不再反复直接请求高延迟的全局显存（Global Memory/DRAM），将显存数据在 Block 线程间进行初步复用。
* **复杂度等级**：★☆☆☆☆
* **对应文件**：
  * 源码：`matmul_tiled.cu`
  * 详解：`matmul_tiled.md`

### 2. 细粒度切块：线程级平铺与寄存器复用 (`matmul_thread_tiled`)
* **核心思想**：引入 **Thread-level Tiling**。每个线程不再只负责计算 1 个输出元素，而是使用私有寄存器（Register File）承载并计算一个 $8 \times 8$ 的输出小矩阵。
* **技术突破**：
  * 将传统的“行乘列点积”重构为**外积累加（Outer Product Accumulation）**形式。
  * 极大提升计算访存比（Arithmetic Intensity），充分榨干 GPU 顶层寄存器文件的极致带宽。
* **复杂度等级**：★★★☆☆
* **对应文件**：
  * 源码：`matmul_thread_tile.cu`
  * 详解：`matmul_thread_tiled.md`

### 3. 硬件极致：128-bit 向量化与 QKV 投影融合 (`matmul_vectorized_qkv`)
* **核心思想**：直接对接 GPT 类大模型的全连接投影层，支持 $Y = XW^T + b$ 且权重无需物理转置，同时将内存带宽利用率推至极限。
* **技术突破**：
  * **128-bit 向量化访存**：全局显存与共享内存全面采用 `float4`（`LDG.128` / `STS.128`），指令吞吐提升 4 倍，打满总线吞吐。
  * **2D 几何重组与对齐**：重构 $(16 \times 16)$ 线程网格与内存地址映射，保证内存完全合并（Coalescing）且无求模/除法等高延迟指令。
  * **Bank Conflict 错峰**：利用 `si_start` 相位偏移技术打散同一 Warp 内各线程对 Shared Memory 的访问，消除串行排队。
  * **算子融合**：初始化寄存器时原地融合 Bias 加法，零额外开销。
* **复杂度等级**：★★★★★
* **对应文件**：
  * 源码：`matmul_vectorized_qkv.cu`
  * 详解：`matmul_vectorized_qkv.md`

---

## 核心专题知识库

在阅读进阶代码之前，建议先掌握以下两篇专题剖析：

1. **`outer_product.md`（外积与点积）**：
   * 详解为什么在寄存器级别做外积展开能够实现更高的缓存复用比。
   * 为什么未做物理转置的权重（$[OC, C]$）能通过两平行行向量的微观内积拼合出宏观外积。
2. **`bank-conflict.md`（Shared Memory 存储体冲突）**：
   * 图解现代 NVIDIA GPU 32 个 Bank 的映射规律与物理节拍。
   * 详解 128-bit 访存下的 4 周期节拍特性，以及为何内层计算循环必须严防死守 Bank Conflict。

---

## 快速编译与运行

在配备支持 CUDA 环境的机器上直接使用 `nvcc` 进行编译运行：

```bash
# 1. 编译并运行 Step 1
nvcc -O3 -arch=native matmul_tiled.cu -o matmul_tiled
./matmul_tiled

# 2. 编译并运行 Step 2
nvcc -O3 -arch=native matmul_thread_tile.cu -o matmul_thread_tile
./matmul_thread_tile

# 3. 编译并运行 Step 3 (QKV 融合投影)
nvcc -O3 -arch=native matmul_vectorized_qkv.cu -o matmul_vectorized_qkv
./matmul_vectorized_qkv
```