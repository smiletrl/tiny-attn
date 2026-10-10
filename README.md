# tiny-attn

面向学习与高性能工程实践的精简高效 Attention 算子演进项目：从原生实现到片上分块，最终通往 FlashAttention 风格的 Online Softmax 前向计算。

## 路线图与演进阶段

| 阶段 | 状态 | 核心模块 | 硬件与算法要点 |
| :--- | :---: | :--- | :--- |
| **01 naive GEMM** |  已完成 | `01_matmul_naive/` | 单线程计算单元素，无片上缓存复用，HBM 访问量为 $O(N^3)$ 级别 |
| **02 tiled GEMM** |  已完成 | `02_matmul_tiled/` | Block 分块 (SRAM) $\rightarrow$ Thread 平铺 (Register 外积) $\rightarrow$ 128-bit 向量化访存、Bank Conflict 消除与 QKV 投影融合 |
| **03 naive attn** | ⏳ 规划中 | `03_attn_naive/` | 标准自注意力计算：物化完整注意力矩阵 $S = QK^T$，观察 $O(N^2)$ 显存瓶颈 |
| **04 tiled attn** | ⏳ 规划中 | `04_attn_tiled/` | 分块计算 Attention，引入片上分块传输，但仍需在 DRAM 中暂存分块分数 |
| **05 flash-style fwd** | ⏳ 规划中 | `05_flash_attn/` | FlashAttention 机制：融合 Online Softmax，完全避免中间 $N \times N$ 矩阵落盘 |

---

## 仓库目录结构

```text
tiny-attn/
├── 01_matmul_naive/               # 阶段一：基础原生矩阵乘法
│   ├── matmul.cu                  # Naive GEMM 实现与 CPU 结果验证
│   └── README.md                  # 模块原理解析文档
├── 02_matmul_tiled/               # 阶段二：高性能分块矩阵乘法进阶
│   ├── matmul_tiled.cu            # Step 1: 基础共享内存分块 (Block-level Tiling)
│   ├── matmul_tiled.md            # Step 1 深度解析
│   ├── matmul_thread_tile.cu      # Step 2: 寄存器级平铺与外积累加 (Thread-level Tiling)
│   ├── matmul_thread_tiled.md     # Step 2 深度解析
│   ├── matmul_vectorized_qkv.cu   # Step 3: 128-bit 向量化访存、Bank 冲突打散与 QKV 融合
│   ├── matmul_vectorized_qkv.md   # Step 3 深度解析
│   ├── bank-conflict.md           # 专题：Shared Memory 物理存储体冲突成因与规避
│   ├── outer_product.md           # 专题：内积 vs 外积形式在 GEMM 中的硬件意义
│   └── README.md                  # 模块总览与进阶路线说明
└── README.md                      # 项目全局说明文档 (当前文件)
```

---

## 核心设计理念

1. **坚持底层认知与硬件对齐**：
   不依赖高级黑盒封装，透彻理解 GPU 存储层级（Global Memory $\rightarrow$ Shared Memory $\rightarrow$ Register File）、线程网格拓扑（Grid $\rightarrow$ Block $\rightarrow$ Warp $\rightarrow$ Thread）以及硬件执行节拍。
2. **循序渐进的优化阶梯**：
   每一个进阶算子均基于上一个算子的性能瓶颈展开（如访存合并、数据复用、指令吞吐压榨、Bank Conflict 打散），并配套详细的数学与硬件推导文档。
3. **直面大模型生产场景**：
   矩阵计算直接采用大模型原生行优先存储格式（免显存转置计算 $Y = XW^T + b$），并融合多头注意力投影中的 QKV 联合变换。

---

## 快速开始

### 环境依赖
* NVIDIA GPU (Compute Capability 7.0+)
* CUDA Toolkit 11.0+
* GCC / G++ 支持 C++14 及以上标准

### 编译与验证示例

* **运行阶段一（Naive 实现）**：
  ```bash
  cd 01_matmul_naive
  nvcc -O3 -arch=native matmul.cu -o matmul
  ./matmul
  ```

* **运行阶段二（QKV 融合向量化算子）**：
  ```bash
  cd 02_matmul_tiled
  nvcc -O3 -arch=native matmul_vectorized_qkv.cu -o matmul_vectorized_qkv
  ./matmul_vectorized_qkv
  ```