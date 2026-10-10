# 问题

我们在上一阶段的工作，使用矩阵乘法的外积形式，以及寄存器内部缓存的方式，每个 thread 负责计算 8 * 8 个输出元素。提升了计算访存比。

在单 thread 读取写入显存时，每次读取/写入一个 float，而实际上 gpu 支持单次读取/写入连续内存空间 4 个 float。这里存在极大的优化空间。

使用 float4，显存访存指令数可以直接减少到原来的 1/4，能够以更少的内存事务（Memory Transactions）高效打满显存总线带宽，极大地降低指令发射单元的压力。

# 数学原理

在这个文档里，我们直接处理大模型中的 `全连接线性层 (Y = X @ W.T + b)`的算子运算。其中 `X` 是输入，`W` 是权重, `b` 是偏置。

我们在这个文档里讨论的场景是将 token 的 word embedding 的维度拓展3倍。一个 token 的词嵌入向量转为 3 个相同长度的子向量，且子向量长度等于原向量长度。这三个子向量就是自注意力计算中大名鼎鼎的 $Q, K, V$ 。

**输出通道**：

一个 token 的词嵌入的维度为 C，转换后的输出维度就是 $OC = 3C$ 。我们一般会说，一个 token 的通道从输入的 C 转为了输出的 OC（3C）。一个 token 的**输出通道**（也就是输出矩阵的列）变成了 OC。这个输出通道的概念，在下文会被频繁引用到。

映射到深度学习中，这一线性层的输入维度是 C，有 OC 个神经元，也就是有 OC 个输出维度。输入层的 batch 为 B*T。

## 数学公式： $Y = X \times W^T$

在 PyTorch 或 Transformer 的线性层中（包括我们这里讨论的代码），输入和权重的原生存储形状是：

* 输入 $X$（即 `inp`）：形状是 $[B\_T, C]$

* 权重 $W$（即 `weight`）：形状是 $[OC, C]$

* 输出 $Y$（即 `out`）：形状是 $[B\_T, OC]$

输出矩阵 $Y$ 的任何一个元素是如何算出来的：

$$Y[i][j] = \sum_{k=0}^{C-1} X[i][k] \times W[j][k]$$

注意这两个索引：

* $i$ 是输出矩阵的**行号**，它来自 $X$ 的**第 $i$ 行**；

* $j$ 是输出矩阵的**列号**，它来自 $W$ 的**第 $j$ 行**！

也就是说：输出矩阵的“列”，在物理上对应的是权重矩阵 $W$ 的“行”！

这个公式 $Y[i][j]$ 跟标准的矩阵乘法公式有差异。我们来做进一步的推导：

### 1. 标准教科书矩阵乘法：

$$Y = X \times W_{math}$$

在标准线性代数中，如果要做乘法，右矩阵的形状必须是 $[C, OC]$：

* 输入 $X$：形状为 $[B\_T, C]$

* 理想数学权重 $W_{math}$：形状为 $[C, OC]$
* 此时按标准定义就是：

$$Y[i][j] = \sum_{k=0}^{C-1} X[i][k] \times W_{math}[k][j]$$

---

### 2. 引入转置关系：

$$W_{math} = W_{stored}^T$$

但大模型（PyTorch/Transformer， 包括我们这里采用的代码）实际存储在显存里的权重矩阵 $W_{stored}$（即代码里的 `weight`）并没有事先转置成 $[C, OC]$，而是直接按 $[OC, C]$ 存放的。

根据矩阵转置的定义：

$$W_{math}[k][j] = (W_{stored}^T)[k][j] = W_{stored}[j][k]$$

把这一项直接代回原来的标准矩阵乘法公式，就变成了：

$$Y[i][j] = \sum_{k=0}^{C-1} X[i][k] \times W_{stored}[j][k]$$

这意味着在当前的数据存储结构下，矩阵乘法中的两个矩阵的共享内存的步长 tile K 方向，都是沿着列方向，从左往右滑动。

## 坐标轴切换

在之前的矩阵乘法工作中，我们将 block 的坐标 `blockiIdx.y` 表示为行，`blockIdx.x` 表示为列。在这个文档的代码实现里，我们切换了。

- `blockIdx.x` 为行
- `blockiIdx.y` 为列

<img src="./assets/coordinate-row-x.png" alt="coordinate-row-x" title="coordinate-row-x" width="300" />

在 `blockIdx` 的划分（任务网格划分）时，我们启动的二维 Block 网格是：

```c++
dim3 gridDim(CEIL_DIV(B_T, 128), CEIL_DIV(OC, 128));
```

这个网格是直接针对**最终输出矩阵 $Y$** （形状是 $[B\_T, OC]$）划分任务的：

* **`blockIdx.x` 负责输出矩阵 $Y$ 的行切片**（每次负责 128 行）。

* **`blockIdx.y` 负责输出矩阵 $Y$ 的列切片**（每次负责 128 列）。

这样子在 grid 内的 block 的坐标轴里，`blockIdx.x` 为行，`blockiIdx.y` 为列。

# 初始指针定位

在 kernel 函数开始进行数据搬运计算之前，我们先统一将函数参数的指针定位到当前 thread 所在的 block 开始的位置。

```c++
// 指针定位到当前 Block 的起点
inp    += 128 * blockIdx.x * C;
weight += 128 * blockIdx.y * C;
out    += 128 * blockIdx.x * OC + 128 * blockIdx.y;
```

当前 Block 要算的是：

* 输出 $Y$ 的行：从 `128 * blockIdx.x` 到 `128 * blockIdx.x + 127`

* 输出 $Y$ 的列：从 `128 * blockIdx.y` 到 `128 * blockIdx.y + 127`

* 输出矩阵每行总宽为 OC，跨行（向下）需要乘 OC，而跨列（向右）连续存储，所以直接加 128 * blockIdx.y 即可。

那么为了算这块区域，当前 Block 需要从全局显存里抓的数据如下

#### ① 取输入 `inp`：

* 它需要 $X$ 的第 `128 * blockIdx.x` 到 `+127` 行。

* 因为 $X$ 在显存里每行长度是 $C$，所以基地址跳过这么多行就是：
```cpp
inp += 128 * blockIdx.x * C;
```

#### ② 取权重 `weight`：

* 它需要对应输出列的权重，而前面说了，**输出列 $j$ 对应的正是 $W$ 的第 $j$ 行**！

* 所以它需要 $W$ 矩阵的第 `128 * blockIdx.y` 到 `+127` 行！

* 而 $W$ 在显存里的形状是 $[OC, C]$，它的每行长度**同样是 $C$**！

* 所以要想跳到第 `128 * blockIdx.y` 行，在物理内存里必须跨过：

```cpp
weight += 128 * blockIdx.y * C;
```

补充说明：我们用一张图来对比不同变量对于`行`的定义。

| 数据 | 形状 | 它的“行”代表什么 | 对应输出 `Y` 的哪个维度 | 用哪个 `blockIdx` 切 |
|---|---|---|---|---|
| `inp` / `X` | `[B_T, C]` | 第 `i` 个 token | 输出行 `i` | `blockIdx.x` |
| `weight` / `W` | `[OC, C]` | 第 `j` 个输出通道 | 输出列 `j` | `blockIdx.y` |
| `out` / `Y` | `[B_T, OC]` | 输出行 `i` | 行 `i` | 行用 `blockIdx.x` |
| `out` / `Y` | `[B_T, OC]` | 输出列 `j` | 列 `j` | 列用 `blockIdx.y` |

所以这里对于 Weight 来说，取行是按照 `blockIdx.y` 来计算。

# 向量化读取显存

我们先看第一步，从全局显存搬运数据到共享内存。

```c++

  for (int so = 0; so < C; so += 32) {
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
  }
```

这一段代码之所以第一眼看起来不如原版的“拉平成 0~255 一维 ID”直观，是因为它**利用了 2D 线程网格 $(16 \times 16)$ 的几何形状，巧妙地重组了“行”与“列”，使得每一个 Warp 内部的 32 个线程完美契合 128-bit（4 个 float）的内存对齐要求**。

我们一步一步把它的几何映射拆解开来：

---

### 1. 目标与总工作量分解

* **要搬运的 Shared Memory 尺寸**： $128 \text{ 行} \times 32 \text{ 列}$ 。


* **列维度（ $K=32$ 维）的向量化特点**：
因为使用了 `float4`，每次读写都是连续的 4 个 float。
32 列以 4 个为一组，可以切分为：

$$\frac{32}{4} = 8 \text{ 个向量块（即列偏移为 } 0, 4, 8, 12, 16, 20, 24, 28\text{）}$$


* **总向量个数**： $128 \text{ 行} \times 8 \text{ 个向量/行} = 1024 \text{ 个 } float4$ 。


* **线程总数**：当前 Block 拥有 $16 \times 16 = 256$ 个线程。


* **每个线程搬运的任务量**：

$$\frac{1024}{256} = 4 \text{ 个 } float4$$



（这正是 `y` 循环步长为 32，从起始行到 128 刚好循环 4 次的原因： $128 / 32 = 4$ ）。



---

### 2. 列方向映射：`xmod8` 与 `xo`

```cpp
int xmod8 = threadIdx.x % 8; // 取值范围 0 ~ 7
int xo    = 4 * xmod8;       // 取值范围 0, 4, 8, 12, 16, 20, 24, 28

```

注意 `threadIdx.x` 的范围是 $0 \sim 15$：

* `threadIdx.x` 的前 8 个线程（ $0 \sim 7$ ），`xmod8` 刚好覆盖 $0 \sim 7$。
* 乘以 4 后，`xo` 覆盖了 **整行 32 列中的全部 8 个 float4 块（即 $0, 4, 8, \dots, 28$ 列）**。
* **结论**：**8 个连续的 `threadIdx.x` 线程，就能刚好横向填满一整行的 32 个元素！**

---

### 3. 行方向映射：`xby8` 与 `y`

如果 8 个线程就能搬完一整行的 32 个元素，那么一行根本不需要 16 个线程。剩下的一半线程怎么办？
答案是：**拿去搬下一行！**

看这两行代码：

```cpp
int xby8  = threadIdx.x / 8;                 // threadIdx.x 在 0~7 时为 0；在 8~15 时为 1
int y_start = 2 * threadIdx.y + xby8;        // 起始行号

```

让我们看一下 `threadIdx.x` 这一行 16 个线程分工的结果：

* 线程 `threadIdx.x = 0 ~ 7`：`xby8 = 0`，它们搬运的是第 `2 * threadIdx.y + 0` 行的全部 8 个 `float4`。
* 线程 `threadIdx.x = 8 ~ 15`：`xby8 = 1`，它们搬运的是第 `2 * threadIdx.y + 1` 行的全部 8 个 `float4`。

也就是说：**一个长度为 16 的 X 轴线程切片，直接同时搞定两整行（第 $2 \times \text{threadIdx.y}$ 行和第 $2 \times \text{threadIdx.y} + 1$ 行）！**

block 内有 16 行 threads （每行有 16 个 threads），而每行 threads 可以读两行显存数据，意味着在一轮里，这个 block 一共可以读 32 行显存数据。所以每轮行的梯度增加 32。

```c++
for (int y = 2 * threadIdx.y + xby8; y < 128; y += 32) {
    st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
    st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
}
```

---

### 4. 观察一个完整的 Warp（32 个线程）是如何工作的

GPU 是以 Warp（32 个线程）为单位执行的。在 `dim3(16, 16)` 的布局下，一个 Warp 由连续的两行线程组成（`threadIdx.y` 为偶数与奇数）：

* Warp 0 包含：
* `threadIdx.y = 0` 的 16 个线程
* `threadIdx.y = 1` 的 16 个线程


* 按照上面的公式代入：
* `threadIdx.y = 0, threadIdx.x = 0~7`   $\rightarrow$ 处理 **第 0 行** 的 8 个 `float4`
* `threadIdx.y = 0, threadIdx.x = 8~15`  $\rightarrow$ 处理 **第 1 行** 的 8 个 `float4`
* `threadIdx.y = 1, threadIdx.x = 0~7`   $\rightarrow$ 处理 **第 2 行** 的 8 个 `float4`
* `threadIdx.y = 1, threadIdx.x = 8~15`  $\rightarrow$ 处理 **第 3 行** 的 8 个 `float4`



**结论：一个 Warp（32 个线程）单次执行，刚好整整齐齐地搬完了连续的 4 整行（ $4 \text{ 行} \times 8 \text{ 个 } float4 = 32 \text{ 个 } float4$ ）！**
这种排布保证了线程访存是绝对严格合并（Coalesced）的。

---

### 5. 循环步长 `y += 32`

整个 Block 一共有 256 个线程，即 8 个 Warp。

* 1 个 Warp 搬 4 行；
* 8 个 Warp（整个 Block）在一次循环中，刚好搬运：

$$8 \times 4 = 32 \text{ 行}$$


* 所以在第一轮中，256 个线程覆盖了 **第 $0 \sim 31$ 行**。
* 接着执行 `y += 32`：


* 第二轮循环：搬运 **第 $32 \sim 63$ 行**
* 第三轮循环：搬运 **第 $64 \sim 95$ 行**
* 第四轮循环：搬运 **第 $96 \sim 127$ 行**


* 4 轮循环刚好打满 128 行，完全覆盖整个 `lhs_s[128][32]`。



---

### 总结对比

| 特性 | 原版 `matmul_thread_tiled.cu` | 进阶版 `matmul_vectorized_kernel` |
| :--- | :--- | :--- |
| **访存单位** | 标量 `float` (32-bit) | 向量 `float4` (128-bit) |
| **线程组织思路** | 先把二维线程拉成一维 (0 ~ 255)，再算除法/取模 | 保留二维几何特性，把 16 个线程拆成两组 8 线程处理 2 行 |
| **单行所需线程数** | 32 个线程搬 1 行（每人 1 个 float） | 8 个线程搬 1 行（每人 1 个 float4） |
| **Warp 覆盖** | 1 个 Warp 覆盖 1 行 (32 个 float) | 1 个 Warp 覆盖连续 4 行 (32 个 float4) |
| **生成的 SASS 指令** | `LDG.E` / `STS` (大量小指令) | `LDG.E.128` / `STS.128` (指令数减为原来的 1/4) |

通过这种映射，代码无需对全局线程 ID 执行昂贵的大整数除法和取模运算。

编译器指令代价：

- `threadIdx.x % 8` 在底层被编译器直接优化成位与操作 `threadIdx.x & 7`；
- `threadIdx.x / 8` 被优化成右移操作 `threadIdx.x >> 3`；
- `4 * xmod8` 被优化成左移操作 `xmod8 << 2`；
全程**没有任何除法、求模等高延迟指令**，全都是单周期的位运算。

达成 128-bit 对齐合并访存。

# 向量化读取共享内存到寄存器

我们在上面的步骤中，将全局显存的数据按照 128 x 32 的格式搬运到了共享内存中。从共享内存中，我们需要继续搬运数据到寄存器中。

```c++
    // 打散同一 Warp 内线程访问 Shared Memory 的起始相位
    int si_start = 4 * (16 * threadIdx.y + threadIdx.x);

    // 2. 沿 C 轴以 32 为步长推进
    for (int so = 0; so < C; so += 32) {
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
```

因为我们采用了 float4 向量化读取，我们的每次请求可以拿到4个连续的值。所以tile步长就是 4。同时为了避免 [bank conflict](./bank-conflict.md)，每个 thread 开始读取的索引是不同的，不是上篇文档的固定从 0 开始。而是在 32 的索引范围之内，用 4 的倍数，比如 0， 4， 8 ... 28 这种，利用线程的二维坐标构造固定的线性起始偏移 $4 * (16 * threadIdx.y + threadIdx.x)$，在 Warp 内确定性地产生错峰相位差，从而将 32 个线程的访存分散到不同的 Bank，避免并发冲突。

```c++
int si_start = 4 * (16 * threadIdx.y + threadIdx.x);
```

对于输入 inp 跟权重 weight，参照上文数学推导，这里都是基于相同的列，取不同行的值，做的外积计算。

我们的矩阵乘法公式是 $Y = X \times W^T$ ， 外积就是用 X 的列跟 $W^T$ 的行做计算。换成实际对标物，就是用 X 的 不同token，外积 $W^T$ 的不同输出通道。这两个对标物，都分别对应了 lhs（inp）/rhs（weight） 中的行。且 rhs 的行对应的坐标轴，跟上文一样，都是 `blockIdx.y`。

```c++

for (int u = 0; u < 8; ++u) {
    rhs[u] = ld_vec(&rhs_s[u + 8 * threadIdx.y][k_sub]);
}

for (int ii = 0; ii < 8; ++ii) {
    float4 lhs = ld_vec(&lhs_s[ii + 8 * threadIdx.x][k_sub]);
    #pragma unroll
    for (int ji = 0; ji < 8; ++ji) {
        vals[ii][ji] += lhs.x * rhs[ji].x;
        // ...
    }
}
```

这里，我们按行 ii 开始，先读出一行的 inp 的值 lhs，然后再用 weight（rhs） 的 8 行 （也就是输出通道里 8 个值）的数值，依次跟 lhs 相乘，得到不同列 ji 的点积。

```c++
vals[ii][ji] += lhs.x * rhs[ji].x;
vals[ii][ji] += lhs.y * rhs[ji].y;
vals[ii][ji] += lhs.z * rhs[ji].z;
vals[ii][ji] += lhs.w * rhs[ji].w;
```

引用[Outer Product](./outer_product.md)中的例子，

设矩阵 $A$ 和 $B$ 分别为：

$$
A = \begin{bmatrix} 1 & 2 \\\\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 5 & 6 \\\\ 7 & 8 \end{bmatrix}
$$

这里 $k = 2$。

#### 1. 拆分向量

* **$A$ 拆为列向量：**

$$
\mathbf{a}_1 = \begin{bmatrix} 1 \\\\ 3 \end{bmatrix}, \quad \mathbf{a}_2 = \begin{bmatrix} 2 \\\\ 4 \end{bmatrix}
$$

* **$B$ 拆为行向量：**

$$
\mathbf{b}_1^T = \begin{bmatrix} 5 & 6 \end{bmatrix}, \quad \mathbf{b}_2^T = \begin{bmatrix} 7 & 8 \end{bmatrix}
$$

#### 2. 分别计算外积

* **第一个外积 $\mathbf{a}_1 \mathbf{b}_1^T$：**

$$
\mathbf{a}_1 \mathbf{b}_1^T = \begin{bmatrix} 1 \\\\ 3 \end{bmatrix} \begin{bmatrix} 5 & 6 \end{bmatrix} = \begin{bmatrix} 1 \times 5 & 1 \times 6 \\\\ 3 \times 5 & 3 \times 6 \end{bmatrix} = \begin{bmatrix} 5 & 6 \\\\ 15 & 18 \end{bmatrix}
$$

* **第二个外积 $\mathbf{a}_2 \mathbf{b}_2^T$：**

$$
\mathbf{a}_2 \mathbf{b}_2^T = \begin{bmatrix} 2 \\\\ 4 \end{bmatrix} \begin{bmatrix} 7 & 8 \end{bmatrix} = \begin{bmatrix} 2 \times 7 & 2 \times 8 \\\\ 4 \times 7 & 4 \times 8 \end{bmatrix} = \begin{bmatrix} 14 & 16 \\\\ 28 & 32 \end{bmatrix}
$$


#### 3. 外积矩阵相加

$$
AB = \mathbf{a}_1 \mathbf{b}_1^T + \mathbf{a}_2 \mathbf{b}_2^T = \begin{bmatrix} 5 & 6 \\\\ 15 & 18 \end{bmatrix} + \begin{bmatrix} 14 & 16 \\\\ 28 & 32 \end{bmatrix} = \begin{bmatrix} 5+14 & 6+16 \\\\ 15+28 & 18+32 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\\\ 43 & 50 \end{bmatrix}
$$

#### 4. 对比：传统行乘列（内积）计算

作为对比，按传统的“行向量与列向量内积”方式，直接计算各个位置的元素：

$$
AB = \begin{bmatrix} 
\begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 5 \\\\ 7 \end{bmatrix} & \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 6 \\\\ 8 \end{bmatrix} \\\\
\begin{bmatrix} 3 & 4 \end{bmatrix} \begin{bmatrix} 5 \\\\ 7 \end{bmatrix} & \begin{bmatrix} 3 & 4 \end{bmatrix} \begin{bmatrix} 6 \\\\ 8 \end{bmatrix}
\end{bmatrix} = \begin{bmatrix} 
1 \times 5 + 2 \times 7 & 1 \times 6 + 2 \times 8 \\\\ 
3 \times 5 + 4 \times 7 & 3 \times 6 + 4 \times 8 
\end{bmatrix} = \begin{bmatrix} 19 & 22 \\\\ 43 & 50 \end{bmatrix}
$$

就相当于用 float2（我们实际用的是 float4），一次性读了

$$
\begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 5 \\ 7 \end{bmatrix}
$$

我们沿着矩阵相乘的数学形式与代码实现一步步推导核对：

### 1. 运算目标的标量表达式

在计算输出矩阵的一个目标元素 $Y[i][j]$ 时，若当前沿着内积维度前进 2 步（即用简化版 `float2` 代替代码中的 `float4`），数学定义为对应位置的分量相乘再累加：


$$Y[i][j] \mathrel{+}= X[i][k] \times W_{math}[k][j] + X[i][k+1] \times W_{math}[k+1][j]$$

### 2. 用矩阵/向量乘法表示

按照线性代数的标准乘法规则（行向量乘以列向量）：

* 输入 $X$ 取出的是**一行**中的连续 2 个元素（行向量）：

$$\begin{bmatrix} X[i][k] & X[i][k+1] \end{bmatrix} = \begin{bmatrix} 1 & 2 \end{bmatrix}$$


* 理想数学权重 $W_{math}$ 取出的是**一列**中的连续 2 个元素（列向量）：

$$\begin{bmatrix} W_{math}[k][j] \\ W_{math}[k+1][j] \end{bmatrix} = \begin{bmatrix} 5 \\ 7 \end{bmatrix}$$


* 两者相乘的结果为一个标量（即点积）：

$$\begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 5 \\ 7 \end{bmatrix} = 1 \times 5 + 2 \times 7 = 19$$

### 3. 与代码的实际执行逻辑对齐

在实际代码中，由于 `weight` 在显存中存储的形状是 $[OC, C]$（每一行存放一个通道的权重，即 $W_{stored}[j][k] = W_{math}[k][j]$）：

* 代码中通过 `ld_vec` 读取的 `lhs`（对应输入）取出的分量是：`lhs.x = 1, lhs.y = 2`。


* 代码中通过 `ld_vec` 读取的 `rhs`（对应权重第 $j$ 行通道）取出的分量是：`rhs.x = 5, rhs.y = 7`。

* 随后执行累加：

```cpp
vals[i][j] += lhs.x * rhs.x;  // 1 * 5
vals[i][j] += lhs.y * rhs.y;  // 2 * 7
```

这与行向量乘以列向量的形式完全等价。

因此，代码里的向量化内积展开，本质就是用行向量与列向量在单次迭代中做局部的点积累加。

**总结：** 在当前 $K=4$ 步长下，微观上做的是 4 维子向量的点积（Dot Product）；宏观上则是通过 8 行输入与 8 列权重的两两组合，完成 $8 \times 8$ 输出子矩阵的外积（Outer Product）展开。

# 从寄存器搬运回全局显存

将计算结果搬回全局显存

```c++
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
```

这里比较简单，计算出 thread 在该 block 内的标准坐标，赋值即可。同样，单次赋值 4 个元素。
