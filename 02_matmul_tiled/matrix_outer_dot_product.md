# 矩阵乘法外积形式

两个矩阵相乘 $AB$ 除了常见的“行乘列”（点积形式），还可以表示为 **$A$ 的列向量与 $B$ 的行向量的外积（Outer Product）之和**。

如果 $A$ 是 $m \times k$ 矩阵，$B$ 是 $k \times n$ 矩阵，设：

$A$ 按列划分为 $k$ 个列向量（每个维度为 $m \times 1$）：

$$
A = \begin{bmatrix} \mathbf{a}_1 & \mathbf{a}_2 & \cdots & \mathbf{a}_k \end{bmatrix}
$$

$B$ 按行划分为 $k$ 个行向量（每个维度为 $1 \times n$）：

$$
B = \begin{bmatrix}
\mathbf{b}_1^T \\\\
\mathbf{b}_2^T \\\\
\vdots \\\\
\mathbf{b}_k^T
\end{bmatrix}
$$

则乘积可以展开为：

$$
AB = \sum_{i=1}^k \mathbf{a}_i \mathbf{b}_i^T = \mathbf{a}_1 \mathbf{b}_1^T + \mathbf{a}_2 \mathbf{b}_2^T + \cdots + \mathbf{a}_k \mathbf{b}_k^T
$$

每个项 $\mathbf{a}_i \mathbf{b}_i^T$ 都是一个秩为 1（Rank-1）的 $m \times n$ 矩阵。

---

### 具体示例

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

---

### 直观意义与应用

* **低秩分解的本质**：这种视角表明，任何矩阵相乘的结果都可以看作一系列 **秩为 1 矩阵的叠加**。
* **SVD（奇异值分解）**：SVD 的展开式 $A = \sum \sigma_i \mathbf{u}_i \mathbf{v}_i^T$ 正是外积之和思想的典型体现，用于数据降维、矩阵截断压缩与 PCA 分析。