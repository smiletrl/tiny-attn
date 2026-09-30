# About

教学导向的一个精简 attention 算子：naive → tiled → online-softmax flash-style forward。

| 阶段 | 状态 | 要点 |
| --- | --- | --- |
| 01 naive GEMM | 进行中 | 一个 thread 算一个输出，HBM 复杂度 $(2n^3)$ |
| 02 tiled GEMM | 进行中 | shared memory tile，复杂度大约 $1/16$ |
| 03 naive attn | 未开始 | 物化 $(S = QK^{\top})$，看 $(N^2)$ 显存 |
| 04 tiled attn | 未开始 | 分块但仍要存分数 |
| 05 flash-style fwd | 未开始 | online softmax，不物化 $(N \times N)$ |
