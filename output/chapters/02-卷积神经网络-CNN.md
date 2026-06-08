# 卷积神经网络 CNN

这一章按 PDF 顺序复习：图像特性 -> 类视皮层启发 -> 卷积计算 -> padding -> stride -> 多通道卷积 -> 池化 -> CNN 结构。页码均指本章 PDF 页码。

## 图像数据的特征：关联性

页码：p3-p5

图像有空间结构。相邻像素通常更相关，距离越远，关联通常越弱。课件中强调：真实图像主要表现为局部关联，但长程关联也不能完全忽略，因此更接近幂级数衰减，而不是单一的快速指数衰减。

\begin{definitionbox}
\textbf{图像关联性：} 图像数据具有局部关联性，并且整体上更接近幂级数衰减；也就是说，近处像素关联强，远处像素关联弱但不一定迅速消失。CNN 的局部连接设计正是为了利用这种空间关联。
\end{definitionbox}

如果两个像素点完全独立、互不影响，那么它们的相关函数或互信息量为 0。这个结论常用于判断“两个位置是否还有统计关联”。

| 模型 | 形式 | 记忆 |
| --- | --- | --- |
| 指数衰减 | $y=e^{-kx}$ | 下降较快 |
| 幂级数衰减 | $y=\frac{1}{x^k}$ | 下降较慢，长程关联更明显 |

幂级数衰减可以看作许多指数衰减成分的叠加：

$$
\int_0^\infty e^{-\alpha t}\,d\alpha = \frac{1}{t}
$$

## 图像数据的特征：不变性

页码：p6

图像分类希望模型识别“物体是什么”，而不是死记每个像素的绝对位置。因此，物体发生小幅平移、旋转、伸缩或光照变化时，类别通常不应改变。

\begin{definitionbox}
\textbf{图像不变性：} 图像通常具有平移不变性、旋转不变性和伸缩不变性；这些变化不应轻易改变图像类别。
\end{definitionbox}

\begin{definitionbox}
\textbf{不是置换不变：} 图像通常不具有置换不变性；如果把像素顺序完全打乱，空间结构会被破坏，识别结果也会受到影响。
\end{definitionbox}

## 大脑如何处理图像：类视皮层设计

页码：p7-p10

大脑视觉系统处理图像时，不是一次性理解整张图，而是先在局部区域检测简单特征，再逐步汇总成更复杂的信息。这个部分主要用于理解 CNN 的设计动机。

\begin{definitionbox}
\textbf{类视皮层设计：} CNN 借鉴了“先局部、后整体”的图像处理思路，前面层提取边缘、方向等简单局部特征，后面层逐步组合成更复杂的表示。
\end{definitionbox}

课件中的对应关系可以这样记：

| 视觉皮层概念 | 作用 | CNN 中的对应 |
| --- | --- | --- |
| 简单细胞 | 对位置、边缘和方向敏感；不同简单细胞有不同感受野 | 卷积层：用局部卷积核提取边缘、方向等局部特征 |
| 复杂细胞 | 整合多个简单细胞的输入 | 后续汇总模块，尤其是全连接层：把局部特征组合成更整体的表示 |

因此，CNN 的设计思路可以理解为：卷积层先像简单细胞一样提取局部特征，再通过池化、全连接等模块逐步汇总信息。

## 卷积运算

页码：p11-p13

卷积核是一个小窗口，例如 `3 x 3`。它在图像上滑动，每次覆盖一个局部区域，把局部区域和卷积核对应位置相乘后求和。

\begin{definitionbox}
\textbf{卷积运算：} 卷积核在输入图像或特征图上滑动，每次对局部窗口做对应位置相乘再求和，得到输出特征图中的一个值。
\end{definitionbox}

卷积核中的权重通常是随机初始化后通过训练学习得到的，并不是固定不变的手工模板。卷积层的偏置（bias）也是可学习参数；每个输出通道通常对应一个偏置标量。

\begin{center}
\includegraphics[width=0.84\linewidth]{output/assets/cnn_figures/cnn_p11_convolution-11.png}
\end{center}

\begin{examplebox}
\textbf{例子 1：单通道卷积得到多个输出位置。} 输入为 $4\times 4$，卷积核为 $3\times 3$，不加 padding，stride 为 $1$。

输入 $X$：

$$
\begin{bmatrix}
1 & 2 & 3 & 4\\
5 & 6 & 7 & 8\\
9 & 10 & 11 & 12\\
13 & 14 & 15 & 16
\end{bmatrix}
$$

卷积核 $K$：

$$
\begin{bmatrix}
1 & 0 & 2\\
0 & -1 & 1\\
2 & 1 & 0
\end{bmatrix}
$$

输出大小为 $2\times 2$，因为 $3\times 3$ 窗口在 $4\times 4$ 输入上有四个可放置位置。每个位置都做“对应元素相乘，再求和”：

$$
\begin{aligned}
Y_{1,1}&=1\times1+2\times0+3\times2+5\times0+6\times(-1)\\
&\quad +7\times1+9\times2+10\times1+11\times0=36
\end{aligned}
$$

$$
\begin{aligned}
Y_{1,2}&=2\times1+3\times0+4\times2+6\times0+7\times(-1)\\
&\quad +8\times1+10\times2+11\times1+12\times0=42
\end{aligned}
$$

$$
\begin{aligned}
Y_{2,1}&=5\times1+6\times0+7\times2+9\times0+10\times(-1)\\
&\quad +11\times1+13\times2+14\times1+15\times0=60
\end{aligned}
$$

$$
\begin{aligned}
Y_{2,2}&=6\times1+7\times0+8\times2+10\times0+11\times(-1)\\
&\quad +12\times1+14\times2+15\times1+16\times0=66
\end{aligned}
$$

所以输出特征图为

$$
\begin{bmatrix}
36 & 42\\
60 & 66
\end{bmatrix}
$$
\end{examplebox}

卷积核继续向右或向下滑动，就得到下一个位置的输出。所有位置的输出组成新的特征图。

| 模型 | 连接方式 | 对图像结构的利用 |
| --- | --- | --- |
| MLP | 常把图像展平成向量，全连接 | 容易弱化二维空间结构 |
| CNN | 局部连接，卷积核滑动 | 直接利用局部关联 |

## Zero Padding

页码：p14

Padding 是在图像边缘补值，常见是补 0。作用是控制卷积后图像大小，避免图像尺寸过快变小。

\begin{definitionbox}
\textbf{Padding：} Padding 是在输入边缘补值，常见是补 0，用来控制卷积后的空间尺寸，并让边缘位置也能参与更多卷积计算。
\end{definitionbox}

Same Padding 指通过合适的 padding 让输出空间尺寸和输入空间尺寸相同。填充本身只是补值，不会引入可学习参数。

\begin{center}
\includegraphics[width=0.84\linewidth]{output/assets/cnn_figures/cnn_p14_padding-14.png}
\end{center}

\begin{examplebox}
\textbf{例子 2：Zero padding。} 原图像为 $3 \times 3$：

$$
\begin{bmatrix}
1 & 2 & 3\\
4 & 5 & 6\\
7 & 8 & 9
\end{bmatrix}
$$

padding 为 $1$ 后变成 $5 \times 5$：

$$
\begin{bmatrix}
0 & 0 & 0 & 0 & 0\\
0 & 1 & 2 & 3 & 0\\
0 & 4 & 5 & 6 & 0\\
0 & 7 & 8 & 9 & 0\\
0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$
\end{examplebox}

使用 `3 x 3` 卷积核、stride=1 时：

\begin{examplebox}
\textbf{例子 3：padding 对输出大小的影响。} 使用 `3 x 3` 卷积核、stride=1 时，不加 padding：$3 \times 3 \rightarrow 1 \times 1$；加 padding 为 $1$：$3 \times 3 \rightarrow 3 \times 3$。
\end{examplebox}

## Stride

页码：p15

Stride 是卷积核滑动步长。课件用 `stride(i, j)` 表示横向跨 `i` 步、纵向跨 `j` 步。

\begin{definitionbox}
\textbf{Stride：} Stride 是卷积核每次滑动的步长。stride 越大，卷积核采样位置越少，输出空间尺寸通常越小。
\end{definitionbox}

\begin{center}
\includegraphics[width=0.84\linewidth]{output/assets/cnn_figures/cnn_p15_stride-15.png}
\end{center}

\begin{examplebox}
\textbf{例子 4：对照 stride 图理解滑动步长。} 上图第一行是 stride$(1,1)$：卷积核每次向右或向下移动 $1$ 格，所以相邻窗口高度重叠，输出位置较多。

上图第二行是 stride$(2,2)$：卷积核每次向右或向下移动 $2$ 格，中间会跳过一些位置，所以输出位置减少，空间尺寸更小。
\end{examplebox}

## 多通道卷积

页码：p16

RGB 图像有 3 个输入通道。多通道卷积中，一个完整卷积核需要覆盖所有输入通道。

\begin{definitionbox}
\textbf{多通道卷积：} 对于有 $C_{\mathrm{in}}$ 个输入通道的数据，一个完整卷积核的深度也必须覆盖 $C_{\mathrm{in}}$ 个通道；卷积核个数通常记为 $C_{\mathrm{out}}$，也就是输出通道数。
\end{definitionbox}

\textbf{单个输出通道的计算：}

下面公式中的 $*$ 表示单通道卷积操作：先在同一个输入通道内做局部窗口乘加，再把所有输入通道的结果相加。

$$
Y_j=\sum_{c=1}^{C_{\mathrm{in}}} X_c * K_{j,c}+b_j
$$

其中，$X_c$ 是第 $c$ 个输入通道，$K_{j,c}$ 是第 $j$ 个卷积核在第 $c$ 个输入通道上的部分，$b_j$ 是第 $j$ 个输出通道的偏置（bias）。对 $j=1,\dots,C_{\mathrm{out}}$ 都计算一次，就得到 $C_{\mathrm{out}}$ 个输出通道。

\begin{center}
\includegraphics[width=0.84\linewidth]{output/assets/cnn_figures/cnn_p16_multichannel-16.png}
\end{center}

\begin{examplebox}
\textbf{例子 5：多通道卷积。} RGB 输入图像有 $3$ 个通道。R 通道局部区域与卷积核 R 部分做一次单通道卷积，G 通道局部区域与卷积核 G 部分做一次单通道卷积，B 通道局部区域与卷积核 B 部分做一次单通道卷积；三个通道结果相加，再加偏置（bias），就是这个卷积核在当前位置的输出。
\end{examplebox}

\begin{examplebox}
\textbf{例子 6：通道数。} 输入为 $32 \times 32 \times 3$，一个完整卷积核大小为 $5 \times 5 \times 3$。如果卷积核个数为 $6$，则输出空间大小另算，输出通道数为 $6$。
\end{examplebox}

## 卷积层尺寸计算总结

页码：p14-p16

卷积层的计算围绕三件事：空间大小怎么变，通道数怎么变，参数量怎么算。

\begin{definitionbox}
\textbf{卷积层输出形状：} 输入为 $H \times W \times C_{\mathrm{in}}$，卷积核个数为 $C_{\mathrm{out}}$ 时，输出形状为 $H_{\mathrm{out}} \times W_{\mathrm{out}} \times C_{\mathrm{out}}$；输出通道数由卷积核个数决定。
\end{definitionbox}

\textbf{输入、卷积核和输出形状：}

$$
\text{输入：}H\times W\times C_{\mathrm{in}}
$$

$$
\text{一个卷积核：}K\times K\times C_{\mathrm{in}}, \qquad
\text{卷积核个数：}C_{\mathrm{out}}
$$

$$
\text{输出：}H_{\mathrm{out}}\times W_{\mathrm{out}}\times C_{\mathrm{out}}
$$

$H,W$ 是输入高和宽；$C_{\mathrm{in}}$ 是输入通道数；$K$ 是卷积核大小；$C_{\mathrm{out}}$ 是卷积核个数，也就是输出通道数。

\textbf{输出空间尺寸公式：}

$$
H_{\mathrm{out}}=\left\lfloor\frac{H+2P-K}{S}\right\rfloor+1
$$

$$
W_{\mathrm{out}}=\left\lfloor\frac{W+2P-K}{S}\right\rfloor+1
$$

其中，$P$ 是 padding 大小，$S$ 是 stride 大小，$\lfloor\cdot\rfloor$ 表示向下取整。

\textbf{尺寸影响：}

| 参数 | 增大后通常会怎样 | 直观理解 |
| --- | --- | --- |
| padding | 输出空间尺寸变大 | 边缘补了一圈，能滑动的位置更多 |
| stride | 输出空间尺寸变小 | 卷积核跳得更远，采样位置更少 |
| 卷积核大小 | 输出空间尺寸变小 | 窗口更大，可放置的位置更少 |
| 卷积核个数 | 输出通道数变多 | 一个卷积核产生一个输出通道 |

\textbf{参数量计算：}

卷积层通常有偏置（bias）；每个输出通道对应 1 个偏置。

$$
\#\text{params}=K\times K\times C_{\mathrm{in}}\times C_{\mathrm{out}}
$$

\textbf{考虑偏置（bias）：}

$$
\#\text{params}=K\times K\times C_{\mathrm{in}}\times C_{\mathrm{out}}+C_{\mathrm{out}}
$$

\begin{examplebox}
\textbf{例子 7：参数量计算。} 输入通道数 $C_{\mathrm{in}}=3$，卷积核大小 $K=5$，卷积核个数 $C_{\mathrm{out}}=6$。不考虑偏置时，参数量为 $5 \times 5 \times 3 \times 6 = 450$；考虑偏置时，参数量为 $5 \times 5 \times 3 \times 6 + 6 = 456$。
\end{examplebox}

\begin{examplebox}
\textbf{例子 8：padding 保持大小。} 输入为 $32 \times 32 \times 3$，使用 $3 \times 3$ 卷积核，卷积核个数为 $16$，padding 为 $1$，stride 为 $1$。

$$
H_{\mathrm{out}} = \frac{32 + 2 \times 1 - 3}{1} + 1 = 32
$$

因此输出为 $32 \times 32 \times 16$。
\end{examplebox}

\begin{examplebox}
\textbf{例子 9：stride 让尺寸减小。} 输入为 $32 \times 32 \times 3$，使用 $3 \times 3$ 卷积核，卷积核个数为 $16$，padding 为 $1$，stride 为 $2$。

$$
H_{\mathrm{out}} = \left\lfloor\frac{32 + 2 \times 1 - 3}{2}\right\rfloor + 1 = 16
$$

因此输出为 $16 \times 16 \times 16$。
\end{examplebox}

## 池化 Pooling

页码：p18

池化层用于对局部区域做汇总，常见方式是最大池化和平均池化。池化通常用于降低空间尺寸、减少计算量，一般没有可学习参数，也通常不改变通道数。

\begin{definitionbox}
\textbf{池化 Pooling：} 池化是在局部窗口内做汇总操作。最大池化取窗口最大值，保留最强响应；平均池化取窗口平均值，保留整体平均水平。
\end{definitionbox}

\begin{examplebox}
\textbf{例子 10：最大池化和平均池化。} 对局部区域

$$
\begin{bmatrix}
1 & 3\\
2 & 4
\end{bmatrix}
$$

做最大池化时，输出为 $\max(1,3,2,4)=4$；做平均池化时，输出为 $\frac{1+3+2+4}{4}=2.5$。
\end{examplebox}

\begin{examplebox}
\textbf{例子 11：池化尺寸。} 输入为 $32 \times 32 \times 16$，最大池化窗口为 $2 \times 2$，stride 为 $2$，输出为 $16 \times 16 \times 16$。
\end{examplebox}

\textbf{全局平均池化 GAP：} GAP 对每个通道在整个空间范围内取平均，把每个通道压成 1 个数。它改变空间尺寸，但不改变通道数；接在全连接层前通常可以大幅减少参数量。

## CNN 整体结构与层的组合

页码：p20-p22

p20 给出了一个典型 CNN 结构：

```text
输入 RGB 三通道图像
-> 卷积 + ReLU + 池化
-> 卷积 + ReLU + 池化
-> 展平
-> 全连接
-> Softmax 输出类别概率
```

\begin{definitionbox}
\textbf{CNN 结构组合：} CNN 没有唯一固定的网络格式；卷积层、激活函数、池化层、全连接层等可以按任务需要组合。
\end{definitionbox}

\begin{examplebox}
\textbf{例子 12：经典 LeNet 结构。} 输入图像 $\rightarrow$ 卷积 $\rightarrow$ 池化 $\rightarrow$ 卷积 $\rightarrow$ 池化 $\rightarrow$ 展平 $\rightarrow$ 全连接 $\rightarrow$ 输出。
\end{examplebox}

只要前一层输出形状能作为后一层输入，卷积、激活、池化、展平、全连接等层就可以灵活搭配。不同 CNN 的差别主要体现在层数、顺序、卷积核个数和是否下采样。

## 关键记忆

\begin{keybox}
\begin{itemize}
\item 图像具有局部关联性和一定不变性；图像关联性更接近幂级数衰减，近处关联强，远处关联弱但不一定迅速消失。p3-p6
\item 图像不具有置换不变性；完全打乱像素顺序会破坏空间结构。p6
\item 两个像素点完全独立时，相关函数或互信息量为 0。p3-p5
\item 指数衰减下降较快，幂级数衰减下降较慢；幂级数衰减可以看作许多指数衰减成分的叠加。p3-p5
\item 类视皮层设计启发 CNN：先提取局部简单特征，再逐步整合成复杂信息。p7-p10
\item 卷积的核心是局部窗口加权求和；卷积核和偏置（bias）都是可训练参数。p11-p13
\item Padding 在边缘补值，用来控制输出空间大小；Same Padding 可保持空间尺寸不变。p14
\item Stride 是卷积核滑动步长；stride 越大，输出空间尺寸通常越小。p15
\item 多通道卷积中，一个完整卷积核覆盖所有输入通道；一个卷积核产生一个输出通道。p16
\item 卷积输出空间尺寸看 $H, W, K, P, S$；输出通道数看卷积核个数 $C_{\mathrm{out}}$。p14-p16
\item 卷积参数量看 $K, C_{\mathrm{in}}, C_{\mathrm{out}}$；有偏置（bias）时再加 $C_{\mathrm{out}}$，不要乘 $H_{\mathrm{out}}$ 和 $W_{\mathrm{out}}$。p14-p16
\item 池化汇总局部信息，降低空间尺寸，通常没有可学习参数，也通常不改变通道数；GAP 把每个通道压成 1 个数，能减少后续全连接层参数量。p18
\item CNN 没有唯一固定格式；卷积、激活、池化、展平、全连接、softmax 等模块可以灵活组合，但相邻层的输入输出尺寸必须匹配。p20-p22
\end{itemize}
\end{keybox}

## 思考题

1. 判断：池化操作一定只能由专门的池化层实现，不能用卷积实现类似效果。
   答案：错误。例如使用带 stride 的卷积可以在提取特征的同时降低空间尺寸，起到类似下采样的作用。
2. 判断：池化操作可以和卷积操作在同一层的设计中部分合并，例如用 stride 大于 1 的卷积完成下采样。
   答案：正确。
3. 填空：`1 x 1` 卷积核不改变单个位置的空间邻域大小，但可以改变______数。
   答案：通道。
4. 计算：输入为 `32 x 32 x 3`，卷积核为 `3 x 3`，padding=1，stride=1，卷积核个数为 16，输出尺寸是______。
   答案：`32 x 32 x 16`。
5. 计算：一个卷积层中 $K=5$，$C_{\mathrm{in}}=3$，$C_{\mathrm{out}}=6$，且每个卷积核有偏置（bias），参数量是______。

   答案：`5 x 5 x 3 x 6 + 6 = 456`。
