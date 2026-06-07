# 卷积神经网络 CNN

这一章按 PDF 顺序复习：图像特性 -> 类视皮层启发 -> 卷积计算 -> padding -> stride -> 多通道卷积 -> 池化 -> CNN 结构。页码均指本章 PDF 页码。

## 图像数据的特征：关联性

页码：p3-p5

图像有空间结构。相邻像素通常更相关，距离越远，关联通常越弱。

\begin{definitionbox}
\textbf{图像局部关联性：} 图像中距离较近的像素或局部区域通常更相关，距离越远，相关性通常越弱。CNN 的局部连接设计正是为了利用这种特点。
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

图像识别不应只记住像素位置。物体轻微平移、旋转、伸缩、局部扰动或光照稍变时，类别通常不应改变。

\begin{definitionbox}
\textbf{不变性：} 输入发生小幅平移、扰动或光照变化时，模型仍应尽量识别出相同语义类别。
\end{definitionbox}

需要注意：图像通常不具有置换不变性。如果把像素顺序完全打乱，图像的空间结构会被破坏，识别结果也会受到影响。

## 大脑如何处理图像：类视皮层设计

页码：p7-p10

图像处理可以理解为：先提取局部简单特征，再逐步整合成复杂信息。

| 类比对象 | 作用 | CNN 中的对应模块 |
| --- | --- | --- |
| 简单细胞 | 对位置、边缘、方向敏感 | 卷积层 |
| 复杂细胞 | 整合多个局部输入 | 池化、全连接等后续模块 |

| CNN 模块 | 作用 |
| --- | --- |
| 卷积层 | 提取局部特征 |
| 池化层 | 汇总局部信息 |
| 全连接层 | 综合特征 |
| 输出层 | 输出类别或概率 |

## 卷积运算

页码：p11-p13

卷积核是一个小窗口，例如 `3 x 3`。它在图像上滑动，每次覆盖一个局部区域，把局部区域和卷积核对应位置相乘后求和。

\begin{definitionbox}
\textbf{卷积运算：} 卷积核在输入图像或特征图上滑动，每次对局部窗口做对应位置相乘再求和，得到输出特征图中的一个值。
\end{definitionbox}

卷积核中的权重通常是随机初始化后通过训练学习得到的，并不是固定不变的手工模板。卷积层的 bias 也是可学习参数；每个输出通道通常对应一个 bias 标量。

\begin{center}
\includegraphics[width=0.84\linewidth]{output/assets/cnn_figures/cnn_p11_convolution-11.png}
\end{center}

\begin{examplebox}
\textbf{单通道卷积例子：}

图像局部区域 $X$：

$$
\begin{bmatrix}
1 & 2 & 0\\
3 & 1 & 2\\
0 & 1 & 1
\end{bmatrix}
$$

卷积核 $K$：

$$
\begin{bmatrix}
1 & 0 & -1\\
1 & 0 & -1\\
1 & 0 & -1
\end{bmatrix}
$$

一次卷积输出为

$$
1 \times 1 + 2 \times 0 + 0 \times (-1)
+ 3 \times 1 + 1 \times 0 + 2 \times (-1)
+ 0 \times 1 + 1 \times 0 + 1 \times (-1)
= 1
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

\newpage

\begin{center}
\includegraphics[width=0.84\linewidth]{output/assets/cnn_figures/cnn_p14_padding-14.png}
\end{center}

\begin{examplebox}
\textbf{Zero padding 例子：} 原图像为 $3 \times 3$：

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
不加 padding：$3 \times 3 \rightarrow 1 \times 1$。加 padding 为 $1$：$3 \times 3 \rightarrow 3 \times 3$。
\end{examplebox}

## Stride

页码：p15

Stride 是卷积核滑动步长。课件用 `stride(i, j)` 表示横向跨 `i` 步、纵向跨 `j` 步。

\begin{definitionbox}
\textbf{Stride：} Stride 是卷积核每次滑动的步长。stride 越大，卷积核采样位置越少，输出空间尺寸通常越小。
\end{definitionbox}

\newpage

\begin{center}
\includegraphics[width=0.84\linewidth]{output/assets/cnn_figures/cnn_p15_stride-15.png}
\end{center}

\begin{examplebox}
\textbf{Stride 例子：} 输入长度方向位置为 $1,2,3,4,5$，卷积核大小为 $3$。

stride 为 $1$ 时，窗口为 $[1,2,3]$、$[2,3,4]$、$[3,4,5]$，输出长度为 $3$。

stride 为 $2$ 时，窗口为 $[1,2,3]$、$[3,4,5]$，输出长度为 $2$。
\end{examplebox}

\newpage

## 多通道卷积

页码：p16

RGB 图像有 3 个输入通道。多通道卷积中，一个完整卷积核需要覆盖所有输入通道。

\begin{definitionbox}
\textbf{多通道卷积：} 对于有 $C_{in}$ 个输入通道的数据，一个完整卷积核的深度也必须覆盖 $C_{in}$ 个通道；一个卷积核产生一个输出通道。
\end{definitionbox}

\begin{center}
\includegraphics[width=0.84\linewidth]{output/assets/cnn_figures/cnn_p16_multichannel-16.png}
\end{center}

\begin{examplebox}
\textbf{多通道卷积例子：} RGB 输入图像有 $3$ 个通道。R 通道局部区域与卷积核 R 部分相乘求和，G 通道局部区域与卷积核 G 部分相乘求和，B 通道局部区域与卷积核 B 部分相乘求和；三个通道结果相加，再加 bias，就是这个卷积核在当前位置的输出。
\end{examplebox}

\begin{examplebox}
\textbf{通道数例子：} 输入为 $32 \times 32 \times 3$，一个完整卷积核大小为 $5 \times 5 \times 3$。如果卷积核个数为 $6$，则输出空间大小另算，输出通道数为 $6$。
\end{examplebox}

## 卷积层尺寸计算总结

页码：p14-p16

卷积层的计算重点记三件事：空间大小怎么变，通道数怎么变，参数量怎么算。

\begin{definitionbox}
\textbf{卷积层输出形状：} 输入为 $H \times W \times C_{in}$，卷积核个数为 $C_{out}$ 时，输出形状为 $H_{out} \times W_{out} \times C_{out}$；输出通道数由卷积核个数决定。
\end{definitionbox}

\textbf{输入、卷积核和输出的形状：}

```text
输入：H x W x C_in

一个卷积核：K x K x C_in
卷积核个数：C_out

输出：H_out x W_out x C_out
```

\textbf{符号含义：} `H, W` 是输入高和宽；`C_in` 是输入通道数；`K` 是卷积核大小，例如 `3 x 3` 卷积核中 `K = 3`；`C_out` 是卷积核个数，也就是输出通道数；`H_out, W_out` 是输出高和宽。

\textbf{输出空间尺寸公式：}

```text
H_out = floor((H + 2P - K) / S) + 1
W_out = floor((W + 2P - K) / S) + 1
```

其中，`P` 是 padding 大小，`S` 是 stride 大小，`floor` 表示向下取整。

\textbf{padding、stride、卷积核大小的影响：}

| 参数 | 增大后通常会怎样 | 直观理解 |
| --- | --- | --- |
| padding | 输出空间尺寸变大 | 边缘补了一圈，能滑动的位置更多 |
| stride | 输出空间尺寸变小 | 卷积核跳得更远，采样位置更少 |
| 卷积核大小 | 输出空间尺寸变小 | 窗口更大，可放置的位置更少 |
| 卷积核个数 | 输出通道数变多 | 一个卷积核产生一个输出通道 |

\textbf{参数量计算：}

卷积层通常有 bias。每个卷积核产生一个输出通道，因此每个输出通道对应 1 个 bias。

```text
不考虑 bias：
参数量 = K x K x C_in x C_out

考虑 bias：
参数量 = K x K x C_in x C_out + C_out
```

\begin{examplebox}
\textbf{例子 1：参数量计算。} 输入通道数 $C_{in}=3$，卷积核大小 $K=5$，卷积核个数 $C_{out}=6$。不考虑 bias 时，参数量为 $5 \times 5 \times 3 \times 6 = 450$；考虑 bias 时，参数量为 $5 \times 5 \times 3 \times 6 + 6 = 456$。
\end{examplebox}

\begin{examplebox}
\textbf{例子 2：padding 保持大小。} 输入为 $32 \times 32 \times 3$，使用 $3 \times 3$ 卷积核，卷积核个数为 $16$，padding 为 $1$，stride 为 $1$。

$$
H_{out} = \frac{32 + 2 \times 1 - 3}{1} + 1 = 32
$$

因此输出为 $32 \times 32 \times 16$。
\end{examplebox}

\begin{examplebox}
\textbf{例子 3：stride 让尺寸减小。} 输入为 $32 \times 32 \times 3$，使用 $3 \times 3$ 卷积核，卷积核个数为 $16$，padding 为 $1$，stride 为 $2$。

$$
H_{out} = \left\lfloor\frac{32 + 2 \times 1 - 3}{2}\right\rfloor + 1 = 16
$$

因此输出为 $16 \times 16 \times 16$。
\end{examplebox}

## 池化 Pooling

页码：p18

池化层用于对局部区域做汇总，常见方式是最大池化和平均池化。池化通常用于降低空间尺寸、减少计算量，一般没有可学习参数，也通常不改变通道数。

\begin{definitionbox}
\textbf{池化 Pooling：} 池化是在局部窗口内做汇总操作。最大池化取窗口最大值，保留最强响应；平均池化取窗口平均值，保留整体平均水平。
\end{definitionbox}

\begin{definitionbox}
\textbf{全局平均池化 GAP：} GAP 对每个通道的所有空间位置取平均，把每个通道压成一个数；它改变空间尺寸，但不改变通道数。
\end{definitionbox}

GAP 常用于减少后续全连接层的参数量。例如输入为 $7\times 7\times 512$，GAP 后变成 $1\times 1\times 512$，每个通道只保留一个平均值。

\begin{examplebox}
\textbf{池化例子：} 对局部区域

$$
\begin{bmatrix}
1 & 3\\
2 & 4
\end{bmatrix}
$$

做最大池化时，输出为 $\max(1,3,2,4)=4$；做平均池化时，输出为 $\frac{1+3+2+4}{4}=2.5$。
\end{examplebox}

\begin{examplebox}
\textbf{尺寸例子：} 输入为 $32 \times 32 \times 16$，最大池化窗口为 $2 \times 2$，stride 为 $2$，输出为 $16 \times 16 \times 16$。
\end{examplebox}

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

\begin{examplebox}
\textbf{经典 LeNet 结构：} 输入图像 $\rightarrow$ 卷积 $\rightarrow$ 池化 $\rightarrow$ 卷积 $\rightarrow$ 池化 $\rightarrow$ 展平 $\rightarrow$ 全连接 $\rightarrow$ 输出。
\end{examplebox}

需要注意：CNN 没有唯一固定的网络格式。卷积层、激活函数、池化层、全连接层等都可以按任务需要组合。

常见组合方式：

```text
卷积 -> 激活 -> 卷积 -> 激活 -> 池化
卷积 -> 激活 -> 池化 -> 卷积 -> 激活 -> 池化
卷积 -> 激活 -> 展平 -> 全连接 -> 输出
```

只要前一层的输出形状能作为后一层的输入，这些层就可以灵活搭配。不同 CNN 结构的差别，往往就体现在层的数量、顺序、卷积核个数、是否使用池化等设计上。

## 关键记忆

\begin{keybox}
\begin{itemize}
\item 图像具有局部关联性和一定不变性；CNN 的设计正是为了利用这些图像特征。p3-p6
\item 图像不具有置换不变性；完全打乱像素顺序会破坏空间结构。p6
\item 两个像素点完全独立时，相关函数或互信息量为 0。p3-p5
\item 指数衰减下降较快，幂级数衰减下降较慢；幂级数衰减可以看作许多指数衰减成分的叠加。p3-p5
\item 类视皮层设计启发 CNN：先提取局部简单特征，再逐步整合成复杂信息。p7-p10
\item 卷积的核心是局部窗口加权求和；卷积核和 bias 都是可训练参数。p11-p13
\item Padding 在边缘补值，用来控制输出空间大小；Same Padding 可保持空间尺寸不变。p14
\item Stride 是卷积核滑动步长；stride 越大，输出空间尺寸通常越小。p15
\item 多通道卷积中，一个完整卷积核覆盖所有输入通道；一个卷积核产生一个输出通道。p16
\item 卷积输出空间尺寸看 $H, W, K, P, S$；输出通道数看卷积核个数 $C_{out}$。p14-p16
\item 卷积参数量看 $K, C_{in}, C_{out}$；有 bias 时再加 $C_{out}$，不要乘 $H_{out}$ 和 $W_{out}$。p14-p16
\item 池化汇总局部信息，降低空间尺寸，通常没有可学习参数，也通常不改变通道数；GAP 会把每个通道压成一个数。p18
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
5. 计算：一个卷积层中 `K=5`，`C_in=3`，`C_out=6`，且每个卷积核有 bias，参数量是______。  
   答案：`5 x 5 x 3 x 6 + 6 = 456`。
6. 判断：全局平均池化 GAP 会把每个通道压成一个数，通常可以减少后续全连接层参数量。
   答案：正确。
