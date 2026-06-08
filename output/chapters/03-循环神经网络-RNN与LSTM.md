# 循环神经网络 RNN 与 LSTM

这一章按 PDF 顺序复习：语言任务特点 -> 数据准备 -> 词元化 -> one-hot 与 embedding -> next token prediction -> RNN 设计 -> BPTT -> LSTM。本章帮助学生理解：文本是可变长度序列，RNN 为什么适合处理序列，以及 LSTM 的门控机制解决了什么问题。

## 自然语言处理 NLP 的任务特点

页码：p3-p5

NLP 的目标是让计算机理解、生成和处理自然语言。语言有层次结构、语法、上下文相关性和长程依赖；更重要的是，文本天然是**可变长度的离散符号序列**。

\begin{definitionbox}
\textbf{自然语言处理 NLP：} NLP 是让计算机理解、生成和处理人类自然语言的技术方向。
\end{definitionbox}

文本和图像的一个关键区别是：文本序列长度通常不固定，不同句子的 token 数可能不同。

\begin{examplebox}
\textbf{例子：图像与文本输入。} 图像通常可以统一 resize 成固定大小，例如 $32 \times 32 \times 3$；文本句子却长短不一，例如“我喜欢 AI”和“我非常喜欢人工智能这门课”的 token 数不同。
\end{examplebox}

语言数据的几个特点：

| 特点 | 含义 | 例子 |
| --- | --- | --- |
| 离散符号序列 | 文本由词、子词、字符等符号组成 | “我 / 喜欢 / AI” |
| 顺序重要 | 词语顺序会影响含义 | “我打他”和“他打我” |
| 上下文相关 | 当前词含义依赖上下文 | “上海的交通大学”和“上海的交通” |
| 长程依赖 | 较远位置的信息可能影响后面理解 | 前文主语影响后文指代 |

\textbf{MLP、CNN 与 RNN 的区别：}

| 模型 | 输入特点 | 是否天然适合可变长度序列 | 对历史信息的处理 |
| --- | --- | --- |
| MLP | 常需要固定长度向量 | 不适合 | 没有专门的历史记忆 |
| CNN | 擅长局部窗口特征 | 不天然适合 | 主要看局部邻域 |
| RNN | 逐 token 处理序列 | 适合 | 用隐藏状态传递历史信息 |

RNN 不要求一次性输入固定长度向量，而是按时间步读入 token，并把历史压缩到隐藏状态中。

\begin{definitionbox}
\textbf{RNN 处理可变长度序列的核心：} RNN 按时间步逐个处理输入 token，并在不同时间步共享同一组参数；同一个 RNN 单元可以重复使用任意多次，因此可以处理不同长度的序列。
\end{definitionbox}

## 数据准备

页码：p6-p7

数据准备是语言模型训练的基础。文本进入模型前通常需要清洗和处理。

\begin{definitionbox}
\textbf{数据预处理：} 数据预处理把原始文本变成模型可用的数据，通常包括收集、过滤、去重、词元化和向量化。
\end{definitionbox}

| 步骤 | 作用 |
| --- | --- |
| 数据收集 | 获得足够数量的文本 |
| 质量过滤 | 去掉质量很差、无意义或噪声很大的文本 |
| 敏感内容过滤 | 减少有毒内容、隐私信息等风险 |
| 数据去重 | 减少重复样本，避免模型过度记忆重复文本 |
| 词元化 | 把文本切成 token |
| 向量化 | 把 token 变成神经网络能处理的数值向量 |

数据量增加通常有助于提升性能；数据质量高可以节约算力、增强稳定性、减少错误输出；大量重复数据可能让模型过度记忆重复模式，反而降低上下文处理能力。

## 分词、one-hot 与 embedding

页码：p8-p11

文本不能直接进入神经网络，通常要先切成 token，再变成向量。

```text
文本 -> tokenization -> token 序列 -> one-hot 或 embedding 向量
```

\begin{definitionbox}
\textbf{Tokenization：} Tokenization 是把原始文本切分成 token 序列的过程；token 可以是词、子词、字符或特殊符号。
\end{definitionbox}

\textbf{常见词元化方法：}

下面这些方法只需要知道是常见切分方式，不需要背具体算法细节。

| 方法 | 记忆 |
| --- | --- |
| 基于规则 | 按空格、标点、词典等规则切分 |
| BPE | 从基本字符开始，逐步合并高频相邻片段 |
| WordPiece | 常见子词切分方法 |
| Unigram | 用统计模型选择合适的子词切分 |

\begin{examplebox}
\textbf{例子 1：Tokenization。} “我喜欢人工智能”可以切成词级 token：$[\text{我}, \text{喜欢}, \text{人工智能}]$；也可以切成子词级 token：$[\text{我}, \text{喜欢}, \text{人工}, \text{智能}]$。
\end{examplebox}

\textbf{常见特殊符号：}

下面这些符号只是常见约定，用来帮助读图和理解输入格式，不需要逐个背诵。

| 特殊符号 | 常见含义 |
| --- | --- |
| `CLS` | 句首或分类标记 |
| `SEP` | 句尾或分隔标记 |
| `MASK` | 掩码标记 |
| `PAD` | 填充标记，用于补齐长度 |
| `UNK` | 未登录词或未知 token |

Token 还需要变成数值向量，常见表示方式是 one-hot 和 embedding。

\begin{definitionbox}
\textbf{One-hot：} One-hot 是用词表长度的稀疏向量表示一个 token，只有该 token 对应位置为 1，其余位置为 0。
\end{definitionbox}

\begin{definitionbox}
\textbf{Embedding：} Embedding 是把 token 映射成低维稠密向量的表示方法，通常由可训练的 embedding 矩阵得到。
\end{definitionbox}

| 表示方式 | 特点 | 记忆 |
| --- | --- | --- |
| one-hot | 高维、稀疏 | 不表达词义相似性 |
| embedding | 低维、稠密、可训练 | 可以学习语义关系 |

\begin{examplebox}
\textbf{例子 2：One-hot 与 embedding。} 设词表为“我、喜欢、人工智能、这门课”，token “喜欢”的 one-hot 可以写成 $[0,1,0,0]$，维度等于词表大小 $4$。如果 embedding 维度设为 $3$，它会通过可训练矩阵映射为一个三维向量，例如 $[0.12,-0.38,0.51]$。
\end{examplebox}

Embedding 向量由 one-hot 向量乘上一个可训练矩阵得到。设词表大小为 $N$，embedding 维度为 $d$：

$$
\varepsilon_i \in R^{N}, \qquad W_{\mathrm{emb}} \in R^{d \times N}
$$

$$
x_i = W_{\mathrm{emb}}\varepsilon_i, \qquad x_i \in R^{d}
$$

其中，$\varepsilon_i$ 是第 $i$ 个 token 的 one-hot 向量，$W_{\mathrm{emb}}$ 是可训练的 embedding 矩阵，$x_i$ 是该 token 的 embedding 向量。

## Next Token Prediction

页码：p13-p16

Next Token Prediction 是根据已有前文预测下一个 token。它是现代语言模型常用的训练框架。

\begin{definitionbox}
\textbf{Next Token Prediction：} 给定前文 token 序列，模型预测下一个 token；训练时每个位置的目标通常是输入序列中下一个位置的 token。
\end{definitionbox}

Next Token Prediction 是一种自监督训练方式：训练标签直接来自原始文本中的“下一个 token”，不需要额外人工标注。

\begin{center}
\includegraphics[width=0.88\linewidth]{output/assets/rnn_figures/rnn_p14_next_token_training-14.png}
\end{center}

训练时，每个位置预测下一个 token；推理时，模型生成一个 token 后，把它接到上下文后继续生成。图中的 $X^{in}$ 是输入序列，$\hat{X}^{out}$ 是预测分布，$X^{out}$ 是输入序列整体向后移动一位后的目标序列。推理可用贪婪解码，也可按概率采样；生成到终止符（如 `SEP`）时停止。

训练时常用交叉熵损失，真实 token 的概率越高，loss 越小。图中给出的交叉熵损失可以写成：

$$
\mathcal{L}(X, \hat{X})
= -\frac{1}{n}\sum_{i=2}^{n+1}\sum_{j=1}^{d}x_{ij}\log \hat{x}_{ij}
$$

其中，$x_{ij}$ 表示真实下一个 token 的 one-hot 标签，$\hat{x}_{ij}$ 表示模型预测的概率。

## RNN 的设计

页码：p17-p18

RNN 是循环神经网络，适合处理序列数据。它的核心设计是：当前隐藏状态不仅看当前输入，也看上一时刻隐藏状态。

\begin{center}
\includegraphics[width=0.82\linewidth]{output/assets/rnn_figures/rnn_p18_rnn-18.png}
\end{center}

\begin{definitionbox}
\textbf{隐藏状态：} RNN 的隐藏状态 $S_t$ 是模型在第 $t$ 步保存的历史信息摘要，它由当前输入 $x_t$ 和上一时刻隐藏状态 $S_{t-1}$ 共同决定。
\end{definitionbox}

RNN 的隐藏状态可以理解为“目前为止读过内容的压缩记忆”。序列长度不同也没关系，因为 RNN 可以重复使用同一个计算单元，按时间步逐个处理输入。

RNN 的基本公式：

$$
S_0 = 0
$$

$$
S_t = f(Ux_t + WS_{t-1} + b), \qquad t = 1,2,\dots,n
$$

$$
o_t = g(VS_t + b_y)
$$

其中：

设输入维度为 $d_x$，隐藏状态维度为 $d_h$，输出维度为 $d_y$：

| 符号 | 含义 | 形状 |
| --- | --- | --- |
| $x_t$ | 第 $t$ 个输入向量 | $R^{d_x}$ |
| $S_t$ | 第 $t$ 步隐藏状态 | $R^{d_h}$ |
| $S_{t-1}$ | 上一时刻隐藏状态 | $R^{d_h}$ |
| $o_t$ | 第 $t$ 步输出 | $R^{d_y}$ |
| $U$ | 输入到隐藏状态的权重矩阵 | $R^{d_h \times d_x}$ |
| $W$ | 隐藏状态到隐藏状态的权重矩阵 | $R^{d_h \times d_h}$ |
| $V$ | 隐藏状态到输出的权重矩阵 | $R^{d_y \times d_h}$ |
| $b$ | 隐藏状态偏置项 | $R^{d_h}$ |
| $b_y$ | 输出偏置项 | $R^{d_y}$ |

$f$ 常用 tanh，$g$ 在分类或预测 token 时常用 softmax。

\textbf{RNN 参数量计算：}

$$
\#\text{params}
= d_h d_x + d_h d_h + d_y d_h + d_h + d_y
$$

其中，$d_h d_x$ 来自 $U$，$d_h d_h$ 来自 $W$，$d_y d_h$ 来自 $V$，$d_h$ 来自隐藏状态偏置 $b$，$d_y$ 来自输出偏置 $b_y$。

\begin{examplebox}
\textbf{RNN 参数量例子：} 若输入维度 $d_x=4$，隐藏维度 $d_h=3$，输出维度 $d_y=2$，则

$$
\#\text{params}=3\times 4+3\times 3+2\times 3+3+2=32
$$
\end{examplebox}

\begin{definitionbox}
\textbf{RNN 公式中的三件事：}
\begin{enumerate}
\item $S_t$ 同时依赖当前输入 $x_t$ 和历史状态 $S_{t-1}$。
\item 参数 $U, W, V$ 在不同时间步共享。
\item 输入多少个 token，就重复多少次同样的 RNN 单元，所以可以处理可变长度序列。
\end{enumerate}
\end{definitionbox}

\textbf{Encoder-Decoder RNN：} Encoder-Decoder RNN 先把输入序列编码成上下文表示，再逐步解码生成输出序列，可以处理输入输出长度不同的任务，例如机器翻译。这部分理解思想即可。

## BPTT 与长序列训练

页码：p20

BPTT 是 BackPropagation Through Time，意思是随时间反向传播。训练 RNN 时，误差要沿多个时间步向前传播回去。

\begin{definitionbox}
\textbf{BPTT：} BPTT 是随时间反向传播，把 RNN 按时间步展开后，沿时间方向反向计算梯度。
\end{definitionbox}

\begin{center}
\includegraphics[width=0.95\linewidth]{output/assets/rnn_figures/Picture1.png}
\end{center}

图中上半部分表示 RNN 按时间步展开；下半部分的红色箭头表示误差沿展开后的时间方向反向传播，计算梯度并用梯度下降更新参数。

长序列训练时，完整 BPTT 需要展开并回传很多时间步，计算和显存开销都很大。实际训练中常用截断 BPTT，只向前回看有限步数，而不是对无限长历史完整反传。

普通 RNN 对长程依赖的建模能力有限。下一节的 LSTM 可以理解为在 RNN 中加入记忆状态和门控机制，用来更好地保留长期信息。

## LSTM

页码：p21-p25

LSTM 是 Long Short-Term Memory，长短期记忆网络。它在 RNN 基础上引入记忆状态，并通过门控机制控制信息流动。

\begin{center}
\includegraphics[width=0.94\linewidth]{output/assets/rnn_figures/rnn_p24_lstm_gates-24.png}
\end{center}

\begin{definitionbox}
\textbf{LSTM：} LSTM 是一种改进的 RNN，它通过记忆状态和门控机制控制信息的保留、写入和输出，用来缓解普通 RNN 难以保留长期信息的问题。
\end{definitionbox}

| 结构 | 作用 |
| --- | --- |
| 记忆状态 | 保存较长期的信息 |
| 遗忘门 | 控制保留多少旧记忆 |
| 输入门 | 控制写入多少新信息 |
| 输出门 | 控制输出多少记忆信息 |

\begin{definitionbox}
\textbf{LSTM 三条设计原则：}
\begin{enumerate}
\item 信息使用前先做可学习的线性变换。
\item 控制比例的“门”用 sigmoid，输出在 0 到 1 之间。
\item 提取候选信息常用 tanh。
\end{enumerate}
\end{definitionbox}

\textbf{门控机制：} 门控机制用 0 到 1 之间的比例控制信息流动：遗忘门控制保留多少旧记忆，输入门控制写入多少新信息，输出门控制输出多少记忆信息。这里理解概念即可，不需要记复杂公式。

普通 RNN 与 LSTM 的区别可以概括为：

\begin{tabularx}{\linewidth}{p{0.16\linewidth}X X}
\hline
对比 & 普通 RNN & LSTM \\
\hline
保存历史信息 & 主要依靠隐藏状态 $S_t$ & 额外引入记忆状态 $C_t$ \\
信息控制 & 缺少显式门控 & 用遗忘门、输入门、输出门控制信息比例 \\
长程依赖 & 长序列中长期信息更难保留 & 通过记忆状态和门控机制缓解长程依赖 \\
很长序列训练 & 常配合截断 BPTT & 仍可配合截断 BPTT，但更容易保留长期信息 \\
\hline
\end{tabularx}

截断 BPTT 解决的是“序列太长，不能无限展开反传”的训练开销问题；LSTM 解决的是“普通 RNN 难以保留长期信息”的建模问题。

## 关键记忆

\begin{keybox}
\begin{itemize}
\item NLP 处理的是离散符号序列，词语顺序和上下文都很重要。
\item 文本数据通常是可变长度的；MLP/CNN 不天然适合记忆历史信息，RNN 通过隐藏状态逐步处理序列。
\item 文本进入神经网络前通常要经过数据清洗、tokenization 和向量化。
\item Tokenization 是把文本切成 token；常见词元化方法和特殊符号只需理解用途，不需要作为背诵清单。
\item One-hot 高维稀疏，本身不表达词义相似性；embedding 由 one-hot 乘可训练矩阵得到，低维稠密，可以通过训练学习语义关系。
\item Next Token Prediction 是根据前文预测下一个 token，是自监督训练框架，标签来自文本中的下一个 token。
\item RNN 的当前隐藏状态依赖当前输入和上一时刻隐藏状态，公式为 $S_t = f(Ux_t + WS_{t-1} + b)$；参数量计算要包含 $U,W,V,b,b_y$。
\item BPTT 是随时间反向传播；长序列训练时，完整 BPTT 计算和显存开销大，常用截断 BPTT。
\item LSTM 通过记忆状态和门控机制控制信息保留、写入和输出，用来缓解普通 RNN 难以保留长期信息的问题。
\end{itemize}
\end{keybox}

## 思考题

1. 判断：文本数据通常是可变长度序列，因此普通 MLP 不天然适合直接处理任意长度文本。
   答案：正确。
2. 填空：把文本切成 token 的过程叫做______。
   答案：tokenization 或词元化。
3. 填空：RNN 的基本公式中，当前隐藏状态可以写成 $S_t = f(Ux_t + W\,\underline{\qquad} + b)$。
   答案：$S_{t-1}$。
4. 判断：RNN 可以处理可变长度序列，一个重要原因是同一个 RNN 单元可以在不同时间步重复使用。
   答案：正确。
5. 选择：LSTM 中控制保留多少旧记忆的是：A. 输入门 B. 遗忘门 C. 输出门 D. Softmax。
   答案：B。
6. 计算：若 RNN 的输入维度 $d_x=4$，隐藏维度 $d_h=3$，输出维度 $d_y=2$，参数量是多少？
   答案：$3\times4+3\times3+2\times3+3+2=32$。
