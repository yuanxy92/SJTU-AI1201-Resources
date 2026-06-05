# 循环神经网络 RNN 与 LSTM

这一章按 PDF 顺序复习：语言任务特点 -> 数据准备 -> 词元化 -> one-hot 与 embedding -> next token prediction -> RNN 设计 -> BPTT -> LSTM。重点是让学生理解：文本是可变长度序列，RNN 为什么适合处理序列，以及 LSTM 的门控机制解决了什么问题。

## 自然语言处理 NLP 的任务特点

页码：p3-p5

NLP 的目标是让计算机理解、生成和处理自然语言。常见任务包括机器翻译、文本分类、语音识别、信息检索、智能客服等。

自然语言有层次结构、语法结构、上下文相关性和长程依赖。更重要的是，文本天然是**可变长度的离散符号序列**。

\begin{definitionbox}
\textbf{自然语言处理 NLP：} NLP 是让计算机理解、生成和处理人类自然语言的技术方向。
\end{definitionbox}

### 文本数据的可变长度

文本和图像的一个关键区别是：文本序列长度通常不固定。不同句子的 token 数可能不同：

```text
我喜欢 AI
我非常喜欢人工智能这门课
这部电影虽然节奏很慢，但是后半段非常精彩
```

这和图像任务很不一样。图像通常可以统一 resize 成固定大小，例如 `32 x 32 x 3`；文本句子却天然长短不一，并且词语顺序会影响含义。

语言数据的几个特点：

| 特点 | 含义 | 例子 |
| --- | --- | --- |
| 离散符号序列 | 文本由词、子词、字符等符号组成 | “我 / 喜欢 / AI” |
| 顺序重要 | 词语顺序会影响含义 | “我打他”和“他打我” |
| 上下文相关 | 当前词含义依赖上下文 | “上海的交通大学”和“上海的交通” |
| 长程依赖 | 较远位置的信息可能影响后面理解 | 前文主语影响后文指代 |

### MLP、CNN 与 RNN 的区别

| 模型 | 输入特点 | 是否天然适合可变长度序列 | 对历史信息的处理 |
| --- | --- | --- |
| MLP | 常需要固定长度向量 | 不适合 | 没有专门的历史记忆 |
| CNN | 擅长局部窗口特征 | 不天然适合 | 主要看局部邻域 |
| RNN | 逐 token 处理序列 | 适合 | 用隐藏状态传递历史信息 |

RNN 能处理可变长度文本的原因是：它不是一次性要求输入固定长度向量，而是按时间步逐个读入 token，并把已经读过的信息压缩到隐藏状态中。

\begin{definitionbox}
\textbf{RNN 处理可变长度序列的核心：} RNN 按时间步逐个处理输入 token，并在每一步更新隐藏状态；同一个 RNN 单元可以重复使用任意多次，因此可以处理不同长度的序列。
\end{definitionbox}

```text
短句：x1 -> x2 -> x3
长句：x1 -> x2 -> x3 -> x4 -> x5 -> ...

同一个 RNN 单元可以在不同时间步重复使用。
```

## 数据准备

页码：p6-p7

数据准备是语言模型训练的基础。考试不考复杂数据工程流程，但要知道文本数据进入模型前通常需要清洗和处理。

| 步骤 | 作用 |
| --- | --- |
| 数据收集 | 获得足够数量的文本 |
| 质量过滤 | 去掉质量很差、无意义或噪声很大的文本 |
| 敏感内容过滤 | 减少有毒内容、隐私信息等风险 |
| 数据去重 | 减少重复样本，避免模型过度记忆重复文本 |
| 词元化 | 把文本切成 token |
| 向量化 | 把 token 变成神经网络能处理的数值向量 |

关键理解：数据不是直接进入 RNN。文本通常要先经过清洗、切分和向量化，才能变成神经网络输入。

## 分词、one-hot 与 embedding

页码：p8-p11

文本进入神经网络通常要经过两步：先把文本切成 token，再把 token 变成向量。

```text
文本 -> tokenization -> token 序列 -> one-hot 或 embedding 向量
```

### Tokenization 词元化

Tokenization 是把文本切成 token。Token 可以是词、子词、字符或特殊符号。

\begin{definitionbox}
\textbf{Tokenization：} Tokenization 是把原始文本切分成 token 序列的过程；输入是文本，输出是 token 序列。
\end{definitionbox}

词元化可以基于规则，也可以基于统计。考试重点不是背具体分词算法，而是知道“文本如何变成 token 序列”。

\begin{examplebox}
\textbf{Tokenization 例子：} 原始句子“我喜欢人工智能”可以切成词级 token：$[\text{我}, \text{喜欢}, \text{人工智能}]$；也可以切成子词级 token：$[\text{我}, \text{喜欢}, \text{人工}, \text{智能}]$。不同 tokenization 方式会产生不同 token 序列。
\end{examplebox}

特殊符号不要求逐个背，但要知道它们也是 token。比如 `PAD` 常用于把不同长度句子补齐到同一长度，`UNK` 常用于表示词表中没有出现过的词。

### one-hot 与 embedding

One-hot 用一个很长的向量表示词表中的一个 token，向量中只有一个位置是 1，其余位置是 0。

Embedding 把 token 映射成低维稠密向量。Embedding 矩阵是可训练参数，可以在训练中学习词语之间的关系。

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
\textbf{例子：} 设词表为“我、喜欢、人工智能、这门课”，token “喜欢”的 one-hot 可以写成 $[0,1,0,0]$，维度等于词表大小 $4$。如果 embedding 维度设为 $3$，它可能被映射为 $[0.12,-0.38,0.51]$。
\end{examplebox}

Embedding 的计算可以理解为矩阵乘法，也可以理解为查表。设词表大小为 $N$，embedding 维度为 $d$：

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

\begin{center}
\includegraphics[width=0.88\linewidth]{output/assets/rnn_figures/rnn_p14_next_token_training-14.png}
\end{center}

大量文本都可以转化为“预测下一个 token”的任务：训练时，每个位置预测下一个 token；推理时，模型生成一个 token 后，把它接到上下文后继续生成。图中的 $X^{in}$ 表示输入 token 序列，模型输出 $\hat{X}^{out}$，目标输出 $X^{out}$ 是输入序列整体向后移动一位后的 token 序列。

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

考试要会识别 RNN 的基本公式：

$$
S_0 = 0
$$

$$
S_t = f(Ux_t + WS_{t-1} + b), \qquad t = 1,2,\dots,n
$$

$$
o_t = g(VS_t)
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
| $b$ | 偏置项 | $R^{d_h}$ |

$f$ 常用 tanh，$g$ 可以是 softmax。

\newpage

这一段的重点只记三件事：

1. $S_t$ 同时依赖当前输入 $x_t$ 和历史状态 $S_{t-1}$。
2. 参数 $U, W, V$ 在不同时间步共享。
3. 输入多少个 token，就重复多少次同样的 RNN 单元，所以可以处理可变长度序列。

Encoder-Decoder RNN 可以处理输入输出长度不同的任务，例如机器翻译。

## BPTT 与梯度问题

页码：p20

BPTT 是 BackPropagation Through Time，意思是随时间反向传播。训练 RNN 时，误差要沿多个时间步向前传播回去。

\begin{definitionbox}
\textbf{BPTT：} BPTT 是随时间反向传播，把 RNN 按时间步展开后，沿时间方向反向计算梯度。
\end{definitionbox}

\begin{center}
\includegraphics[width=0.95\linewidth]{output/assets/rnn_figures/Picture1.png}
\end{center}

长序列中，梯度需要经过很多次链式相乘，容易出现两个问题：

| 问题 | 含义 | 影响 |
| --- | --- | --- |
| 梯度消失 | 梯度越来越小 | 较早时间步的信息难以学习 |
| 梯度爆炸 | 梯度越来越大 | 训练不稳定 |

因此，普通 RNN 对长程依赖的建模能力有限。

如果序列非常长，理论上完整 BPTT 要展开并回传很多时间步，计算和显存开销都很大。实际训练中常用截断 BPTT，只向前回看有限步数，而不是对无限长历史完整反传。

\newpage

## LSTM

页码：p21-p25

LSTM 是 Long Short-Term Memory，长短期记忆网络。它在 RNN 基础上引入记忆状态，并通过门控机制控制信息流动。

\begin{center}
\includegraphics[width=0.92\linewidth]{output/assets/rnn_figures/rnn_p24_lstm_gates-24.png}
\end{center}

\begin{definitionbox}
\textbf{LSTM：} LSTM 是一种改进的 RNN，它通过记忆状态和门控机制控制信息的保留、写入和输出，用来缓解普通 RNN 的长程依赖问题。
\end{definitionbox}

| 结构 | 作用 |
| --- | --- |
| 记忆状态 | 保存较长期的信息 |
| 遗忘门 | 控制保留多少旧记忆 |
| 输入门 | 控制写入多少新信息 |
| 输出门 | 控制输出多少记忆信息 |

门通常使用 sigmoid，因为 sigmoid 输出在 0 到 1 之间，适合表示“保留多少”“写入多少”这种比例。候选记忆常用 tanh 提取。

\begin{definitionbox}
\textbf{门控机制：} 门控机制用 0 到 1 之间的比例控制信息流动：遗忘门控制旧记忆保留多少，输入门控制新信息写入多少，输出门控制当前输出多少记忆信息。
\end{definitionbox}

LSTM 设计里最重要的点：

1. 多了一条记忆状态 $C_t$，用于保存较长期信息。
2. 用门控制比例，而不是简单地全保留或全丢弃。
3. 遗忘门决定旧记忆保留多少。
4. 输入门决定新信息写入多少。
5. 输出门决定当前输出多少记忆信息。

\newpage

普通 RNN 与 LSTM 的区别可以概括为：

\begin{tabularx}{\linewidth}{p{0.16\linewidth}X X}
\hline
对比 & 普通 RNN & LSTM \\
\hline
保存历史信息 & 主要依靠隐藏状态 $S_t$ & 额外引入记忆状态 $C_t$ \\
信息控制 & 缺少显式门控 & 用遗忘门、输入门、输出门控制信息比例 \\
长程依赖 & 长序列中更容易梯度消失或梯度爆炸 & 通过记忆状态和门控机制缓解长程依赖 \\
很长序列训练 & 常配合截断 BPTT & 仍可配合截断 BPTT，但更容易保留长期信息 \\
\hline
\end{tabularx}

截断 BPTT 解决的是“序列太长，不能无限展开反传”的训练开销问题；LSTM 解决的是“普通 RNN 难以保留长期信息”的建模问题。LSTM 不能保证完全消除所有梯度问题，但它可以缓解普通 RNN 的长程依赖和梯度消失问题。

## 关键记忆

\begin{keybox}
\begin{itemize}
\item NLP 处理的是离散符号序列，词语顺序和上下文都很重要。
\item 文本数据通常是可变长度的；MLP/CNN 不天然适合记忆历史信息，RNN 通过隐藏状态逐步处理序列。
\item 文本进入神经网络前通常要经过数据清洗、tokenization 和向量化。
\item Tokenization 是把文本切成 token；token 可以是词、子词、字符或特殊符号。
\item One-hot 高维稀疏，本身不表达词义相似性；embedding 低维稠密，可以通过训练学习语义关系。
\item Next Token Prediction 是根据前文预测下一个 token，是现代语言模型常用训练框架。
\item RNN 的当前隐藏状态依赖当前输入和上一时刻隐藏状态，公式为 $S_t = f(Ux_t + WS_{t-1} + b)$。
\item BPTT 是随时间反向传播；长序列中容易出现梯度消失或梯度爆炸。
\item LSTM 通过记忆状态和门控机制控制信息保留、写入和输出，可以缓解长程依赖问题。
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
