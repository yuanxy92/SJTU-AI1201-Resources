# Transformer

这一章按 PDF 顺序复习：为什么需要 Transformer -> next token prediction 回顾 -> 位置编码 -> 注意力机制与 Q/K/V -> causal mask -> Transformer block -> 输出层。本章围绕 Transformer 如何用注意力机制建模 token 之间的关系，以及为什么它比 RNN 更适合并行计算。

## 为什么需要 Transformer

页码：p3-p9

Transformer 出现前，序列任务常用 RNN/LSTM。RNN/LSTM 能处理序列，但有两个明显限制：长程依赖不容易保留，并且计算通常要按时间步顺序进行，不方便并行。

\begin{center}
\includegraphics[width=0.86\linewidth]{output/assets/transformer_figures/transformer_p03_why.png}
\end{center}

\begin{definitionbox}
\textbf{Transformer：} Transformer 是以注意力机制为核心的序列模型，能够直接建模 token 之间的关系，更容易捕捉全局上下文，也更适合并行计算。
\end{definitionbox}

Transformer 的直观思想是：当前 token 更新自己时，直接计算“应该关注哪些 token”，再按关注程度汇总信息。原始 Transformer 用于机器翻译，采用 encoder-decoder 结构；现在常见生成式大语言模型多使用 decoder-only 结构。

\begin{center}
\includegraphics[width=0.92\linewidth]{output/assets/transformer_figures/rnn_lstm_vs_transformer_position_encoding.png}
\end{center}

图中两点最重要：RNN/LSTM 按时间步串行传递信息，5 个 token 通常要做 5 次串行计算；Transformer 可把整段 token 同时送入模型，并行得到输出，但 token 向量本身没有顺序概念，因此必须加入位置编码。$x_i$ 是输入 token，$p_i$ 是位置编码，$y_i$ 是输出 token。

\begin{tabularx}{\linewidth}{p{0.16\linewidth}X X}
\hline
模型 & 处理序列的方式 & 主要特点 \\
\hline
RNN/LSTM & 按时间步顺序读入 token & 有隐藏状态，但并行不方便，长程信息传递更困难 \\
Transformer & 直接计算 token 之间的关系 & 注意力机制更适合全局关系和并行计算 \\
\hline
\end{tabularx}

\begin{examplebox}
\textbf{例子 1：注意力的直观类比。} 如果要估计某位同学缺失的化学成绩，可以先找“和他各科表现相似”的同学，再参考这些同学的化学成绩。注意力机制做的事情也类似：先计算相关性，再按相关性汇总信息。
\end{examplebox}

## Next Token Prediction 回顾

页码：p10-p13

Next token prediction 已经在“循环神经网络 RNN 与 LSTM”章的 4.4 节讲过：给定前文 token，模型预测下一个 token，并常用交叉熵损失。Transformer 这一章不重复展开，后面只保留这个联系：decoder 做生成任务时必须保证当前位置不能看到未来 token，因此需要 causal mask。

\begin{definitionbox}
\textbf{Next Token Prediction 回顾：} 生成任务中，每个位置预测下一个 token；decoder 训练和生成时不能提前看到未来 token。
\end{definitionbox}

## Embedding 与位置编码

页码：p15-p18

Embedding 已经在“循环神经网络 RNN 与 LSTM”章的 4.3 节讲过：token 先变成 one-hot，再通过可训练 embedding 矩阵映射成低维稠密向量。Embedding 矩阵通常随机初始化，并在训练过程中持续更新。本节讨论位置编码。

Transformer 会把一组 token 同时送入模型。Embedding 只表示“这个 token 是什么”，不表示“它在第几个位置”；位置编码用于补上顺序信息。

\begin{center}
\includegraphics[width=0.84\linewidth]{output/assets/transformer_figures/transformer_p17_embedding_position.png}
\end{center}

\begin{definitionbox}
\textbf{位置编码：} 位置编码为模型提供 token 的位置信息；输入表示通常由 token embedding 和位置编码相加得到。位置编码不改变词表大小。
\end{definitionbox}

\textbf{输入表示：}

$$
X^{(1)} = X^{emb}+X^{pos}
$$

其中，$X^{emb}\in R^{n\times d_m}$ 是 token embedding，$X^{pos}\in R^{n\times d_m}$ 是位置编码，因此 $X^{(1)}\in R^{n\times d_m}$。$n$ 是序列长度，$d_m$ 是每个 token 的向量维度。

\begin{examplebox}
\textbf{例子 3：为什么需要位置编码。} “我 爱 小 猫”和“猫 爱 小 我”包含相同 token，但顺序不同，含义也不同。位置编码就是告诉 Transformer：这些 token 分别出现在第几个位置。
\end{examplebox}

复杂位置编码属于阅读材料，不展开推导；记住输入表示通常是 embedding 加位置编码。

## 注意力机制与 Q、K、V

页码：p21-p31

注意力主要计算 token 之间的相关性，本身不负责记录“谁在第几个位置”，所以输入 Transformer 前要先把位置编码加进去。

\begin{center}
\includegraphics[width=0.86\linewidth]{output/assets/transformer_figures/transformer_p25_qkv.png}
\end{center}

\begin{definitionbox}
\textbf{Q、K、V：} Q 是 Query，表示当前 token 发出的查询；K 是 Key，表示每个 token 可被匹配的键；V 是 Value，表示真正被加权汇总的信息内容。
\end{definitionbox}

\textbf{总公式：缩放点积注意力}

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

\textbf{四步理解：}

1. 线性映射得到 $Q,K,V$。
   在 self-attention 中，输入是 $X^{(l)}\in R^{n\times d_m}$，其中 $n$ 是 token 数量，$d_m$ 是每个 token 的隐藏向量维度。$Q,K,V$ 不是新的输入数据，而是由同一个 $X^{(l)}$ 通过三个可训练线性映射得到：

$$
\begin{aligned}
Q^{(l)}&=X^{(l)}W^{Q(l)}, & W^{Q(l)}&\in R^{d_m\times d_k}\\
K^{(l)}&=X^{(l)}W^{K(l)}, & W^{K(l)}&\in R^{d_m\times d_k}\\
V^{(l)}&=X^{(l)}W^{V(l)}, & W^{V(l)}&\in R^{d_m\times d_v}
\end{aligned}
$$

   输出为 $Q^{(l)},K^{(l)}\in R^{n\times d_k}$，$V^{(l)}\in R^{n\times d_v}$。这里 $K$ 和 $V$ 的第一维相同，表示每个 key 都对应一个 value；$Q$ 和 $K$ 的最后一维必须相同，才能做点积；$V$ 的最后一维可以不同。一般注意力中，Query 的个数可以和 Key/Value 的个数不同，也就是 $Q$ 的第一维可以不同；self-attention 中三者来自同一段序列，所以 token 数相同。

2. 用 $Q$ 和 $K$ 计算关联强度。
   输入是 $Q^{(l)},K^{(l)}\in R^{n\times d_k}$：

$$
A^{(l)} = \frac{Q^{(l)}(K^{(l)})^T}{\sqrt{d_k}}, \qquad A^{(l)}\in R^{n\times n}
$$

   $A_{ij}$ 表示第 $i$ 个 token 对第 $j$ 个 token 的关注分数；除以 $\sqrt{d_k}$ 是为了让训练更稳定。

3. 对关联强度做 softmax 得到注意力权重。
   输入是 $A^{(l)}\in R^{n\times n}$：

$$
\operatorname{Attn}^{(l)}
=\operatorname{softmax}(\operatorname{mask}(A^{(l)})), \qquad \operatorname{Attn}^{(l)}\in R^{n\times n}
$$

   mask 是可选遮挡步骤，生成任务里的 causal mask 见下一节；如果不需要遮挡，可直接对 $A^{(l)}$ 做 softmax。

4. 用注意力权重对 $V$ 加权求和，再用 $W^O$ 调整维度。
   输入是 $\operatorname{Attn}^{(l)}\in R^{n\times n}$，$V^{(l)}\in R^{n\times d_v}$：

$$
X^{qkv(l)}=\operatorname{Attn}^{(l)}V^{(l)}, \qquad X^{qkv(l)}\in R^{n\times d_v}
$$

再通过输出矩阵 $W^{O(l)}$ 映射回模型维度：

$$
X^{pr(l)}=X^{qkv(l)}W^{O(l)}, \qquad W^{O(l)}\in R^{d_v\times d_m}
$$

   最终输出 $X^{pr(l)}\in R^{n\times d_m}$。$W^O$ 的作用是把 $n\times d_v$ 变回 $n\times d_m$，这样才能和原输入 $X^{(l)}$ 做残差相加。

\begin{examplebox}
\textbf{例子 4：注意力加权求和。} 如果某个 token 对三个位置的注意力权重是 $[0.5,0.3,0.2]$，三个 value 向量的第二个分量分别是 $2,4,1$，则输出向量的第二个分量为

$$
0.5\times 2+0.3\times 4+0.2\times 1=2.4
$$
\end{examplebox}

## Causal Mask

页码：p27-p29

在生成任务中，预测当前位置时不能看到未来 token，否则模型就相当于提前看到了答案。Causal mask 用来遮住未来位置。

\begin{center}
\includegraphics[width=0.86\linewidth]{output/assets/transformer_figures/transformer_p28_mask_softmax.png}
\end{center}

\begin{definitionbox}
\textbf{Causal Mask：} Causal mask 会把未来位置的注意力分数设为很小的值，使 softmax 后这些位置的权重接近 0，从而保证预测时不能看未来 token。
\end{definitionbox}

对第 $i$ 个 token 来说，causal mask 只允许它关注第 $1$ 到第 $i$ 个 token。这样就保证 next token prediction 的因果性。

\begin{examplebox}
\textbf{例子 5：Causal mask 允许看哪些位置。} 对序列“我 爱 小 猫”，每个位置能关注的位置如下：

\begin{center}
\begin{tabular}{c|c|c}
当前位置 & 允许关注 & 不能关注 \\
\hline
我 & 我 & 爱、小、猫 \\
爱 & 我、爱 & 小、猫 \\
小 & 我、爱、小 & 猫 \\
猫 & 我、爱、小、猫 & 无 \\
\end{tabular}
\end{center}

对应的 causal mask 可以写成一个矩阵。行表示当前位置，列表示被关注的位置；$0$ 表示允许看，$-\infty$ 表示遮住：

$$
M=
\begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

计算时可以理解为先把 mask 加到注意力分数上：

$$
\operatorname{Attn}=\operatorname{softmax}(A+M)
$$

因此，当模型在“我 爱”之后预测下一个 token 时，可以利用“我”和“爱”，但不能提前看到“小”或“猫”。
\end{examplebox}

## Transformer Block、LayerNorm 与 FNN

页码：p31-p37

Transformer block 可以理解为“先让 token 之间交流，再分别处理每个 token”。

\begin{center}
\includegraphics[width=0.86\linewidth]{output/assets/transformer_figures/transformer_p31_attention_block.png}
\end{center}

\begin{definitionbox}
\textbf{Transformer Block：} 一个 Transformer block 通常包含注意力模块、残差连接、LayerNorm 和前馈全连接网络 FNN。
\end{definitionbox}

\textbf{Block 中四个核心部件：}

1. 注意力模块：负责 token 之间的信息交互。注意力输出经 $W^O$ 后变回 $n\times d_m$，才能和输入做残差相加。
2. 残差连接：把子层输入直接加到子层输出上，基本形式是

$$
y=x+F(x)
$$

3. LayerNorm：逐个 token 做归一化。对 $n\times d_m$ 的矩阵，它对每一行 token 向量内部的 $d_m$ 个数计算均值和方差，不在不同 token 之间混合。归一化后还有可训练仿射参数 $\gamma,\beta$。
4. FNN：逐 token 作用的小 MLP；不同位置共享同一套 FNN 参数。FNN 不负责 token 之间的信息交互，token 之间的 interaction 主要发生在 attention 里。

\textbf{LayerNorm 公式示意：}

$$
X^{ao(l)}=\operatorname{LayerNorm}(X^{(l)}+X^{pr(l)})
$$

\begin{examplebox}
\textbf{例子 6：LayerNorm 和 FNN 都是逐 token。} 如果输入有 4 个 token，每个 token 是 128 维向量，LayerNorm 会分别对 4 个 token 各自的 128 个数做归一化；FNN 也分别作用在 4 个 token 上。不同 token 之间的信息交换主要已经在注意力模块中完成。
\end{examplebox}

## 输出投影与解码

页码：p38-p39

输出投影层把每个 token 的隐藏向量映射到词表大小，再经过 softmax 得到下一个 token 的概率分布。

\begin{center}
\includegraphics[width=0.86\linewidth]{output/assets/transformer_figures/transformer_p38_output_projection.png}
\end{center}

\begin{definitionbox}
\textbf{输出投影层：} 输出投影层把隐藏向量映射到词表维度，softmax 后得到每个候选 token 的概率。
\end{definitionbox}

\textbf{输出公式：}

$$
X^{out}=X^{do(L)}W^{proj}+b^{proj}, \qquad W^{proj}\in R^{d_m\times d}
$$

其中，$d$ 是词表大小，所以输出的最后一维对应词表中每个 token 的分数。

语言模型训练通常用交叉熵损失衡量预测分布和真实下一个 token 之间的差距；真实 token 的预测概率越高，loss 越小。

\begin{tabularx}{\linewidth}{p{0.18\linewidth}X X}
\hline
解码方式 & 含义 & 结果特点 \\
\hline
贪婪解码 & 每一步取概率最大的 token & 确定性强；同一输入通常得到同一输出 \\
采样解码 & 按概率分布随机抽样 & 有随机性；同一输入可能得到不同输出 \\
\hline
\end{tabularx}

\begin{examplebox}
\textbf{例子 7：贪婪解码和采样解码的区别。} 假设下一个 token 的概率为：猫 $0.70$、狗 $0.20$、鼠 $0.10$。贪婪解码一定选择概率最大的“猫”；采样解码大多数时候会抽到“猫”，但也有可能抽到“狗”或“鼠”。所以采样解码更有随机性。
\end{examplebox}

## 关键记忆

\begin{keybox}
\begin{itemize}
\item Transformer 用注意力机制直接建模 token 之间的关系，比 RNN 更适合并行计算。p3-p9
\item 原始 Transformer 是 encoder-decoder 结构；现在常见生成式大语言模型多用 decoder-only。p3-p9
\item RNN/LSTM 按时间步串行处理 token；Transformer 可以并行处理 token，但需要位置编码提供顺序信息。p3-p18
\item Embedding 矩阵随机初始化并可训练；位置编码提供 token 位置信息，不改变词表大小。p15-p18
\item 注意力公式是 $\operatorname{softmax}(QK^T/\sqrt{d_k})V$；Q 是查询，K 是匹配用的键，V 是被汇总的信息。p21-p31
\item K 和 V 的数量必须相同；Q 和 K 的最后一维必须相同。一般注意力中，Q 的数量可以和 K/V 不同；self-attention 中三者来自同一段序列，所以 token 数相同，注意力权重形状通常是 $n\times n$。p21-p31
\item Causal mask 保证预测当前位置时不能看到未来 token。p27-p29
\item Block 中 attention 负责 token 交互；FNN 逐 token 作用；LayerNorm 逐 token 计算，并有可训练参数 $\gamma,\beta$。p31-p37
\item 残差连接公式是 $y=x+F(x)$，用于保留原信息并缓解深层训练困难。p31-p37
\item 输出投影层把隐藏向量映射到词表维度；贪婪解码取最大概率 token，采样解码具有随机性。p38-p39
\item Transformer 整体流程：输入 token -> Embedding + 位置编码 -> 多层 block -> 输出投影 -> Softmax。p15-p39
\item 多头注意力是了解项：多组注意力并行工作，不同头可以关注不同类型的信息。p21-p31
\end{itemize}
\end{keybox}

## 思考题

1. 判断：Transformer 使用注意力机制来建模 token 之间的关系。
   答案：正确。
2. 填空：Transformer 需要加入 \underline{\hspace{3em}}，用来表示 token 在序列中的位置。
   答案：位置编码。
3. 选择：Q、K、V 中，用来真正汇总信息内容的是：A. Q；B. K；C. V；D. mask。
   答案：C。
4. 判断：Q 和 K 的最后一维必须相同，因为要计算 $QK^T$。
   答案：正确。
5. 判断：LayerNorm 是把不同 token 混在一起计算均值和方差。
   答案：错误。LayerNorm 是逐 token 在向量内部计算。
6. 填空：缩放点积注意力的核心公式可以写成 $\operatorname{softmax}(QK^T/\sqrt{d_k})\underline{\hspace{2em}}$。
   答案：$V$。
