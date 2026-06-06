# Transformer

这一章按 PDF 顺序复习：为什么需要 Transformer -> next token prediction -> embedding 与位置编码 -> 注意力机制与 Q/K/V -> causal mask -> Transformer block -> 输出层。重点是理解 Transformer 如何用注意力机制建模 token 之间的关系，以及为什么它比 RNN 更适合并行计算。

## 为什么需要 Transformer

页码：p3-p9

Transformer 出现前，序列任务常用 RNN/LSTM。RNN/LSTM 能处理序列，但有两个明显限制：长程依赖不容易保留，并且计算通常要按时间步顺序进行，不方便并行。

\begin{center}
\includegraphics[width=0.86\linewidth]{output/assets/transformer_figures/transformer_p03_why.png}
\end{center}

\begin{definitionbox}
\textbf{Transformer：} Transformer 是以注意力机制为核心的序列模型，能够直接建模 token 之间的关系，更容易捕捉全局上下文，也更适合并行计算。
\end{definitionbox}

Transformer 的直观思想可以理解为：当前 token 更新自己时，不是只看前一个隐藏状态，而是主动计算“我应该关注哪些 token”，再按关注程度汇总信息。

| 模型 | 处理序列的方式 | 主要特点 |
| --- | --- | --- |
| RNN/LSTM | 按时间步顺序读入 token | 有隐藏状态，但并行不方便 |
| Transformer | 直接计算 token 之间的关系 | 注意力机制更适合全局关系和并行计算 |

\begin{examplebox}
\textbf{例子 1：注意力的直观类比。} 如果要估计某位同学缺失的化学成绩，可以先找“和他各科表现相似”的同学，再参考这些同学的化学成绩。注意力机制做的事情也类似：先计算相关性，再按相关性汇总信息。
\end{examplebox}

## Next Token Prediction 框架

页码：p10-p13

Transformer 语言模型常用于 next token prediction：给定前文 token 序列，预测下一个 token。训练时，输入序列和目标序列通常相差一个位置。

\begin{definitionbox}
\textbf{Next Token Prediction：} 给定前文 token，模型预测下一个 token；训练时每个位置的目标通常是下一个位置的真实 token。
\end{definitionbox}

\textbf{输入和输出：}

设序列长度为 $n$，词表大小为 $d$，one-hot 表示后的输入和目标可以写成：

$$
X^{in}\in R^{n\times d}, \qquad X^{out}\in R^{n\times d}
$$

如果输入是“我 爱 小 猫”，目标可以理解为整体向后移动一位：

\begin{examplebox}
\textbf{例子 2：输入和目标错位。}

\begin{center}
\begin{tabular}{c|c}
输入位置 & 目标 token \\
\hline
我 & 爱 \\
爱 & 小 \\
小 & 猫 \\
猫 & ！ \\
\end{tabular}
\end{center}
\end{examplebox}

\textbf{交叉熵损失：}

如果 $x_{ij}$ 是真实下一个 token 的 one-hot 标签，$\hat{x}_{ij}$ 是模型预测概率，则常用交叉熵损失：

$$
\mathcal{L}(X,\hat{X})
=-\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{d}x_{ij}\log \hat{x}_{ij}
$$

真实 token 的预测概率越高，loss 越小。

## Embedding 与位置编码

页码：p15-p18

Token 是离散符号，不能直接进入神经网络。Embedding 层把 one-hot token 映射成低维稠密向量。

\begin{center}
\includegraphics[width=0.84\linewidth]{output/assets/transformer_figures/transformer_p15_embedding.png}
\end{center}

\begin{definitionbox}
\textbf{Embedding：} Embedding 是把 token 映射成低维稠密向量的表示方法，通常由可训练的 embedding 矩阵得到。
\end{definitionbox}

\textbf{Embedding 公式：}

设词表大小为 $d$，embedding 维度为 $d_m$：

$$
X^{in}\in R^{n\times d}, \qquad W^{emb}\in R^{d\times d_m}
$$

$$
X^{emb}=X^{in}W^{emb}, \qquad X^{emb}\in R^{n\times d_m}
$$

\begin{examplebox}
\textbf{例子 3：embedding 矩阵大小。} 如果词表大小 $d=10000$，embedding 维度 $d_m=128$，则 embedding 矩阵大小为 $10000\times 128$。
\end{examplebox}

只有 embedding 还不够，因为注意力机制本身不天然知道 token 的顺序。位置编码用于告诉模型每个 token 在序列中的位置。

\begin{center}
\includegraphics[width=0.84\linewidth]{output/assets/transformer_figures/transformer_p17_embedding_position.png}
\end{center}

\begin{definitionbox}
\textbf{位置编码：} 位置编码为模型提供 token 的位置信息；输入表示通常由 token embedding 和位置编码相加得到。
\end{definitionbox}

\textbf{输入表示：}

$$
X^{(1)} = X^{emb}+X^{pos}
$$

其中，$X^{pos}$ 可以是可学习参数，也可以是人为设计的位置编码。课件后面的复杂位置编码属于阅读材料，复习时不展开推导。

## 注意力机制与 Q、K、V

页码：p21-p31

注意力机制可以分成四步：先把输入线性变换成 Q、K、V；再用 Q 和 K 算关联强度；然后经过 mask 和 softmax 得到注意力权重；最后用这些权重对 V 加权求和。

\begin{center}
\includegraphics[width=0.86\linewidth]{output/assets/transformer_figures/transformer_p25_qkv.png}
\end{center}

\begin{definitionbox}
\textbf{Q、K、V：} Q 是 query，用来发出查询；K 是 key，用来和 query 匹配；V 是 value，是真正被加权汇总的信息。
\end{definitionbox}

设第 $l$ 层输入为 $X^{(l)}\in R^{n\times d_m}$：

$$
Q^{(l)}=X^{(l)}W^{Q(l)}, \qquad W^{Q(l)}\in R^{d_m\times d_k}
$$

$$
K^{(l)}=X^{(l)}W^{K(l)}, \qquad W^{K(l)}\in R^{d_m\times d_k}
$$

$$
V^{(l)}=X^{(l)}W^{V(l)}, \qquad W^{V(l)}\in R^{d_m\times d_v}
$$

因此：

$$
Q^{(l)},K^{(l)}\in R^{n\times d_k}, \qquad V^{(l)}\in R^{n\times d_v}
$$

\textbf{注意力权重：}

$$
A^{(l)} = \frac{Q^{(l)}(K^{(l)})^T}{\sqrt{d_k}}
$$

$$
\operatorname{Attn}^{(l)}
=\operatorname{softmax}(\operatorname{mask}(A^{(l)}))
$$

其中，除以 $\sqrt{d_k}$ 主要是为了训练更稳定。

\textbf{加权求和：}

$$
X^{qkv(l)}=\operatorname{Attn}^{(l)}V^{(l)}, \qquad X^{qkv(l)}\in R^{n\times d_v}
$$

\begin{examplebox}
\textbf{例子 4：注意力加权求和。} 如果某个 token 对三个位置的注意力权重是 $[0.7,0.2,0.1]$，三个 value 分别是 $[1,3,5]$，则汇总结果为

$$
0.7\times 1+0.2\times 3+0.1\times 5=1.8
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

\begin{examplebox}
\textbf{例子 5：不能偷看未来。} 已知前文是“我 爱”，模型要预测下一个 token 时，可以看“我”和“爱”，但不能提前看后面的“小 猫”。
\end{examplebox}

对第 $i$ 个 token 来说，causal mask 只允许它关注第 $1$ 到第 $i$ 个 token。这样就保证 next token prediction 的因果性。

## Transformer Block、LayerNorm 与 FNN

页码：p31-p37

注意力输出后，还会经过线性变换、残差连接和 LayerNorm。之后再接前馈全连接网络 FNN。多个 Transformer block 可以复制堆叠。

\begin{center}
\includegraphics[width=0.86\linewidth]{output/assets/transformer_figures/transformer_p31_attention_block.png}
\end{center}

\begin{definitionbox}
\textbf{Transformer Block：} 一个 Transformer block 通常包含注意力模块、残差连接、LayerNorm 和前馈全连接网络 FNN。
\end{definitionbox}

\textbf{注意力输出后的线性变换：}

$$
X^{pr(l)}=X^{qkv(l)}W^{O(l)}, \qquad W^{O(l)}\in R^{d_v\times d_m}
$$

\textbf{残差连接与 LayerNorm：}

$$
X^{ao(l)}=\operatorname{LayerNorm}(X^{(l)}+X^{pr(l)})
$$

LayerNorm 是逐个 token 做归一化：每个 token 自己减均值、除方差，再做可训练的缩放和平移。复习时只需知道它用于稳定训练，不需要展开复杂变体。

\textbf{FNN：}

FNN 是前馈全连接网络，对每个 token 分别作用。它通常不混合不同 token 的信息，token 之间的信息交互主要来自注意力模块。

\begin{examplebox}
\textbf{例子 6：逐 token 过 FNN。} 对句子“我 爱 小 猫”，FNN 会分别作用在“我”“爱”“小”“猫”对应的隐藏向量上；不同 token 之间的信息交换主要已经在注意力模块中完成。
\end{examplebox}

## 输出投影与解码

页码：p38-p39

最后，输出投影层把每个 token 的隐藏向量映射回词表大小，再经过 softmax 得到下一个 token 的概率分布。

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

| 解码方式 | 含义 | 结果是否固定 |
| --- | --- | --- |
| 贪婪解码 | 每一步取概率最大的 token | 通常固定 |
| 采样解码 | 按概率分布随机抽样 | 可能不同 |

\begin{examplebox}
\textbf{例子 7：贪婪解码。} 如果词表中“猫”的概率是 $0.70$，“狗”的概率是 $0.20$，“鼠”的概率是 $0.10$，贪婪解码会选择“猫”。
\end{examplebox}

## 关键记忆

\begin{keybox}
\begin{itemize}
\item Transformer 用注意力机制直接建模 token 之间的关系，比 RNN 更适合并行计算。p3-p9
\item Next token prediction 是根据前文预测下一个 token，常用交叉熵损失。p10-p13
\item Embedding 把 one-hot token 映射成低维稠密向量；位置编码提供 token 位置信息。p15-p18
\item Q 和 K 用于计算关联强度，V 是真正被加权汇总的信息。p21-p26
\item 注意力的核心计算是 $QK^T$、mask、softmax、再对 $V$ 加权求和。p26-p31
\item Causal mask 保证预测当前位置时不能看到未来 token。p27-p29
\item Transformer block 通常包含注意力模块、残差连接、LayerNorm 和 FNN。p31-p37
\item 输出投影层把隐藏向量映射到词表维度；贪婪解码取最大概率 token，采样解码具有随机性。p38-p39
\end{itemize}
\end{keybox}

## 思考题

1. 判断：Transformer 使用注意力机制来建模 token 之间的关系。  
   答案：正确。
2. 填空：Transformer 需要加入 \underline{\hspace{3em}}，用来表示 token 在序列中的位置。  
   答案：位置编码。
3. 选择：Q、K、V 中，用来真正汇总信息内容的是：A. Q B. K C. V D. mask。  
   答案：C。
4. 判断：Decoder 做 next token prediction 时，可以看到未来 token。  
   答案：错误。需要 causal mask 遮住未来位置。
5. 计算：注意力权重为 $[0.6,0.4]$，两个 value 为 $[2,5]$，加权求和结果是多少？  
   答案：$0.6\times 2+0.4\times 5=3.2$。
