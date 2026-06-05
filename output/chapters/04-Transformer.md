# Transformer

这一章按“为什么需要 Transformer -> embedding 与位置编码 -> 注意力机制 -> Q/K/V -> causal mask -> Transformer block 与输出”的顺序复习。重点是理解 Transformer 如何用注意力机制建模 token 之间的关系。

## 为什么需要 Transformer

Transformer 出现前，序列任务常用 RNN 和 LSTM。RNN/LSTM 能处理序列，但有两个明显限制：

| 问题 | 含义 |
| --- | --- |
| 长程依赖困难 | 很远位置的信息不容易传到当前时刻 |
| 并行计算不方便 | RNN 通常按时间步顺序计算 |

Transformer 使用注意力机制直接建模 token 之间的关系，更容易捕捉全局语义关系，也更适合并行计算。

## Embedding 与位置编码

Token 是离散符号，进入神经网络前需要先变成向量。

```text
token -> embedding 向量
```

但仅有 embedding 还不够。同一个 token 出现在不同位置时，句子含义可能不同，因此模型还需要位置信息。

```text
输入表示 = token embedding + 位置编码
```

位置编码用于告诉模型 token 在序列中的位置。位置编码可以是可学习的，也可以是人为设计的。

## 注意力机制直观理解

注意力机制可以分成两步：

```text
第一步：算“我应该关注谁”
第二步：按关注程度汇总信息
```

对某个 token 来说，模型会先计算它和其他 token 的关联强度，再把这些关联强度变成权重，对其他 token 的信息加权求和。

| 概念 | 含义 |
| --- | --- |
| 注意力分数 | token 之间的关联强度 |
| Softmax | 把分数变成权重 |
| 注意力权重 | 通常非负，总和为 1 |
| 加权求和 | 按关注程度汇总信息 |

注意力权重越大，说明该位置对当前 token 的更新影响越大。

## Q、K、V

注意力机制中常见三个向量：Q、K、V。

| 符号 | 英文 | 作用 |
| --- | --- | --- |
| Q | Query | 当前 token 发出的查询 |
| K | Key | 其他 token 可被匹配的键 |
| V | Value | 真正被汇总的信息内容 |

Q、K、V 都由输入向量经过可训练线性变换得到。Q 和 K 用来计算相关性，V 用来做加权求和。

```text
Q 和 K -> 算关联强度
Softmax -> 得到注意力权重
注意力权重 和 V -> 加权求和
```

## Causal Mask

在生成任务中，模型预测当前位置时不能看到未来 token，否则就等于提前看到了答案。

```text
已知：我 喜欢
预测：人工
不能偷看：智能
```

Decoder 使用 causal mask 遮住未来位置。常见做法是把未来位置的注意力分数设为负无穷，经过 Softmax 后，这些位置的权重接近 0。

Causal mask 保证 next token prediction 的因果性：预测当前位置时只能看当前位置和前文，不能看未来。

## Transformer Block、FNN 与输出

一个 Transformer block 通常包含注意力模块和前馈网络 FNN，并配合残差连接和 LayerNorm。

```text
输入
-> 注意力模块
-> 残差连接 + LayerNorm
-> FNN
-> 残差连接 + LayerNorm
-> 输出
```

FNN 是前馈全连接网络，对每个 token 分别作用。多个 Transformer block 可以复制堆叠，形成深层 Transformer。

输出层把隐藏向量映射到词表大小，再经过 Softmax 得到下一个 token 的概率分布。

| 解码方式 | 含义 | 结果是否固定 |
| --- | --- | --- |
| 贪婪解码 | 每步取概率最大的 token | 通常固定 |
| 采样解码 | 按概率分布随机抽样 | 可能不同 |

## 关键记忆

\begin{keybox}
\begin{itemize}
\item Transformer 的核心思想之一是注意力机制，用来建模 token 之间的关系。
\item Transformer 仍然需要位置信息；位置编码用于表示 token 在序列中的位置。
\item 注意力机制先计算关联强度，再用 Softmax 得到权重，最后对信息加权求和。
\item Q 和 K 用于计算相关性，V 是真正被汇总的信息内容。
\item Decoder 做 next token prediction 时不能看到未来 token，需要 causal mask。
\item Transformer block 通常由注意力模块、FNN、残差连接和 LayerNorm 等部分组成。
\item 输出层维度通常对应词表大小；Softmax 把输出转成下一个 token 的概率分布。
\item 采样解码具有随机性，同一输入可能生成不同结果。
\end{itemize}
\end{keybox}

## 思考题

1. 判断：Transformer 使用注意力机制来建模 token 之间的关系。  
   答案：正确。
2. 填空：Transformer 需要加入______，用来表示 token 在序列中的位置。  
   答案：位置编码。
3. 选择：Q、K、V 中，用来真正汇总信息内容的是：A. Q B. K C. V D. mask。  
   答案：C。
4. 判断：Decoder 做 next token prediction 时，可以看到未来 token。  
   答案：错误。需要 causal mask 遮住未来位置。
5. 判断：采样解码按概率分布随机抽样，因此同一输入可能生成不同结果。  
   答案：正确。
