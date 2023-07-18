# 处理文本数据 Processing Text Data

## Tokenization (Text -> Words)

My dog is cute, he likes playing -> [my, dog, is, cute, he, likes, play, ##ing]

- Upper case -> Lower case.
- Remove stop words. e.g. "the", "a", "of" etc.
- Typo correction

## Count  Word Frequencies

- 建立字典（HashMap）统计词频
- 按照词频递减顺序排序
- 将词频替换为index（从1开始）
- 删除低频词（Name Entities / Typos）减小vocabulary，减少参数，防止overfitting

## One-Hot Encoding

Words -> Indices -> One-Hot Encoding 如果一个词不在字典中，可以忽略或编码为0

## Align Sequences

序列长度不同：假设最后长度为w

- 过长的序列：仅保留前w或后w个token
- 过短的序列：零填充 zero-padding

# 词嵌入 Word Embedding

将One-Hot向量投影为低维向量：

$$
x_i^{d\times i}=P^Te_i^{v\times 1}
$$

- $P^{v\times d}$：参数矩阵( $d$：词向量维度，$v$：字典大小)
- $e_i$：One-Hot向量

代码实现：

```python
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras import optimizers

model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dim, input_length=word_num))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs, batch_size, validation_data=(x_valid, y_valid))
loss_and_acc = model.evaluate(x_test, labels_test)
print(f'loss = {loss_and_acc[0]}')
print(f'acc = {loss_and_acc[1]}')
```

# 循环神经网络 Recurrent Neural Networks (RNNs)

<img src=".\figures\RNN&自然语言处理\image-20230331191827896.png" alt="image-20230331191827896" style="zoom: 50%;" />

## Simple RNN

$$
h_t=\tanh(A^{h\times (h+d)}\times\mathrm{Concat}(h_{t-1}, x_t))
$$

tanh：normalization，将特征重新缩到[-1, 1]，防止数值计算出问题

<img src=".\figures\RNN&自然语言处理\image-20230331192234060.png" alt="image-20230331192234060"  />

代码实现：

```python
from keras.models import Sequential
from keras.layers import SimpleRNN, Embedding, Dense

model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dim, input_length=word_num))
model.add(SimpleRNN(state_dim, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
```

shortcoming：擅长短期依赖，不擅长长期依赖。$h_{100}$ is almost irrelevant to $x_1$：$\frac{\partial h_{100}}{\partial x_1}$ is near zero.

## 长短期记忆模型 Long Short-Term Memory (LSTM)

> Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." *Neural computation* 9.8 (1997): 1735-1780.

![image-20230331195512568](.\figures\RNN&自然语言处理\image-20230331195512568.png)

结构：

- 传送带Conveyor Belt：过去的信息直接流向未来，不会发生太大的变化，避免梯度消失。
- 遗忘门Forget Gate：$f_t=\sigma(W_f\times\mathrm{Concat}(h_{t-1}, x_t))$
- 输入门Input Gate：$i_t=\sigma(W_i\times\mathrm{Concat}(h_{t-1}, x_t))$；$\tilde{C}_T=\tanh(W_c\times\mathrm{Concat}(h_{t-1}, x_t))$

$$
C_t=f_t\cdot C_{t-1}+i_t\cdot\tilde{C}_t
$$

- 输出门Output Gate：$o_t=\sigma(W_o\times\mathrm{Concat}(h_{t-1}, x_t))$；$h_t=o_t\cdot \tanh(C_t)$

参数数量 $=4\times \mathrm{Shape}(h)\times [\mathrm{Shape}(h) + \mathrm{Shape}(x)]$

代码实现：将SimpleRNN替换为LSTM

## 多层RNN Stack RNN

<img src=".\figures\RNN&自然语言处理\image-20230331233706854.png" alt="image-20230331233706854" style="zoom:50%;" />

把很多RNN堆叠起来，构成多层RNN。

代码实现：

```python
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dim, input_length=word_num))
model.add(LSTM(state_dim, return_sequences=True))
model.add(LSTM(state_dim, return_sequences=True))
model.add(LSTM(state_dim, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
```

## 双向RNN Bidirectional RNN

<img src=".\figures\RNN&自然语言处理\image-20230331234357722.png" alt="image-20230331234357722" style="zoom:50%;" />

如果有多层RNN，就把==$[y_1, y_2, \dots, y_t]$==作为上层RNN的输入

如果没有多层RNN，就返回==$\mathrm{Concat}(h_t, h_t')$==作为结果

代码实现

```python
from keras.layers import LSTM, Embedding, Dense, Bidirectional

model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dim, input_length=word_num))
model.add(Bidirectional(LSTM(state_dim, return_sequences=False)))
model.add(Dense(1, activation='sigmoid'))
```

## 预训练 Pretrain

Embedding层占据了绝大多数的参数量，在小数据集上训练很容易过拟合，这时候无论怎么调下层模型都无法获得比较好的效果。

1. 在更大的数据集上训练模型（可以是不同的问题 / 可以是不同的上层模型）主要是学习Embedding层，任务最好相似。
2. 训练结束之后只保留Embedding层。 
3. 让Embedding层不要训练，在这个层上搭建RNN和输出层训练原任务。

# 文本生成 Text Generation

- 将文章分成等长的segments（带重叠）
  - 一个segment做输入，它的下一个字符做标签。训练集：(segment, next_char) pairs
  - 输出过一下Softmax，转换成多分类问题

```python
import numpy as np
from keras import layers

model = Sequential()
model.add(LSTM(128, input_shape=(seg_len, vocabulary_size)))
model.add(Dense(vocabulary_size, activation='softmax'))

optimizer = optimizers.Adam(lr=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.fit(x, y, batch_size, epochs)

# Predict the Next Char
pred = model.predict(x_input, verbose=0)[0]  # 概率值

next_index = np.argmax(pred)  # greedy selection

next_onehot = np.random.multinomial(1, pred, 1)  # sampling from the multinomial distribution
next_index = np.argmax(next_onehot)

pred = pred ** (1 / tempearture)  # adjusting the multinomial distribution
pred = pred / np.sum(pred)
next_onehot = np.random.multinomial(1, pred, 1)
next_index = np.argmax(next_onehot)
```

根据概率生成下一个字符的策略：

- greedy selection：deterministic，empirically not good
- sampling from the multinomial distribution：maybe too random
- adjusting the multinomial distribution：between greedy and multinomial（hyperparameter ==temperature==越小越极端）

# 机器翻译 Machine Translation & Seq2Seq

## Machine Translation with LSTM

1. Tokenization & Build Dictionary （char-level / word-level），目标字典要额外增加起始符和终止符。
2. Training Seq2Seq Model

![image-20230401024517881](.\figures\RNN&自然语言处理\image-20230401024517881.png)

3. Improvement
   - Encoder RNN换成Bidirectional的
   - 多任务学习 Multi-Task Learning（生成各种语言，训练更好的Encoder）

## Machine Translation with Attention

> Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." *arXiv preprint arXiv:1409.0473* (2014).

- 极大提升Seq2Seq模型性能
- 不会忘记源语言输入
- decoder知道应该重视哪里
- 计算量大得多

<img src=".\figures\RNN&自然语言处理\image-20230401033249184.png" alt="image-20230401033249184" style="zoom: 67%;" />

每次更新状态$s_t$：

1. 计算一个权重 Weight：$\alpha_i=\mathrm{align}(h_i, s_{t-1})$

   1. 原论文提出的
      $$
      \tilde{\alpha}_i = v^T\cdot\tanh(W\times \mathrm{Concat}(h_i, s_{t-1}))\\
      [\alpha_1, \cdots, \alpha_m] = \mathrm{Softmax}([\tilde{\alpha}_1, \cdots, \tilde{\alpha}_m])
      $$
      
   2. Transformer提出的，更常用
   
      1. 线性变换 Linear Maps
         $$
         k_i = W_K\times h_i\quad \mathrm{for}\;i=1\;to\;m\\
         q_{t-1} = W_Q\times s_{t-1}
         $$
   
      2. 内积 Inner Product
   
      $$
      \tilde{\alpha_i} = k_i^Tq_{t-1}\quad\mathrm{for}\;i=1\;to\;m\\
      $$
   
      3. Normalization
   
      $$
      [\alpha_1, \cdots, \alpha_m] = \mathrm{Softmax}([\tilde{\alpha}_1, \cdots, \tilde{\alpha}_m])
      $$
   
2. 计算上下文向量 Context Vector：$c_0=\sum_{i=1}^m\alpha_ih_i=\alpha_1h_1+\cdots+\alpha_mh_m$
3. 在原有RNN基础上，输入再与Context Vector进行Concat

![image-20230401033647394](.\figures\RNN&自然语言处理\image-20230401033647394.png)

## Self-Attention

> Cheng, Jianpeng, Li Dong, and Mirella Lapata. "Long short-term memory-networks for machine reading." *arXiv preprint arXiv:1601.06733* (2016).

改善Encoder，每次更新状态$h_t$与$c_t$：

1. 计算$h_t$：$h_t=\tanh(A\times \left[\begin{matrix}x_t\\c{t-1}\end{matrix}\right] + b)$ 或 $h_t=\tanh(A\times \left[\begin{matrix}x_t\\c{t-1}\\h_{t-1}\\\end{matrix}\right] + b)$
2. 计算一个权重 Weight：$\alpha_i=\mathrm{align}(h_i, h_t)$
3. 计算上下文向量Context Vector：$c_t=\sum_{i=1}^t\alpha_ih_i=\alpha_1h_1+\cdots+\alpha_th_t$

