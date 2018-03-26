---
layout: post
title: "循环神经网络介绍"
description: "RNN"
categories: [machine learning]
tags: [tensorflow, RNN]
redirect_from: 
  - /2018/03/24/
---  
* Kramdown table of contents
{:toc .toc}
---

## 1. 介绍
请参考这篇介绍循环神经网络和LSTM的[里程碑级的论文](https://www.tensorflow.org/tutorials/recurrent)。

## 2. 语言模型
在本篇教程中，我们将展示如何在语言模型中训练循环神经网络。
训练的目标是拟合句子的概率模型。它将会根据文本中的历史词汇给出下个词的预测。
基于本用途我们将使用 Penn Tree Bank(PTB)数据集，这是一个很小
且很容易训练的通测量模型。  
语言模型是许多有趣问题的关键技术，如对话识别，机器翻译，图像标注。
这是非常有趣的。可以参考如下链接。  
本篇教程的目的是重现 Zaremba et al.,2014的结果，该方法在PTB集合上达到了非常好的效果。  

## 3. 教程用的文件
教程用到了如下这些文件，大家可以在 TensorFlow models repo 的 `models/tutorials/rnn/ptb`中找到。  

| File  | Purpose |  
| ------------- | ------------- |  
| ptb_word_lm.py  | 在PTB数据集上训练语言模型的代码 |  
| reader.py  | 	读取数据集的代码  |  

## 4. 下载和准备数据  
本篇教程要求的数据在  PTB dataset from Tomas Mikolov's webpage 的`data/`目录下。  
这个数据集已经被预先处理过，包含超过10000个不同的单词，包含句尾标志和稀有词汇的特殊只是(\<unk>)。
在`reader.py`中，我们将每个词转换为唯一的整数，以便在神经网络中处理更简单。  


## 5. 模型  
### 5.1 LSTM  
包含LSTM cell的模型的核心是一次处理一个单词并且计算本句中下个词出现的概率。神经网络的记忆状态
被初始化为向量0，并且读取每个词后都会被更新。由于计算的原因，我们将以`batch_size` 的大小
在mini-batches中处理数据。在本例中需要特别注意到 `current_batch_of_words`与句子的词不相关。
一批数据中的每个词都有一个关联时间 t。TensorFlow将会计算每批数据中的梯度和。  
例如

>
 t=0  t=1    t=2  t=3     t=4
[The, brown, fox, is,     quick]
[The, red,   fox, jumped, high]

words_in_dataset[0] = [The, The]
words_in_dataset[1] = [brown, red]
words_in_dataset[2] = [fox, fox]
words_in_dataset[3] = [is, jumped]
words_in_dataset[4] = [quick, high]
batch_size = 2, time_steps = 5

基本的伪代码如下：
```python
words_in_dataset = tf.placeholder(tf.float32, [time_steps, batch_size, num_features])
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
# Initial state of the LSTM memory.
hidden_state = tf.zeros([batch_size, lstm.state_size])
current_state = tf.zeros([batch_size, lstm.state_size])
state = hidden_state, current_state
probabilities = []
loss = 0.0
for current_batch_of_words in words_in_dataset:
    # The value of state is updated after processing each batch of words.
    output, state = lstm(current_batch_of_words, state)

    # The LSTM output can be used to make next word predictions
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities.append(tf.nn.softmax(logits))
    loss += loss_function(probabilities, target_words)
```

### 5.2 截断反向传播(Truncated Backpropagation)  
依靠设计来说，循环神经网络(RNN)的输出以来与任意距离的输入。不幸的是，这将会导致
反向传播计算很困难。为了让学习过程更温和，它联系创建了不回滚的神经网络，此网络中
包含了固定数量(num_steps)的 LSTM输入和输出。此模型在RNN有效的逼近中训练。它可以由
一次性输入长度和每次输入块的反向传递构成。  
这是一个用于创建截断反向传播模型计算图的简单代码块。  
```python  
# Placeholder for the inputs in a given iteration.
words = tf.placeholder(tf.int32, [batch_size, num_steps])

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
# Initial state of the LSTM memory.
initial_state = state = tf.zeros([batch_size, lstm.state_size])

for i in range(num_steps):
    # The value of state is updated after processing each batch of words.
    output, state = lstm(words[:, i], state)

    # The rest of the code.
    # ...

final_state = state
```
并且下面是如何通过迭代处理整个数据集的过程:

```python  
# A numpy array holding the state of LSTM after each batch of words.
numpy_state = initial_state.eval()
total_loss = 0.0
for current_batch_of_words in words_in_dataset:
    numpy_state, current_loss = session.run([final_state, loss],
        # Initialize the LSTM state from the previous iteration.
        feed_dict={initial_state: numpy_state, words: current_batch_of_words})
    total_loss += current_loss
```

### 5.3 输入  
在将单词标识符输入LSTM前，首先要被插入到密集集合中。这项操作可以让模型更好的理解知识。
它可以简单写为：
```python 
# embedding_matrix is a tensor of shape [vocabulary_size, embedding size]
word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids)
```
这个插入的矩阵将会被随机初始化，然后通过观察数据学习不同单词的意思。  

### 5.4 损失函数  
我们可以通过球对目标词汇的负对数的概率： 
公式  
它计算起来非常困难，但是 `sequence_loss_by_example`已经可用，所以我们可以直接使用它。  
论文中典型的测量报告是平均每词的熵，它等于
公式  
我们将会在训练过程中监控它的值。  

### 5.5 存储多倍 LSTMs
为了提供模型更好的表现，我们增加了LSTMs多个层来处理数据。第一个层的输出变成第二个层的输入，等等。
我们有个类叫 `MultiRNNCell`这可以无缝执行。  
```python 
def lstm_cell():
  return tf.contrib.rnn.BasicLSTMCell(lstm_size)
stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    [lstm_cell() for _ in range(number_of_layers)])

initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
for i in range(num_steps):
    # The value of state is updated after processing each batch of words.
    output, state = stacked_lstm(words[:, i], state)

    # The rest of the code.
    # ...

final_state = state
```

## 6. 运行代码  
在运行代码前，就像教程中一样的下载PTB数据集。然后，解压缩PTB数据集到你的工作目录。

>
tar xvfz simple-examples.tgz -C $HOME

现在，从 GitHub 中克隆代码 TensorFlow models repo。 运行如下命令：

>
cd models/tutorials/rnn/ptb
python ptb_word_lm.py --data_path=$HOME/simple-examples/data/ --model=small

在教程代码中有3个支持模型配置:small,medium 和 large. 他们之间不同是是 LSTMs的大小
和训练中的超参集。  
较大的模型将会取得更好的训练效果。经过数小时的训练，`small`模型的熵可以在120以下，而`large`
的熵可以收敛到80以下。

## 7. 下一步  
本文中未提到一些可以让模型变得更好的技巧，如：
* 下降学习速率调度
* 在LSTM中丢弃部分结果  
学习代码并且进一步的提升模型。  
 


## 6. 参考文献
1.  TensorFlow 官方教程      [TensorFlow官方教程](https://www.tensorflow.org/tutorials/image_retraining)
