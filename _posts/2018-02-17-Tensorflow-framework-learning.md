---
layout: post
title: "Tensorflow 框架"
description: "Tensorflow framework"
categories: [tensorflow]
tags: [tensorflow,machine learning]
redirect_from:
  - /2018/02/17/
---  
  
## TensorFlow 基本的训练框架    
1. 对模型参数进行初始化。通常采用随机初始化，简单模型初始化为0。  
2. 读取训练数据(包括每个数据样本及其期望输出)，通常在这些样本送入模型之前，随机打乱样本的次序。  
3. 在训练数据上执行推断模型，获取每个训练样本的输出值。   
4. 计算损失。通过计算损失评估模型的性能。
5. 调整模型参数。给定损失函数，通过大量训练步骤改善各参数的值，从而将损失最小化。最常用的是梯度下降算法。

```python
# TF code scaffolding for building simple models.

import tensorflow as tf

# initialize variables/model parameters

# define the training loop operations
def inference(X):
    # compute inference model over data X and return the result
    return

def loss(X, Y):
    # compute loss over training data X and expected values Y
    return

def inputs():
    # read/generate input training data X and expected outputs Y
    return

def train(total_loss):
    # train / adjust model parameters according to computed total loss
    return

def evaluate(sess, X, Y):
    # evaluate the resulting trained model
    return


# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:

    tf.initialize_all_variables().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and learning purposes, see how the loss gets decremented thru training steps
        if step % 10 == 0:
            print("loss: ", sess.run([total_loss]))

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()

```

## TensorFlow 训练存档  
为了避免计算机长时间训练后停电，借助 tf.train.Saver类可以将数据流图中的变量保存到专门的二进制文件中(checkpoint)。可以周期性保存变量，并从对应变量恢复训练。

1. tf.train.Saver 存储当前会话；
2. tf.train.get_checkpoint_state 验证之前是否有检查点文件被保存下来；
3. tf.train.Saver.restore 可以恢复会话。

```python  
with tf.Session() as sess:  
    initial_step = 0
    
    # 验证之前是否训练有存档
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
    if ckpt and ckpt.model_checkpoint_path:
        # 从检查点恢复模型参数
        saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-',1)[1]) # 很奇特的用法
    for step in range(training_steps):
        sess.run([train_op])
        
        if step % 1000 == 0:
            saver.save(sess, 'my-model', global_step = step))
	# 模型评估
	saver.save(sess,'my-model',global_step=training_steps)
	sess.close()
```



## 参考文献
1. 《面向机器智能 TensorFlow实践》  Sam Abrahams, Danijar Hafner, Erik Erwitt, Ariel Scarpinelli  
