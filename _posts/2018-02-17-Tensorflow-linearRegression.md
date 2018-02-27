---
layout: post
title: "Tensorflow 模型——线性回归"
description: "Tensorflow Linear Regression"
categories: [tensorflow]
tags: [tensorflow,machine learning]
redirect_from:
  - /2018/02/18/
---  
  
## TensorFlow 线性回归    
1. 问题背景  
    通过年龄，体重与血液脂肪含量的关联数据来预测指定年龄的血液脂肪含量。  
2. 训练样本  
    [年龄，体重(公斤)]，对应的血液脂肪含量 blood_fat_content
    

```python
import tensorflow as tf

with tf.name_scope('parameters'):
    with tf.name_scope('Weight'):
        W = tf.Variable(tf.zeros([2, 1]), name="Weight")
        tf.summary.histogram('Weight', W)
    with tf.name_scope('Bias'):
        b = tf.Variable(0., name="Bias")
        tf.summary.histogram('Bias', b)


def inference(X):
    return tf.matmul(X, W) + b


def loss(X, Y):
    with tf.name_scope('Y_Predicted'):
        Y_Predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_Predicted))

def inputs():
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44],
                  [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34],
                  [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308,
                         220, 311, 181, 274, 303, 244]

    return tf.to_float(weight_age), tf.to_float(blood_fat_content)


def train(total_loss):
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
    print(sess.run(inference([[80., 25.]])))  # ~ 303
    print(sess.run(inference([[84., 46.]])))  # ~ 303
    print(sess.run(inference([[65., 25.]])))  # ~ 256


with tf.Session() as sess:
    tf.initialize_all_variables().run()
    with tf.name_scope('Data'):
        X, Y = inputs()   # X: weight_age, Y: blood_fat

    with tf.name_scope('loss'):
        total_loss = loss(X, Y)
        tf.summary.scalar('loss', total_loss)
    with tf.name_scope('train'):
        train_op = train(total_loss)
    with tf.name_scope('init'):
        initOP = tf.global_variables_initializer()
        print (type(initOP))

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./my_graph", sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    training_steps = 1000
    sess.run(initOP)
    for step in range(training_steps):
        sess.run([train_op])
        #if step % 50 == 0:
        rs = sess.run(merged)
        writer.add_summary(rs, step)
        print("loss: ", sess.run([total_loss]))

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    writer.close()
    sess.close()


```



## 参考文献
1. 《面向机器智能 TensorFlow实践》  Sam Abrahams, Danijar Hafner, Erik Erwitt, Ariel Scarpinelli  
