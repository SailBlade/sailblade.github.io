---
layout: post
title: "Tensorflow 实践——softmax 分类"
description: "softmax"
categories: [machine learning]
tags: [tensorflow, tensorboard, softmax]
redirect_from: 
  - /2018/03/06/
---  
* Kramdown table of contents
{:toc .toc}
---


## 1. softmax预测鸢尾花种类
根据鸢尾花的特征萼片的长宽，花瓣的长宽，推断鸢尾花的种类。{萼片的长度:sepal_length，萼片的宽度:sepal_width，花瓣的长度:petal_length，花瓣的宽度:petal_width，鸢尾花的种类:label}。  
本例中的鸢尾花种类有如下三种{ Iris-setosa，Iris-versicolor，Iris-virginica}，数据格式如下  
![鸢尾花样本](http://p30p0kjya.bkt.clouddn.com/%E9%B8%A2%E5%B0%BE%E8%8A%B1%E6%95%B0%E6%8D%AE.PNG)
  
## 2. 训练框架  
![矩阵元素](http://p30p0kjya.bkt.clouddn.com/%E7%9F%A9%E9%98%B5%E5%85%83%E7%B4%A0.PNG)  
![对数几率回归](http://p30p0kjya.bkt.clouddn.com/%E5%AF%B9%E6%95%B0%E5%9B%9E%E5%BD%92%E5%AD%A6%E4%B9%A0.PNG)  

## 3. Feature Of This Framework
1. tf.train.shuffle_batch 是将队列中数据打乱后，再读取出来。
2. 如果需要观察训练中的正确率，需要在训练中增加正确率的信息。
3. 如果样本量太小，如本例中一次所需数据为100，但是样本总共只有150，容易导致编译异常，扩大样本量后解决。

## 4. 训练结果
 本例的预测率随样本量增加趋近于100%正确率，不得不说很神奇。  
![鸢尾花的预测正确率](http://p30p0kjya.bkt.clouddn.com/%E9%B8%A2%E5%B0%BE%E8%8A%B1accuracy.PNG)  
 Loss函数如下所示，对于本例来说正确率似乎更适合表达模型的性能。  
![鸢尾花的Loss函数](http://p30p0kjya.bkt.clouddn.com/%E9%B8%A2%E5%B0%BE%E8%8A%B1loss.PNG)  
 鸢尾花的参数直方图。  
![鸢尾花的直方图](http://p30p0kjya.bkt.clouddn.com/%E9%B8%A2%E5%B0%BE%E8%8A%B1%E7%9B%B4%E6%96%B9%E5%9B%BE.PNG)  

## 5. softmax预测分类案例的思考  
1. 


## 6. 源码  

```python  
# Softmax example in TF using the classical Iris dataset
# Download iris.data from https://archive.ics.uci.edu/ml/datasets/Iris
# Be sure to remove the last empty line of it before running the example

import tensorflow as tf
import os

with tf.name_scope('parameters'):
    # this time weights form a matrix, not a column vector, one "weight vector" per class.
    W = tf.Variable(tf.zeros([4, 3]), name="weights")
    # so do the biases, one per class.
    b = tf.Variable(tf.zeros([3]), name="bias")
    tf.summary.histogram('weight', W)
    tf.summary.histogram('bias', b)


def combine_inputs(X):
    return tf.matmul(X, W) + b


def inference(X):
    return tf.nn.softmax(combine_inputs(X))


def loss(sess, X, Y):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=combine_inputs(X)))
    tf.summary.scalar('loss', loss)
    evaluate(sess, X, Y)
    return loss


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(os.path.abspath(__file__)) + "\\" + file_name])
    print (os.path.dirname(os.path.abspath(__file__)) + "\\" + file_name)
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    print (filename_queue)
    print (value)

    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)

def inputs():

    sepal_length, sepal_width, petal_length, petal_width, label = \
        read_csv(100, "iris.data", [[0.0], [0.0], [0.0], [0.0], [""]])

    # convert class names to a 0 based class index.
    label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([
        tf.equal(label, ["Iris-setosa"]),
        tf.equal(label, ["Iris-versicolor"]),
        tf.equal(label, ["Iris-virginica"])
    ])), 0))

    # Pack all the features that we care about in a single matrix;
    # We then transpose to have a matrix with one example per row and one feature per column.
    features = tf.transpose(tf.stack([sepal_length, sepal_width, petal_length, petal_width]))

    return features, label_number

def train(total_loss):
    learning_rate = 0.01
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
    return train

def evaluate(sess, X, Y):
    predicted = tf.cast(tf.arg_max(inference(X), 1), tf.int32)
    # print (sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))
    tf.summary.scalar('accuracy', tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32)))

# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    X, Y = inputs()

    total_loss = loss(sess, X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./my_graph", sess.graph)
    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        rs = sess.run(merged)
        writer.add_summary(rs, step)
        # for debugging and learning purposes, see how the loss gets decremented thru training steps

        if step % 10 == 0:
            print ("loss: ", sess.run([total_loss]))

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()


```  
      

## 7. 参考文献
1. 《面向机器智能 TensorFlow实践》  Sam Abrahams, Danijar Hafner, Erik Erwitt, Ariel Scarpinelli  
2.  鸢尾花种类预测                  [Github 地址](https://github.com/backstopmedia/tensorflowbook)  
3.  交叉熵在分类分类问题的优势      [交叉熵在分类分类问题的优势](https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/)
4.  TensorFlow 官方对交叉熵的说明   [tensorflow 谈交叉熵](http://colah.github.io/posts/2015-09-Visual-Information/)
