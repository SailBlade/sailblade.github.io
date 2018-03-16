---
layout: post
title: "Tensorflow 实践——卷积神经网络"
description: "CNN"
categories: [machine learning]
tags: [tensorflow, CNN]
redirect_from: 
  - /2018/03/11/
---  
* Kramdown table of contents
{:toc .toc}
---

## 1. 卷积神经网络介绍    
卷积神经网络是当前处理图片分类的瑰宝级的架构。卷积神经网络依靠对图像像素一系列的滤波抽象出更高维度的特征。CNN包含如下三个部分：  
1. 卷积层(Convolutional layers)  
应用指定的卷积滤波数在图片上，每个子区域利用卷积产生一个单独值。卷积层通常应用ReLU激活函数。

2. 池化层(Pooling layers)    
通过降采样卷积层输出的特征维度来降低处理时间。通常使用 2*2 最大池化。丢弃除最大值外的其他值。  

3. 密集层(全连接层)(Dense (fully connected) layers)  
对提取后的卷积层和降采样后的池化层的结果进行分类。本层的每个节点都对应着预测层的每个节点。

## 2. 通过CNN识别MNIST    
提供了若干 7*7 = 28 像素的图像，每个图像对应一个0 ~ 9的手写数字。识别图像中的数字。

### 2.1 CNN识别模型的软件框架  
![CNN识别模型软件框架](http://p30p0kjya.bkt.clouddn.com/CNN%E6%9E%B6%E6%9E%842.png)  
 
### 2.2 样本数据格式    
本案例中的[MINST数据集](http://yann.lecun.com/exdb/mnist/)有两个集合，共四个文件:  
1. 60000张图片训练集  
   1) train-images-idx3-ubyte: training set images (图像的像素集合)  
   2) train-labels-idx1-ubyte: training set labels (图像对应的数字)  
2. 10K 张图片的测试集，其中前5K是训练集中的图像，后5K是专门用于测试的图像。后5K的图片识别难度更大。  
   1) t10k-images-idx3-ubyte:  test set images (图像的像素集合)   
   2) t10k-labels-idx1-ubyte:  test set labels (图像对应的数字)

训练集存储图像对应数字的文件格式：(train-labels-idx1-ubyte)  

| [offset] | [type]          | [value]          | [description] |
| ---------- | ------------ | ----------- | -----------| 
| 0000     | 32 bit integer  | 0x00000801(2049) | magic number (MSB first) |
| 0004     | 32 bit integer  | 60000            | number of items |
| 0008     | unsigned byte   | ??               | label |
| 0009     | unsigned byte   | ??               | label |
| ........ |                 |                  | 
| xxxx     | unsigned byte   | ??               | label |

The labels values are 0 to 9.

训练集存储图像像素的文件格式：(train-images-idx3-ubyte 格式)  

| [offset] | [type]        |   [value]       |    [description] | 
| ---------- | ------------ | ----------- | -----------| 
| 0000     | 32 bit integer  | 0x00000803(2051) | magic number | 
| 0004     | 32 bit integer  | 60000     |        number of images | 
| 0008     | 32 bit integer  | 28          |      number of rows | 
| 0012     | 32 bit integer  | 28            |    number of columns | 
| 0016     | unsigned byte   | ??            |    pixel | 
| 0017     | unsigned byte   | ??            |    pixel | 
| ........ |                 |                |         | 
| xxxx     | unsigned byte  |  ??             |   pixel | 

Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

测试集存储图像对应数字的文件格式：(t10k-labels-idx1-ubyte)  

| [offset] | [type]         |  [value]    |       [description] | 
| ---------- | ------------ | ----------- | -----------| 
| 0000     | 32 bit integer  | 0x00000801(2049) | magic number (MSB first) | 
| 0004     | 32 bit integer  | 10000            | number of items | 
| 0008     | unsigned byte   | ??               | label | 
| 0009     | unsigned byte   | ??               | label | 
| ........ |                 |                  |       | 
| xxxx     | unsigned byte   | ??               | label | 

The labels values are 0 to 9.

测试集存储图像像素的文件格式： (t10k-images-idx3-ubyte)  

| [offset]   | [type]       |    [value]  |          [description] | 
| ---------- | ------------ | ----------- | -----------| 
| 0000       | 32 bit integer |  0x00000803(2051) |  magic number | 
| 0004     | 32 bit integer | 10000               | number of images | 
| 0008     | 32 bit integer |  28                 |  number of rows | 
| 0012     | 32 bit integer |  28                 | number of columns | 
| 0016     | unsigned byte  |  ??                 | pixel | 
| 0017     | unsigned byte  |  ??                 | pixel | 
| ........ |                |                     |       | 
| xxxx     | unsigned byte  |  ??                 | pixel | 


### 2.3 数据预处理  


### 2.4 CNN框架
1. 卷积层#1  
   Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
2. 池化层#1  
   Performs max pooling with a 2x2 filter and stride of 2 (which specifies that pooled regions do not overlap)
3. 卷积层#2  
   第二个卷积层，应用64通道 5x5 滤波，使用ReLU激活函数。
4. 池化层#2
   第二个池化层呢过，进行2x2滤波，而且滑动间隔为2.
5. 密集层 #1  
   1024个神经元, 丢失正则化率为 0.4. 训练过程中任何元素丢弃概率为 0.4.  
6. 密集层 #2  
   10 个神经元，从0到9。    


  

## 3. Feature Of This Framework

## 4. 训练结果

## 5. softmax预测分类案例的思考  


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
1.  TensorFlow 官方教程  [TensorFlow官方教程](http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html)  
2.  SHIHUC 个人博客      [SHIHUC个人博客](https://www.cnblogs.com/shihuc/p/6648130.html)