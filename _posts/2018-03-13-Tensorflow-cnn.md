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
本案例中的[MNIST数据集](http://yann.lecun.com/exdb/mnist/)有两个集合，共四个文件:  
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
定义数据对象 MNIST, 该对象有三个数据集组成：  
    1) train 用于训练时的数据集；  
	2) validation 从训练集中取出了前5000个图像，不清楚有什么具体用途；  
	3) test  用于测试时的数据集。

### 2.4 输入层  
测试模型的输入张量的形状定义为 `[batch_size, image_width, image_height, channels]`：  
* batch_size 训练中执行梯度下降的样本子集的大小；  
* image_width 单个图像的宽；  
* image_height 单个图像的高；    
* channels 样本图像的颜色数量。一般为3(RGB),对于灰度图是1。  
对于本例来说，输入层的形状为 [batch_size, 28, 28, 1]。  
为了转换输入特征 map 到本形状，我们可以使用 reshape 函数
`input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])`  
batch_size 值 -1指定了输入特征的维度是动态配置的。这代表batch_size像超参可以被调整。  

### 2.5 卷积层 #1(Convolutional Layer #1)    	
在第一卷积层，我们对输入层使用32通道的5 *5 滤波，然后再使用 ReLU激活函数。我们可以使用`conv2d()` 搭建本层。 
  
```python   
conv1 = tf.layers.conv2d( 
	  inputs=input_layer, 
	  filters=32, 
	  kernel_size=[5, 5], 
	  padding="same", 
	  activation=tf.nn.relu) 
```  
 
输入参数必须是形状为 `[batch_size, image_width, image_height, channels]`的装量。本例中为 `[batch_size, 28, 28, 1]`  
`filter`参数指定了滤波的目标，本例中为32。 并且 `kernel_size` 指定了滤波的的维度 `[width, height]`,这里为 `[5, 5]`。  
`padding`为枚举型{valid (default value), same}。如果指定输出张量与输入张量的高和宽一致，选择`padding=same`
在本例中，将通过在输入张量的图片边缘添加0来保证高宽都为28。
(如果不进行padding，一个5\*5卷积处理 28\*28的张量将生成一个 24*24的张量。)  
`activation` 参数指定的卷积输出的激活函数，本例中我们使用 ReLU。  
本例中通过 `con2d()`输出张量形状为 `[batch_size, 28, 28, 32]`，等同于输入张量，但是每个滤波器有32个通道。  
	
### 2.6 池化层 #1(Pooling Layer #1)    
现在可以使用 `max_pooling2d()` 来搭建 2 * 2的滤波并且跳跃为2。  
`pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)`  
本层的输入张量形状为`[batch_size, image_width, image_height, channels]`。本例中输入张量为 `conv1`，
即第一卷积层的输出，对应输出张量的形状为`[batch_size, 28, 28, 32]`  
`pool_size`指定了最大池化滤波器的`[width, height]`,本例中为[2,2]。  
`strides` 指定了跳跃的大小。本例中的跳跃度2，指示了滤波抽象间隔度宽和高分别为2个像素。  
`max_pooling2d()`的输出张量的形状为 `[batch_size, 14, 14, 32]`: 2 * 2的滤波将减少高宽各 50%。  

### 2.6 卷积层 #2 和池化层 #2  
然后使用`conv2d()`和`max_pooling2d()` 创建第二卷积层。对于本层来说，
配置了 64通道 5\*5的滤波器和ReLU的激活函数。并且对于池化层#2，我们仍然配置了和池化层#1相同的参数。
(2*2最大池化，跳跃度为2)  

```python    
conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
```  

注意卷积层#2使用的是第一池化层的输出做为输入，生成张量的形状为  `[batch_size, 14, 14, 64]`，
与池化层1相同的高宽配置，和64通道的滤波器。  
池化层#2使用卷积层#2的输出做为输入。池化层#2输出张量的形状为 `[batch_size, 7, 7, 64]`
(相比卷积层#2减少了一半的高和宽)。  

### 2.7 连接层(Dense Layer)  
然后添加一个连接层(1024个神经元和 ReLU激活函数)给CNN模型来根据卷积池化抽象出的结果进行分类。  
在连接本层前，我么必须要想特征转换为2维`[batch_size, features]`。  
`pool2_flat = tf.reshape(pool2, [-1, 7 \* 7 \* 64])`  
在上面的`reshape()`操作中，-1代表 batch_size 的维度将会有输入数据动态计算。
每个样本有 7 * 7 * 64 个特征，所以特征的维度为 7 \* 7 \* 64 。本层输出张量形状为 `[batch_size, 3136]`。  
现在，我们可以使用 `dense()`连接层。  
`dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)`  
`inputs`参数指定了输入张量：特征map。  
`units`指定了连接层的神经元数量(1024).   
`activation` 参数指定了激活函数为 ReLU。  
为了改善模型，我们使用 dropout正则化到连接层。
`dropout = tf.layers.dropout(
inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)`  
`inputs` 指定了输入张量，即连接层的输出张量。  
`rate`指定了丢弃率，本例中的0.4代表在训练中随机丢弃 40%的结果。  
`training`只是本模型是否在训练模式。dropout只有在训练模式才会丢弃。  
本层的输出张量形状为 `[batch_size, 1024]`。 

### 2.8 Logits 层(对数层)  
神经网络的最后一层为对数层，这将返回我们预测的最终结果。
本例中我们用10个神经元搭建了连接层，并使用了线性激活函数。  
`logits = tf.layers.dense(inputs=dropout, units=10)`  	
CNN最终的输出张量的形状为`[batch_size, 10]`。  
	
### 2.9 生成预测结果  
CNN模型的对数层返回了预测的原始结果`[batch_size, 10]` 。
我们需要将此结果转换为模型可以返回的两种格式：  

* 每个图像的预测值： 数字 0~ 9.  
* 每个图像每个可能目标的概率。 
 
给定样本后，预测结果是最大概率的对数张量。我们可以通过 `tf.argmax`找到:  
```python 
tf.argmax(input=logits, axis=1)
```  

`input` 指定了提取出的最大值得张量。  
`axis`指定了 `input`张量找最大值的轴。这里我们使用1 关联我们的预测。  
我们可以使用 `tf.nn.softmax` 求导对数层来应用 softmax 激活。  
`tf.nn.softmax(logits, name="softmax_tensor")`  
我们在字典里编译预测，并返回 EstimatorSpec对象。  

```python   
predictions = {
	"classes": tf.argmax(input=logits, axis=1),  
	"probabilities": tf.nn.softmax(logits, name="softmax_tensor")  
}  
if mode == tf.estimator.ModeKeys.PREDICT:  
	return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)  
```  


	

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
1. ReLU 激活函数的用途是什么？为什么第二层连接层没有？  
2. 如果确定卷积核的大小 5 * 5 ？  
3. 如何确定通道数32，又如何将通道数扩充到 64 ？  
4. 计算正确率的Op比较奇怪，从计算图中看到应该和正常训练走相同的流程。是一种冗余么？  



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