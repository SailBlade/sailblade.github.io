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
提供了若干 7*7 = 28 像素的图像，每个图像对应一个0 ~ 9的手写数字。识别如下图像中的数字。  
![待识别的数字图像](http://images.sailblade.com/mnist_0-9.png)  

### 2.1 CNN MNIST识别模型的软件框架  
![CNN识别模型软件框架](http://images.sailblade.com/CNN%E6%9E%B6%E6%9E%842.png)  
 
#### 2.1.1 样本数据格式    
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


#### 2.1.2 数据预处理  
定义数据对象 MNIST, 该对象有三个数据集组成：  
    1) train 用于训练时的数据集；  
	2) validation 从训练集中取出了前5000个图像，不清楚有什么具体用途；  
	3) test  用于测试时的数据集。

#### 2.1.3 输入层  
测试模型的输入张量的形状定义为 `[batch_size, image_width, image_height, channels]`：  
* batch_size 训练中执行梯度下降的样本子集的大小；  
* image_width 单个图像的宽；  
* image_height 单个图像的高；    
* channels 样本图像的颜色数量。一般为3(RGB),对于灰度图是1。  

对于本例来说，输入层的形状为 [batch_size, 28, 28, 1]。  
为了转换输入特征 map 到本形状，我们可以使用 reshape 函数
```python 
input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
```  
batch_size 值 -1指定了输入特征的维度是动态配置的。这代表batch_size像超参可以被调整。  

#### 2.1.4 卷积层 #1(Convolutional Layer #1)    	
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
	
#### 2.1.5 池化层 #1(Pooling Layer #1)    
现在可以使用 `max_pooling2d()` 来搭建 2 * 2的滤波并且跳跃为2。  
`pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)`  
本层的输入张量形状为`[batch_size, image_width, image_height, channels]`。本例中输入张量为 `conv1`，
即第一卷积层的输出，对应输出张量的形状为`[batch_size, 28, 28, 32]`  
`pool_size`指定了最大池化滤波器的`[width, height]`,本例中为[2,2]。  
`strides` 指定了跳跃的大小。本例中的跳跃度2，指示了滤波抽象间隔度宽和高分别为2个像素。  
`max_pooling2d()`的输出张量的形状为 `[batch_size, 14, 14, 32]`: 2 * 2的滤波将减少高宽各 50%。  

#### 2.1.6 卷积层 #2 和池化层 #2  
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

#### 2.1.7 连接层(Dense Layer)  
然后添加一个连接层(1024个神经元和 ReLU激活函数)给CNN模型来根据卷积池化抽象出的结果进行分类。  
在连接本层前，我么必须要想特征转换为2维`[batch_size, features]`。  
`pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])`  
在上面的`reshape()`操作中，-1代表 batch_size 的维度将会有输入数据动态计算。
每个样本有 7 * 7 * 64 个特征，所以特征的维度为 7 \* 7 \* 64 。本层输出张量形状为 `[batch_size, 3136]`。  
现在，我们可以使用 `dense()`连接层。  

```python    
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
```  

`inputs`参数指定了输入张量：特征map。  
`units`指定了连接层的神经元数量(1024).   
`activation` 参数指定了激活函数为 ReLU。  
为了改善模型，我们使用 dropout正则化到连接层。
```python 
dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)  
```  
`inputs` 指定了输入张量，即连接层的输出张量。  
`rate`指定了丢弃率，本例中的0.4代表在训练中随机丢弃 40%的结果。  
`training`只是本模型是否在训练模式。dropout只有在训练模式才会丢弃。  
本层的输出张量形状为 `[batch_size, 1024]`。 

#### 2.1.8 Logits 层(对数层)  
神经网络的最后一层为对数层，这将返回我们预测的最终结果。
本例中我们用10个神经元搭建了连接层，并使用了线性激活函数。  
`logits = tf.layers.dense(inputs=dropout, units=10)`  	
CNN最终的输出张量的形状为`[batch_size, 10]`。  
	
#### 2.1.9 生成预测结果  
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

#### 2.1.10 计算损失  
为了训练和评估模型，我们需要定义损失函数来测量预测值和目标值有多近。
对于像MNIST之类的多元分类问题，交叉熵是最常用的损失函数。如下代码在TRAIN
和 EVAL模式下计算交叉熵。    

```python  
onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels, logits=logits)
```  
如上所示，label张量是包含了一组预测范围{0~9}的值。为了计算交叉熵，我们首先
把label张量转换为 one-hot 编码：
```python  
[[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 ...]
```  
我们使用 `tf.one_hot`函数进行转换，tf.one_hot()有两个参数：  
* `indices`.在one-hot张量中对应位置上。本例中第一行对应的值为1.  
* `depth`. one-hot张量的深度。目标分类的个数，本例中为10.  
通过下面的代码我们为label创造了one-hot张量。  
`onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)`  
下一步，我们需要计算 onehot_labels的交叉熵，并通过softmax预测对数层输出的结果。
`tf.losses.softmax_cross_entropy()`  使用了 `onehot_labels` 和 `logits `
做为参数，对对数结果进行softmax激活，计算交叉熵，返回损失函数。  
```python  
loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels, logits=logits)
```  

#### 2.1.11 配置训练 Op  
最后我们使用步长为0.001的随机梯度算法做为优化算法。    
```python  
if mode == tf.estimator.ModeKeys.TRAIN:
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
	train_op = optimizer.minimize(
		loss=loss,
		global_step=tf.train.get_global_step())
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
```  

#### 2.1.12 增加评估度量  
为了为本模型增加正确率度量，我们定义了` eval_metric_ops`    
```python  
eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
return tf.estimator.EstimatorSpec(
    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
```  

### 2.2 训练和评估CNN MNIST分类器  
#### 2.2.1 获取训练和测试数据  
通过增加 main()获取训练和测试数据。  
```python 
def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
```  

#### 2.2.2 创建评估器  
创建`Estimator`——一个TensorFlow执行高级训练，评估，推理的类。  
```python 
# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
```  
`model_fn`指定了训练，评估，预测的模型。  
`model_dir` 指定了训练模型被保存的位置。  

#### 2.2.3 建立打印回调  
为了跟踪训练过程，所以我们可以使用`tf.train.SessionRunHook` 
搭建一个`tf.train.LoggingTensorHook`来打印softmax层的概率。  
```python 
# Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
```  
我们可以将张量存储到 `tensors_to_log`中。
每个我们选择的关键信息可以被打印到LOG中，并且对应的标签是TensorFlow计算图中的标签。
这里我们打印了 `softmax_tensor` 中的概率。  
下一步，我们搭建了 `tf.train.LoggingTensorHook`，并传递`tensors_to_log`到对应类中。
我们设置 `every_n_iter=50`，代表训练50步将会进行LOG输出。  

#### 2.2.4 训练模型  
我们现在可以在`train_input_fn` 中调用 `train()`和 `mnist_classifier`来启动训练。 
```python 
# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])
```   
在`numpy_input_fn`中，我们传递了特征信息和标签。并且设置一次采样100。  
`num_epochs=None` 代表模型会一直训练达到训练次数。
我们通过设置`shuffle=True`来传递训练数据。  
在`train`函数调用中，我们设置`steps=20000`。  
我们传递了 `logging_hook`来在训练中记录数据。  

#### 2.2.5 评估模型  
一旦训练完成，我们可以通过在MNIST的测试集评估模型。
我们可以通过使用 `evaluate`函数，该函数可以通过在模型 `model_fn`中的
Op `eval_metric_ops`调用 evaluate`来实现度量。  
```python 
# Evaluate the model and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
```  
搭建 `eval_input_fn`,并且`num_epochs=1`,所以每个数据处理时期都会返回结果。  
我们可以通过设置 `shuffle=False`来逐次操作。  

#### 2.2.6 训练结果展示
在训练中，我们可以看到如下的LOG输出,我们可以看到测试集达到了 97.3%的准确率。  
```python
INFO:tensorflow:loss = 2.36026, step = 1
INFO:tensorflow:probabilities = [[ 0.07722801  0.08618255  0.09256398, ...]]
...
INFO:tensorflow:loss = 2.13119, step = 101
INFO:tensorflow:global_step/sec: 5.44132
...
INFO:tensorflow:Loss for final step: 0.553216.

INFO:tensorflow:Restored model from /tmp/mnist_convnet_model
INFO:tensorflow:Eval steps [0,inf) for training step 20000.
INFO:tensorflow:Input iterator is exhausted.
INFO:tensorflow:Saving evaluation summary for step 20000: accuracy = 0.9733, loss = 0.0902271
{'loss': 0.090227105, 'global_step': 20000, 'accuracy': 0.97329998}
```  
  

## 3. CNN学习后的心得
1. 英文官方的源码模块化非常漂亮，代码很整洁。  


## 4. CNN MNIST分类案例的思考  
1. 什么是激活函数，什么时候用?  
2. 如何定义卷积核的shape，本案例中是[5 \* 5]?  
3. 如何确定卷积层输出通道数，本例从 1 -> 32 -> 64?  
4. 随机梯度下降的步长如何确定?  



## 5. 源码  

### 5.1 中文官方教程源码 
```python  
# -*- coding: utf-8 -*-
'''卷积神经网络测试MNIST数据'''
#导入MNIST数据

import tensorflow as tf

#导入input_data用于自动下载和安装MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
# 创建一个交互式的Session， 这与之前的 tf.Session()不同
# InteractiveSession 执行后变为默认的session，
# 后续的Tensor.eval() 和 Operation.run() 都会默认使用InteractiveSession。
# 可以减少默认Session的传递，特别适用于Shell
sess = tf.InteractiveSession()

#创建两个占位符，数据类型是float。x占位符的形状是[None，784]，即用来存放图像数据的变量，图像有多少张
#是不关注的。但是图像的数据维度有784维。因为MNIST处理的图片都是28*28的大小，将一个二维图像
#展平后，放入一个长度为784的数组中。
#y_占位符的形状类似x，只是维度只有10，因为输出结果是0-9的数字，所以只有10种结构。
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())
# 权重初始化函数,用一个较小的正数来初始化偏置项
# 通过函数的形式定义权重变量。变量的初始值，来自于截取正态分布中的数据。
# 截断正态分布是截断分布(Truncated Distribution)的一种，
# 那么截断分布是什么？截断分布是指，限制变量[Math Processing Error] 取值范围(scope)的一种分布。
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#通过函数的形式定义偏置量变量，偏置的初始值都是0.1，形状由shape定义。
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#定义卷积函数，其中x是输入，W是权重，也可以理解成卷积核，strides表示步长，或者说是滑动速率，包含长宽方向
#的步长。padding表示补齐数据。 目前有两种补齐方式，一种是SAME，表示补齐操作后（在原始图像周围补充0），实
#际卷积中，参与计算的原始图像数据都会参与。一种是VALID，补齐操作后，进行卷积过程中，原始图片中右边或者底部
#的像素数据可能出现丢弃的情况。
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#这步定义函数进行池化操作，在卷积运算中，是一种数据下降采样的操作，降低数据量，聚类数据的有效手段。常见的
#池化操作包含最大值池化和均值池化。这里的2*2池化，就是每4个值中取一个，池化操作的数据区域边缘不重叠。
#函数原型：def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None)。对ksize和strides
#定义的理解要基于data_format进行。默认NHWC，表示4维数据，[batch,height,width,channels]. 下面函数中的ksize，
#strides中，每次处理都是一张图片，对应的处理数据是一个通道（例如，只是黑白图片）。长宽都是2，表明是2*2的
#池化区域，也反应出下采样的速度。
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#定义第一层卷积核。shape在这里，对应卷积核filter。
#其中filter的结构为：[filter_height, filter_width, in_channels, out_channels]。这里，卷积核的高和宽都是5，
#输入通道1，输出通道数为32，也就是说，有32个卷积核参与卷积。
W_conv1 = weight_variable([5, 5, 1, 32])

#偏置量定义，偏置的维度是32.
b_conv1 = bias_variable([32])

#将输入tensor进行形状调整，调整成为一个28*28的图片，因为输入的时候x是一个[None,784]，有与reshape的输入项shape
#是[-1,28,28,1]，后续三个维度数据28,28,1相乘后得到784，所以，-1值在reshape函数中的特殊含义就可以映射程None。即
#输入图片的数量batch。
x_image = tf.reshape(x, [-1,28,28,1])

#把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，
#将2维卷积的值加上一个偏置后的tensor，进行relu操作，一种激活函数，关于激活函数，有很多内容需要研究，在此不表。
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

#对激活函数返回结果进行下采样池化操作。
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积，卷积核大小5*5，输入通道有32个，输出通道有64个，从输出通道数看，第二层的卷积单元有64个。
#卷积核的大小如何确定？ 卷积核的输出通道如何决定。
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#类似第一层卷积操作的激活和池化
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#图片尺寸减小到7x7，加入一个有1024个神经元的全连接层，用于处理整个图片。把池化层输出的张量reshape成一些
#向量，乘上权重矩阵，加上偏置，然后对其使用ReLU激活操作。
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

#将第二层池化后的数据进行变形
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#进行矩阵乘，加偏置后进行relu激活
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#为了减少过拟合，在输出层之前加入dropout
keep_prob = tf.placeholder("float")
#对第二层卷积经过relu后的结果，基于tensor值keep_prob进行保留或者丢弃相关维度上的数据。这个是为了防止过拟合，快速收敛。
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#添加一个softmax层，就像softmax regression一样
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
#最后，添加一个softmax层，就像前面的单层softmax regression一样。softmax是一个多选择分类函数，其作用和sigmoid这个2值
#分类作用地位一样，在我们这个例子里面，softmax输出是10个。
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#实际值y_与预测值y_conv的自然对数求乘积，在对应的维度上上求和，该值作为梯度下降法的输入
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

#下面基于步长1e-4来求梯度，梯度下降方法为AdamOptimizer。
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#首先分别在训练值y_conv以及实际标签值y_的第一个轴向取最大值，比较是否相等
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

#对correct_prediction值进行浮点化转换，然后求均值，得到精度。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#先通过tf执行全局变量的初始化，然后启用session运行图。
sess.run(tf.initialize_all_variables())

#训练
for i in range(1000):
    # 从mnist的train数据集中取出50批数据，返回的batch其实是一个列表，元素0表示图像数据，元素1表示标签值
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        # 计算精度，通过所取的batch中的图像数据以及标签值还有dropout参数，带入到accuracy定义时所涉及到的相关变量中，进行
        # session的运算，得到一个输出，也就是通过已知的训练图片数据和标签值进行似然估计，然后基于梯度下降，进行权值训练。
        train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print ("-->step %d, training accuracy %.4f"%(i, train_accuracy))
    # 此步主要是用来训练W和bias用的。基于似然估计函数进行梯度下降，收敛后，就等于W和bias都训练好了。
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#对测试图片和测试标签值以及给定的keep_prob进行feed操作，进行计算求出识别率。就相当于前面训练好的W和bias作为已知参数。
print ("卷积神经网络测试MNIST数据集正确率: %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```  

### 5.2 英文官方教程源码  

```python  
#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
```  
      

## 6. 参考文献
1.  TensorFlow 官方教程      [TensorFlow官方教程](https://www.tensorflow.org/tutorials/layers)
2.  TensorFlow 中文官方教程  [TensorFlow中文官方教程](http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html)  
3.  SHIHUC 个人博客          [SHIHUC个人博客](https://www.cnblogs.com/shihuc/p/6648130.html)
