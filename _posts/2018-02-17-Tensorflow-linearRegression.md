---
layout: post
title: "Tensorflow 实践——线性回归"
description: "Linear Regression"
categories: [machine learning]
tags: [tensorflow, tensorboard, linear regression]
redirect_from: 
  - /2018/02/18/
---  
* Kramdown table of contents
{:toc .toc}
---
  
  

## 1. 通过线性回归预测线性方程[y = weight * x + bias]的参数 weight, bias  

1. 利用随机数生成入参 x ；  
2. 使用线性方程 y = weight * x + bias, (weight = 0.3, bias = 0.1) 生成训练样本 y；  
3. 将 x 代入线性方程中求得预测值 y_prediction；  
4. 计算预测值 y_prediction 与训练样本 y 的方差；  
5. 利用梯度下降求weight，bias 使得方差最小；  
如下图所示, 随着x_axis 样本点的增加，训练方差收敛到 0。  
![loss收敛](http://images.sailblade.com/%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C2018022803.PNG)  
如下图所示，随着x_axis 样本点的增加，weight收敛到 0.3， bias收敛到 0.1。  
![分布图](http://images.sailblade.com/%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C2018022802.PNG)  
5. 获得线性方程的参数 weight = 0.3, bias = 0.1。  
如下图所示, 随着y_axis 样本点的增加, 参数 weight, bias 收敛到一个区间内。  
![柱状图](http://images.sailblade.com/%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C2018022801.PNG)  


源码如下：
```python
import tensorflow as tf
import numpy as np

with tf.name_scope('data'):
       x_data = np.random.rand(100).astype(np.float32)
       y_data = 0.3*x_data+0.1                               # 待预测的线性方程
       
with tf.name_scope('parameters'):
      with tf.name_scope('weights'):
            weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
            tf.summary.histogram('weight',weight)            # 在 histogram 页面生成‘weight’随样本增加的收敛柱状图
      with tf.name_scope('biases'):
            bias = tf.Variable(tf.zeros([1]))
            tf.summary.histogram('bias',bias)
with tf.name_scope('y_prediction'):
      y_prediction = weight*x_data+bias
with tf.name_scope('loss'):
      loss = tf.reduce_mean(tf.square(y_data-y_prediction))
      tf.summary.scalar('loss',loss)                         # 在 scalars 页面生成‘loss’随样本增加的收敛曲线
optimizer = tf.train.GradientDescentOptimizer(0.5)
with tf.name_scope('train'):
    train = optimizer.minimize(loss)
with tf.name_scope('init'):
    init = tf.global_variables_initializer()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./my_graph", sess.graph)
sess.run(init)

for step  in  range(101):
     sess.run(train)
     rs=sess.run(merged)
     writer.add_summary(rs, step)
```  



## 2. 通过线性回归依据(年龄，体重)预测血脂     
问题：
    训练样本提供了对应年龄，体重下的血脂，根据训练后的参数，推测给定年龄，体重下的血脂。  
    
1. `inputs()`获取训练样本； 
2. `inference()`将训练样本中的年龄，血脂代入线性方程 tf.matmul(X, W) + b 中，获得预测值 y_prediction;
3. `loss()`利用步骤2 的 y_prediction 和训练样本中的 血脂求方差并累计求和 ；(利用广义最小二乘性，方差和累积最小，则函数越接近)  
如下图所示，可以看到随着 x_axis 样本量的增加，方差在逐渐缩小，但相比本章案例1的损失函数，该损失较大 5.3 * power(10, 6), 收敛不明显  
![损失函数](http://images.sailblade.com/%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.PNG)  
4. `train()`利用梯度下降找到最小最近似的参数。  
如下图所示：
    * Bias 随着 x_axis 样本量的增加收敛到 1.01；
    * Weight 随着 x_axis 样本量的增加不会收敛，但是从柱状图可以明显看到分布于 1.6,3.6附近的区域。可能后续需要其它手段进行优化。  

    
![参数收敛曲线图](http://images.sailblade.com/%E5%8F%82%E6%95%B0%E6%94%B6%E6%95%9B.PNG)

![参数收敛柱状图](http://images.sailblade.com/%E5%8F%82%E6%95%B0%E6%94%B6%E6%95%9B2.PNG)

## 3. 对线性回归案例的思考  
   1. 从预测血脂量的案例看，预测值与实际值相差较大，如何评判该模型训练后准确率？ 
      目前想到的只能是从训练样本中抽取测试样本，单独用来评估准确率。
   2. 如果线性模型预测结果不尽如意，后续该如何处理？是尝试换模型，还是换求最小损失的函数或者步长？

## 4. 源码  
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


## 5. 参考文献
1. 《面向机器智能 TensorFlow实践》  Sam Abrahams, Danijar Hafner, Erik Erwitt, Ariel Scarpinelli  
2.  根据年龄，体重预测血脂的代码    [Github 地址](https://github.com/backstopmedia/tensorflowbook)  

