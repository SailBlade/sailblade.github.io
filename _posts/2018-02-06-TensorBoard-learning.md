---
layout: post
title: "TensorBoard 学习"
description: "感恩"
categories: [life]
tags: [tensorflow,tensorboard]
redirect_from:
  - /2018/02/13/
---  
  
## 启动TensorBoard  
在完成数据流图生成后，在训练模型的同级目录输入 `tensorboard --logdir="C:\tensorflowLearning" --port 1234 --debug`  


```python  
import tensorflow as tf

a = tf.constant(5,name = 'input_a')  
b = tf.constant(3,name = 'input_b')

c = tf.multiply(a,b,name = 'mul_c')
d = tf.add(a,b,name = 'add_d')
e = tf.add(c,d,name = 'add_e')

sess = tf.Session()
output = sess.run(e)
writer = tf.summary.FileWriter('./my_graph',sess.graph)

writer.close()
sess.close()
```  
对应的数据流图如下  
![数据流图](http://p30p0kjya.bkt.clouddn.com/tensorboardLearning0213.PNG)  
  TensorFlow Operation 也称 Op, 是一些利用 Tensor对象执行运算的节点，为创造Op，需要在Python中调用其构造方法。调用时，需要传入计算机所需的所有Tensor参数以及对应的Op的属性。
Python 构造方法将返回一个指向所创建Op的输出的句柄。

### name_scope
如果希望在一个数据流图中对不同Op复用相同的name参数，则可以利用name_scope 将这些运算组织在一起，实现封装。  

### TensorFlow Session  
  Session类负责数据流图的执行，构造方法 tf.Session()可以接受如下3个可选参数  
  * target 指定了所使用的执行引擎。主要用于分布式设置中连接不同的实例。  
  * graph 参数指定了将要在Session对象中加载的Graph对象，其默认值为 None，表示使用当前的默认数据流图  
  * config 参数允许用户执行配置 Session 对象所需的选项，如限制 CPU 或 GPU的使用数目，为数据流图设置优化参数及日志选项等。  



## 参考文献
1. 《面向机器智能 TensorFlow实践》  Sam Abrahams, Danijar Hafner, Erik Erwitt, Ariel Scarpinelli  
