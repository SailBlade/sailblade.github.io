---
layout: post
title: "Tensorflow 实践——对数几率回归"
description: "logistic Regression"
categories: [machine learning]
tags: [tensorflow, tensorboard, logistic regression]
redirect_from: 
  - /2018/03/01/
---  
* Kramdown table of contents
{:toc .toc}
---


## 1. 通过对数几率回归预测泰坦尼克号上乘客的生存率

  
## 训练框架  
![对数几率回归](http://p30p0kjya.bkt.clouddn.com/%E5%AF%B9%E6%95%B0%E5%9B%9E%E5%BD%92%E5%AD%A6%E4%B9%A0.PNG)

## 3. 对数几率回归案例的思考  
1. 本案例对训练结果进行评估。根据features 预测生存率并给出和真实生存率的预测成功率。  
2. 对数几率回归其实是线性回归的变形，这点从 inference() 即可看出。  
3. 如何设置数据的行列，目前比较困难。因为在函数定义的时候打印是无法看出数据的格式，单步调试也无实际意义。估计有其它方法。  
4. 从 “train.csv” 中读取数据的处理比较奇特，不理解是如何处理的，后续如果能看到API应该会大有裨益。  
5. 很想通过数据图的格式观察分布，但是目前没有。  


## 源码  

```python  
import tensorflow as tf
import os

W = tf.Variable(tf.zeros([5,1]), name='weights')
b = tf.Variable(0., name="bias")

def combine_inputs(X):
    return tf.matmul(X,W) + b

def inference(X):
    return tf.sigmoid(combine_inputs(X))

def loss(X, Y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=combine_inputs(X)))

def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(), file_name)])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

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
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = \
        read_csv(100, "train.csv", [[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]])

    # convert categorical data
    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))

    gender = tf.to_float(tf.equal(sex, ["female"]))

    # Finally we pack all the features in a single matrix;
    # We then transpose to have a matrix with one example per row and one feature per column.
    print(is_first_class, is_second_class, is_third_class, gender, age)
    # 可以使用 tf.stack 方法将所有布尔值打包进单个张量中
    print(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))
    print(tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age])))

    # 最终将所有特征排列在一个矩阵中，然后对该矩阵转置，使其每行对应一个样本，每列对应一种特征
    features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))
    survived = tf.reshape(survived, [100, 1])

    return features, survived


def train(total_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):

    predicted = tf.cast(inference(X) > 0.5, tf.float32)

    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))

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

    import time
    time.sleep(5)

    coord.request_stop()
    coord.join(threads)
    sess.close()
```  
      

## 4. 参考文献
1. 《面向机器智能 TensorFlow实践》  Sam Abrahams, Danijar Hafner, Erik Erwitt, Ariel Scarpinelli  
2.  根据年龄，体重预测血脂的代码    [Github 地址](https://github.com/backstopmedia/tensorflowbook)  
3.  交叉熵在分类分类问题的优势      [交叉熵在分类分类问题的优势](https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/)
4.  TensorFlow 官方对交叉熵的说明   [tensorflow 谈交叉熵](http://colah.github.io/posts/2015-09-Visual-Information/)
