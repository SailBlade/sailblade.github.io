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
在完成数据流图生成后，在Anaconda Prompt中首先激活TensorFlow的环境，然后执行命令 `tensorboard --logdir="C:\tensorflowLearning" --port 1234 --debug`  


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

### TensorFlow Session.Run()  
  1. fetches 参数  
     fetches 参数接收任意的数据流图元素(Op 或 Tensor对象)，后者指定了用户希望执行的对象。  
	 如果请求对象为Tensor对象，则run()的输出将为NumPy数组；  
	 如果请求对象为Op，则输出将为None。  
  2. feed_dict 参数  
     本参数用于覆盖数据流图中的Tensor对象值，它需要Python字典对象做为输入。字典中的‘键’为指向应当被覆盖的Tensor对象的句柄，而字典的‘值’可以是数字，字符串，列表，NumPy数组。

```python  
a = tf.add(2,5)
b = tf.mul(a,3)

sess = tf.Session()  
# 定义一个字段，将a的值替换为15  
```

	 
### placeholder
占位符，类似定义一个变量，等待传入数据。  
为了给占位符传值，需要使用 Session.run()中的 feed_dict参数  
	 
```python  
import tensorflow as tf  
import numpy as np

# 创建一个长度为2，数据类型为 int32的占位向量  
a = tf.placeholder(tf.int32, shape=[2], name = "my_input") # 调用placeholer()时，dtype参数必须指定， 而shape参数可选  

# 将给占位向量视为其它任意 Tensor对象，加以使用
b = tf.reduce_prod(a, name ="prod_b")
c = tf.reduce_sum(a,name = "sum_c")

# 完成数据流图的定义
d = tf.add(b,c, name = "add_d")

replace_dict = {a : 15}
sess.run(b,feed_dict = replace_dict)   # 返回值45
sess = tf.Session()  
input_dict = {a: np.array([5,3],dtype = np.int32)}  
sess.run(d,feed_dict = input_dict)  
```

### Variable 对象
1. 创建 Variable 对象

```python  
my_var = tf.Variable(3, name = "my_variable")
add = tf.add(5, my_var)
zeros = tf.zeros([2,2]) # 2 *2的零矩阵  
ones  = tf.ones([6])    # 长度为6的全1向量  
uniform = tf.random_uniform([3,3,3],minval = 0, maxval = 10) # 3*3*3 d的张量，其元素服从 0~10的均匀分布  
normal  = tf.random_normal([3,3,3],mean = 0.0, stddev = 2.0) # 3*3*3 d的张量，其元素服从0均值，标准差为2的正态分布  
random_var = tf.Variable(tf.truncated_normal([2,2]))  
```

2. Variable 对象的初始化  
在Session对象中对所有 Variable 对象初始化  

```python  
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
```

只需要对一个Variable对象初始化  

```python  
var1 = tf.Variable(0,name = "initialize_me")  
var2 = tf.Variable(1,name = "no_initialization")  
init = tf.initialize_variables([var1],name = "init_var1")
sess = tf.Session()
sess.run(init)  
```  

3. Variable 对象的修改  

```python  
# 创建一个初值为1的Variable对象
my_var = tf.Variable(1)
# 创建一个Op，使其在每次运行时都将 Variable对象乘以 2
my_var_times_two = my_var.assign(my_var * 2)
# 初始化 Operation  
init = tf.initialize_all_variables()  
# 启动一个会话  
sess = tf.Session()
# 初始化Variable对象  
sess.run(init)
# 将Variable对象乘以2，并返回  
sess.run(my_var_times_two) # 输出 2
# 将Variable对象乘以2，并返回  
sess.run(my_var_times_two)  # 输出 4
```  

###  name scope(名称作用域)  
name scope 的用途是将Op划分到 Block中进行封装，获得更好的可视化效果。  

```python  
import tensorflow as tf

with tf.name_scope("Scope_A"):
    a = tf.add(1, 2, name="A_add")
    b = tf.multiply(a, 3, name="A_mul")

with tf.name_scope("Scope_B"):
    c = tf.add(4, 5, name="B_add")
    d = tf.multiply(c, 6, name="B_mul")

e = tf.add(b, d, name="output")
writer = tf.summary.FileWriter('./my_graph', graph = tf.get_default_graph())
writer.close()

```  


![nameScope](http://p30p0kjya.bkt.clouddn.com/nameScope2.PNG)  


###  数据流图
```python  
import tensorflow as tf
import numpy as np

# Explicitly create a Graph object
graph = tf.Graph()

with graph.as_default():
    with tf.name_scope("variables"):
        # Variable to keep track of how many times the graph has been run
        global_step = tf.Variable(0, dtype=tf.int32, name="global_step")

        # Increments the above `global_step` Variable, should be run whenever the graph is run
        increment_step = global_step.assign_add(1)

        # Variable that keeps track of previous output value:
        previous_value = tf.Variable(0.0, dtype=tf.float32, name="previous_value")

    # Primary transformation Operations
    with tf.name_scope("exercise_transformation"):
        # Separate input layer
        with tf.name_scope("input"):
            # Create input placeholder- takes in a Vector
            a = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_a")

        # Separate middle layer
        with tf.name_scope("intermediate_layer"):
            b = tf.reduce_prod(a, name="product_b") # 计算指定维度的元素相乘的总和
            c = tf.reduce_sum(a, name="sum_c")      # 计算指定维度的元素总和

        # Separate output layer
        with tf.name_scope("output"):
            d = tf.add(b, c, name="add_d")
            output = tf.subtract(d, previous_value, name="output")
            update_prev = previous_value.assign(output)

    # Summary Operations tf.summary.scalar().
    with tf.name_scope("summaries"):
        tf.summary.scalar("output_summary", output)  # Creates summary for output node
        tf.summary.scalar("prod_summary", b)
        tf.summary.scalar("sum_summary", c)

    # Global Variables and Operations
    with tf.name_scope("global_ops"):
        # Initialization Op
        init = tf.initialize_all_variables()
        # Collect all summary Ops in graph
        merged_summaries =  tf.summary.merge_all()

# Start a Session, using the explicitly created Graph
sess = tf.Session(graph=graph)

# Open a SummaryWriter to save summaries
writer = tf.summary.FileWriter('./my_graph', graph)

# Initialize Variables
sess.run(init)


def run_graph(input_tensor):
    """
    Helper function; runs the graph with given input tensor and saves summaries
    """
    feed_dict = {a: input_tensor}
    output, summary, step = sess.run([update_prev, merged_summaries, increment_step], feed_dict=feed_dict)
    writer.add_summary(summary, global_step=step)


# Run the graph with various inputs
run_graph([2, 8])
run_graph([3, 1, 3, 3])
run_graph([8])
run_graph([1, 2, 3])
run_graph([11, 4])
run_graph([4, 1])
run_graph([7, 3, 1])
run_graph([6, 3])
run_graph([0, 2])
run_graph([4, 5, 6])

# Writes the summaries to disk
writer.flush()

# Flushes the summaries to disk and closes the SummaryWriter
writer.close()

# Close the session
sess.close()

# To start TensorBoard after running this file, execute the following command:
# $ tensorboard --logdir='./improved_graph'
```

![exercise_transformation](http://p30p0kjya.bkt.clouddn.com/exercise_transformation.PNG)
![Variable](http://p30p0kjya.bkt.clouddn.com/variables.PNG)
![Summary](http://p30p0kjya.bkt.clouddn.com/Summary.PNG)

## 参考文献
1. 《面向机器智能 TensorFlow实践》  Sam Abrahams, Danijar Hafner, Erik Erwitt, Ariel Scarpinelli  
