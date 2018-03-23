---
layout: post
title: "Tensorflow 实践——对新样本最终层的再训练"
description: "Transfer learning"
categories: [machine learning]
tags: [tensorflow, Transfer learning]
redirect_from: 
  - /2018/03/22/
---  
* Kramdown table of contents
{:toc .toc}
---

现代识别模型拥有上百万的参数，以及长达数周的训练。
Transfer learning 是对一组类型全训练的便捷技术，
如ImageNet。并且可以使用已经存在的权重识别新类。
在本例中，我们涂鸦中重训练最终层。

## 1. 根据花的图片进行分类  
本脚本加载了pre-trained Inception v3 mode，
移去了老的最高层，并且依据下载的花的照片训练了新层。
在原始的ImageNet识别网中没有任何对于花的训练。
transfer learning的的奇特之处在于已经被训练好的底层
可以不经适配直接识别其它对象。

## 2. 瓶颈  
本脚本的第一步操作是分析磁盘中的图片，并对每张图片计算瓶颈值。 
瓶颈是我们在最终输出层之前实际分类时的非官方说法。
倒数第二层已经被训练并输出了一组可以被分类器用来对所有
被要求分的类很好的输出集。这代表着它是一个对一张图像的汇总和说明。
它用非常少的值已经足够分类器进行准确的判断。我们的最终层
重训练后可以工作在新的分类的原因是它在1000个种类中识别
的信息同样可以应用在新种类的识别上。
由于每个图片在训练中使用多次并且每次计算瓶颈都需要占用
较多时间，所以可以通过降瓶颈值储存在在磁盘里而不用重复计算。
默认存放在 `/tmp/bottleneck`路径下，如果你重新运行脚本，
它们会被重新使用而不必浪费这部分时间。

## 3. 训练
一旦瓶颈计算完成，则识别网络的最高层训练开始。
你可以看到每步训练的输出序列，训练的正确率，有效的正确率，
交叉熵。训练正确率展示了使用当前训练集识别的标签的正确率。
有效正确率是从不同集合里随机选择识别对象的正确率。
它们之间不同的关键之处在于训练正确率基于神经网络能够学习的图像，
所以神经网络可能会过拟合干扰信息。
一个真正测量神经网络性能的的指标不应该包含训练数据，所以
我们可以使用有效正确率评估。
如果训练正确率太高而有效正确率较低，这表明识别网络过拟合，
并且在训练图片中记录的特征并不十分有效。
交叉熵是一个损失函数，它可以告诉我们学习过程达到多好了。
训练目标是尽可能的减少损失，所以如果学习工作的损失函数保持
趋势向下，则可以忽略短期的干扰。

默认的基本可以训练4000步，每一个步随机从训练集中选出
10个图像。从缓冲区里找到对应的瓶颈值，并且灌入最终层
获得预测值。这些预测值将通过对比实际标签来通过反向
传播(back-propagation)调整最终层的权重。
做为过程你可以观察到正确率的提升，并且所有训练步数
完成以后，最终的测试正确率可以通过运行非训练和非有效的
图片获得。测试评估是对训练模型分类任务最好的评估。
尽管正确率在训练过程中会随机浮动，但都会在90%和95%之间。
正确率基于被完全训练模型后的正确标签在测试集中的比例。

## 4. 使用重训练模型   
本脚本将会使用 Inception v3 神经网络和被新种类训练的最后一层。
默认放在 /tmp/output_graph.pb, 并且将标签单独存放在一个文本文件，
/tmp/output_labels.txt。这些都是C++和Python图片识别类可以被读取的，
所以你可以很快开始你的新模型。从你替换最高层开始，你需要在脚本中
指定新名字，例如使用标志 `--output_layer=final_result`。
这有个例子，如何使用重训练的计算图运行标签图。
```python
python tensorflow/examples/label_image/label_image.py \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--input_layer=Mul \
--output_layer=final_result \
--input_mean=128 --input_std=128 \
--image=$HOME/flower_photos/daisy/21652746_cc379e0eea_m.jpg
```
你将看到一组花名标签，大多是情况菊花在前。你可以替换`--image`
参数用你自己的图片。  
如果你希望用你自己的Python图片，则上面的 `label_image script `
是个合理的起点。`label_image`目录也包含C++代码，你可以使用模板
将TensorFlow整合到你的程序里。  
如果你认为默认的 Inception v3 model太大或是太慢，可以参考
Other Model Architectures section 选择加速你的网络。  



      

## 6. 参考文献
1.  TensorFlow 官方教程      [TensorFlow官方教程](https://www.tensorflow.org/tutorials/layers)
2.  TensorFlow 中文官方教程  [TensorFlow中文官方教程](http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html)  
3.  SHIHUC 个人博客          [SHIHUC个人博客](https://www.cnblogs.com/shihuc/p/6648130.html)
