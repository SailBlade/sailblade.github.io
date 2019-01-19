---
layout: post
title: "matplotlib 绘图技巧"
description: "python"
categories: [uncategorized]
tags: [technology, githbub pages, markdown]
redirect_from:
  - /2019/01/01/
---
* Kramdown table of contents
{:toc .toc}
---
##  基本元素解释   
      
![](http://images.sailblade.com/2019011901071369114.png)    
    
### Figure    
它代表整个图像。包含了所有的子图，图相关的一些元素(标题，图例等等)，画布。一个`figure`可能会有多个`Axes`子图，但至少有一个。
用pyplot画图的方法如下:
    
```python    
fig = plt.figure()  # an empty figure with no axes
fig, ax_lst = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes
```   
    
### Axes    
一般来说，这就是我们理解的单个图，它是指包含数据部分的图像区域。一个`Figure`里面会有多个`Axes`，但是一个`Axes`只可能包含在一个`figure`中。
一个`Axes`中包含两个坐标轴`Axis`(3D下包含三个`Axis`)。`Axis`的设置需要考虑到数据的范围(`set_xlim()`和`set_ylim()`)。
每个图像`Axes`可以有一个标题(`set_title()`)，和X轴的名称(`set_xlabel()`), Y轴的名称(`set_ylabel()`)。    
    
### Axis    
坐标轴`Axis`是一个数字型对象，它们限定了图像的范围并生成坐标轴网格以及对应标签。坐标轴网格对应位置通过`Locator`对象生成，坐标轴网格标签通过`Formatter`对象生成。
恰当的使用`Locator`和`Formatter`可以很好的控制坐标轴网格以及标签。    
    
### Artist    
你能在`figure`看到的所有东西都称之为`artist` (包含`figure`，`Axes`，`Axis`)。它还包含了`Text`，`Line2D`,`collection`，`Patch`对象等等。
Basically everything you can see on the figure is an artist (even the Figure, Axes, and Axis objects). 当`figure`被绘制时，所有的`artist`会被绘制在画布(`canvas`)上。
大多数`Artist`与`Axes`紧密相关; 一个`Artist`不能被多个`Axes`分享，也不能被移动另外一个`Axes`中。    
    
### Colors    
在matplotlib中可以使用下面的颜色。    
![](http://images.sailblade.com/sphx_glr_named_colors_001.png)    
![](http://images.sailblade.com/sphx_glr_named_colors_002.png)    
![](http://images.sailblade.com/sphx_glr_named_colors_003.png)    