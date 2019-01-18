---
layout: post
title: "基于 decision stump 的 Adaboost"
description: "ML"
categories: [uncategorized]
tags: [technology, githbub pages, markdown]
redirect_from:
  - /2019/01/01/
---
* Kramdown table of contents
{:toc .toc}
---
##  前言
从目前学习到的机器学习方法中，Adaboost 自适应学习看起来是第一个接触到的具有很强适应性的一个分类算法。毕竟林轩田老师靠它拿下了一届机器学习比赛的冠军。   
断断续续耗在Adaboost上的时间已近数月。从最后的调试结果看确实难度不大，但是对PPT的理解偏差多次代入坑里，花了很大的代价才爬了处理，有一些教训，希望以后可以汲取。

1. 码代码前最好还是出个方案。避免编写代码时会对算法的本义产生误解，毕竟磨刀不误砍柴功；    
2. 重视数据的可视化。在前期无法准确领会讲义内容时，一个可以良好回馈学习信息的图形，可以更快的帮助定位问题。


##  AdaBoost的原理

1. AdaBoost 算法通过弱分类算法(本例中的 decision stump) 找到 E_in，以及对应 E_in的对应训练目标 s, theta；    
2. 对于不满足步骤1的训练数据权重提高,而对于满足步骤1的训练数据权重减少；     
3. 获得本次训练的权重 alpha；
4. 重复1 ~ 3 直至满足训练次数；
5. 根据每次训练的 g(x) * alpha 相加所得 G(x);
6. 利用G(x) 验证训练集中的所有数据获得 E_in。

##  方案设计    
   
![](http://images.sailblade.com/decisionStump01191.png)    

	
##  结果验收    
	
![](http://images.sailblade.com/0119train%200.png)    

![](http://images.sailblade.com/0119train%201.png)    

![](http://images.sailblade.com/0119train%202.png)    

![](http://images.sailblade.com/0119train%203.png)    

![](http://images.sailblade.com/0119train%204.png)    

![](http://images.sailblade.com/0119train%205.png)   

![](http://images.sailblade.com/0119train%2019.png)   