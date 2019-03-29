---
layout: post
title: "Pool().join() 无法等到子进程结束"
description: "python"
categories: [uncategorized]
tags: [python]
redirect_from:
  - /2019/03/30/
---
* Kramdown table of contents
{:toc .toc}
---
    
## 简介    
调试多任务期间，`Pool.join()` 无法接收到创建的异步进程的结束信号量，导致挂死，经过彻夜排查，终于发现是发现在包含的文件中定义了一个 `conn` 做为全局变量，并未关闭。自己挖的一口大坑啊！     

## 背景     
pycharm 调试python多任务进程，使用了如下代码：    

```python   
import maintainDb as maintainDb 
import os
import time
from multiprocessing import Pool

def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(5)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))


if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

```    
等啊等，等啊等，只出现了如下结果：    

```    
Parent process 18868.
Waiting for all subprocesses done...
Run task 0 (16700)...
Run task 1 (26808)...
Run task 2 (36372)...
Run task 3 (21204)...
Task 0 runs 5.00 seconds.
Run task 4 (16700)...
Task 1 runs 5.00 seconds.
Task 2 runs 5.00 seconds.
Task 3 runs 5.00 seconds.
Task 4 runs 5.00 seconds.
```

此处略过遍读python各种参考资料的痛苦过程。今日灵光乍现，重新创建了一个新的工程后，可以拿到正确答案。    

```    
Parent process 32504.
Waiting for all subprocesses done...
Run task 0 (34456)...
Run task 1 (9964)...
Run task 2 (15220)...
Run task 3 (35800)...
Task 0 runs 5.00 seconds.
Run task 4 (34456)...
Task 1 runs 5.00 seconds.
Task 2 runs 5.00 seconds.
Task 3 runs 5.00 seconds.
Task 4 runs 5.00 seconds.
All subprocesses done.

Process finished with exit code 0
```

这个愁死个人了，于是乎遍寻代码，并未发现任何蹊跷。峰回路转之际，发现了 pycharm的并发调试功能 `Concurrency Diagram for \"filename\"`, 点击运行：    
![](http://images.sailblade.com/%E5%A4%9A%E4%BB%BB%E5%8A%A1%E8%B0%83%E8%AF%95%E7%AA%97%E5%8F%A3.png)     
真是强大的功能，直接显示出来 tushareinterface.py:12 创建了新的进程，太令人振奋了。检查代码，一个全局变量的 conn定义导致，删除后解决问题。    
 
```python    
# coding: UTF-8
import tushare as ts
import pandas as pd
import numpy as np
import datetime

from threading import Thread
import functools
import time

global cons
cons = ts.get_apis()

```
    
     
      
