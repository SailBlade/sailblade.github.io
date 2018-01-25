---
layout: post
title: "Jekyll的目录分析"
description: "Build Blog"
categories: [uncategorized]
tags: [technology, jekyll]
redirect_from:
  - /2018/01/23/
---
* Kramdown table of contents
{:toc .toc}
---

# 1. Jekyll 的目录层次鸟瞰：  
![alt text](http://p30p0kjya.bkt.clouddn.com/2018-01-24_005752%20Jekyll.png "Jekyll的目录结构")

# 2. 各目录或文件的意义  
#2.1   _data  

#2.2   _includes 

#2.3   _layouts 

#2.4   _posts  
本目录下存放博客内容，文件名格式为  

> yyyy-mm-dd-blogTitle.md  

如 2018-01-23-technology-Jekyll.md.  

#2.5   _sass  

#2.6   _site  
![alt text](http://p30p0kjya.bkt.clouddn.com/2018-01-24_010916_site.png "_site 的目录结构")  
_site的目录如上所示，从实际的测试来看对应Blog的代码在修改后，Jekyll都会动态的在_site目录里生成新的代码，而此代码直接对应网页的显示。  
如上图所示的 index.html就对应着blog的主页代码，修改对应代码内容，会直接体现在主页上。但是一段时间后，该值会被刷新为默认值。

#2.7   assets 

#2.8   blog 

#2.9   config.yml  
本文件是网站对应的基础配置
home: 配置了主页的标题

#2.10   Gemfile 

#2.11   Gemfile.lock  

#2.12   index.html 

#2.13   search.json  

#2.14   _includes 

#2.15   _layouts 

# 3. Jekyll 的语法剖析:  
Jekeyll涉及到的语法有三种：
## 3.1 Markdown语法  (GitHub方言)

 [github 的 Markdown 语法](https://github.com/github/linguist/blob/master/lib/linguist/languages.yml)

## 3.2 Html  

## 3.3 JavaScript  





