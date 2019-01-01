---
layout: post
title: "Git 技巧汇总"
description: "Git"
categories: [Git]
tags: [Git]
redirect_from: 
  - /2018/04/08/
---  
* Kramdown table of contents
{:toc .toc}
---

## 如何查看 Git 跟踪的文件  

> git ls-files  

## 如何删除对某些文件的跟踪  

1. 在Git根目录下的 `.gitignore` 中加入需要取消跟踪的文件或者文件夹  

> /picture/*  

2. 删除Git的缓存，重新添加跟踪文件夹

> git rm -r --cached .
git add .
git commit -m 'update .gitignore'

3. 当工作目录无需更新 `git pull` 有冲突时

> git fetch --all  
  git reset --hard origin\master  
  git pull  
  



