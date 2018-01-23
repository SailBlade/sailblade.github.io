---
layout: post
title: "Jekyll建立自己的Blog"
description: "Build Blog"
categories: [uncategorized]
tags: [technology, jekyll]
redirect_from:
  - /2018/01/22/
---
* Kramdown table of contents
{:toc .toc}
---

# 1. Jekyll搭建博客方面的技术
## 2018-01-22  使用Jekyll 模板simple-texture搭建博客
按照如下链接 [simple-texture]([https://github.com/yizeng/jekyll-theme-simple-texture](https://github.com/yizeng/jekyll-theme-simple-texture) "Title") 搭建了一个博客系统，看起来还比较舒服，记录下来，方便学习。  
1. 首先需要配置本地的Ruby环境；由于我使用的是win10，网上很多都推荐linux，所以一路心惊胆颤的完成了环境配置，并成功安装成功jekyll。具体如下
## Ruby安装
在windows下，可以使用Rubyinstaller安装。
## RubyDevKit安装  
从这里下载DevKit，注意版本要与Ruby版本一致  
下载下来的是一个很有意思的sfx文件，如果你安装有7-zip，可以直接双击，它会自解压到你所选择的目录。  
解压完成之后，用cmd进入到刚才解压的目录下，运行下面命令，该命令会生成config.yml。  

>$ruby dk.rb init  

config.yml文件实际上是检测系统安装的ruby的位置并记录在这个文件中，以便稍后使用。
执行如下命令，执行安装：  

> $ruby setup.rb  

或是

> $ruby dk.rb install

##Rubygems安装
Rubygems是类似Radhat的RPM、centOS的Yum、Ubuntu的apt-get的应用程序打包部署解决方案。Rubygems本身基于Ruby开发，在Ruby命令行中执行。我们需要它主要是因为jekyll的执行需要依赖很多Ruby应用程序，如果一个个手动安装比较繁琐。jekyll作为一个Ruby的应用，也实现了Rubygems打包标准。只要通过简单的命令就可以自动下载其依赖。
解压后，用cmd进入到解压后的目录，执行命令即可：

>$ruby setup.rb  

就像yum仓库一样，仓库本身有很多，如果希望加快应用程序的下载速度，特别绕过“天朝”的网络管理制度，可以选择国内的仓库镜像，taobao有一个：https://ruby.taobao.org/。配置方法这个链接里面很完全。

##安装jekyll

>$gem install jekyll

jekyll依赖的组件会自动下载安装

## 测试jekyll服务
安装好之后就可以测试我们的环境了。用cmd进入到上一节我们创建的目录，执行下面命令：

>$jekyll serve --safe --watch

jekyll此时会在localhost的4000端口监听http请求，用浏览器访问http://localhost:4000/index.html，之前的页面出现了！


## 下载github上选中的jekyll模板

>git clone git@github.com:[YOUR_USERNAME]/jekyll-theme-simple-texture.git

Delete starter-kit folder and jekyll-theme-simple-texture.gemspec file (they're for people installing via gem)

Install Bundler if haven't done so.

> gem install bundler

Update the Gemfile to look like the following:

> source "https://rubygems.org"
gem 'jekyll', '= 3.5.2' # locked in to be consistent GitHub Pages.  
group :jekyll_plugins do  
  gem 'jekyll-feed'  
  gem 'jekyll-redirect-from'  
  gem 'jekyll-seo-tag'  
  gem 'jekyll-sitemap'  
end  

Run bundle install to install dependencies.  
Run Jekyll with  

> bundle exec jekyll serve  

Hack away at http://localhost:4000!  

