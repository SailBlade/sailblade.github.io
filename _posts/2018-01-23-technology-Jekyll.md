---
layout: post
title: "Jekyll 如何运行"
description: "Build Blog"
categories: [uncategorized]
tags: [technology, jekyll]
redirect_from:
  - /2018/01/23/
---
* Kramdown table of contents
{:toc .toc}
---

# 1. Jekyll 的目录结构  

>  
.  
├── _config.yml  
├── _data  
|   └── members.yml  
├── _drafts  
|   ├── begin-with-the-crazy-ideas.md  
|   └── on-simplicity-in-technology.md  
├── _includes  
|   ├── footer.html  
|   └── header.html  
├── _layouts  
|   ├── default.html  
|   └── post.html  
├── _posts  
|   ├── 2007-10-29-why-every-programmer-should-play-nethack.md  
|   └── 2009-04-26-barcamp-boston-4-roundup.md  
├── _sass  
|   ├── _base.scss  
|   └── _layout.scss  
├── _site  
├── .jekyll-metadata  
└── index.html # can also be an 'index.md' with valid YAML Frontmatter  


# 2. Jekyll 目录作用  

|            FILE / DIRECTORY                |	                           DESCRIPTION                                               |  
| ------------------------------------------ | --------------------------------------------------------------------------------------|  
|_config.yml |  存储基本配置。 Many of these options can be specified from the command line executable but it’s easier to specify them here so you don’t have to remember them. |  
|_drafts     |  存储未发布的博客草稿。Drafts are unpublished posts. The format of these files is without a date: title.MARKUP. Learn how to work with drafts.|  
|_includes   |  These are the partials that can be mixed and matched by your layouts and posts to facilitate reuse. The liquid tag  { % include file.ext % }   can be used to include the partial in _includes/file.ext.|  
|_layouts    |  These are the templates that wrap posts. Layouts are chosen on a post-by-post basis in the YAML Front Matter, which is described in the next section. The liquid tag  `{ { content } }` is used to inject content into the web page. |  
|_posts      |  Your dynamic content, so to speak. The naming convention of these files is important, and must follow the format: YEAR-MONTH-DAY-title.MARKUP. The permalinks can be customized for each post, but the date and markup language are determined solely by the file name.|  
|_data       |  Well-formatted site data should be placed here. The Jekyll engine will autoload all data files (using either the .yml,  .yaml, .json or .csv formats and extensions) in this directory, and they will be accessible via `site.data`. If there's a file  members.yml under the directory, then you can access contents of the file through site.data.members.|  
|_sass       |  These are sass partials that can be imported into your main.scss which will then be processed into a single stylesheet  main.css that defines the styles to be used by your site.|  
|_site       |  This is where the generated site will be placed (by default) once Jekyll is done transforming it. It’s probably a good idea to add this to your .gitignore file.|  
|.jekyll-metadata | This helps Jekyll keep track of which files have not been modified since the site was last built, and which files will need to be regenerated on the next build. This file will not be included in the generated site. It’s probably a good idea to add this to your .gitignore file.|  
| index.html or index.md and other HTML, Markdown, Textile files |  Provided that the file has a YAML Front Matter section,  it will be transformed by Jekyll. The same will happen for any .html, .markdown,  .md, or .textile file in your site’s root directory or directories not listed above.|  
| Other Files/Folders | Every other directory and file except for those listed above—such as css and images folders,  favicon.ico files, and so forth—will be copied verbatim to the generated site. There are plenty of sites already using Jekyll if you’re curious to see how they’re laid out. | 


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

 [github 的 Markdown 语法](http://sailblade.com/blog/2018/01/25/technology-Markdown/)

## 3.2 Html  

## 3.3 JavaScript  

# 4. 参考资料
[Jekyll 官网](https://jekyllrb.com/)



