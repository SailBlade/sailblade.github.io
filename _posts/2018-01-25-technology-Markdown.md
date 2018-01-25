---
layout: post
title: "github pages的 Markdown语法"
description: "Build Blog"
categories: [uncategorized]
tags: [technology, githbub pages, markdown]
redirect_from:
  - /2018/01/25/
---
* Kramdown table of contents
{:toc .toc}
---
# github pages的 Markdown语法  

####  插入表格
| First Header  | Second Header |  
| ------------- | ------------- |  
| Content Cell  | Content Cell  |  
| Content Cell  | Content Cell  |  


| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |

注：表格前后都需要空行


####  插入图片  

> \![alt text\]\(/path/img.jpg "title"\)  

####  代码语法高亮显示(GitHub markd 特性语法)
代码语法高亮显示支持的语言有很多种，最常用的有 \{C, C#，C++, ruby, Python等\}，其余语言可参考 [github 语法高亮](https://github.com/github/linguist/blob/master/lib/linguist/languages.yml)

> \`\`\`ruby  
require 'redcarpet'  
markdown = Redcarpet.new("Hello World!")  
puts markdown.to_html  
\`\`\`

对应高亮显示后的格式如下:
```ruby  
require 'redcarpet'  
markdown = Redcarpet.new("Hello World!")  
puts markdown.to_html  
```  