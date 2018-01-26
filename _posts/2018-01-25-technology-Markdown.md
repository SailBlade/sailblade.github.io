---
layout: post
title: "GitHub 的 Markdown语法"
description: "Build Blog"
categories: [uncategorized]
tags: [technology, githbub pages, markdown]
redirect_from:
  - /2018/01/25/
---
* Kramdown table of contents
{:toc .toc}
---
# 1. 标题格式
章节标题可以通过 \# 指示，最多可以到第6层 \#\#\#\#\#\#。

># \# 第一级标题  
##  \#\# 第二级标题  
###### \#\#\#\#\#\# 第六级标题

# 2. 格式文本

|   格式     |     语法     |      例子   |     输出   |      
| ---------- | ------------ | ----------- | -----------| 
|   粗体     | \*\* \*\* 或 \_\_ \_\_ | \*\*粗体\*\* | **粗体** |  
|   斜体     | \* \* 或 \_  \_ | \*斜体\* | *斜体* |  
|  删除线    | ~~ ~~ | \~\~ 斜体\~\~  | ~~ 斜体~~ |  
| 粗斜体  | \*\* \*\* 和 \_\_ | \*\*\_粗斜体\_\*\* | **_粗斜体_**| 

# 3. 引用文本  
可以通过使用 \> 来引用文本  
\> 爱因斯坦隔壁家王老二说过，知识使人有钱

> 爱因斯坦隔壁家王老二说过，知识使人有钱

# 4. 引用代码  
可以通过使用 `Code`，完成单个代码内容的引用  

爱因斯坦隔壁家王老二说过: \`知识使人有钱\`   
 
爱因斯坦隔壁家王老二说过: `知识使人有钱`  

当然也可以通过 \`\`\`Code\`\`\`完成代码段的引用

>\`\`\`  
git status  
git add  
git commit  
\`\`\`

```
git status  
git add  
git commit  
```

# 5. 超级链接
\[文本内容\]\(https://pages.github.com/\)  

# 6. 图片链接
\[图片标题\](\图片链接\)
> \![alt text\]\(/path/img.jpg "title"\)  

# 7. 列表
列表的符号分别有 \- 或 \*
可以通过数字表示列表
1. 首先  
2. 其次  
3. 最后  


# 8. 插入表格
表格格式如下，需要在表头和表尾各插入一个空行。  
| First Header  | Second Header |  
| ------------- | ------------- |  
| Content Cell  | Content Cell  |  
| Content Cell  | Content Cell  |  


| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |

注：表格前后都需要空行



# 9. 代码语法高亮
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

# 10. Markdown 特殊符号  

| Markdown 特殊符号  |    用途     |  
| ------------- | ------------- |  
|     \\        | 转义字符，取消Markdown语法解析器解析后续符号     |  
|     \`        | TAB键上的反引号，在单句里使用可以标注命令或代码  |  
|     &emsp     | 全角的缩进                                       |
|     &ensp     | 半角的缩进                                       |


# 11. 参考文档  
[GitHub Help](https://help.github.com/articles/basic-writing-and-formatting-syntax/)