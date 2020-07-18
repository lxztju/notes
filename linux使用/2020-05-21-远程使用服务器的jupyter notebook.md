---
layout: post
title: "远程使用服务器的jupyter notebook"
date: 2020-05-21
description: "linux使用"
tag: linux使用 
--- 

## 1、在服务器端设置安装jupyter,并配置

1、安装jupyter， 使用如下的命令直接安装
```
conda install jupyter
```

2、生成config文件
```
jupyter notebook --generate-config
```

3、生成密码

```
#进入python
#然后输入如下的命令
from notebook.auth import passwd
passwd()
#然后输入两次密码，就会得到sha1
#输入两次密码
#Enter password: 
#Verify password: 
#然后出现
#'sha1:xxxxxxxxxxxxxxxxx'
```
然后将带引号的'sha1:xxxxxxxx'复制下来

4、修改之前生成的config文件

首先进入config文件
```
vim ~/.jupyter/jupyter_notebook_config.py
```
然后进行如下修改（直接粘贴）：
```
c.NotebookApp.ip='*' #表示侦听所有的ip
c.NotebookApp.password = u'sha:xxxxxx'#刚才复制的
c.NotebookApp.open_browser = False
c.NotebookApp.port =8123 #随便指定一个
```

## 修改xshell

文件-打开-选择服务器-右击属性-隧道： 
![](https://raw.githubusercontent.com/lxztju/lxztju.github.io/master/blog_images/remote_jupyter1.PNG)

点击添加，然后在目标端口处填入上边config中的8123端口，侦听端口随便填，例如填写8123（本地打开时使用）
![](https://raw.githubusercontent.com/lxztju/lxztju.github.io/master/blog_images/remote_jupyter2.PNG)

点击确定，完成设置

## 服务器端打开jupyter
```
nohup jupyter notebook --no-browser --port=8123 &
```

然后在本地的浏览器中写入
```
127.0.0.1:8123
```
输入密码就可以链接远程的jupyter

# jupyter使用conda环境

安装ipythonkernel
```
conda install ipykernel
```

首先激活对应的conda环境
```
source activate 环境名称
```

将环境写入notebook的kernel中
```
python -m ipykernel install --user --name 环境名称 --display-name "Python (环境名称)"
```

然后运行,就可以在jupyter中看到对用的环境
```
jupyter notebook
```


**[更多技术文章请点击查看](https://lxztju.github.io/tags/)**