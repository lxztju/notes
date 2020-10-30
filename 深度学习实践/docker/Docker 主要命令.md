# Docker 主要命令

这篇文章的主要命令：

1. Docker相关

2. image相关
    1. 拉取镜像
    2. 列出所有镜像
    3. 删除镜像

3. container的主要命令
    1. 列出容器
    2. 停止容器
    3. 删除容器
    4. 启动容器
    5. 进入容器


4. 运行image（包括交互式运行）（创建容器）
    挂载数据盘

## Docker相关

```shell
Docker --version  #查看docker版本
Docker info    #查看docker信息
docker --help   #列出帮助信息
```

## image相关命令

1. 从远程仓库拉去镜像（下载image）
```shell 
docker search ****      #查看所需包的可用版本
docker image pull ****  # 拉取对应的镜像
```

2. 列出计算机中所有镜像
```
docker images
```

3. 删除镜像
```shell
# 依据第二步得到的image信息
docker image rm ***    # ***为待删除image_id的前三位
```

## container 主要命令

1. 列出容器
```shell
docker ps  #列出所有正在运行的容器
docker ps -a   #列出所有的容器，包括已经停止的
```

2. 停止容器
```shell
docker stop container_id  ##可以仅仅采用container_id的前三位
```

3. 删除容器

```shell
docker rm container_id  ##可以仅仅采用container_id的前三位
```

4. 启动容器

```shell
docker start container_id  ##可以仅仅采用container_id的前三位
```

4. 进入容器

```shell
docker attach container_id  ##可以仅仅采用container_id的前三位
```

## 运行image(创建容器)

```shell
docker run image_id(image_name)
docker -it run image_id(image_name) /bin/bash  # (交互式运行)

```
### 挂载数据盘
利用-v挂载数据盘
```shell
docker run -it -v 源文件地址:挂载数据盘位置 image_name /bin/bash   
```