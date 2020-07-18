# Docker 安装及镜像加速

这篇文章内容：
1. Docker安装（ubuntu）
2. Docker hub的阿里云镜像

## Docker 安装
可参考如下网址安装：https://www.runoob.com/docker/ubuntu-docker-install.html
使用Docker仓库

### 设置仓库

1. 更新apt包索引
```shell
sudo apt-get update
```
2. 安装依赖
```shell
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
        software-properties-common
```
3. 设置apt仓库地址，添加阿里云的apt仓库
```shell
添加gpg密钥
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
     "deb [arch=amd64] https://mirrors.aliyun.com/docker-ce/linux/ubuntu \
     $(lsb_release -cs) \
     stable"
```
### 安装Docker
```
sudo apt-get update
sudo apt-get install docker-ce
```
查看docker版本
```shell
docker --version
```

### 启动docker
Ubuntu 16启动docker的方法（只需做一次即可）：
```
sudo systemctl enable docker
sudo systemctl start docker
```

避免每次都使用sudo的方法：建立docker组，将用户加入到docker组中
```
sudo groupadd docker
sudo usermod -aG docker $USER
sudo gpasswd -a ${USER} docker
sudo service docker restart
newgrp - docker
```

## 阿里云镜像加速

1. 获取阿里云镜像加速
登录阿里云控制台[容器hub服务](https://cr.console.aliyun.com/cn-hangzhou/instances/mirrors),点击左侧镜像加速，得到加速器地址，复制保存。

*我的个人加速器*：`https://rjm5nlh8.mirror.aliyuncs.com`

2. 打开文件`sudo vim /etc/docker/daemon.json`

添加如下代码：
{
  "registry-mirrors": ["https://rjm5nlh8.mirror.aliyuncs.com"]
}

3. 重启docker

```
sudo systemctl daemon-reload
sudo systemctl restart docker
```
