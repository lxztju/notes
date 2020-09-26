# Ubuntu16安装nvidia驱动与cuda pytorch

pytorch更新了好多东西， 我还停留在pytorch1.1, 这里phoenix倒腾了一下电脑， 重装了cuda12与pytorch1.6



驱动以及cuda下载地址：

cuda：[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)

nvidia driver [https://www.nvidia.cn/Download/Find.aspx?lang=cn](https://www.nvidia.cn/Download/Find.aspx?lang=cn)

这里先选择cuda，然后按照对应的cuda后边的数字选择对应的驱动版本

## 1. 卸载原有驱动

```shell
sudo apt remove nvidia-*
sudo apt autoremove
```



## 2. 安装显卡驱动

按下`CTRL+ALT+F1`， 进入命令行界面

```shell
sudo service lightdm stop  #禁用图形界面
sudo chmod +x NVIDIA-Linux-x86_64-440.33.01.run  # 赋予可执行权限

# -no-opengl-files这个参数可以放置循环启动， 不过有时候不加它倒是正常，加上了反而循环启动
sudo sh NVIDIA-Linux-x86_64-440.33.01.run -no-x-check -no-nouveau-check -no-opengl-files  # 安装

# 测试
nvidia-smi

sudo service lightdm start #启用图形界面 
```



## 3. 卸载原有cuda

```shell
sudo /usr/local/cuda-10.0/bin/uninstall_cuda_10.0.pl

sudo rm -rf /usr/local/cuda-10.0
```



## 4. 安装cuda



```shell
sudo chmod +x cuda_10.2.89_440.33.01_linux.run

sh cuda_10.2.89_440.33.01_linux.run

# accept
```

可能会出现以下的东西：

```
CUDA Installer

-[X] driver
-[X]CUDA ToolKit
-[X]CUDA Samples
-[X]CUDA Demo Suit
-[X]CUDA Documentation
Options
Install
```

由于之前已经安装过显卡驱动， 这里选中driver，点击回车将X号去掉， 然后选择Install回车即可正常安装

## 5. 安装pytorch

打开官网直接安装

```shell
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

