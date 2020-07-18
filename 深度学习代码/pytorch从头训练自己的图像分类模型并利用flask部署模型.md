本文推荐一个完整的pytorch图像分类代码，轻松上手，可以实现很好的分类性能。

利用pytorch实现图像分类，其中包含的densenet，resnext，mobilenet，efficientnet等图像分类网络，可以根据需要再行利用torchvision扩展其他的分类算法

##### github代码地址：[https://github.com/lxztju/pytorch_classification](https://github.com/lxztju/pytorch_classification)
如果有用欢迎star
## 实现功能
* 基础功能利用pytorch实现图像分类
* 包含带有warmup的cosine学习率优化
* 多模型融合预测，加权与投票融合
* 利用flask实现模型云端api部署
* 冻结相应层不训练（减轻过拟合程度）
* 实现多gpu训练，单gpu或者cpu预测

## 运行环境
* python3.7
* pytorch 1.1
* torchvision 0.3.0

## 代码仓库的使用

### 数据集形式
原始数据集存储形式为，同个类别的图像存储在同一个文件夹下，所有类别的图像存储在一个主文件夹datasets下。

利用preprocess.py将数据集格式进行转换（个人习惯这种数据集的方式）

转换后的数据集为，将训练集的路径与类别存储在train.txt文件中，测试机存储在val.txt中

### 模型介绍
仓库中模型densenet，mobilenet,resnext模型来自于torchvision

efficientnet来自于

build_model.py 构建模型

### 训练

* 在`cfg.py`中修改合适的参数，并在train.py中选择合适的模型
```shell
python train.py
```

### 预测
在cfg.py中TRAINED_MODEL参数修改为指定的权重文件存储位置
```shell
python predict.py
```
* 多模型融合的使用方式类似于预测，使用`multi-model_predict.py`进行多模型融合

### flask云端部署

将训练存储好的权重文件，存储在`flask_deployment`文件夹中

然后修改`server.py`中路径运行即可
利用`client.py`进行调用


