---
layout: post
title: "Domain Adaptation"
date: 2020-07-05
description: "leetcode"
tag: leetcode 
--- 

Domain Adaptation(领域适应)是一个迁移学习中很常见的问题，描述的主要是对于源域数据与目标域数据分布不同的情况下，利用源域数据训练的模型在目标域数据上获得相对较好的表现。实际中就是将分布不同的源域与目标域的数据映射到一个相同的特征空间中，这样基于源域训练的模型在目标域中也会有比较好的效果。

在一个图像分类任务中，如果源域数据集（用来训练模型的训练集，有label标注）的获取方式与目标域（测试集， 无label）获取方式不同，例如在训练模型用的数据集在实验室环境中采集，而域实际场景贴近的实际应用中数据具有一定的差异，例如拍摄环境还有光照等一系列因素导致二者数据存在差别。

领域自适应就是解决这个实际问题，主要的思路就是减小源域域目标域的差异。采用不同的方法来减小不同目标域之间的差异，以提升模型的泛化能力。

Deep Transfer Learning with Joint Adaptation Networks这篇文章在训练的时候减小source域target的分布的差异，让网络的泛化能力更强，能够同时学习到两个域上的信息。


Learning Transferable Features with Deep Adaptation Networks，这篇文章在上一篇文章的基础上，采用了多层来减小不同网络间的差异。


首先Domain Adaptation基本思想是既然源域和目标域数据分布不一样，那么就把数据都映射到一个特征空间中，在特征空间中找一个度量准则，使得源域和目标域数据的特征分布尽量接近，于是基于源域数据特征训练的判别器，就可以用到目标域数据上。

Domain-Adversarial Training of Neural Networks这篇文章利用对抗网络来减小源域域目标域之间的分布差异


## 文献

* [Deep Transfer Learning with Joint Adaptation Networks](https://arxiv.org/pdf/1605.06636.pdf)
* [Learning Transferable Features with Deep Adaptation Networks](http://proceedings.mlr.press/v37/long15.pdf)
* [Deep Domain Confusion: Maximizing for Domain Invariance](https://arxiv.org/pdf/1412.3474.pdf)
* [Domain-Adversarial Training of Neural Networks](https://dl.acm.org/doi/abs/10.5555/2946645.2946704)
* [https://github.com/zhaoxin94/awesome-domain-adaptation](https://github.com/zhaoxin94/awesome-domain-adaptation)

参考文章：

1、https://chenrudan.github.io/blog/2017/12/15/domainadaptation1.html

2、https://zhuanlan.zhihu.com/p/50710267

**[更多技术文章请点击查看](https://lxztju.github.io/tags/)**