# Anchor-free第一篇-CornerNet 论文总结

> 作者： 小哲
>
> 微信公众号： 小哲AI
>
> github： [https://github.com/lxztju/notes](https://github.com/lxztju/notes)



[TOC]




## 1. 论文摘要

文章提出一种新型的目标检测算法， CornerNet， 利用一个单一的卷积神经网络检测一对关键点（左上角与右下角）作为一个对象的bbox。 CornerNet利用一组关键点检测目标物体，消除了在通常的one-stage算法中anchor的设计。论文中提出了一种新型的pooling层帮助更好的定位corner——corner pooling， CornerNet在COCO上实现了42.1AP， 超出所有显存的one-stage感知器。





## 2. 算法实现效果

在COCO数据集上的实验结果对比， 在各种one-stage方法中得到了SOTA的结果。

![cornernet_1](/home/luxiangzhe/git/notes/目标检测/images/cornernet_1.png)

## 3. 论文主要思想及创新点

主要思想：将目标检测的问题看作一个关键点检测的问题， 利用物体目标bbox的左上与右下角作为得到预测的bbox，因此在cornetnet中就不存在anchor的概念。因此避免了anchor使用的两个弊端：

1. 正负样本的极度不均衡， anchor的数目往往很多，造成负样本过多
2. 引入大量的设计超参数，例如anchor的数目，大小，长宽比。

因此这种算法是一种anchor free的算法。

**总结这种算法：主要就是两个关键点(top left and bottom right)的*检测*与*匹配*问题**





## 4. 论文架构

算法的主要架构：

![cornernet_2](/home/luxiangzhe/git/notes/目标检测/images/cornernet_2.png)

主要结构：

	1. 利用hourglasses作为backbone提取特征
 	2. 生成两组feature map（heat map）分别表示top-left与bottom-right
 	3. 然后根据一定的方法匹配两个关键点。

![cornernet_4](/home/luxiangzhe/git/notes/目标检测/images/cornernet_4.png)

论文的整体结构图，如上图所示，经过backbone之后的特征图经过corner pooling之后，生成heatmap， embedding与offsets，

embedding vector用来评估两个关键点之间的距离，offsets是为了对预测bbox进行微调，因为在将下采样后的feature map映射回原图时进行取整会产生量化误差。

主要的结构部分：

* hourglass backbone
* corner pooling



## 5. 论文的细节

### 1. 检测角点

预测两个feature map，一个代表了左上角， 一个代表右下角， 每一组角点都预测CxHxW， 其中HW为heatmap的长宽，C为类别的数目（不包括背景类别）。如下图对于每一个corner都有一个positive的ground-truth， 其他的位置都是负值。对于负样本的惩罚随着距离真值的距离不断递减的（依据2维高斯函数递减）。

![cornernet_6](/home/luxiangzhe/git/notes/目标检测/images/cornernet_6.png)



### 2. 分组匹配角点

一张图像上可能有多个物体，因此会检测出很多的角点，采用embedding的方法进行匹配，网络对于每一个角点预测一个embedding向量，对于属于同一个object的top-left与bottom-right的点， 他们的embedding向量之间的距离非常小。（利用二者之间的距离来匹配对应的top-left与bottom-down角点）。



### 3. corner pooling

这是论文中的一种新提出的方法， 直接上图看看怎么计算（以top-lerft为例）：

![cornernet_5](/home/luxiangzhe/git/notes/目标检测/images/cornernet_5.png)

判断某个点是否为top-left角点， 上下两组特征图分别表示left与top的特征图，对于一个给定的特征图对于left， 分别从当前位置到最右侧的vector位置取maxpooling，同理top是从当前位置到最下端的vector取maxpooling， 然后将二者相加。

这个层在预测module中产生heatmap， embedding与offsets。



### 4. hourglass Network

这个网络的hourglass模块先下采样，然后进行上采样。由于下采样的过程中会丢失大量的信息，加入了skip layers，这种架构里采用了一个整体的架构能够轻松的实现全局与局部信息的捕捉。

hourglass 论文：[https://arxiv.org/pdf/1611.05424](https://arxiv.org/pdf/1611.05424)



---

CornerNet的论文地址：[https://arxiv.org/abs/1808.01244](https://arxiv.org/abs/1808.01244)