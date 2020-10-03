# Anchor-free第二篇-CerterNet 论文总结

> 作者： 小哲
>
> 微信公众号： 小哲AI
>
> github： [https://github.com/lxztju/notes](https://github.com/lxztju/notes)



[TOC]




## 1. 论文摘要

在目标检测中，由于缺少对于裁剪框中的内容作进一步的校验，基于关键点的方法通常会有大量的不正确目标框的问题。论文介绍了基于最小代价情况下探索裁剪框中视觉模式的方法。本文所采用的方法是基于一步法的关键点检测的目标检测方法。CenterNet检测每一个目标为一个三元组（triplet）而不是一对关键点（cornernet）。这种方法提升了精确率与召回率。相对应的，设计了两个自定义的模块，分别为cascade corner pooling和center pooling。这两个模块的作用分别为丰富左上角与右下角的信息， 提升更多中心区域的识别信息。

在MSCOCO数据集上，CenterNet实现了47.0%AP， 远远超出了之前提出的所有的一步法检测器，与此同时，CenterNet与两步法的检测器也有一定的可比性并且拥有较大的检测速度。



## 2. 算法实现效果

CenterNet在coco数据集上的实验效果如下表所示：

![centernet_6](/home/luxiangzhe/git/notes/目标检测/images/centernet_6.png)






## 3. 论文主要思想及创新点

### 基于anchor的方法的不足

1. 需要大量的anchor导致负样本的数量远远大于正样本的数目，导致正负样本的极度不均衡，这是目标检测相关方法的一个极大的痛点
2. anchor的使用引入了大量的超参数，例如anchor的数目，尺寸大小，长宽比等需要手动设计
3. anchor通常与ground-truth不能对齐，这不利于分类任务的进行。

### cornernet方法的缺陷

1. cornernet使用左上右下两个关键点来检测物体，虽然解决了基于anchor搭的弊端，但是依然存在大量的不足
2. 由于cornernet缺少对物体全局信息的观察，检测出大量不正确的bbox，尤其是当iou较小时，这种情况更加严重。也即是说每个物体由一对关键点检测得到，导致算法对bbox目标框边界比较敏感。



![centernet_1](/home/luxiangzhe/git/notes/目标检测/images/centernet_1.png)

### 创新点：

	1. 使用三元组的三个关键点（左上角右下角中心点）来解决在cornernet中出现的大量不正确的边界框的问题。利用一个额外的关键点来获取建议区域的中心区域，思想就是如果一个预测的bbox与GT有大的iou那么其中心点在预测框的中心区域预测同样类别的得分就会很高。通过检测是否有同类别的中心点出现在器中信区域。
 	2. centerpooling：  在预测中心关键点的分支网络中使用。Center pooling帮助中心关键点获取更多目标中可识别的视觉信息，对proposal中心部分的感知会更容易。通过在预测中心关键点的特征图上，对关键点响应值纵向和横向值的求和的最大值来获取最大响应位置。
 	3. cascade corner pooling使得原始corner pooling 模块具有感知内部信心的能力。我们通过在特征图上获得目标边界和内部方向最大响应值的和来预测角点。




## 4. 论文架构

论文整体的架构：

![centernet_2](/home/luxiangzhe/git/notes/目标检测/images/centernet_2.png)

论文使用corner net的方法作为baseline， cornernet会检测出左上与右下两个heatmaps，embedding来评估两个关键点是否属于同一个object，offset来将角点从heatmap映射回原始图像。根据得分值大小，**从热力图上分别获得左上角点和右下角点的前top-k个来生成目标框**。**计算一对角点的向量（embedding vector）距离来确定这对角点是否来自同个目标。当一对角点的距离小于一个特定阈值，即生成一个目标框，目标框的置信度是这对角点热力值的平均值**。

centernet利用一个中心关键点与一对角点来检测目标物体， 整合center keypoint的heatmap到cornernet的baseline中， 然后生成topk个bbox，采用如下的过程来滤除大量的不正确的bbox。

1. 按照得分的高低，选择topk的center 关键点

2. 使用对应的offsets将中心点映射回原图

3. 为每一个bbox定义一个中心区域并检查这个区域是否包含一个中心点（中心区域与中心点的类别一致）。如果一个区域包含一个中心点，那么这个bbox就会被保存，得分就是三个关键点的得分平均值。如果该区域不包含中心点，那么该bbox就会被移除。如下图：

   ![centernet_2](/home/luxiangzhe/git/notes/目标检测/images/centernet_2.png)


## 5. 论文的细节

### 5.1 自适应的中心区域的大小

bbox中选取的中心区域大小影响了检测的结果，如果中心区域小，那么检测结果就回更加准确那么就会有更低的召回率， 如果中心区域较大，那么召回率就回更高，但是准确率就会下降，因此对于这个问题采用自适应的中心区域尺寸。



1. 对于小物体生成较大的中心区域（相对应的大小）
2. 对于大物体生成较小的中心区域

公式如下：

![centernet_7](/home/luxiangzhe/git/notes/目标检测/images/centernet_7.png)

其中t表示top-left， b表示bottom-right， c表示中心区域的左上与右下角，在论文实验中n=3（bbox小于150）和5（大于150）.

![centernet_3](/home/luxiangzhe/git/notes/目标检测/images/centernet_3.png)

### 5.2 center pooling

物体的几何中心通常不会传递视觉可识别的较强的视觉模式（visual pattern），例如一个人的几何中心是在身体上，而头部具有较强的可是别视觉信息。为了解决这个问题，采用center pooling， 处理过程为：首先backbone会输出一个特征图，为了确定特征图上的某个像素是否为中心关键点，需要在水平和垂直方向寻找最大值，并且将最大值相加。 center pooling有利于更好检测中心关键点。



![centernet_4](/home/luxiangzhe/git/notes/目标检测/images/centernet_4.png)



### 5.3 cascade corner pooling

通常情况下，角点存在于物体之外，缺乏局部外观特性。CornerNet用corner Pooling来解决此问题，通过沿边界方向找到最大值从而确定角点。但是其使得角点对边界特别敏感（因为是针对边界的特征信息做的池化操作，受边界信息变化影响较大）。为此，**本文作者提出让角点可看到更多的目标视觉模式信息（即获取物体内部的信息，而不仅仅是边界的），见图4（c），原理是沿边界寻找最大值，根据边界沿物体内部方向寻找最大值，将两个最大值相加。该方法获取的角点 带有边界信息以及物体内部的数据模式信息**。



![centernet_5](/home/luxiangzhe/git/notes/目标检测/images/centernet_5.png)



---

论文地址：[https://arxiv.org/pdf/1904.08189.pdf](https://arxiv.org/pdf/1904.08189.pdf)