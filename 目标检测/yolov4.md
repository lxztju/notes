3. 

[TOC]



# YoloV4论文总结

论文链接: [https://arxiv.org/pdf/2004.10934.pdf](https://arxiv.org/pdf/2004.10934.pdf)



这篇论文提出了一个检测精度高,速度快的目标检测模型, 融合介绍当前常用的各种目标检测的各种tricks,得到速度快,精度高的one-stage模型.

当前目标检测器主要由几个部分组成:

* **输入图像**: (图像, 图像块(RCNN)使用, 图像金字塔)

* **Backbones(骨干网络)**: VGG16, ResNet-50, SpineNet, EfficientNet-B0/B7, CSPResNeXt-50, CSPDarkNet53

* **Neck**: 

  * 额外的模块(additional blocks): SPP, ASPP, RFB, SAM
  * 融合模块: FPN, PAN, NAS-FPN, Fully-connected FPN, BiFPN, ASFF, SFAM

* **Heads**: 

  * Dense prediction(one-stage):
    * RPN, SSD, YOLO, RetainNet ( anchor base)
    * CornerNet, CenterNet, MatrixNet, FCOS(anchor free)
  * Sparse Prediction(two-stage):
    * Faster RCNN, R-FCN, Mask RCNN(anchor based)
    * RepPoints(anchor free)

  

  



下边就总结一下下边论文提到的在yolov4中的各种work或者不work的各种ticks.



## 1 Bag of freebies

这种方法是仅仅改变训练策略,增加训练成本的方法称为: `bag of freebies`, 通常采用的方法是数据增广, 目的时提升训练数据的多样性

### 1.1 data augmentation



#### 像素级别的调整

* 几何畸变(geometic distortion)
  * 随机缩放(random scaling)
  * 随机裁剪( random cropping)
  * 随机反转(random flipping)
  * 随机旋转(random rotating)
* 光度畸变(photometic distortion)
  * 亮度(brightness)
  * 对比度(contrast)
  * 色调(hue)
  * 饱和度(saturation)
  * 噪声(noise)

#### 目标遮挡(object occlusion)

* 随机擦除(random erase)

  随机选取一块区域,用随机像素值填充

* cutout(不知道怎么翻译,就是从图像上裁下来一块)

  随机选取方形的区域用0填充

如果在对于目标遮挡的思想应用在特征图上,就是:

* DropOut

* DropConnect

* DropBlock

  由于dropout是随机drop掉一些输出,对于全连接层有效,但是对于卷积层无效,卷积层在空间上是相关的,当这些特性相互关联时，即使有dropout，有关输入的信息仍然可以发送到下一层，这会导致网络overfit.

  一个feature map的输出,不再随机选取一些特征, 而是直接将feature map的一个相邻区域drop掉.

  每个feature层都设置自己的dropblock size与drop率更好.

#### 多图的融合

* **mixup**

  两张图同比例混合, label也按照一定的比例混合

* **cutmix**

  cut与mixup的结合, 将切除的一块区域,采用另一图像中的区域混合

* **mosaic data augmentation**

  cutmix混合了两张图, 而mosaic混合了四张图

* **style transfer gan**

  cnn学到的是纹理特征,因此通过gan引入风格化的数据集,提升模型的泛化能力

### 1.2 数据不均衡

目标检测样本不均衡, 低召回(所有正类样本中,检测出了多少个). 

* **two-stage: 将误分类的样本再次输入图片进行训练**

  * **hard negative example mining**

  ​       hard negitive mining, 在faster rcnn中体现在RPN与RoIHead链接的地方, 直接采用正负样本1:3roi进行RoiHead的训练.而且在faster rcnn中不能直接采用iou小于0.1的作为hard negtive,由于这些很容易检测为负样本,这个应该算是easy negitive.在faster rcnn的代码中选择是iou为[0.1-0.5]的.

  这种思路在SSD中同样存在.

  * **online hard  example mining**

  与上边的hard negitive mining不同的是,上边的方法考虑的是hard negitive, 而这里考虑的是所有的hard example. 

  *Training Region-based Object Detectors with Online Hard Example Mining*将难分样本挖掘(hard example mining)机制嵌入到SGD算法中，使得Fast R-CNN在训练的过程中根据**区域提议的损失，自动选取合适的区域提议**作为正负例训练。

  主要思想是首先将所有ROI区域输入fast rcnn模块求得每个ROI的loss,对所有ROI的loss从大到小排序,选择前N个最大的loss的ROI样本进行训练网络更新fast rcnn的参数.

* **one-stage: 降低易分类样本的损失, 让网络更加关注困难或者错分样本**

  * focal loss
  
  Focal loss是一种改进了的交叉熵(cross-entropy, CE)loss，它通过在原有的CE loss上乘了个使易检测目标对模型训练贡献削弱的指数式，从而使得Focal loss成功地解决了在目标检测时，正负样本区域极不平衡而目标检测loss易被大批量负样本所左右的问题

### 1.3 one-hot硬编码不能表示类别间的关系

* label smothing(标签平滑)

  将硬标签编码转换为软标签编码

* 知识蒸馏: 设计获得更好的soft label



### 1.4 bbox回归

传统的目标检测器用MSE作为回归的损失, 但是直接评估每个点的坐标值,把bbox的坐标点看作时独立的变量.

* l1

$$
l1 = |x|
$$



* l2

$$
l2 = x^2
$$



* smooth l1

$$
Smooth_{l1} = \begin{cases} 
	0.5x^2    &   if |x| < 1 \\
	|x| - 0.5 &   otherwise
    \end{cases}
$$





* IOU loss

计算bbox与GT的IOU,然后取对数,作为损失函数具体的公式为 , 损失函数与IOU成正比.
$$
L_{iou} = -1 \times log(IOU)
$$
也有定义为:
$$
L_{iou} = log( 1 - iou)
$$


* GIOU loss

当IOU为0, 损失函数不可导, IOU loss无法优化这种情况, iou无法反映两个bbox是如何相交(正, 斜)

GIOU就是采用预测bbox与GT的最小外接矩形C, 用C减去二者的并集,除以C得到一个数, 用真实框与预测框的IOU减去这个数值得到GIOU
$$
L_{Giou} = 1 - GIOU \\
其中, GIOU = IOU - \frac {C - (AUB)}{C}
$$ { }
GIOU具有尺度不变性, IOU的范围为0到1, GIOU的范围为-1到1, 此时IOU与GIOU均为1,当A, B不相交且距离无限远(不可能的,只是值会趋近与-1)时, GIOU = -1

* DIOU loss

好的目标框回归损失应该考虑三个重要的几何因素：重叠面积(IOUloss与GIOUloss)，中心点距离，长宽比。DIoU Loss,相对于GIoU Loss收敛速度更快，该Loss考虑了重叠面积和中心点距离，但没有考虑到长宽比。

当目标框完全包裹预测框的时候，IoU和GIoU的值都一样，此时GIoU退化为IoU, 无法区分其相对位置关系；此时作者提出的DIoU因为加入了中心点归一化距离，所以可以更好地优化此类问题.

作者思考两个问题:

- 直接最小化预测框与目标框之间的归一化距离是否可行，以达到更快的收敛速度。(DIOU loss)
- 如何使回归在与目标框有重叠甚至包含时更准确、更快。 ( CIOU loss)

通常基于IOU的损失函数可以表示为: $L = 1 - IOU + R(B, B^T)$, 其中R为惩罚项.

在DIOU loss中:
$$
L_{DIOU} = 1 - IoU + \frac{\rho ^ 2(b, b^{gt})}{c^2}
$$
其中, $\rho$表示欧式距离, $b, b^{gt}$分别表示bbox的中心点, c表示最小外接矩形对角线的距离.

 当两个框完全重合, 那么$L_{GIOU} = L_{DIOU} = L_{IOU} = 0$

* CIOU loss

综合考虑了重叠面积, 中心点距离与长宽比
$$
L_{CIOU} =1 - IOU +\frac{\rho ^ 2(b, b^{gt})}{c^2} + \alpha v
$$


$\alpha$是权重, v用来度量长宽比的相似性
$$
v = \frac 4 {\pi^2} (arctan(\frac{w^gt}{h^gt}) - arctan(\frac{w}{h}) )^2
$$


## 2 Bag of specials



增加很少的推理成本嫩够提高目标检测精度的方法.



###  2.1 扩大感受野

* SPP

空间金字塔池化

![spp](/home/luxiangzhe/git/notes/目标检测/images/spp.png)

* ASPP

空洞卷积

![aspp](/home/luxiangzhe/git/notes/目标检测/images/aspp.png)

* RFB

同样利用空洞卷积

![rfb](/home/luxiangzhe/git/notes/目标检测/images/rfb.png)



### 2.2 注意力机制

* SE- Net, channel-wise attention

对于每张特征图赋予不同的权重, 类似于CAM中的做法, 在gpu上利用这种注意力机制很大的增加推理时间,很少的增加计算量, 适合在移动端设备使用

![senet](/home/luxiangzhe/git/notes/目标检测/images/senet.png)

* SAM, CBAM point-wise attention

在CBAM中对于CHW的特征图按照通道进行最大和平均池化,得到2HW的特征图,然后用1x1的卷积得到HW的特征图,然后sigmoid激活,得到HW个权重,然后乘以相对应的特征图

![CBAM](/home/luxiangzhe/git/notes/目标检测/images/CBAM.png)

* modified SAM

直接对CHW的特征图进行1x1的卷积得到CHW的特征层,然后sigmoid激活后直接乘以相对应的值





### 2.3  特征融合

* FPN

![FPN](/home/luxiangzhe/git/notes/目标检测/images/FPN.png)

* SFAM

M2Det文章中

![M2det](/home/luxiangzhe/git/notes/目标检测/images/M2det.png)

* ASFF

不采用element-wise或者concat, 采用加权相加, 权重因子采用1x1卷积后得到.

![ASFF](/home/luxiangzhe/git/notes/目标检测/images/ASFF.png)

* BiFPN

![BiFPN](/home/luxiangzhe/git/notes/目标检测/images/BiFPN.png)

### 2.4 激活函数

* ReLU
* LReLU
* PReLU
* ReLU6
* Scaled Exponential Linear Unit
* Swish
* hard-Swish
* Mish

ReLU6, hard-Swish为量化网络设计



### 2.5 NMS

* nms

将与基准框的IOU的差值大于thresh的删掉

* soft nms

将与基准框的IOU的差值大于thresh的bbox的置信度降低.不直接删掉.

线性修正:
$$
si = \begin{cases}
si & iou(M, b) < thresh \\
si( 1-iou(M, b) & iou(M, b) >= thresh
\end{cases}
$$
高斯惩罚修正
$$
si = si * e^{\frac{iou(M, b) ^2}{\sigma}}
$$


* DIoU nms

就是在NMS中将IOU的计算换成DIOU.



## 3 yolov4的主要内容

### 3.1输入端

* mosiac 数据增广
* SAT对抗训练
* Cross mini-batch Normalization(CmBN)

### 3.2 backbone

* CSPDarknet53

  *  CSP: Cross Stage Paritial Network
* Mish激活函数
  *  backbone使用这个激活函数, 后边的部分依然使用leaky relu
* Dropblock

### 3.3 Neck

采用SPP, FPN+PAN ( PANet中)



### 3.4 Head

* CIOU-loss

* DIoU-NMS



参考链接: [https://zhuanlan.zhihu.com/p/143747206](https://zhuanlan.zhihu.com/p/143747206)

































