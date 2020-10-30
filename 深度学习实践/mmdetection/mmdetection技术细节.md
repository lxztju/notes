# mmdetection 技术细节

在这一部分，将会介绍训练一个detector的主要的细节:
* data pipeline
* model
* iteration pipeline

## data pipeline

依然是非常传统的使用，我们使用`Dataset`和`DataLoader`来使用多线程进行数据的加载。
`Dataset`将会返回一个与模型前向方法相对应的数据字典。因为在目标检测中数据可能不是相同的尺寸（例如图形尺寸，bbox的尺寸等），我们引入了一个新的`DataContainer`类型来帮助收集与分类这些不同尺寸的数据。更多的细节请看[here](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py)。

数据的传输通道与数据集的定义分离开，通常数据集定义如何处理annotations，数据的pipeline定义准备数据字典的所有的步骤。一个pipeline由一系列的运算组成，每一次运算都会将一个字典作为输入，而且为下一次的转换运算输出一个字典。

在下边我们提出了一个经典的pipeline，蓝色的模块是pipeline操作，在pipline运算的过程中，每一次的操作都可以在结果中加入新的keys（标记为绿色）或者更新已经存在的keys（标记为橘黄色）
![pipeline figure](./data_pipeline.png)

这些运算被分为数据读取，数据预处理，格式化，测试时增强四类

Here is an pipeline example for Faster R-CNN.
```python
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
```

For each operation, we list the related dict fields that are added/updated/removed.

### Data loading

`LoadImageFromFile`
- add: img, img_shape, ori_shape

`LoadAnnotations`
- add: gt_bboxes, gt_bboxes_ignore, gt_labels, gt_masks, gt_semantic_seg, bbox_fields, mask_fields

`LoadProposals`
- add: proposals

### Pre-processing

`Resize`
- add: scale, scale_idx, pad_shape, scale_factor, keep_ratio
- update: img, img_shape, *bbox_fields, *mask_fields, *seg_fields

`RandomFlip`
- add: flip
- update: img, *bbox_fields, *mask_fields, *seg_fields

`Pad`
- add: pad_fixed_size, pad_size_divisor
- update: img, pad_shape, *mask_fields, *seg_fields

`RandomCrop`
- update: img, pad_shape, gt_bboxes, gt_labels, gt_masks, *bbox_fields

`Normalize`
- add: img_norm_cfg
- update: img

`SegRescale`
- update: gt_semantic_seg

`PhotoMetricDistortion`
- update: img

`Expand`
- update: img, gt_bboxes

`MinIoURandomCrop`
- update: img, gt_bboxes, gt_labels

`Corrupt`
- update: img

### Formatting

`ToTensor`
- update: specified by `keys`.

`ImageToTensor`
- update: specified by `keys`.

`Transpose`
- update: specified by `keys`.

`ToDataContainer`
- update: specified by `fields`.

`DefaultFormatBundle`
- update: img, proposals, gt_bboxes, gt_bboxes_ignore, gt_labels, gt_masks, gt_semantic_seg

`Collect`
- add: img_meta (the keys of img_meta is specified by `meta_keys`)
- remove: all other keys except for those specified by `keys`

### Test time augmentation

`MultiScaleFlipAug`

## 模型

在框架中，模型基本被分为四类：

- backbone:通常是一个全卷积网络来提取特征
- neck:在backbone和head之间的部分，例如FPN
- head: 对于特定的任务的部分，例如bbox预测
- roi extractor: 从特征图上提取特征的部分，例如RoI Align

我们利用上边的组件，也写了一些常规的感知器pipeline，例如单步法与两步法的感知器。

### 用基本的组件构建一个模型

利用基本的pipeline，模型架构可以通过config文件很简单的自定义。

如果想要来利用一些新的组件，例如path aggregation FPN structure，[Path Aggregation Network for Instance Segmentation](https://arxiv.org/abs/1803.01534)，我们需要做两个工作：


1. create a new file in `mmdet/models/necks/pafpn.py`.

    ```python
    from ..registry import NECKS

    @NECKS.register
    class PAFPN(nn.Module):

        def __init__(self,
                    in_channels,
                    out_channels,
                    num_outs,
                    start_level=0,
                    end_level=-1,
                    add_extra_convs=False):
            pass

        def forward(self, inputs):
            # implementation is ignored
            pass
    ```

2. Import the module in `mmdet/models/necks/__init__.py`.

    ```python
    from .pafpn import PAFPN
    ```

2. modify the config file from

    ```python
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5)
    ```

    to

    ```python
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5)
    ```

我们将会开源更多的组件，例如backbones， necks， heads。

### 写一个新模型

为了写一个新的检测pipeline，我们需要继承自`BaseDetector`，在这个类中我么们定义了如下的方法：

- `extract_feat()`: given an image batch of shape (n, c, h, w), extract the feature map(s).
- `forward_train()`: forward method of the training mode
- `simple_test()`: single scale testing without augmentation
- `aug_test()`: testing with augmentation (multi-scale, flip, etc.)

[TwoStageDetector](https://github.com/hellock/mmdetection/blob/master/mmdet/models/detectors/two_stage.py)是一个展示了如何使用的很简单的例子

## Iteration pipeline

我们采用了分离式的训练，既可以在单机上训练，也可以多机并行训练。
假设有一个8 gpu的服务器，我们将会启动8进程，每个进程运行在每个gpu上。

每个进程都保持一个孤立的模型，数据提取器，优化器
模型的参数仅仅在开始的时候同步一次。
在一次前向传播和反向传播中，在所有gpu上的梯度都会下降，优化器将会更细所有的参数，因为梯度是同时下降，模型的参数将会保持一致，

## Other information

For more information, please refer to our [technical report](https://arxiv.org/abs/1906.07155).
