# 开始使用mmdetection

## 利用预训练模型进行推测

提供了评估各个数据集（COCO， VOC，cityspace）的测试脚本，也有一些对于其他项目整合的高层api

### 测试数据集

框架支持的操作
* 单gpu测试
* 多gpu测试
* 可视化测试结果

利用如下的命令去测试数据集：
```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

可选择的参数有：
* `RESULT_FILE`：如果指定这个参数那么输出的结果将会保存在一个pkl文件中，如果不指定结果就不会保存
* `EVAL_METRICS`： 这个参数依赖于数据集，对于COCO，`proposal_fast`, `proposal`, `bbox`, `segm`，对于VOC数据集，`mAP`, `recall`， 对于cityspace， 除了支持所有的COCOmetrics还有，`cityscapes`
*  `--show`：如果指定这个参数，检测的结果就会显示在一个窗口中，这个设置只在单gpu测试时可用，同时确保GUI可用，否则可能会遇到这个错误 `cannot connect to X server`

在检测整个数据集时，不要指定`--show`这个参数

实施举例：

假设你已经下载预训练权重到`/checkpoint`

1. 测试Faster RCNN并且可视化结果，按任意键检测下一张图片
```shell
python tools/test.py configs/faster_rcnn_r50_fpn_1x.py \
    checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth \
    --show
```

2. 在pascal voc上测试Faster RCNN 并且计算MAP，并不保存测试文件
```shell
python tools/test.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc.py \
    checkpoints/SOME_CHECKPOINT.pth \
    --eval mAP
```

3. 利用8gpu测试maskrcnn， 并评估bbox和mask ap

```shell
./tools/dist_test.sh configs/mask_rcnn_r50_fpn_1x.py \
    checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth \
    8 --out results.pkl --eval bbox segm
```

4. 利用8gpu在coco测试集上测试maskrcnn，并生成可直接提交到评估服务器的json文件

```shell
./tools/dist_test.sh configs/mask_rcnn_r50_fpn_1x.py \
    checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth \
    8 --format_only --options "jsonfile_prefix=./mask_rcnn_test-dev_results"
```
最终会得到两个json文件：`mask_rcnn_test-dev_results.bbox.json` and `mask_rcnn_test-dev_results.segm.json`.

5. 利用8gpu在cityspace上测试maskrcnn，并且生成可提交到评估服务器的txt和png文件

```shell
./tools/dist_test.sh configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py \
    checkpoints/mask_rcnn_r50_fpn_1x_cityscapes_20200227-afe51d5a.pth \
    8  --format_only --options "outfile_prefix=./mask_rcnn_cityscapes_test_results"
```

### Webcam demo

框架提供了一个webcam demo展示这个检测结果

```shell
python demo/webcam_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${GPU_ID}] [--camera-id ${CAMERA-ID}] [--score-thr ${SCORE_THR}]
```

例子：
```shell
python demo/webcam_demo.py configs/faster_rcnn_r50_fpn_1x.py \
    checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
```

### 测试图像的高层API

#### 同步接口

这是一个构建模型测试图片的例子

```python
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
show_result(img, result, model.CLASSES)
# or save the visualization results to image files
show_result(img, result, model.CLASSES, out_file='result.jpg')

# test a video and show the results
video = mmcv.VideoReader('video.mp4')
for frame in video:
    result = inference_detector(model, frame)
    show_result(frame, result, model.CLASSES, wait_time=1)
```

另外有一个ipython版本[demo/inference_demo.ipynb](https://github.com/open-mmlab/mmdetection/blob/master/demo/inference_demo.ipynb).

#### 异步接口（仅支持python 3.7+）

异步接口允许不阻塞GPU绑定的推理代码上的CPU，并为单线程应用程序提供更好的CPU/GPU利用率。可以在不同的输入数据样本之间或某个推理管道的不同模型之间并发地进行推理。

利用`tests/async_benchmark.py`来对比同步与一部接口的速度

```python
import asyncio
import torch
from mmdet.apis import init_detector, async_inference_detector, show_result
from mmdet.utils.contextmanagers import concurrent

async def main():
    config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
    device = 'cuda:0'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    # queue is used for concurrent inference of multiple images
    streamqueue = asyncio.Queue()
    # queue size defines concurrency level
    streamqueue_size = 3

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device=device))

    # test a single image and show the results
    img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once

    async with concurrent(streamqueue):
        result = await async_inference_detector(model, img)

    # visualize the results in a new window
    show_result(img, result, model.CLASSES)
    # or save the visualization results to image files
    show_result(img, result, model.CLASSES, out_file='result.jpg')


asyncio.run(main())

```

## 训练模型

mmdetection采用分布式训练与非分布式的训练

使用`MMDistributedDataParallel` 和 `MMDataParallel`实现分布式训练

所有的输出（日志文件与权重文件）保存在由`work_dir`指定的工作目录下

通常每1epoch就执行一次evaluation，可以通过修改间隔语句来进行保存。
```python
evaluation = dict(interval=12)  # This evaluate the model per 12 epoch.
```

**\*Important\***: 在config文件中默认的学习率是8gpus，每张gpu2张图像，
如果在训练过程中采用不同的硬件配置，依据所用的batch_size，按照 [Linear Scaling Rule](https://arxiv.org/abs/1706.02677)进行设置

### 单gpu训练

```shell
python tools/train.py ${CONFIG_FILE}
```

### 多gpu训练

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

* 可选参数：
- `--validate` (**strongly recommended**): 在训练过程中可以按照 [this](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn_r50_fpn_1x.py#L174)) 调整评估的间隔，默认每个epoch执行评估.
- `--work_dir ${WORK_DIR}`: 重新指定工作目录。
- `--resume_from ${CHECKPOINT_FILE}`: 重新制定重新训练读取文件的位置

`resume_from` 和 `load_from` 的不同点：

`resume_from`：不仅要从checkpoint文件中读取权重也需要得到特定的优化器状态和epoch数目， 用于程序运行过程中中断后继续训练
`load_from`： 仅仅读取模型权重用于fine-tune

### 在多个机器上训练

这一部分近期不会接触到，因此没有看这一部分的内容

### 在一台服务上部署多个训练任务

如果在一台服务器上部署多个训练任务，需要设置不同的port，默认的port是29500.

如果使用`dist_train.sh`进行训练，需要在命令行中设置port
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

## 有用的工具
框架在`./tools`中提供了很多有用的工具

### 分析日志

可以利用一个训练的log日志直接plot 损失曲线，首先`pip install seaborn` 安装所需的依赖。

```shell
python tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

**一些例子**
- 绘制分类损失曲线

```shell
python tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
```

- 绘制分类回归损失并且保存figure为pdf文件

```shell
python tools/analyze_logs.py plot_curve log.json --keys loss_cls loss_reg --out losses.pdf
```

- 比较两个训练过程的bbox_mAP

```shell
python tools/analyze_logs.py plot_curve log1.json log2.json --keys bbox_mAP --legend run1 run2
```

- 可以计算平均训练速度

```shell
python tools/analyze_logs.py cal_train_time ${CONFIG_FILE} [--include-outliers]
```
训练速度分析计算的大致输出如下表示：
```
-----Analyze train time of work_dirs/some_exp/20190611_192040.log.json-----
slowest epoch 11, average time is 1.2024
fastest epoch 1, average time is 1.1909
time std over epochs is 0.0028
average iter time: 1.1959 s/iter
```
### 得到 FLOPs 和 params (正在实验过程中，还不稳定)

框架提供了一个脚本用来计算FLOPs和parameters， 脚本文件来自于 [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) 

```shell
python tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

会得到类似如下的结果：
```
==============================
Input shape: (3, 1280, 800)
Flops: 239.32 GMac
Params: 37.74 M
==============================
```

**Note**: 这个工具依然在实验过程中，不能保证得到的数据正确，可以利用这个数据做简单的对比，但是在被用做论文中时需要多次检查

1. FLOPs是依赖于输入的shape，而params不是。默认的输入形状是（1，3，1280，800）
2. 一些自定义的操作及GN等没有参与FLOP的计算，也可以用一些其他的方法进行修正[`mmdet/utils/flops_counter.py`](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/utils/flops_counter.py).
3. 两步法的FLOPs依赖于proposal的数目

### 公布一个模型

在上传一个模型到AWS之前，你需要做如下的工作：
1. 转换model weights为cpu tensor
2. 删除优化器状态
3. 计算checkpoint file的hash，并且应用hash id为文件的名字

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/publish_model.py work_dirs/faster_rcnn/latest.pth faster_rcnn_r50_fpn_1x_20190801.pth
```
最终的输出文件是`faster_rcnn_r50_fpn_1x_20190801-{hash id}.pth`.

### 检测感知器的鲁棒性

请参考[ROBUSTNESS_BENCHMARKING.md](ROBUSTNESS_BENCHMARKING.md).

### 转换为ONNX（正在实验过程中还不稳定）

框架提供了一个脚本文件去将模型转换为[ONNX](https://github.com/onnx/onnx)格式，转换完的模型可以利用[Netron](https://github.com/lutzroeder/netron)可视化.

```shell
python tools/pytorch2onnx.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out ${ONNX_FILE} [--shape ${INPUT_SHAPE}]
```

**Note**: 这个工具依然在实验过程中，自定义的操作不支持转换，为`RoIPool` and `RoIAlign`我们动态设置了`use_torchvision=True`.

## 入门指南

### 使用我们自己的数据集

最简单的方式就是转换数据集为COCO格式

下边展示一个例子，利用一个5个类别的自定义数据集。假设是COCO格式

In `mmdet/datasets/my_dataset.py`:
```python
from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class MyDataset(CocoDataset):

    CLASSES = ('a', 'b', 'c', 'd', 'e')
```

In `mmdet/datasets/__init__.py`:

```python
from .my_dataset import MyDataset
```

然后在config文件中，可以使用MyDataset。

如果不想要转换为COCO或者PASCAL格式，实际上，我们也定义了一个简单的标注格式并且所有的数据集都可以处理，所有现有的数据集
在线或离线处理以与之兼容。

数据集的标注是一个字典列表，每个字典对应了一个图像，有三个field，`filename`, `width`, `height`,这三个参数在测试时使用，
并且，一个额外的field `ann`用来训练， `ann`也是一个field，它包含了至少两个field：`bboxes` and `labels`，这两者都是numpy.一些数据集可能还有crowd/difficult/ignored bboxes， 我们使用`bboxes_ignore` and `labels_ignore`来代替。

这是一个例子
```
[
    {
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        'ann': {
            'bboxes': <np.ndarray, float32> (n, 4),
            'labels': <np.ndarray, int64> (n, ),
            'bboxes_ignore': <np.ndarray, float32> (k, 4),
            'labels_ignore': <np.ndarray, int64> (k, ) (optional field)
        }
    },
    ...
]
```


有两个方法去转换自定义数据集

- 在线转换
可以用继承自 `CustomDataset`的数据集类来写一个新的数据集类, 并且重新写两个方案。`load_annotations(self, ann_file)` and `get_ann_info(self, idx)`，
就像 [CocoDataset](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/coco.py) and [VOCDataset](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/voc.py).

- 离线转换

转换标注文件的格式为希望的格式，并且保存为pkl或者json文件的格式

### 自定义优化器

在 `mmdet/core/optimizer/copy_of_sgd.py`中定义了一个自定义优化器的例子`CopyOfSGD`

In `mmdet/core/optimizer/my_optimizer.py`:

```python
from .registry import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module
class MyOptimizer(Optimizer):

```

In `mmdet/core/optimizer/__init__.py`:

```python
from .my_optimizer import MyOptimizer
```

Then you can use `MyOptimizer` in `optimizer` field of config files.

### 开发新组件

可以将模型组件归类为4个类型

- backbone: usually an FCN network to extract feature maps, e.g., ResNet, MobileNet.
- neck: the component between backbones and heads, e.g., FPN, PAFPN.
- head: the component for specific tasks, e.g., bbox prediction and mask prediction.
- roi extractor: the part for extracting RoI features from feature maps, e.g., RoI Align.

下边我们显示如何开发新组件

1. Create a new file `mmdet/models/backbones/mobilenet.py`.

```python
import torch.nn as nn

from ..registry import BACKBONES


@BACKBONES.register_module
class MobileNet(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(self, x):  # should return a tuple
        pass

    def init_weights(self, pretrained=None):
        pass
```

2. Import the module in `mmdet/models/backbones/__init__.py`.

```python
from .mobilenet import MobileNet
```

3. Use it in your config file.

```python
model = dict(
    ...
    backbone=dict(
        type='MobileNet',
        arg1=xxx,
        arg2=xxx),
    ...
```

更多的信息将会展示在 [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md).