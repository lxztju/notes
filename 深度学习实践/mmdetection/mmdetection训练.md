# mmdetection训练

## config文件的修改，以cascade_rcnn_r50_fpn_1x.py为例
```
#参考部分，https://blog.csdn.net/hajlyx/article/details/85991400
model = dict(
    type='CascadeRCNN',  #model的名称类型
    num_stages=3,     #stage的数目为3
    pretrained='torchvision://resnet50',   #预训练模型的参数
    backbone=dict(
        type='ResNet',                      #backbone的类型
        depth=50,                           #网络的层数
        num_stages=4,                       #backbone的stage的数目
        out_indices=(0, 1, 2, 3),           #输出的stage的数目
        frozen_stages=1,                    #冻结的stage的数量，-1表示所有的stage都更新参数
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),                   # 网络风格：如果设置pytorch，则stride为2的层是conv3x3的卷积层；如果设置caffe，则stride为2的层是第一个conv1x1的卷积层
    neck=dict(
        type='FPN',                         # neck类型为FPN
        in_channels=[256, 512, 1024, 2048], # 输入的各个stage的通道数
        out_channels=256,                   # 输出特征层的通道数
        num_outs=5),                           # 输出的特征层的数量
    rpn_head=dict(
        type='RPNHead',                     # RPN网络类型
        in_channels=256,                    #RPN的输入通道数
        feat_channels=256,                  # 特征层的通道数
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],      # anchor的宽高比
        anchor_strides=[4, 8, 16, 32, 64],  #anchor对应于原图的感受野
        target_means=[.0, .0, .0, .0],         #均值
        target_stds=[1.0, 1.0, 1.0, 1.0],       #方差
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, loss_weight=1.0),  #use_sigmoid=False则采用softmax
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',              # RoIExtractor类型
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),  # ROI具体参数：ROI类型为ROIalign，输出尺寸为7，sample数为2
        out_channels=256,           #输出通道数
        featmap_strides=[4, 8, 16, 32]),  ## 特征图的步长
    bbox_head=[
        ## 这里对应cascade rcnn的三个stage
        dict(
            type='SharedFCBBoxHead',    #全连接层的类型
            num_fcs=2,                  #全连接层的数目
            in_channels=256,            #输入通道
            fc_out_channels=1024,          #输出通道数
            roi_feat_size=7,            # ROI特征层尺寸
            num_classes=144,            #类别数，为分类的类别数加一，包含一个背景类
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=True,      #是否采用class_agnostic的方式来预测，class_agnostic表示输出bbox时只考虑其是否为前景，后续分类的时候再根据该bbox在网络中的类别得分来分类，也就是说一个框可以对应多个类别
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=144,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.05, 0.05, 0.1, 0.1],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=144,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.033, 0.033, 0.067, 0.067],
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
    ])
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',              # RPN网络的正负样本划分
            pos_iou_thr=0.4,                    #正样本的IOU阈值
            neg_iou_thr=0.4,                    #负样本的IOU阈值
            min_pos_iou=0.4,                    # 正样本的iou最小值。如果assign给ground truth的anchors中最大的IOU低于0.3，则忽略所有的anchors，否则保留最大IOU的anchor
            ignore_iof_thr=-1),                 # 忽略bbox的阈值，当ground truth中包含需要忽略的bbox时使用，-1表示不忽略
        sampler=dict(
            type='OHEMSampler',                 # 正样本提取器的类型
            num=256,                            # 需提取的正负样本数量
            pos_fraction=0.5,                   # 正样本比例
            neg_pos_ub=-1,                      # 最大负样本比例，大于该比例的负样本忽略，-1表示不忽略
            add_gt_as_proposals=False),         # 把ground truth加入proposal作为正样本
        allowed_border=0,
        pos_weight=-1,                          # 正样本权重，-1表示不改变原始的权重
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,                # 在所有的rpn层内做nms
        nms_pre=2000,                           #nms处理前保留最大的proposal的数目
        nms_post=2000,                          #nms处理后保存到最大的数目
        max_num=2000,                           #处理后的proposal的数量
        nms_thr=0.7,                            #nms阈值
        min_bbox_size=0),                          #最小的额bbox的尺寸
    rcnn=[
        #三个stage的设置
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='OHEMSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                ignore_iof_thr=-1),
            sampler=dict(
                type='OHEMSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                ignore_iof_thr=-1),
            sampler=dict(
                type='OHEMSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ],
    stage_loss_weights=[1, 0.5, 0.25])
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='soft_nms', iou_thr=0.5), max_per_img=100))# 这里可以换为soft_nms
# dataset settings
dataset_type = 'CocoDataset'                #数据类型为COCO格式
data_root = '/home/luxiangzhe/haihua/train/'        #数据集的存储位置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(960,540), keep_ratio=True), #这里可以更换多尺度[(),()]
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
        img_scale=(960,540),
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
data = dict(
    imgs_per_gpu=4,                         #每个gpu的图像的数目，类似于每个gpu的batchsize
    workers_per_gpu=4,                      #每个gpu的线程数目
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_nouncat.json', # 更换自己的json文件
        img_prefix=data_root , # images目录
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'train_nouncat.json',
        img_prefix=data_root ,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +'train_nouncat.json',
        img_prefix=data_root ,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001) # lr = 0.00125*batch_size，不能过大，否则梯度爆炸。
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[20, 70, 110])
checkpoint_config = dict(interval=1)    ##每interval保存一个权重文件
# yapf:disable
log_config = dict(
    interval=64,                #每interval写入日志文件
    hooks=[
        dict(type='TextLoggerHook'), # 控制台输出信息的风格
        # dict(type='TensorboardLoggerHook') # 需要安装tensorflow and tensorboard才可以使用
    ])
# yapf:enable
# runtime settings
total_epochs = 150          # 一共需要运行的epoch数目
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/luxiangzhe/haihua/mm_workdir/cascade_rcnn_r50_fpn_1x' # 日志目录
# load_from = '/home/luxiangzhe/haihua/mm_workdir/cascade_rcnn_r50_fpn_1x/cascade_rcnn_r50_fpn_1x_20190501-3b6211ab.pth' # 模型加载目录文件
# load_from = '/home/luxiangzhe/haihua/mm_workdir/cascade_rcnn_r50_fpn_1x/epoch_coco_pretrained_weights_classes_144.pth'
load_from = '/home/luxiangzhe/haihua/mm_workdir/cascade_rcnn_r50_fpn_1x/epoch_1.pth'
resume_from =  None
workflow = [('train', 1)]
```

## 单gpu训练
```
python ./tools/train.py configs/cascade_rcnn_r50_fpn_1x.py --gpus 1
```

## 多gpu训练
```
./tools/dist_train.sh configs/cascade_rcnn_r50_fpn_1x.py 2
```
