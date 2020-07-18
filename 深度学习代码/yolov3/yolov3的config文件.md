# yolov3的config文件

If you open the configuration file, you will see something like this.


```
[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear
```
We see 4 blocks above. Out of them, 3 describe convolutional layers, followed by a shortcut layer. 
A shortcut layer is a skip connection, like the one used in ResNet. 
There are 5 types of layers that are used in YOLO:

## Convolutional
```
[convolutional]
batch_normalize=1  
filters=64  
size=3  
stride=1  
pad=1  
activation=leaky
```
## Shortcut
```
[shortcut]
from=-3  
activation=linear  
```
A shortcut layer is a skip connection, akin to the one used in ResNet. 
The from parameter is -3, which means the output of the shortcut layer is obtained by adding feature maps 
from the previous and the 3rd layer backwards from the shortcut layer.

## Upsample
```
[upsample]
stride=2
```
Upsamples the feature map in the previous layer by a factor of stride using bilinear upsampling.

## Route
```
[route]
layers = -4

[route]
layers = -1, 61
```
The route layer deserves a bit of explanation. 
It has an attribute layers which can have either one, or two values.

When layers attribute has only one value, it outputs the feature maps of the layer indexed by the value. 
In our example, it is -4, so the layer will output feature map from the 4th layer backwards from the Route layer.

When layers has two values, it returns the concatenated feature maps of the layers indexed by it's values. In our example it is -1, 61, and the layer will output feature maps from the previous layer (-1) and the 61st layer, concatenated along the depth dimension.

## YOLO
```
[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .5
truth_thresh = 1
random=1
```
YOLO layer corresponds to the Detection layer described in part 1. 
The anchors describes 9 anchors, but only the anchors which are indexed by attributes of the mask tag are used. Here, the value of mask is 0,1,2, which means the first, second and third anchors are used. 
This make sense since each cell of the detection layer predicts 3 boxes. 
In total, we have detection layers at 3 scales, making up for a total of 9 anchors.

## Net
```
[net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=16
width= 320
height = 320
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1
```
There's another type of block called net in the cfg, but I wouldn't call it a layer as it only describes information about the network input and training parameters. It isn't used in the forward pass of YOLO. However, it does provide us with information like the network input size, which we use to adjust anchors in the forward pass.