# mmdetection的安装

mmdetection的github链接：https://github.com/open-mmlab/mmdetection
mmdetection的官方文档：https://mmdetection.readthedocs.io/en/latest/
mmdetection的预训练权重：https://mmdetection.readthedocs.io/en/latest/MODEL_ZOO.html

## mmdetection的安装

1. 创建conda环境
```
conda create -n mmdetection python=3.6
source activate mmdetection
```
2. 安装pytorch及torchvision

```
conda install pytorch torchvision -c pytorch
#也可以上官网选择特定的版本
# https://pytorch.org/ 
```

3. 安装cython
```
conda install cython
```
4. 安装mmcv
```
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install .  #有个点，不能忘
```

5. clone mmdetection的代码库
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
```

6. 安装mmdetection

```
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -v -e . # or "python setup.py develop"
```

