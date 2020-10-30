# Docker 构建自己的镜像

这篇文章主要内容：
1. 构建自己的image镜像文件
2. 在容器中修改image并提交修改
3. 保存新镜像
4. 删除原有的镜像
5. load镜像
6. 封装镜像

* 镜像文件夹需要包含的文件：
1. Dockerfile
2. 源代码文件（我这里主要是python文件）名称为run.py
3. requirements.txt，这个文件包含需要安装的配置环境包

## Dockerfile的编辑
```shell
from python
workdir /code
copy . /code
run pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
cmd ["python", "run.py"]
```

* from： 所有的Dockerfile都要有from命令，有且只有一个，是一个基容器镜像，在这个基容器镜像上构建自己的容器镜像。
* workdir：定义应用所在的顶层目录 
* copy ： 将本地系统中的文件复制到文件系统中，第一个参数为`.`是将文件夹下所有的文件移动 
* run ： 安装指定所需包
* cmd ：执行命令运行代码文件

## 构建image
```
docker build -t image_name .    #image_name为自定义的image name
```

## 在容器中修改
1. 进入容器，执行修改
2. 利用exit推出容器
3. 执行如下命令提交
```
sudo docker commit -m "message"  container_id image_name:Tag
```
4. 利用`docker images`即可看到修改后的image


由于修改后的镜像依赖于原始镜像直接删除则删除不掉，需要先保存修改后的镜像，然后删除原有的镜像。

## 保存新镜像
```
docker save -o 自定义名字.tar 上边修改后的image的名称与tag
```


## 删除原有镜像，然后load新镜像
```
docker image rm image_id
```

```
docker load -i ./上文保存的tar文件的名称
```


## 封装镜像
```
docker save test_docker:latest | gzip > test_docker_today.tar.gz
```