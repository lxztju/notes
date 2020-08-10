---
layout: post
title: "纯python实现经典机器学习算法之SVM算法"
date: 2020-08-01
description: "纯python实现经典机器学习算法"

tag: 纯python实现经典机器学习算法 
--- 

不调库，纯python实现经典的机器学习算法之SVM

代码放置在github上：[https://github.com/lxztju/machine_learning_python](https://github.com/lxztju/machine_learning_python)

理论知识请参考李航老师统计学习方法，

代码的主要内容包括，软间隔，SMO序列最小最优化算法，高斯核技巧，文中代码注释清晰，易读

## 非线性SVM

### 第一部分，读取mnist数据集

```python
def loadData(fileName):
    '''
    加载Mnist数据集
    :param fileName:要加载的数据集路径
    :return: list形式的数据集及标记
    '''
    # 存放数据及标记的list
    dataArr = []
    labelArr = []
    # 打开文件
    fr = open(fileName, 'r')
    # 将文件按行读取
    for line in fr.readlines():
        # 对每一行数据按切割福','进行切割，返回字段列表
        curLine = line.strip().split(',')

        # Mnsit有0-9是个标记，由于是二分类任务，所以仅仅挑选其中的0和1两类作为正负类进行分类
        # if int(curLine[0]) != 0 or int(curLine[0]) !=1: continue
        if int(curLine[0]) == 0 or int(curLine[0]) == 1:
            if int(curLine[0]) == 0:
                labelArr.append(1)
            else:
                labelArr.append(-1)
            dataArr.append([int(num) / 255 for num in curLine[1:]])
        # 存放标记
        # [int(num) for num in curLine[1:]] -> 遍历每一行中除了以第一个元素（标记）外将所有元素转换成int类型
        # [int(num)/255 for num in curLine[1:]] -> 将所有数据除255归一化(非必须步骤，可以不归一化)
        # dataArr.append([int(num)/255 for num in curLine[1:]])

    # 返回data和label
    return dataArr, labelArr
```



### 第二部分计算核函数矩阵



可以在训练过程中 进行计算，这里采用保存所有的矩阵，使用过程中直接查表



```python
    def calcKernel(self):
        '''
        计算核函数矩阵，采用高斯核
        :return: 高斯核矩阵
        '''

        # 高斯核矩阵的大小为m×m
        K = [[0] * self.m for _ in range(self.m)]

        # 遍历Xi， 这个相当于核函数方程中的x
        for i in range(self.m):

            if i % 100 == 0:
                logging.info('Construct The Gaussian Kernel: ({}/{}).'.format(i, self.m))

            Xi = self.traindataArr[i]
            #遍历Xj，相当于公式中的Z
            for j in range(self.m):
                Xj = self.traindataArr[j]
                # 计算||xi-xj||^2
                diff = np.dot((Xi - Xj), (Xi - Xj).T)
                # nisan高斯核参数矩阵
                K[i][j] = np.exp((-1/2) * (diff/(self.sigma ** 2 )))

        # 返回高斯核
        return K
```



### 利用SMO训练SVM

```python
    def train(self, iter = 100):
        '''
        训练SVM分类器
        :param iter: 最大的迭代次数
        :return:  无返回值，训练SVM
        '''
        iterStep = 0   # 迭代的次数，超过迭代次数依然没有收敛，则强制停止
        parameterChanged = 1 # 参数是否发生更改的标志，如果发生更改，那么这个值为1,如果不更改，说明算法已经收敛

        # 迭代训练SVM
        while iterStep < iter and parameterChanged > 0:
            logging.info('Iter:{}/{}'.format(iterStep, iter))

            iterStep += 1
            # 初始化参数变化值为0,如果参数改变，说明训练过程正在进行，那么parameterChanged置一
            parameterChanged = 0

            # 利用SMO更新的两个变量
            E1, E2, i, j = self.getAlpha()

            y1 = self.trainlabelArr[i]
            y2 = self.trainlabelArr[j]

            alpha1Old = self.alpha[i]
            alpha2Old = self.alpha[j]

            # 计算边界
            if y1 == y2:
                L = max(0, alpha2Old+alpha1Old-self.C)
                H = min(self.C, alpha2Old + alpha1Old)
            else:
                L = max(0, alpha2Old-alpha1Old)
                H = min(self.C, self.C+alpha2Old+alpha1Old)

            if L == H:
                continue
            # print(L, H, alpha1Old, alpha2Old)
            k11 = self.kernel[i][i]
            k22 = self.kernel[j][j]
            k12 = self.kernel[i][j]
            k21 = self.kernel[j][i]

            eta = (k11 + k22 - 2*k12)

            # 如果eta为0,在后边的分母中会报错
            if eta <= 0:
                continue

            alpha2NewUnc = alpha2Old + y2 * (E1-E2)/ eta
            # print(E1, E2, eta, alpha2Old, alpha2NewUnc)
            if alpha2NewUnc <L:
                alpha2New = L
            elif alpha2NewUnc > H:
                alpha2New = H
            else:
                alpha2New = alpha2NewUnc
            # print(alpha2New, alpha2Old)
            alpha1New = alpha1Old + y1 * y2 * (alpha2Old - alpha2New)

            b1New = -1 * E1 - y1 * k11 * (alpha1New - alpha1Old) \
                    - y2 * k21*(alpha2NewUnc - alpha2Old) + self.b

            b2New = -1 * E2 - y1 * k12 * (alpha1New - alpha1Old) \
                    - y2 * k22 * (alpha2New - alpha2Old) + self.b

            # 依据α1和α2的值范围确定新b
            if (alpha1New > 0) and (alpha1New < self.C):
                bNew = b1New
            elif (alpha2New > 0) and (alpha2New < self.C):
                bNew = b2New
            else:
                bNew = (b1New + b2New) / 2

            self.alpha[i] = alpha1New
            self.alpha[j] = alpha2New
            self.b = bNew

            self.E[i] = self.calc_Ei(i)
            self.E[j] = self.calc_Ei(j)
            # parameterChanged = 1
            # print(math.fabs(alpha2New - alpha2Old))
            # 如果α2的改变量过于小，就认为该参数未改变，不增加parameterChanged值
            # 反之则自增1
            if math.fabs(alpha2New - alpha2Old) >= 0.00001:
                parameterChanged = 1
            # break
        #全部计算结束后，重新遍历一遍α，查找里面的支持向量
        for i in range(self.m):
            #如果α>0，说明是支持向量
            if self.alpha[i] > 0:
                #将支持向量的索引保存起来
                self.supportVecIndex.append(i)

        logging.info('Training process is Done !!!!')
```







完整代码请移步github：[https://github.com/lxztju/machine_learning_python](https://github.com/lxztju/machine_learning_python)







其他算法的实现：

[纯python实现经典机器学习算法](https://zhuanlan.zhihu.com/p/163688301)





参考链接：https://github.com/Dod-o/Statistical-Learning-Method_Code
		https://github.com/fengdu78/lihang-code

