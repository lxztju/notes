---
layout: post
title: "纯python实现经典机器学习算法之AdaBoost"
date: 2020-08-02
description: "纯python实现经典机器学习算法"

tag: 纯python实现经典机器学习算法 
--- 

不调库，纯python实现经典的机器学习算法之AdaBoost

代码放置在github上：[https://github.com/lxztju/machine_learning_python](https://github.com/lxztju/machine_learning_python)

理论知识请参考李航老师统计学习方法，

代码部分主要实现单层树（利用阈值分类）作为基分类器，然后实现Adaboost，算法简单易于实现


### 基分类器的实现
```python
class SingleTree:
    def __init__(self, traindataList, trainlabelList):
        '''
        构建单层的决策树作为AdaBoost的基分类器
        :param traindataList:  输入的数据集的list格式
        :param trainlabelList: 输入训练集的label的list格式
        :param D: 训练数据集的权重
        '''
        self.traindataArr = np.array(traindataList)
        self.trainlabelArr = np.array(trainlabelList)
        self.m, self.n = self.traindataArr.shape
        self.D = [1/ self.m] * self.m  # 初始化数据集权重为均匀分布


    def calcError(self, prediction, trainlabelArr, D):
        '''
        计算在训练数据集上的分类误差率
        :param prediction:  决策树预测出的prediction，与trainlabelArr长度相同
        :param trainlabelArr:  ground truth
        :param D:  训练数据集的权重
        :return: 返回训练误差率
        '''
        # 初始化error
        error = 0

        for i in range(trainlabelArr.size):
            if prediction[i] != trainlabelArr[i]:
                error += D[i]
        return error


    def singleTree(self):
        '''
        构建单层决策树，作为基分类器
        :return:
        '''
        # 利用字典构建一棵树
        # print(self.D)
        tree = {}
        # 切分点，由于数据集读取的过程中，每个特征的取值均为0 和 1,因此选择三个切分点，第一个小于0,第二个0,1之间，第三个大于1
        divides = [-0.5, 0.5, 1.5]
        # 指定规则，对于某个特征，less为小于切分点阈值的为1,大于的为-1
        #                     Over为大于切分点阈值的为-1, 小于的为1
        rules = ['Less', 'Over']
        # 最大的误差值为1,因此初始化为1
        min_error = 1
        # 遍历每个特征，找寻能够使得误差最小值的切分店，与切分规则还有特征值
        for i in range(self.n):
            for divide in divides:

                for rule in rules:
                    #初始化预测的结果为predicition
                    prediction = np.ones(self.m)
                    if rule == 'Less':
                        # 当切分规则为Less时，大于切分点的样本置为-1,因为一开始一开始初始化为1，因此预测为1的可不进行赋值处理
                        prediction[self.traindataArr[:,i] >divide] = -1
                    else:
                        # 当切分点为Over时，小于切分店的样本置为-1
                        prediction[self.traindataArr[:, i] <= divide] = -1
                    # 对于给定的特征、切分点、切分规则，计算相对应的错误率
                    error = self.calcError(prediction, self.trainlabelArr, self.D)
                    # 找到最小的错误率来构建树
                    if error < min_error:
                        # print(prediction, self.traindataArr[:, i], trainlabelList)
                        tree['error'] = error
                        tree['rule'] = rule
                        tree['divide'] = divide
                        tree['feature'] = i
                        tree['Gx'] = prediction
                        min_error = error
        # print(tree, error)
        return tree
```

### AdaBoost实现

继承，基分类器，然后进行boosting

```python
class Adaboost(SingleTree):
    def __init__(self, traindataList, trainlabelList, treeNum = 50):
        super().__init__(traindataList, trainlabelList)

        self.treeNum = treeNum

        self.trees = self.BoostingTree()



    def BoostingTree(self):
        '''
        构建Adaboost
        :return: 返回构建完成的Adaboost模型
        '''
        # 初始化树的列表，每个元素代表一棵树，从前到后一层层
        tree = []
        # 最终的预测值列表，每个元素表示对于每个样本的预测值
        finalPrediction = np.zeros(self.trainlabelArr.size)
        #迭代生成treeNum层的树
        for i in range(self.treeNum):
            # 构建单层的树
            curTree = self.singleTree()
            # 根据公式8.2,计算alpha
            alpha = 1/2 * np.log((1-curTree['error']) / curTree['error'])
            # 保留这一层树的预测值，用于后边权重值的计算
            Gx = curTree['Gx']

            # 计算数据集的权重
            # 式子8.4的分子部分，是一个向量，在array中 *与np.multiply表示元素对应相乘
            # np.dot()是向量点乘
            w = self.D * ( np.exp( -1 * alpha * self.trainlabelArr * Gx))
            # 训练集的权重分布
            self.D = w / sum(w)
            curTree['alpha'] = alpha
            # print(curTree)

            tree.append(curTree)

            #################################
            # 计算boosting的效果，提前中止
            finalPrediction += alpha * Gx
            # print(finalPrediction, self.trainlabelArr, alpha)
            correct_num = sum(np.sign(finalPrediction) == self.trainlabelArr)
            # print(correct_num, finalPrediction, self.trainlabelArr)
            accuracy = correct_num / self.trainlabelArr.size
            logging.info("The {}th Tree, The train data's accuracy is:{}".format(i, accuracy))
            # 如果在训练集上转却率已经达到1,提前中止
            if accuracy == 1:
                break
        return tree
```

完整代码请移步github：[https://github.com/lxztju/machine_learning_python](https://github.com/lxztju/machine_learning_python)



其他算法的实现：

[纯python实现经典机器学习算法](https://zhuanlan.zhihu.com/p/163688301)





参考链接：
		https://github.com/Dod-o/Statistical-Learning-Method_Code

​		https://github.com/fengdu78/lihang-code