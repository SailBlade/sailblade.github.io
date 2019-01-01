---
layout: post
title: "Linear SVM 分类"
description: "machine learning"
categories: [machine learning]
tags: [machine learning]
redirect_from: 
  - /2018/11/02/
---  
* Kramdown table of contents
{:toc .toc}
---

##  背景   
SVM的分类过程一般为如下所示，从中可知求解二次规划是SVM学习之路上的必经之路，这篇文章给出了CVXOPT求解的典型范例，方便大家学习。      
1. 将训练问题转换为二次规划问题，并拆出二次规划对应的因式；   
2. 将因式代入二次规划的python解决包cvxopt求得目标函数的w，b；
3. 将训练出的w，b绘制在样本点上获得线性分类目标函数


##  linear SVM 分类范例   
#### 问题描述：   
利用linear SVM 对已知的线性数据进行分类 `Y = sign(w1*X1 + w2*X2 + b)`。待训练数据格式如下：   

| X1 | X2 | Y |   
| ------------- | ------------- |  ------------- |   
| 0.568304|  0.568283|  1|    
| 0.310968|  0.310956| -1|    
| 0.103376|  0.103373| -1|    
| 0.0531882|  0.053218| -1|    
| 0.97006|  0.970064|  1 |    
| …… |  ……|  …… |

可视化如下：   
![](http://images.sailblade.com/%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE%E5%88%86%E5%B8%832.png)


### 1、将训练问题转换为二次规划问题，并拆出二次规划对应的因式    
![](http://images.sailblade.com/%E5%9B%BE%E7%89%873.png)   
需要将上述SVM的最大间隔转化为如下形式：   
![](http://images.sailblade.com/%E5%9B%BE%E7%89%872.png)    
 

### 2、对上述待训练数据的处理过程   
#### 1) 由于上述训练数据分别有两个参数，X1，X2，所以需要转换的二次规划问题目标为：   
![](http://images.sailblade.com/%E5%85%AC%E5%BC%8F5.png)   

#### 2) 对目标函数限制条件转换为二次规划的转换流程为：   
![](http://images.sailblade.com/%E5%85%AC%E5%BC%8F2018110301.png)   

#### 3) 由上述流程转换后可得：   
![](http://images.sailblade.com/%E5%85%AC%E5%BC%8F2018110302.png)   

#### 4) 由上述流程转换后可得：   
![](http://images.sailblade.com/%E5%85%AC%E5%BC%8F2018110303.png)   

#### 5) 利用CVXOPT求得最优解如下：   
![](http://images.sailblade.com/%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE%E5%88%86%E5%B8%833.png)


##  源码     

```python       
import numpy as np
import cvxopt as cvx
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

def DoLinearSVM():

    class LinearSVM():
        def __init__(self):
            #self._b = np.zeros(1,dtype=np.float64)
            self._w = np.zeros(2)

        def __sign__(self,a):
            if a >= 0.0:
                return 1
            elif a < 0.0:
                return -1

        def __generateTrainData__(self):
            X1 = np.random.uniform(-1.0,1.0,200)
            X2 = np.random.uniform(-1.0,1.0,200)
            B  = 0.03
            W1=  0.3
            W2=  0.8
            Y = np.zeros(200)
            for loop in range(Y.shape[0]):
                Y[loop] = self.__sign__(W1 * X1[loop] + W2 * X2[loop] + B)

            '''
            noiseArray = np.random.rand(Y.shape[0])
            for loop in range(Y.shape[0]):
                if (noiseArray[loop]) > 0.9:
                    Y[loop] = -1 * Y[loop]
            '''

            Xarray = np.stack((X1, X2), axis=-1)

            self._X  = Xarray
            self._Y = Y
            self.sketchTrainDataMap()




        def train(self):
            Q = cvx.matrix(np.diag([0.0, 1.0,1.0])) # Q =[[0,0,0],[0,1,0],[0,0,1]]

            for loop in range(self._Y.shape[0]):
                if loop == 0:
                    A = np.array([[-1.0*self._Y[loop],
                                 np.multiply(self._Y[loop] *-1.0,self._X[loop][0]),
                                 np.multiply(self._Y[loop] *-1.0,self._X[loop][1])]])
                else:
                    rowArray = np.array([[-1.0*self._Y[loop],
                                 np.multiply(self._Y[loop] * -1.0,self._X[loop][0]),
                                 np.multiply(self._Y[loop] * -1.0,self._X[loop][1])]])
                    A = np.concatenate((A, rowArray), axis=0)

            cn_array = np.multiply(-1,np.ones((200,)))
            cn = cvx.matrix(cn_array)

            p = cvx.matrix([0.0, 0.0, 0.0])
            A = cvx.matrix(A)
            cn = cvx.matrix(cn)

            sol = cvx.solvers.qp(Q, p, A, cn)  # 调用优化函数solvers.qp求解
            self.result = sol['x']
            self._b = np.float64(self.result[0])
            self._w1 =  np.float64(self.result[1])
            self._w2 =  np.float64(self.result[2])
            self.sketchMap()


        def sketchMap(self):
            correctListX1 = []
            correctListX2 = []

            errorListX1 = []
            errorListX2 = []

            for loop in range(self._Y.shape[0]):
                if self._Y[loop] == 1:
                    correctListX1.append(self._X[loop][0])
                    correctListX2.append(self._X[loop][1])
                else:
                    errorListX1.append(self._X[loop][0])
                    errorListX2.append(self._X[loop][1])

            correctArrayX1 = np.array(correctListX1)
            correctArrayX2 = np.array(correctListX2)
            errorArrayX1   = np.array(errorListX1)
            errorArrayX2    = np.array(errorListX2)

            optimalY = []

            for loop in range(self._X.shape[0]):
                val = self._w1 * self._X[loop][0] + self._w2 * self._X[loop][1]+self._b
                optimalY.append(val)

            optimalYArray = np.array(optimalY)

            fig, ax = plt.subplots(1, 1)
            ax = fig.gca(projection='3d')
            ax.scatter(correctArrayX1, correctArrayX2,c='g', marker="x")
            ax.scatter(errorArrayX1, errorArrayX2,c='b', marker="+")

            x1Array,x2Array = np.split(self._X,self._X.shape[1],axis = 1)
            #ax.scatter(x1Array, x2Array, optimalYArray,c='orangered', marker=".")
            x1Array = np.reshape(x1Array,(self._X.shape[0]))
            x2Array = np.reshape(x2Array, (self._X.shape[0]))
            print (type(x1Array),self._X.shape[0],x1Array.shape)
            x1SplitResult = np.split(x1Array, [10, self._X.shape[0]],axis = 0)
            x2SplitResult = np.split(x2Array, [10, self._X.shape[0]],axis = 0)
            x1Array, x2Array = np.meshgrid(x1SplitResult[0], x2SplitResult[0])  # 将坐标向量变为坐标矩阵，列为x的长度，行为y的长度
            print(x1Array.shape,x2Array.shape)
            optimalYArray = self._w1 * x1Array + self._w2 * x2Array +self._b
            ax.plot_surface(x1Array, x2Array, optimalYArray, rstride=1, cstride=1,color='coral', linewidth=0, antialiased=True,alpha=0.1)
            ax.set_xlabel("x1-label", color='r')
            ax.set_ylabel("x2-label", color='g')
            ax.set_zlabel("Y-label", color='b')

            fig.suptitle('Linear SVM ')
            plt.savefig("Linear_SVM.png")
            plt.show()

        def sketchTrainDataMap(self):
            correctListX1 = []
            correctListX2 = []

            errorListX1 = []
            errorListX2 = []

            for loop in range(self._Y.shape[0]):
                if self._Y[loop] == 1:
                    correctListX1.append(self._X[loop][0])
                    correctListX2.append(self._X[loop][1])
                else:
                    errorListX1.append(self._X[loop][0])
                    errorListX2.append(self._X[loop][1])

            correctArrayX1 = np.array(correctListX1)
            correctArrayX2 = np.array(correctListX2)
            errorArrayX1   = np.array(errorListX1)
            errorArrayX2    = np.array(errorListX2)


            fig, ax = plt.subplots(1, 1)
            ax = fig.gca(projection='3d')
            ax.scatter(correctArrayX1, correctArrayX2,c='g', marker="x")
            ax.scatter(errorArrayX1, errorArrayX2,c='b', marker="+")
            fig.suptitle('Linear SVM ')

            plt.show()

        def loadTrainData(self,fileLocation):
            data = []
            label = []
            lineNum = 0
            with open(fileLocation) as f:
                line = f.readline()
                while line:
                    lineArray = line.split()
                    for i in range(len(lineArray) - 1):
                        data.append(float(lineArray[i]))
                    label.append(int(lineArray[2]))
                    line = f.readline()
                    lineNum += 1
            dataArray = np.array(data)
            dataArray = dataArray.reshape(lineNum, 2)

            labelArray = np.array(label)
            self._X  = dataArray
            self._Y = labelArray

    linearSVMObj = LinearSVM()
    linearSVMObj.__generateTrainData__()
    #linearSVMObj.loadTrainData( 'G:\\林轩田教程\\MachineLearningFoundations\\homework5\\data\\question13_TRAIN.txt')
    linearSVMObj.train()


if __name__ == "__main__":
    DoLinearSVM()


```    















