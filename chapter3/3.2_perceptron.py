# -*- coding: utf-8 -*-

# 感知机出现于1957年，由Rosenblatt提出，是一个简单的二分类问题，1960年，Widrow制定了用于实际中训练感知机的过程，也被称为最小平方问题，这两个理念的结合产生了一个不错的线性分类器，就是感知机（Perceptron）

# 感知机的大概思路是：构建一个线性分类模型，这个模型的输入是样本的特征向量，输出为样本的类别（+1或-1），感知机对应于空间中将样本划分为两类的超平面。感知机构建了基于误分类的损失函数，只关注被误分类的样本，利用梯度下降法对损失函数进行最优化。

# 以下代码通过调用sklearn中的感知机，对经典的鸢尾花数据做分类
# 首先导入鸢尾花数据
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
               'iris/iris.data',header=None)
df.tail()
y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
x=df.iloc[0:100,[0,2]].values

# 我们先将导入的数据打印出来
plt.scatter(x[:50,0],x[:50,1],color='blue',marker='x',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1],color='red',marker='o',label='versicolor')
plt.xlabel('petal length (cm)')
plt.ylabel('sepal length (cm)')
plt.legend(loc='upper left')
plt.show()

# 针对以上数据，我们导入sklearn中的Perceptron模块，对数据进行分类，并将分类结果打印出来
plt.scatter(x[:50,0],x[:50,1],color='blue',marker='x',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1],color='red',marker='o',label='versicolor')
plt.xlabel('petal length (cm)')
plt.ylabel('sepal length (cm)')
plt.legend(loc='upper left')

from sklearn.linear_model import Perceptron
clf=Perceptron(fit_intercept=False,n_iter=20,shuffle=False)
clf.fit(x,y)
line_x = np.arange(4,9)
line_y = line_x * (-clf.coef_[0][0] / clf.coef_[0][1]) - clf.intercept_
plt.plot(line_x,line_y)
plt.show()