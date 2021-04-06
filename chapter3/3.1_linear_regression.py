# -*- coding: utf-8 -*-

# 线性回归是利用数理统计中回归分析，来确定两种或两种以上变量间相互依赖的定量关系的一种统计分析方法，运用十分广泛。其表达形式为y = w'x+e，e为误差服从均值为0的正态分布。回归分析中，只包括一个自变量和一个因变量，且二者的关系可用一条直线近似表示，这种回归分析称为一元线性回归分析。如果回归分析中包括两个或两个以上的自变量，且因变量和自变量之间是线性关系，则称为多元线性回归分析。

# 首先我们来看一个一元线性回归的例子，并通过这个例子认识其实际用途

# 以下是城市A房屋面积与价格（万）的一组数据，我们首先通过matplotlib将其画出来

import matplotlib.pyplot as plt

def draw(size=None):
    plt.figure(figsize=size)
    plt.title('房价-面积关系图')
    plt.xlabel('面积（平方米）')
    plt.ylabel('价格（万元）')
    plt.axis([50, 150, 50, 150])
    plt.grid(True)
    return plt
plt = draw()
x = [[53], [75], [92], [110], [135]]
y = [[63], [76], [98], [103], [124]]
plt.plot(x, y, 'k.')
plt.show()

# 针对以上数据，我们导入sklearn中的线性回归模块对数据进行模型构建
import numpy as np
from sklearn import linear_model

model_1 = linear_model.LinearRegression()
model_1.fit(x, y)

#通过以上代码，我们训练出了一个线性模型，是通过简单的最小二乘法训练出来的
#现在我们将模型训练结果打印出来
plt = draw()
plt.plot(x, y, 'k.')
x2 = [[50], [90], [120], [150]]
model = linear_model.LinearRegression()
model.fit(x,y)
y2 = model.predict(x2)
plt.plot(x2, y2, 'g-')
plt.show()