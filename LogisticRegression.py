# -*- coding: gb18030 -*-
__author__ = 'jinyu'
from numpy import *
import math

##
# 读取训练数据
# #
def loadDataSet(dataFileName):
    dataMat = []
    labelMat = []
    fr = open(dataFileName)
    for line in fr:
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))

    return dataMat, labelMat
##
# sigmod函数
# #
def sigmoid(x):
    return 1.0 / (1+math.exp(-x))

##
# 梯度上升法
# #
def gradientAscent(dataMat, labelMat, alpha):
    m = len(dataMat)        #训练集个数
    n = len(dataMat[0])     #数据特征纬度
    theta = [0] * n

    iter = 1000
    while(iter):
        for i in range(m):
            hypothesis = sigmoid(computeDotProduct(dataMat[i], theta))
            error = labelMat[i] - hypothesis
            gradient = computeTimesVect(dataMat[i], error)
            theta = computeVectPlus(theta, computeTimesVect(gradient, alpha))
        iter -= 1
    return theta

##
# 牛顿法
# #
def newtonMethod(dataMat, labelMat, iterNum=10):
    m = len(dataMat)        #训练集个数
    n = len(dataMat[0])     #数据特征纬度
    theta = [0.0] * n

    while(iterNum):
        gradientSum = [0.0] * n
        hessianMatSum = [[0.0] * n]*n
        for i in range(m):
            try:
                hypothesis = sigmoid(computeDotProduct(dataMat[i], theta))
            except:
                continue
            error = labelMat[i] - hypothesis
            gradient = computeTimesVect(dataMat[i], error/m)
            gradientSum = computeVectPlus(gradientSum, gradient)
            hessian = computeHessianMatrix(dataMat[i], hypothesis/m)
            for j in range(n):
                hessianMatSum[j] = computeVectPlus(hessianMatSum[j], hessian[j])

        #计算hessian矩阵的逆矩阵有可能异常，如果捕获异常则忽略此轮迭代
        try:
            hessianMatInv = mat(hessianMatSum).I.tolist()
        except:
            continue
        for k in range(n):
            theta[k] -= computeDotProduct(hessianMatInv[k], gradientSum)

        iterNum -= 1
    return theta

##
# 计算hessian矩阵
# #
def computeHessianMatrix(data, hypothesis):
    hessianMatrix = []
    n = len(data)

    for i in range(n):
        row = []
        for j in range(n):
            row.append(-data[i]*data[j]*(1-hypothesis)*hypothesis)
        hessianMatrix.append(row)
    return hessianMatrix

##
# 计算两个向量的点积
# #
def computeDotProduct(a, b):
    if len(a) != len(b):
        return False
    n = len(a)
    dotProduct = 0
    for i in range(n):
        dotProduct += a[i] * b[i]
    return dotProduct

##
# 计算两个向量的和
# #
def computeVectPlus(a, b):
    if len(a) != len(b):
        return False
    n = len(a)
    sum = []
    for i in range(n):
        sum.append(a[i]+b[i])
    return sum

##
# 计算某个向量的n倍
# #
def computeTimesVect(vect, n):
    nTimesVect = []
    for i in range(len(vect)):
        nTimesVect.append(n * vect[i])
    return nTimesVect

def plotBestFit(dataMat, labelMat, weights):
    import matplotlib.pyplot as plt
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(10.0, 65.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest(tainFileName, testFileName):
    frTrain = open(tainFileName); frTest = open(testFileName)
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = newtonMethod(trainingSet, trainingLabels, 10)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

dataMat, labelMat = loadDataSet("data/ex4.dat")
theta = newtonMethod(dataMat, labelMat, 10)
plotBestFit(dataMat, labelMat, array(theta))

#colicTest('data/horseColicTraining.txt', 'data/horseColicTest.txt')


