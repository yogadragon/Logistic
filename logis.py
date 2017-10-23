from numpy import *
from random import *
class Logis(object):
	def loaddata(self):
		datamat,labelmat = [],[]
		fr = open('testSet.txt')
		for line in fr.readlines():
			linearr = line.strip().split()
			datamat.append([1.0,float(linearr[0]),float(linearr[1])])
			labelmat.append(float(linearr[2]))
		return datamat,labelmat

	def sigmoid(self,x):
		return 1.0/(1.0+exp(-x))

	def classify(self,x,weights):
		prob = self.sigmoid(sum(x*weights))
		if prob>0.5:
			return 1.0
		else:
			return 0.0

	def gradascend(self,datain,labelin):   # 梯度上升方法,不同函数尽量不要用同一个变量名
		datam = mat(datain)
		labelm = mat(labelin).transpose()
		m,n = shape(datam)
		weight = ones((n,1))  # datam * weight = labelm
		alpha = 0.001  # 学习因子
		cycles = 500  # 循环次数
		for k in range(cycles):
			h = self.sigmoid(dot(datam,weight)) # 矩阵乘法
			error = labelm - h
			weight = weight +alpha*datam.transpose()*error
		return weight
	
	def stogradascend(self,datain,labelin,numiter):   # 随机梯度下降算法
		m,n = shape(datain)
		weight = ones(n)
		for j in range(numiter):
			dataindex = list(range(m))
			for i in range(m):
				alpha = 4.0/(1.0+i+j)+0.01   # 每次学习系数都变化,时间越久学习系数越小
				randindex = int(uniform(0,len(dataindex)))
				h = self.sigmoid(sum(datain[randindex]*weight))  # 这里不是矩阵乘法,就是对应相乘,sum是求内积
				error = labelin[randindex] - h
				weight = weight+alpha*error*array(datain[randindex])
				del(dataindex[randindex])
		return weight

	def colictest(self,k):
		frtrain = open('horseColicTraining.txt')
		frtest = open('horseColicTest.txt')
		trainset,trainlabels = [],[]
		for line in frtrain.readlines():
			cline = line.strip().split('\t')
			linearr = []
			for i in range(21):
				linearr.append(float(cline[i]))
			trainset.append(linearr)
			trainlabels.append(float(cline[21]))
		trainweight = self.stogradascend(array(trainset),array(trainlabels),k)
		errorcount = 0
		numtest = 0.0
		for line in frtest.readlines():
			numtest += 1.0
			cline = line.strip().split('\t')
			linearr = []
			for i in range(21):
				linearr.append(float(cline[i]))
			if int(self.classify(array(linearr),trainweight))!=int(cline[21]):
				errorcount += 1
		errorrate = float(errorcount)/numtest
		return errorrate



def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=Logis().loaddata()
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
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

# data,label = Logis().loaddata()
# weights = Logis().stogradascend(data,label,150)
# print(weights)
# plotBestFit(weights)
fr = open('logistic.plt','w')
fr.write('variables = k,err'+'\n')
for k in range(100,1000,50):
	err1 = Logis().colictest(k)
	fr.write(str(float(k))+','+str(float(err1))+'\n')

fr.close()
