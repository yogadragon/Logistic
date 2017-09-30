from numpy import *
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
		return 1.0/(1+exp(-x))

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

data,label = Logis().loaddata()
weights = Logis().gradascend(data,label)
print(weights)
plotBestFit(weights.getA())
