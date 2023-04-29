
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
import math
test_real=[]
test_pred=[]
def computeCorrelation(X,Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0,len(X)):
        diffXXBar = X[i] -xBar
        diffYYBar = Y[i] -yBar
        SSR +=(diffXXBar*diffYYBar)
        varX += diffXXBar**2
        varY += diffYYBar**2
    SST = math.sqrt(varX*varY)
    return SSR/SST
def cal_acc():
    file=open('valfinall.csv',encoding='utf-8-sig')
    for line in file.readlines():
        lineArr = line.strip().split(',')
        test_pred.append(float(lineArr[1]))
        test_real.append(float(lineArr[2]))
    p=computeCorrelation(test_real,test_pred)
    print('The last learn Pearson correlation coefficien is ',p)
    R = r2_score(test_real,test_pred)
    print('The lase learn r2_score is ',R)
    mae = mean_absolute_error(test_real,test_pred)
    print('The last learn Mean absolute error is ', mae)
cal_acc()

