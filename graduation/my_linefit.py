import cv2
import numpy as np
import pandas as pd

##最小二乘法
from scipy.optimize import leastsq  ##引入最小二乘法算法
##需要拟合的函数func :指定函数的形状
def func(p,x):
    k,b=p
    return k*x+b

##偏差函数：x,y都是列表:这里的x,y更上面的Xi,Yi中是一一对应的
def error(p,x,y):
    return func(p,x)-y

def linefit(X,Y):
    #k,b的初始值，可以任意设定,经过几次试验，发现p0的值会影响cost的值：Para[1]
    p0=[1,0]
    
    #把error函数中除了p0以外的参数打包到args中(使用要求)
    Para=leastsq(error,p0,args=(X,Y))
    
    #读取结果
    k,b=Para[0]
    return k,b

if __name__ == '__main__':
    X = []
    Y = []
    #读取类别文件
    label_csv = pd.read_csv("./facedemo/label.csv", index_col = 'index')
    for i in range(9):
        filepath ='D:/MY/zihaoopencv-master/zihaoopencv-master/facedemo/face/'+str(i+1)+'.png'
        img = cv2.imread(filepath)
        b,g,r = cv2.split(img)
        red = np.var(r)/np.mean(r)
        blue = np.var(b)/np.mean(b)
        R = red/blue
        X.append(round(R,2))
        Y.append(label_csv.values[i][1])
    X = np.array(X)
    Y = np.array(Y)
    k,b = linefit(X,Y)