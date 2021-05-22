import cv2 as cv
import numpy as np
from sklearn.linear_model import LinearRegression   #引入多元线性回归算法模块进行相应的训练
from sklearn.model_selection import train_test_split


X = []
#读取人脸图像
for i in range(9):
    filepath = 'C:/Users/10981/Desktop/bishe/test/face/' + str(i+1) +'.jpg'
    img = cv.imread(filepath)
    b,g,r = cv.split(img)
    hb = np.mean(b)
    hg = np.mean(g)
    hr = np.mean(r)
    ir = np.var(r)/hr
    ib = np.var(b)/hb
    R = ir/ib
    x = [round(R,2),round(hg,2),round(hr,2),round(hb,2)]
    X.append(x)
    
U,E,VT = np.linalg.svd(X)

simple2=LinearRegression()
x_train = X
y_train = [97,98,97,99,98,98,98,98,98]
simple2.fit(x_train,y_train)

p = []
p.append(simple2.intercept_)
for i in range(4):
    p.append(simple2.coef_[i])
print(p)

#test
filepath = 'C:/Users/10981/Desktop/bishe/test/face/10.jpg'
img = cv.imread(filepath)
b,g,r = cv.split(img)
hb = np.mean(b)
hg = np.mean(g)
hr = np.mean(r)
ir = np.var(r)/hr
ib = np.var(b)/hb
R = ir/ib
x_test = [[round(R,2),round(hg,2),round(hr,2),round(hb,2)]]
y_predict = simple2.predict(x_test)
print(y_predict)