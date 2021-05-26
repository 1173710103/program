import cv2 as cv
import numpy as np
from sklearn.linear_model import LinearRegression   #引入多元线性回归算法模块进行相应的训练
from sklearn.model_selection import train_test_split


X = []
'''
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
'''
x = [1.40,190,108,90]
X.append(x)
x = [1.41,190,108,90]
X.append(x)
x = [1.43,196,112,95]
X.append(x)
x = [1.45,197,115,95]
X.append(x)
x = [1.51,199,117,99]
X.append(x)
x = [1.52,200,120,99]
X.append(x)
x = [1.57,204,128,105]
X.append(x)
x = [1.59,205,130,106]
X.append(x)
x = [1.63,209,140,110]
X.append(x)
x = [1.65,209,142,110]
X.append(x)
    
U,E,VT = np.linalg.svd(X)

simple2=LinearRegression()
x_train = X
y_train = [93,93,94,94,95,95,96,96,97,97]
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