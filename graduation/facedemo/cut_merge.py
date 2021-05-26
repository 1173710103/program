import cv2 as cv
from PIL import Image

for i in range(9):
    img = cv.imread('D:/MY/zihaoopencv-master/zihaoopencv-master/facedemo/face/'+str(i+1)+'.png')
    #cv.imshow('img',src)
    print('src.shape:',img.shape)
    
    b,g,r = cv.split(img) # 分割后单独显示
    #cv.imshow('b',b)
    print('b.shape:',b.shape)
    #cv.imshow('g',g)
    print('g.shape:',g.shape)
    #cv.imshow('r',r)
    print(type(r+b))
    im = Image.fromarray(r+b)
    print(im.size)
    im.save('D:/MY/zihaoopencv-master/zihaoopencv-master/facedemo/face/'+str(i+1)+'.png')

    
    cv.waitKey(0)
    cv.destroyAllWindows()