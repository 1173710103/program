import cv2
import numpy as np
import detect

#裁剪检测区域
def CropImage4File(image,sx,sy,ex,ey):
    cropImg = image[sx+1:ex,sy+1:ey]  #裁剪图像
    return cropImg
cap = cv2.VideoCapture("./20200320_183950.mp4")
while(1):
    # get a frame
    ret, frame = cap.read()
    img = detect.detect_img(frame)
    cv2.imshow('frame',img)
    # 每5毫秒监听一次键盘动作
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 