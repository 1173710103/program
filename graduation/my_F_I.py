import cv2

#裁剪检测区域
def CropImage4File(image,sx,sy,ex,ey):
    cropImg = image[sx+1:ex,sy+1:ey]  #裁剪图像
    return cropImg

def FaceandEyes(filepath,index):
    # 导入人脸级联分类器引擎，'.xml'文件里包含训练出来的人脸特征，cv2.data.haarcascades即为存放所有级联分类器模型文件的目录
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    # 导入人眼级联分类器引擎吗，'.xml'文件里包含训练出来的人眼特征
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
    
    # 读入一张图片，引号里为图片的路径，需要你自己手动设置
    img = cv2.imread(filepath)
    a = img.shape
    k = 1
    img=cv2.resize(img,(int(a[1]/k),int(a[0]/k)),interpolation=cv2.INTER_AREA)
    
    # 用人脸级联分类器引擎进行人脸识别，返回的faces为人脸坐标列表，1.3是放大比例，5是重复识别次数
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    # 对每一张脸，进行如下操作
    for (x,y,w,h) in faces:
        # 画出人脸框，蓝色（BGR色彩体系），画笔宽度为2
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # 框选出人脸区域，在人脸区域而不是全图中进行人眼检测，节省计算资源
        face_area = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_area,1.3,5)
        # 用人眼级联分类器引擎在人脸区域进行人眼识别，返回的eyes为眼睛坐标列表
        for (ex,ey,ew,eh) in eyes:
            #画出人眼框，绿色，画笔宽度为1
            cv2.rectangle(face_area,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
        if len(eyes) == 2:
                x2 = (eyes[0][0]+eyes[0][0]+eyes[0][2])/4 + (eyes[1][0]+eyes[1][0]+eyes[1][2])/4
                y2 = (eyes[0][1]+eyes[0][1]+eyes[0][3])/4 + (eyes[1][1]+eyes[1][1]+eyes[1][3])/4
                #xre = int((eyes[0][2]+eyes[1][2])/8)
                #yre = int((eyes[0][3]+eyes[1][3])/8)
                #需要修改
                xre = 9
                yre = 9
                de = y2/3
                cv2.rectangle(face_area,(int(x2)-xre,int(y2-de)-yre),(int(x2)+xre,int(y2-de)+yre),(0,0,255),1)
                result= CropImage4File(face_area,int(y2-de)-yre,int(x2)-xre,int(y2-de)+yre,int(x2)+xre)
                blur = cv2.GaussianBlur(result,(5,5),0)
                savepath = 'D:/MY/zihaoopencv-master/zihaoopencv-master/facedemo/face/'+index+'.png'
                cv2.imwrite(savepath,blur)
    # 在"img"窗口中展示效果图
    # 监听键盘上任何按键，如有案件即退出并关闭窗口，并将图片保存为output.jpg
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    for i in range(9):
        filepath ='D:/MY/zihaoopencv-master/zihaoopencv-master/fs/'+str(i+1)+'.png'
        FaceandEyes(filepath,str(i+1))