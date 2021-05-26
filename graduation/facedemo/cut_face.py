from PIL import Image
import os
######## 需要裁剪的图片位置#########
path_img = 'D:/MY/zihaoopencv-master/zihaoopencv-master/fs/'
img_dir = os.listdir(path_img)
print(img_dir)
 
'''
（左上角坐标(x,y)，右下角坐标（x+w，y+h）
'''
 
for i in range(len(img_dir)):
    #####根据图片名称提取id,方便重命名###########
    img = Image.open(path_img + img_dir[i])
    size_img=img.size
    print(size_img)
    x = 0
    y = 0
    ########这里需要均匀裁剪几张，就除以根号下多少，这里我需要裁剪25张-》根号25=5（5*5）####
    w = int(size_img[0]/3)
    h = int(size_img[0]/3)
    i = 1
    for k in range(3):
        for v in range(3):
            region = img.crop((x+k*w, y+v*h, x + w*(k+1), y + h*(v+1)))
            #####保存图片的位置以及图片名称###############
            print('%d' % i  + '.png')
            region.save('D:/MY/zihaoopencv-master/zihaoopencv-master/fs/%d' % i  + '.png')
            i+=1
 