import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import torchvision
import eval_widerface
import torchvision_model
import os

def pad_to_square(img, pad_value):
    _, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def get_args():
    parser = argparse.ArgumentParser(description="Detect program for retinaface.")
    parser.add_argument('--image_path', type=str, default='test.jpg', help='Path for image to detect')
    parser.add_argument('--model_path', type=str, default='model.pt', help='Path for model')
    parser.add_argument('--save_path', type=str, default='./out', help='Path for result image')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--scale', type=float, default=1.0, help='Image resize scale', )
    args = parser.parse_args()

    return args

def detect_img(img):
    args = get_args()
    # Create torchvision model
    return_layers = {'layer2':1,'layer3':2,'layer4':3}
    RetinaFace = torchvision_model.create_retinaface(return_layers)

    # Load trained model
    retina_dict = RetinaFace.state_dict()
    pre_state_dict = torch.load(args.model_path)
    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
    RetinaFace.load_state_dict(pretrained_dict)

    RetinaFace = RetinaFace.cuda()
    RetinaFace.eval()

    # Read image
    img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = img.permute(2,0,1)

    if not args.scale == 1.0:
        size1 = int(img.shape[1]/args.scale)
        size2 = int(img.shape[2]/args.scale)
        img = resize(img.float(),(size1,size2))

    input_img = img.unsqueeze(0).float().cuda()
    picked_boxes, picked_landmarks, picked_scores = eval_widerface.get_detections(input_img, RetinaFace, score_threshold=0.5, iou_threshold=0.3)

    # np_img = resized_img.cpu().permute(1,2,0).numpy()
    np_img = img.cpu().permute(1,2,0).numpy()
    np_img.astype(int)
    img = cv2.cvtColor(np_img.astype(np.uint8),cv2.COLOR_BGR2RGB)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for j, boxes in enumerate(picked_boxes):
        if boxes is not None:
            for box, landmark, score in zip(boxes,picked_landmarks[j],picked_scores[j]):
                cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,0,255),thickness=2)
                cv2.circle(img,(landmark[0],landmark[1]),radius=1,color=(0,0,255),thickness=2)
                cv2.circle(img,(landmark[2],landmark[3]),radius=1,color=(0,255,0),thickness=2)
                cv2.circle(img,(landmark[4],landmark[5]),radius=1,color=(255,0,0),thickness=2)
                cv2.circle(img,(landmark[6],landmark[7]),radius=1,color=(0,255,255),thickness=2)
                cv2.circle(img,(landmark[8],landmark[9]),radius=1,color=(255,255,0),thickness=2)
                
                '''
                x = (landmark[0] + landmark[2]) / 2
                y = landmark[3] - (landmark[3] - box[1]) / 3
                cv2.circle(img,(x,y),radius=5,color=(0,0,255),thickness=1)
                '''
                '''
                start_point_x = (landmark[0] + landmark[2]) / 2
                start_point_y = (landmark[1] + landmark[3]) / 2
                end_point_x = (landmark[8] + landmark[6]) / 2
                end_point_y = (landmark[9] + landmark[7]) / 2
                cv2.line(img , (start_point_x,start_point_y),(landmark[4],landmark[5]),color=(255,100,0),thickness=2)
                cv2.line(img , (landmark[4],landmark[5]),(end_point_x,end_point_y),color=(255,255,100),thickness=2)
                '''
                '''
                cv2.rectangle(img,(landmark[0],landmark[1]),(landmark[8],landmark[9]),(0,0,100),thickness=2)
                '''
                cv2.putText(img, text=str(score.item())[:5], org=(box[0],box[1]), fontFace=font, fontScale=0.5,
                            thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 255))

    return img