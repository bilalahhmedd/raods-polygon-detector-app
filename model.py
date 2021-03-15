import numpy as np
import cv2
#import natsort
import matplotlib.pyplot as plt
#from matplotlib import cm 
from PIL import Image, ImageFilter 
from PIL.ImageQt import ImageQt as ImQT

#from scipy.io import loadmat

import os
import os.path
from pathlib import Path
#from os import listdir
from os.path import join
import sys


import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader ,random_split
from torch.nn import functional as F
import torchvision.transforms.functional as tf
import torch.optim as optim
import torch.nn.functional as F



from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)



def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def show_databatch(inputs, classes):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=None)


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class FCN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(FCN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
model1=FCN(n_channels=3,n_classes=1).to(device)
model1.eval()

model2=FCN(n_channels=3,n_classes=1).to(device)
model2.eval()


# Check

dictionary = torch.load('./models/model_MSE_Road_polygone.pth', map_location=torch.device('cpu'))   #Path to model weights
## load model weights state_dict
model1.load_state_dict(dictionary['model_state_dict'])


dictionary = torch.load('./models/model_MSE_lane.pth',map_location=torch.device('cpu'))         # Path to the pretrained weights
model2.load_state_dict(dictionary['model_state_dict'])


# Will be used by model1 --> Road Detection
A=     transforms.Compose([
        transforms.Resize([288, 352]),
        transforms.ToTensor()
    ])


# Will be used by model2 -- Lane Detection
B=  transforms.Compose([
        transforms.Resize([360, 640]),
        transforms.ToTensor()
    ])

fx = 1600/640
fy = 1200/360
def evaluate_image_lane(im_path):
    model2.eval()
    images = Image.open(im_path)    # Image to be tested

    #images.show() # Comment it after wards
    
    x = B(images).to(device)
    inp =x.cpu()
    inp= torchvision.utils.make_grid(inp.detach())
    #imshow(inp,title=None)
    out=model2(x[None,...])
    out =out.cpu()
    plt.pause(0.00001)
    out=out.detach().numpy()

    #plt.imshow(out[0,0,:,:]>80)
    return out, images # Will be used in the compute_contours

def compute_contours_lane(out):
    im=np.array(out[0,0,:,:]>40).astype('uint8')
    image=np.array(im)
    det=np.array(image)

    kernel1 = np.ones((5,5),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    
    im = cv2.dilate(im,kernel2,iterations = 1)
    image=np.array(im)
    det=np.array(image)

    kernel1 = np.ones((8,8),np.uint8)
    kernel2 = np.ones((8,8),np.uint8)

    contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    loc = []

    for c in np.array(contours, dtype="object"):
      # plt.imshow(img)
      # plt.plot(c[:,0,:])
      A= np.array((c[:,0,:].mean(0)))
      loc.append(A[0])
    
    if loc != []:
        left= np.argmin(loc)
        right = np.argmax(loc)
    else:
        return None,None,None
    
    return left,right,contours

def draw_lane(contours,images, left,right, name=None,identifier=None):
    #is_dir_or_create(dir='processed_lane') # Create a directory
    imaged=images.resize((1600, 1200), Image.ANTIALIAS)
    imaged=imaged.convert('RGB')
    #im = np.array(imaged)

    line1 = contours[left]
    line2 = contours[right]
    fx = 1600/640
    fy = 1200/360
    line1[:,0,0][:] = line1[:,0,0][:] * fx
    line1[:,0,1][:] = line1[:,0,1][:] * fy
    line2[:,0,0][:] = line2[:,0,0][:] * fx
    line2[:,0,1][:] = line2[:,0,1][:] * fy

    img_data =cv2.drawContours(np.array(imaged), contours,-1 , (0,255,0), 8)
    return img_data,imaged,""


def get_rect_points_lane(left,right,contours):

    line1 = contours[left]
    line2 = contours[right]

    S1=(line1[np.argmin(line1[:,0,1])][0]) 
    S4=(line1[np.argmax(line1[:,0,1])][0]) 
    S2=(line2[np.argmin(line2[:,0,1])][0]) 
    S3=(line2[np.argmax(line2[:,0,1])][0]) 
    return S1,S2,S3,S4


def four_point_transform(image, pts,fx = 1,fy = 1):
	rect = np.array(pts,dtype='float32')
	print(rect)
	rect[:,0]=rect[:,0]*(fx)
	rect[:,1]=rect[:,1]*(fy)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

    
# Will be called on button click
def evaluate_image(im_path):
    model1.eval()
    images = Image.open(im_path)    # Image to be tested

    #images.show() # Comment it after wards

    x = A(images).to(device)
    inp =x.cpu()
    inp= torchvision.utils.make_grid(inp.detach())
    #imshow(inp,title=None)
    out=model1(x[None,...])
    out =out.cpu()
    plt.pause(0.00001)
    out=out.detach().numpy()

    #plt.imshow(out[0,0,:,:]>80)
    return out, images # Will be used in the compute_contours


def compute_contours(out):
    im=np.array(out[0,0,:,:]>80).astype('uint8')
    image=np.array(im)

    det=np.array(image)

    kernel1 = np.ones((8,8),np.uint8)
    kernel2 = np.ones((8,8),np.uint8)
    im = cv2.erode(im,kernel1,iterations = 4)
    im = cv2.dilate(im,kernel2,iterations = 4)
    contours, hierarchy = cv2.findContours(im, 1, 3)
    return contours,hierarchy  # Will be used in drawing polygon on the image


def draw_polygon(contours,images, name=None,identifier=None):

    images=images.resize((1600, 1200), Image.ANTIALIAS)
    fx = 1600/352
    fy = 1200/288
    img = images.convert('RGB')
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
        polygone = approx;
        polygone[:,:,0] = approx[:,:,0]*fx
        polygone[:,:,1] = approx[:,:,1]*fy
        img_data = (cv2.drawContours(np.array(img).astype('uint8'), [polygone], -1, (255, 0, 0), 3))

          
    polygone = approx
    return polygone, img_data, img, ""


def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect



def get_rect_points(polygone):
    A = (polygone[:,0])
    S1 = (min(A[:,0]),min(A[:,1]))    # bl
    S2 = (max(A[:,0]),min(A[:,1])) # use          # br
    S3 = (min(A[:,0]),max(A[:,1]))             # tl
    S4 = (max(A[:,0]),max(A[:,1])) # use          #tr
    return S1, S2, S3, S4
