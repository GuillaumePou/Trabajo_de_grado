# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 18:59:05 2018

@author: guillaume
Practicas 3 part 2
"""

#%% Filtros no lineales


from matplotlib import pyplot as plt
import numpy as np
import cv2
from math import *
#from scipy.signal import argrelextrema
from scipy.signal import find_peaks



def intermediates(p1, p2):
#Return a list of nb_points equally spaced points
#   between p1 and p2"""
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    # number of point is calculate with pythagore
    nb_points = (sqrt((p2[0] - p1[0])**2+(p2[1] - p1[1])**2))
    x_spacing = (p2[0] - p1[0]) / nb_points
    y_spacing = (p2[1] - p1[1]) / nb_points
    

    return [[int(p1[0] + i * x_spacing), int(p1[1] +  i * y_spacing)] for i in range(1, int(nb_points+1))]

#img = cv2.imread('/data/estudiantes/gpoullain/sdm_yolo_guillaume/python/SPT_images_test/Labels_init/SPT_image_labels_CAM04_20190131060000_1_0004.jpg') # leer la imagen
#img = cv2.imread('/data/estudiantes/gpoullain/sdm_yolo_guillaume/python/SPT_images/Labels/SPT_image_labels_CAM04_20190131060000_1_0020.jpg') # leer la imagen
img = cv2.imread('/home/gpoullain/Documents/im_incl/SPT_image_labels_CAM04_20171214085300_582107575_0003.jpg') # leer la imagen

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(img,cmap='gray')

classes = [0,1,2,3,4,5,6,7,8]
im_threshold = np.zeros(img.shape)
labels_coord_all = []


P1 = (539, 461)
P2 = (1285, 508)

def histogram_incl(img,P1,P2):
    m,n = img.shape
    highest_pt = np.argmin([P1[1],P2[1]])
    v = abs(P1[1]-P2[1])
    if highest_pt == 0:
        x0 = 0
        hist = np.zeros(n-v)
        while v <= (n-1):
            p1 = (x0,0)
            p2 = (v,m)
            v_hist = 0
            pts_line = intermediates(p1, p2)
            for n_pts in range (len(pts_line)):
                v_hist+= img[pts_line[n_pts][1],pts_line[n_pts][0]]
            hist[x0] = v_hist
            x0+=1
            v+=1
    else:
        x0 = n-1
        hist = np.zeros(n)
        while v >= 0:
            p1 = (x0,0)
            p2 = (n-v,m)
            v_hist = 0
            pts_line = intermediates(p1, p2)
            for n_pts in range (len(pts_line)):
                v_hist+= img[pts_line[n_pts][1],pts_line[n_pts][0]]
            hist[x0] = v_hist
            x0-=1
            v-=1
    return np.convolve(hist/(len(pts_line)), [1,1,1,1,1], mode='full')/5
    
def line_zero(img,P1,P2,valey):
    m,n = img.shape
    highest_pt = np.argmin([P1[1],P2[1]])
    v = abs(P1[1]-P2[1])
    if highest_pt == 0:
        for i in range (len(valey)):
            x0 = valey[i]
            p1 = (x0,0)
            p2 = (v+x0,m)
            pts_line = intermediates(p1, p2)
            for k in range (len(pts_line)):
                x = pts_line[k][0]
                y = pts_line[k][1]
                img[y,x]=0         
                a=21
    else:
        for i in range (len(valey)):
            x0 = n-1-valey[i]
            p1 = (x0,0)
            p2 = (v+x0,m)
            pts_line = intermediates(p1, p2)
            for k in (pts_line):
                x = pts_line[k][0]
                y = pts_line[k][1]
                img[x,y]=0 
    return img

def find_ind_min_local(hist_norm):
    ind_min_local = []
    ind_max = argrelextrema(hist_norm, np.greater)[0]
    ind_min = argrelextrema(hist_norm, np.less)[0]
    val_max = hist_norm[ind_max[:]]
    val_min = hist_norm[ind_min[:]]
    if len(ind_max)>0:
#        if ind_max[0]>ind_min[0]:
#            ind_min = ind_min[1:len(ind_min)]
#            val_min = val_min[1:len(ind_min)]
        for i in range (len(ind_max)-1):
            vmin =val_min[i]
            if val_max[i]-vmin>0.1 and val_max[i+1]-vmin>0.1 and vmin!=0.1:
                ind_min_local.append(vmin)
    return ind_min_local
    
    
img_bw = np.copy(img) 
air_table = [1500,3000,700,3000,7000,1400,3000,700,3000]
#for cl in classes:
cl = 6
air_min = air_table[cl]
v_pixel = int((cl+1)*255/9)

img_bw = np.uint8(img_bw)

img_bw = cv2.medianBlur(img_bw,5) #median blur filter

tresh_min = np.floor(v_pixel*0.95) 
tresh_max = np.ceil(v_pixel*1.05) 
im_bool_sup = img_bw<=tresh_max
im_bool_inf = img_bw>=tresh_min
im_bool = im_bool_sup&im_bool_inf  

im_th = np.uint8(im_bool)*v_pixel        

hist = histogram_incl(im_th,P1,P2)
hist_norm = hist/v_pixel
#        ind_min_local = find_ind_min_local(hist_norm)
neg_hist = np.ones(len(hist_norm))-hist_norm
valey = find_peaks(neg_hist, height=None, threshold=None, distance=None, prominence=(np.max(hist_norm)/5), width=None, wlen=None, rel_height=0.5, plateau_size=None)
valey =valey[0]        
valey2 = []      
if len(valey) !=0:
    for i in range (len(valey)):
        if neg_hist[valey[i]]<0.98:
            valey2.append(valey[i])
    plt.plot(neg_hist)
    plt.plot(valey2, neg_hist[valey2], "x")

im_th2 = line_zero(im_th,P1,P2,valey2)
kernel = np.ones((2,2))
im_th2 = cv2.erode(im_th2,kernel,iterations = 1)

if im_bool.any():
    
    contours, hierarchy = cv2.findContours(im_th2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels_coord = []
    for ct in range (len(contours)):
        cont = contours[ct]
        xmin = np.min(cont[:,:,0])
        ymin = np.min(cont[:,:,1])
        xmax = np.max(cont[:,:,0])
        ymax = np.max(cont[:,:,1])
        air = (xmax-xmin)*(ymax-ymin)
        if air>air_min & (xmax-xmin)>4:
            labels_coord.append([xmin,ymin,xmax,ymax])
else:
    labels_coord = np.zeros((0,4))
im_threshold += im_th2
labels_coord_all.append(labels_coord)
#
#f = plt.figure('bilateralcomp',figsize=(15,5))
#f.add_subplot(231),plt.imshow(img,cmap='gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
#f.add_subplot(234),plt.imshow(img[100:300,310:420],cmap='gray')
#plt.xticks([]), plt.yticks([])
#f.add_subplot(232),plt.imshow(im_threshold,cmap='gray')
#plt.title('median filter 5'), plt.xticks([]), plt.yticks([])
#f.add_subplot(235),plt.imshow(im_threshold[100:300,310:420],cmap='gray')
#plt.xticks([]), plt.yticks([])
#f.add_subplot(233),plt.imshow(im_threshold,cmap='gray')
#plt.title('Threshold +-5%'), #plt.xticks([]), plt.yticks([])
#f.add_subplot(236),plt.imshow(im_threshold[100:300,310:420],cmap='gray')
#plt.xticks([]), plt.yticks([])

plt.figure('erode',figsize=(15,5))
plt.imshow(im_threshold,cmap='gray')

plt.show()

