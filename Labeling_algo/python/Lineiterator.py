# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:11:30 2019

@author: gpoullain
"""
import os
os.environ['PATH']='/home/ubuntu/torch/install/bin:/bin:/usr/bin:/usr/X11R6/bin:/usr/local/bin:/usr/local/cuda/bin:/home/ubuntu/caffe/build/tools/'
import numpy as np
from scipy.signal import find_peaks
from math import *
import cv2

def createLineIterator(P1, P2, img):

   #define local variables for readability
   imageH = img.shape[0]
   imageW = img.shape[1]
   P1X = P1[0]
   P1Y = P1[1]
   P2X = P2[0]
   P2Y = P2[1]

   #difference and absolute difference between points
   #used to calculate slope and relative location between points
   dX = P2X - P1X
   dY = P2Y - P1Y
   dXa = np.abs(dX)
   dYa = np.abs(dY)

   #predefine numpy array for output based on distance between points
   itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
   itbuffer.fill(np.nan)

   #Obtain coordinates along the line using a form of Bresenham's algorithm
   negY = P1Y > P2Y
   negX = P1X > P2X
   if P1X == P2X: #vertical line segment
       itbuffer[:,0] = P1X
       if negY:
           itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
       else:
           itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
   elif P1Y == P2Y: #horizontal line segment
       itbuffer[:,1] = P1Y
       if negX:
           itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
       else:
           itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
   else: #diagonal line segment
       steepSlope = dYa > dXa
       if steepSlope:
           slope = dX/dY
           if negY:
               itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
           else:
               itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
           itbuffer[:,0] = np.linspace(P1X, P2X, dYa).astype(np.int)
       else:
           slope = dY/dX
           if negX:
               itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
           else:
               itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
           itbuffer[:,1] = np.linspace(P1Y, P2Y, dXa).astype(np.int)

   #Remove points outside of image
   colX = itbuffer[:,0]
   colY = itbuffer[:,1]
   itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

   #Get intensities from img ndarray
   itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

   return itbuffer

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
            for n_pts in range (len(pts_line)-1):
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
                img[y,x] = 0         
    else:
        for i in range (len(valey)):
            x0 = n-1-valey[i]
            p1 = (x0,0)
            p2 = (v+x0,m)
            pts_line = intermediates(p1, p2)
            for k in range (len(pts_line)):
                x = pts_line[k][0]
                y = pts_line[k][1]
                img[y,x] = 0 
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
    
def intermediates(p1, p2):
#Return a list of nb_points equally spaced points
#   between p1 and p2"""
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    # number of point is calculated with pythagore
    nb_points = (sqrt((p2[0] - p1[0])**2+(p2[1] - p1[1])**2))
    x_spacing = (p2[0] - p1[0]) / nb_points
    y_spacing = (p2[1] - p1[1]) / nb_points
    

    return [[int(p1[0] + i * x_spacing), int(p1[1] +  i * y_spacing)] for i in range(1, int(nb_points+1))]

def edge_detect(img_bw, classes,P1, P2):
    highest_pt = np.argmin([P1[0],P2[0]])
    nb_points = (sqrt((P2[0] - P1[0])**2+(P2[1] - P1[1])**2))
    im_threshold = np.zeros(img_bw.shape)
    labels_coord_all = []
    size_table = np.array([9,40,5,20,30,40,9,5,40])
    air_table = size_table*nb_points/3
    for cl in classes:
        air_min = air_table[cl]
        size_min = size_table[cl]
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
        valey = find_peaks(neg_hist, height=None, threshold=None, distance=None, prominence=0.1, width=None, wlen=None, rel_height=0.5, plateau_size=None)
        valey =valey[0]        
        valey2 = []      
        if len(valey) !=0:
            for i in range (len(valey)):
                if neg_hist[valey[i]]<0.98:
                    valey2.append(valey[i])
#            plt.plot(neg_hist)
#            plt.plot(valey2, neg_hist[valey2], "x")
        
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
                if air>air_min and (xmax-xmin)>size_min:
                    labels_coord.append([xmin,ymin,xmax,ymax])
        else:
            labels_coord = np.zeros((0,4))
        im_threshold += im_th2
        labels_coord_all.append(labels_coord)
    return(im_threshold,labels_coord_all)