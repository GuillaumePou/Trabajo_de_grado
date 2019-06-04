# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:49:17 2019

@author: gpoullain
"""

def edge_detect(img_bw, classes,P1, P2):
    highest_pt = np.argmin([P1[0],P2[0]])
    im_threshold = np.zeros(img_bw.shape)
    labels_coord_all = []
    air_table = [1500,3000,700,3000,7000,1400,3000,700,3000]
    for cl in classes:
        air_min = air_table[cl]
        v_pixel = int((cl+1)*255/9)
##        (thresh, image_contour) = cv2.threshold(im_bw, tresh_min, tresh_max, 0)
##    #    cv2.imwrite('bw_'+file_name, im_bw)
##        contours, hierarchy = cv2.findContours(image_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##        cv2.drawContours(image_contour, contours, -1, 255, 1)
        img_bw = np.uint8(img_bw)

        img_bw = cv2.medianBlur(img_bw,5) #median blur filter
        
#        kernel = np.ones((3,3),np.uint8)#cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#        img_bw = cv2.erode(img_bw,kernel,iterations = 1) 
        
#        kernel_d = np.ones((3,3),np.uint8)
#        img_bw = cv2.dilate(img_bw,kernel_d,iterations = 1)
#        img_bw = cv2.GaussianBlur(img_bw,(3,3),0) 
        tresh_min = np.floor(v_pixel*0.95) 
        tresh_max = np.ceil(v_pixel*1.05) 
        im_bool_sup = img_bw<=tresh_max
        im_bool_inf = img_bw>=tresh_min
        im_bool = im_bool_sup&im_bool_inf  

        im_th = np.uint8(im_bool)*v_pixel

#        kernel = np.ones((5,5),np.uint8)#cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#        im_th = cv2.erode(im_th,kernel,iterations = 1) 
                
#        im_th = cv2.di1=,.[]late(im_th,kernel,iterations = 1) 
#        im_th = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel)
#        im_th = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel)
        
        
        if im_bool.any():
            
            contours, hierarchy = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        im_threshold += im_th
        labels_coord_all.append(labels_coord)
    return(im_threshold,labels_coord_all)