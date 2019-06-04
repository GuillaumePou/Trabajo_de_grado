# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 09:46:56 2019

@author: gpoullain
"""

import cv2 
from matplotlib import pyplot as plt
import numpy as np
import math


img = cv2.imread('/data/estudiantes/gpoullain/sdm_yolo_guillaume/python/SPT_images_test/Labels/SPT_image_labels_CAM04_20171214085300_582107575_0013.jpg') # leer la imagen
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#kernel = np.ones((5,5),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,5,5))
img_erode = cv2.erode(img_rgb,np.transpose(kernel),iterations = 1)



cv2.imshow("Original", img_rgb)
cv2.imshow("Warped", img_erode)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)