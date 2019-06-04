# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 17:33:29 2018

@author: guillaume
"""

import cv2 
from matplotlib import pyplot as plt
import numpy as np
import math


img_bike = cv2.imread('/data/estudiantes/gpoullain/Videos/CAM04_20171214085300_582107575_linea_1.jpg') # leer la imagen
img_rgb_bike = cv2.cvtColor(img_bike, cv2.COLOR_BGR2RGB)
#%%
"""
cv2.cvtColor(input_image, flag)
where flag determines the type of conversion.
For BGR \rightarrow Gray conversion we use the flags cv2.COLOR_BGR2GRAY. 
Similarly for BGR \rightarrow HSV, we use the flag cv2.COLOR_BGR2HSV.
"""
plt.figure() # nueva figura
plt.imshow(img_rgb_bike) # graficar la imagen
plt.xticks([]), plt.yticks([]) # para no graficar la escala
plt.title('Bicicleta roja') # graficar el titulo

plt.show() #para no graficar 'Text(0.5,1,'RGB')




#%% #Transformacion shear a 45° derecha con matrice

M_shear = np.float32([[1,1,0],[0,1,0],[0,0,1]])
rows,cols,dim = img_rgb_bike.shape #tamano de la imagen

bike_shear = cv2.warpPerspective(img_rgb_bike,M_shear,(rows,cols))


plt.figure() # nueva figura
plt.imshow(bike_shear) # graficar la imagen
plt.xticks([]), plt.yticks([]) # para no graficar la escala
plt.title('Bicicleta roja shear 45 derecha') # graficar el titulo
plt.show()



#%% #Transformacion shear a 45° derecha

pts1 = np.float32([[1000,1000],[1000,1000],[1000,1000]])
pts2 = np.float32([[1000,1000],[1000/math.sqrt(2),1000/math.sqrt(2)],[1000,math.sqrt(2)*1000]])

M_aff = cv2.getAffineTransform(pts1,pts2)
bike_shear = cv2.warpAffine(img_rgb_bike,M_aff,(cols,rows))

plt.figure() # nueva figura
plt.imshow(bike_shear) # graficar la imagen
plt.xticks([]), plt.yticks([]) # para no graficar la escala
plt.title('Bicicleta roja shear 45 derecha') # graficar el titulo
plt.show()

M_shear = np.float32([[ 7.07106689e-01,-7.07106689e-01,2.70710669e+03],
                      [-2.92893311e-01,1.70710669e+00,-1.53553345e+03],
                      [0,0,1]])
bike_shear = cv2.warpPerspective(img_rgb_bike,M_shear,(rows,cols))

plt.figure() # nueva figura
plt.imshow(bike_shear) # graficar la imagen
plt.xticks([]), plt.yticks([]) # para no graficar la escala
plt.title('Bicicleta roja shear 45 derecha') # graficar el titulo
plt.show()

print('rows=',rows)
print('colums=',cols)
print('M_aff = ',M_aff)


