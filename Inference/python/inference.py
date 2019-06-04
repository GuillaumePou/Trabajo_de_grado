#!/usr/bin/python2import cv2
import os
os.environ['PATH']='/home/ubuntu/torch/install/bin:/bin:/usr/bin:/usr/X11R6/bin:/usr/local/bin:/usr/local/cuda/bin:/home/ubuntu/caffe/build/tools/'
import easygui
import cv2
import numpy as np
import ntpath

from Counter import linecounter as lc
from time import sleep
from matplotlib import pyplot as plt
from time import time
from lineiterator import *

# different models
title  ="Which model do you want to use?"
msg = "Select the model"
choices = ["YOLOv2", "YOLOv2tiny", "YOLOv3", "YOLOv3tiny", "YOLOv3-all"]
model = easygui.choicebox(msg, title, choices)
type(model)
print "You selected the model: " + model

if model == "YOLOv2":
## YOLOv2 train to detect on STI
    from darknet import *
    net = load_net("../cfg/yolov2-septima-infe.cfg", "../weight/yolov2-septima_final.weights", 0)
    meta = load_meta("../data/septima.data")

elif model=="YOLOv2tiny":
## YOLOv2 tiny train to detect on STI
    from darknet5 import *
    net = load_net("../cfg/yolov2-tiny-septima-infe.cfg", "../weight/yolov2-tiny-septima_final.weights", 0)
    meta = load_meta("../data/septima.data")

elif model=="YOLOv3":
## YOLOv3 train to detect on STI
    from darknet import *
    net = load_net("../cfg/yolov3-septima-infe.cfg", "../weight/yolov3-septima_final.weights", 0)
    meta = load_meta("../data/septima.data")

elif model=="YOLOv3tiny":
# YOLOv3 tiny train to detect on STI
    from darknet2 import *
    net = load_net("../cfg/yolov3-tiny-septima-infe.cfg", "../weight/yolov3-tiny-septima_final.weights", 0)
    meta = load_meta("../data/septima.data")

elif model=="YOLOv3-all":
# YOLOv3 tiny train to detect on STI
    from darknet import *
    net = load_net("../cfg/yolov3-all-infe.cfg", "../weight/y/oolov3-all_final.weights", 0)
    meta = load_meta("../data/all.data")

msg = "Enter the length of the spatio-temporal image."
title = "STI length"
fieldNames = ["STI length"]
len_SPT = easygui.integerbox(msg, title, default=None, lowerbound=0, upperbound=1000, image=None, root=None)

#start
deCamara=False
count = 0


if deCamara:
    cam = cv2.VideoCapture(0)
else: 
    fn = easygui.fileopenbox(default="/data/estudiantes/gpoullain/Videos/",filetypes = ['*.avi','*.mp4'])
    cam = cv2.VideoCapture(fn)
    MAXW=700
    mindist=200
    ruta,ext=os.path.splitext(fn)
    head,tail = ntpath.split(ruta)
    archsal=ruta+'.csv'

frames=0
ret_val, imgFile2 = cam.read()
frames+=1
if not ret_val:
    print ('Could not open the camera')
    exit()


t_detection = 0
time_s = time()

lineasDeConteo = 1
contadores=[]
for cc in range(lineasDeConteo):
    sleep(1)
    lineaDeConteo=lc.selectLine(imgFile2,ownString='Selecciona la linea de conteo #' +str(cc+1),filename=archsal,linecount=cc+1)
# parameters of each line of counting    
    sleep(1)
    contadores.append(lc.counter(lineaDeConteo.pt1,lineaDeConteo.pt2,filename=archsal,linecount=cc+1,fps=20))

#initialisation of the STI
P1 = lineaDeConteo.pt1
P2 = lineaDeConteo.pt2

b,g,r = cv2.split (cv2.cvtColor(imgFile2,cv2.COLOR_BGR2RGB))
itbufferB = createLineIterator(P1, P2, b)
acc_B = np.ones((len(itbufferB[:,2]),len_SPT))
acc_G = np.ones((len(itbufferB[:,2]),len_SPT))      
acc_R = np.ones((len(itbufferB[:,2]),len_SPT))
number_SPT = 1
countting_class = np.zeros(9)

# list of class with their number and their color for the bounding box
num2clases=['particular', 'bus', 'motorcyclist', 'minivan', 'pedestrian', 'truck', 'taxi', 'cyclist', 'tractomula']
clases={'particular': 0, 'bus': 1, 'motorcyclist': 2, 'minivan': 3, 'pedestrian': 4, 'truck': 5, 'taxi': 6, 'cyclist': 7, 'tractomula': 8}
colour = [(0,0,0),(0,0,255),(0,255,0),(255,0,0),(255,255,255),(0,255,255),(128,0,0),(255,165,0),(255,255,0)]

while True:
    ret_val, imgFile2 = cam.read()
    frames+=1
    
    if not ret_val:#Process the last STI (could be of size inferior to lenght_SPT) and end exit the algorithm

        segframes=cam.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        
        #save the ST image
        path_STI = os.path.join('SPT_images/RGB', "SPT_image_RGB_"+tail+"_%04i.JPEG" %number_SPT)
        cv2.imwrite(path_STI,SPT_RGB )
        
        SPT_out2 = SPT_RGB.copy()
        
        time_start = time()
        r = detect(net, meta, path_STI) 
        t_detection = t_detection + time() - time_start
        print ('Detections: '+str(len(r)))
        print (r)
        
        for i in range(len(r)): #plot bounding box and numbers on the STI image
            w=int(r[i][2][2])
            h=int(r[i][2][3])
            x=int(r[i][2][0])-w/2
            y=int(r[i][2][1])-h/2
            classe = r[i][0] 
            n_classe = clases[classe]
            cv2.rectangle(SPT_out2, (x,y), (x+w,y+h), colour[clases[classe]], thickness=2, lineType=8, shift=0)
            cv2.putText(SPT_out2,classe, (x+w/2,y+h/2), cv2.FONT_HERSHEY_SIMPLEX,1, colour[n_classe])
            countting_class[n_classe] +=1  # add the objects to the counter
        cv2.imshow('STI', SPT_out2) 
        path_label = os.path.join('SPT_images/RGB_label', "SPT_image_RGB_lab_"+tail+model+"_%04i.JPEG" %number_SPT)
        cv2.imwrite(path_label,SPT_out2 )
        
        print ("End of the video, exit")
        cv2.imwrite('last_image.jpg',imgFile3)
        break

    
    imgFile3 = cv2.cvtColor(imgFile2,cv2.COLOR_BGR2RGB)

    b,g,r = cv2.split (cv2.cvtColor(imgFile2,cv2.COLOR_BGR2RGB))
    itbufferB = createLineIterator(P1, P2, b)
    acc_B[:,count] = itbufferB[:,2]
    itbufferG = createLineIterator(P1, P2, g)
    acc_G[:,count] = itbufferG[:,2]        
    itbufferR = createLineIterator(P1, P2, r)
    acc_R[:,count] = itbufferR[:,2]        
    count += 1

    if count == len_SPT: # if we completed a STI then we run detection and create the bounding box and increment the counter
        SPT_RGB = np.uint8(cv2.merge([acc_R,acc_G,acc_B]))

        path_STI = os.path.join('SPT_images/RGB', "SPT_image_RGB_"+tail+"_%04i.JPEG" %number_SPT)
        cv2.imwrite(path_STI,SPT_RGB )
        
        SPT_out2 = SPT_RGB.copy()
        
        time_start = time()
        r = detect(net, meta, path_STI) 
        t_detection = t_detection + time() - time_start
        print ('Detections: '+str(len(r)))
        print (r)
        
        for i in range(len(r)):
            w=int(r[i][2][2])
            h=int(r[i][2][3])
            x=int(r[i][2][0])-w/2
            y=int(r[i][2][1])-h/2
            classe = r[i][0] 
            n_classe = clases[classe]
            cv2.rectangle(SPT_out2, (x,y), (x+w,y+h), colour[clases[classe]], thickness=2, lineType=8, shift=0)
            cv2.putText(SPT_out2,classe, (x+w/2,y+h/2), cv2.FONT_HERSHEY_SIMPLEX,1, colour[n_classe])
            countting_class[n_classe] +=1
        cv2.imshow('STI', SPT_out2)   
        path_label = os.path.join('SPT_images/RGB_label', "SPT_image_RGB_lab_"+tail+model+"_%04i.JPEG" %number_SPT)
        cv2.imwrite(path_label,SPT_out2 )
        
        #re-initialisation of the STI
        b,g,r = cv2.split (cv2.cvtColor(imgFile2,cv2.COLOR_BGR2RGB))
        itbufferB = createLineIterator(P1, P2, b)
        acc_B = np.ones((len(itbufferB[:,2]),len_SPT))
        acc_G = np.ones((len(itbufferB[:,2]),len_SPT))      
        acc_R = np.ones((len(itbufferB[:,2]),len_SPT))   
        number_SPT += 1
        count = 0
        
    k = cv2.waitKey(2)& 0xFF
    if k==ord('q'):    # Esc key=537919515 en linux WTF??? para parar y en mi otro PC 1048689
        print ('Forced user exit...')
        break
#
#for contar in contadores:
#    contar.saveFinalCounts(frames)
#cv2.imwrite('lastphotography.jpg',imgFile3)

cv2.destroyAllWindows()
cam.release()

# to calculate the different times
time_tot = time() - time_s
frame_s = segframes/time_tot

# few lines to create the .csv file        
filename_output=head+'/'+tail+'_'+model+'_'+str(len_SPT)+'_frames'+'.csv'

FILE = open(filename_output,'w')
FILE.write("Label;number\n")
FILE.write("\n")
for jj in range(len(countting_class)):
    FILE.write(str(num2clases[jj])+';'+str(countting_class[jj])+'\n')
    print (str(num2clases[jj])+': '+str(countting_class[jj]))

FILE.write("\n")
FILE.write('execution time = '+';'+str(t_detection)+'s')
FILE.write("\n")
FILE.write('total time = '+';'+str(time_tot)+'s')
FILE.write("\n")
FILE.write('frames/s = '+';'+str(frame_s)+'s')
FILE.write("\n")
FILE.write('SPI length = '+';'+str(len_SPT))
FILE.write("\n")
FILE.write('model : '+';'+model)

print('execution time = '+ str(t_detection)+'s')
print('total time = '+ str(time_tot)+'s')
print('frames/s = '+ str(frame_s)+'s')
print('SPI length = '+ str(len_SPT))
print('model : '+ str(model))


print ('Exit...')
exit()
