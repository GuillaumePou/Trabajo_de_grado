# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:14:54 2019

@author: gpoullain
"""

#!/usr/bin/python2import cv2
import os
os.environ['PATH']='/home/ubuntu/torch/install/bin:/bin:/usr/bin:/usr/X11R6/bin:/usr/local/bin:/usr/local/cuda/bin:/home/ubuntu/caffe/build/tools/'
import easygui
from darknet import *
from Counter import linecounter as lc
from Track import tracking as tr
from time import sleep
#from Line import LineIterator
import numpy as np
import ntpath
from math import *
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
#from scipy import *
import glob

"""
#YOLO VOC
net = load_net("../cfg/yolo-voc.cfg", "../../darknet/yolo-voc.weights", 0)
meta = load_meta("../cfg/voc_py.data")
"""

#NUESTRO YOLO ENTRENADO 80000 iteraciones
net = load_net("../yolo-obj.cfg", "../../weights/yolo-obj_final.weights", 0)
meta = load_meta("../data/obj.data")


"""
#YOLO COCO
net = load_net("yolo.cfg", "../../darknet/yolo.weights", 0)
meta = load_meta("coco_es.data")
"""

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
           itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
       else:
           slope = dY/dX
           if negX:
               itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
           else:
               itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
           itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

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
    # number of point is calculate with pythagore
    nb_points = (sqrt((p2[0] - p1[0])**2+(p2[1] - p1[1])**2))
    x_spacing = (p2[0] - p1[0]) / nb_points
    y_spacing = (p2[1] - p1[1]) / nb_points
    

    return [[int(p1[0] + i * x_spacing), int(p1[1] +  i * y_spacing)] for i in range(1, int(nb_points+1))]

def edge_detect(img_bw, classes,P1, P2):
    highest_pt = np.argmin([P1[0],P2[0]])
    nb_points = (sqrt((P2[0] - P1[0])**2+(P2[1] - P1[1])**2))
    im_threshold = np.zeros(img_bw.shape)
    labels_coord_all = []
    size_table = np.array([9,40,5,20,30,40,9,5,40])/2
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
        valey = find_peaks(neg_hist, height=None, threshold=None, distance=None, prominence=0.09, width=None, wlen=None, rel_height=0.5, plateau_size=None)
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
    
    
#primera vez
charlador=False
pintarTrayectos=True
framesttl=5
deCamara=False
MAXW=550 ## 200 pixeles maximo de ancho permitido
mindist=10

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
    print ('no se pudo abrir la camara, saliendo')
    exit()

imgFile3 = cv2.cvtColor(imgFile2, cv2.COLOR_BGR2RGB)
tama=imgFile2.shape
imgImported=make_image(tama[1],tama[0],tama[2])

imgFileptr,cv_img=get_iplimage_ptr(imgFile3)    
ipl_in2_image(imgFileptr,imgImported)
rgbgr_image(imgImported)

track=tr.tracking(verbose=charlador,mindist=mindist,framesttl=framesttl)#verbose=False,mindist=100
title  ="Cuantas lineas de conteo?"
msg = "Seleccione el numero de lineas de conteo que quiere poner, se recomiendan maximo 6 lineas de conteo"
choices = ["1", "2", "3", "4", "5", "6"]
choice = easygui.choicebox(msg, title, choices)
type(choice)
lineasDeConteo=int(choice)
print "usted ha seleccionado ",lineasDeConteo," lineas de conteo"

contadores=[]
for cc in range(lineasDeConteo):
    sleep(1)
    lineaDeConteo=lc.selectLine(imgFile2,ownString='Selecciona la linea de conteo #' +str(cc+1),filename=archsal,linecount=cc+1)
# parameters of each line of counting    
    sleep(1)
    contadores.append(lc.counter(lineaDeConteo.pt1,lineaDeConteo.pt2,filename=archsal,linecount=cc+1,fps=20))


files = [f for f in glob.glob(head + "**/*.avi")]


for vid in range (len(files)):
    

    #primera vez
    charlador=False
    pintarTrayectos=True
    framesttl=5
    deCamara=False
    MAXW=550 ## 200 pixeles maximo de ancho permitido
    mindist=10
    
    if deCamara:
        cam = cv2.VideoCapture(0)
    else: 
        fn = files[vid]
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
        print ('no se pudo abrir la camara, saliendo')
        exit()
    
    imgFile3 = cv2.cvtColor(imgFile2, cv2.COLOR_BGR2RGB)
    tama=imgFile2.shape
    imgImported=make_image(tama[1],tama[0],tama[2])
    
    imgFileptr,cv_img=get_iplimage_ptr(imgFile3)    
    ipl_in2_image(imgFileptr,imgImported)
    rgbgr_image(imgImported)
    track=tr.tracking(verbose=charlador,mindist=mindist,framesttl=framesttl)#verbose=False,mindist=100
    contadores=[]
    contadores.append(lc.counter(lineaDeConteo.pt1,lineaDeConteo.pt2,filename=archsal,linecount=cc+1,fps=20))
    
    
    count = 0
    len_SPT =416
    number_SPT = 1
    P1 = lineaDeConteo.pt1
    P2 = lineaDeConteo.pt2
    
    itbuffer = createLineIterator(P1, P2, cv2.cvtColor(imgFile2,cv2.COLOR_BGR2GRAY))
    acc_labels = np.ones((len(itbuffer[:,2]),len_SPT))
    acc_labels[:,int(count)] = itbuffer[:,2]
    
    #initialisation of the STI
    b,g,r = cv2.split (cv2.cvtColor(imgFile2,cv2.COLOR_BGR2RGB))
    itbufferB = createLineIterator(P1, P2, b)
    acc_B = np.ones((len(itbufferB[:,2]),len_SPT))
    acc_B[:,int(count)] = itbufferB[:,2]
    itbufferG = createLineIterator(P1, P2, g)
    acc_G = np.ones((len(itbufferG[:,2]),len_SPT))
    acc_G[:,int(count)] = itbufferG[:,2]        
    itbufferR = createLineIterator(P1, P2, r)
    acc_R = np.ones((len(itbufferR[:,2]),len_SPT))
    acc_R[:,int(count)] = itbufferR[:,2]   
        
    
    while True:
        ret_val, imgFile2 = cam.read()
        frames+=1
        
        if not ret_val:#end process at the end of the video
            print ("Fin del video o salida en camara, saliendo")
            cv2.imwrite('ultimofotogramaprocesado.jpg',imgFile3)
            break
        
        segframes=cam.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        tiempoactual=cam.get(cv2.cv.CV_CAP_PROP_POS_MSEC)*20.0/30.0
        imgFile3 = cv2.cvtColor(imgFile2,cv2.COLOR_BGR2RGB)
        imgFile4 = np.zeros((tama[0],tama[1]), np.uint8)
        #imgFile3 = cv2.imread("../data/eagle2.jpg")
        tama=imgFile3.shape
        #imgImported=make_image(tama[1],tama[0],tama[2])
        imgFileptr=copy_iplimage_ptr(imgFile3,imgFileptr,cv_img)
        
        ipl_in2_image(imgFileptr,imgImported)
        #save_image(imgImported,"dog_detect")
        r = detect_img(net, meta, imgImported) 
        if charlador:
            print ('Detecciones: '+str(len(r)))
            print (r)
            
            for i in range(len(r)):
                w=int(r[i][2][2])
                h=int(r[i][2][3])
                x=int(r[i][2][0])-w/2
                y=int(r[i][2][1])-h/2
                #cv2.rectangle(imgFile2, (x,y), (x+w,y+h), (255,255,0), thickness=1, lineType=8, shift=0)
        
        for i in range(len(r)):
            if r[i][2][2]<MAXW:
                track.insertNewObject(r[i][2][0],r[i][2][1],r[i][2][2],r[i][2][3],strFeature=r[i][0])
            else:
                print ("        eliminado objeto por tamanio= ",r[i][2][2])
            #w=int(r[i][2][2])
            #h=int(r[i][2][3])
            #x=int(r[i][2][0])-w/2
            #y=int(r[i][2][1])-h/2
                
        if charlador:
            print('Antes de procesar')
            track.printObjets()
            track.printPaths()
            
        track.processObjectstoPaths()
    
        # show rectangle 
        for j in (range(len(track.p.p[:]))):
            x=int(track.p.p[j].rect.x)
            y=int(track.p.p[j].rect.y)
            u=int(track.p.p[j].rect.u)
            v=int(track.p.p[j].rect.v)
            
            cv2.rectangle(imgFile4, (x,y), (u,v),track.p.p[j].colour, thickness=cv2.cv.CV_FILLED, lineType=8, shift=0)
            if track.p.p[j].contado:
                cv2.putText(imgFile3,str(track.p.p[j].str), (int(track.p.p[j].cp.x),int(track.p.p[j].cp.y)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255))
            else:
                cv2.putText(imgFile3,str(track.p.p[j].str), (int(track.p.p[j].cp.x),int(track.p.p[j].cp.y)), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255))
        if charlador:
            print('Despues de procesar')
            track.printPaths()
    
        #Falta graficar los dos ultimos puntos de los paths procesados
        #y si estos pasan la linea de conteo se suma uno
    
        
        #show line and point chose at the begining
        for contar in contadores:
            cv2.circle(imgFile4,contar.point1,3,(0,0,255),-1)
            cv2.line(imgFile4,contar.point1,contar.point2,(0,0,255),1)
            cv2.circle(imgFile4,contar.point2,3,(255,0,255),-1)
            
        # contar los que trayectos que pasen las lineas de conteo
        
        for idx in range(len( track.p.p)):
            if len(track.p.p[idx].path)>2: # si la longitud del path es mayor a dos
                # toma los dos registros mas recientes y los prueba si pasaron la linea de conteo
                p1=(int(track.p.p[idx].path[-1].x),int(track.p.p[idx].path[-1].y))#mas reciente (cuadro actual)
                p2=(int(track.p.p[idx].path[-2].x),int(track.p.p[idx].path[-2].y))#anterior     (cuadro anterior)
                cv2.line(imgFile4,p1,p2,track.p.p[idx].colour,1)
    
                for contar in contadores:
                    if (contar.testLine(p2,p1) and not track.p.p[idx].contadores[contar.linecount]):
                        direct=contar.crossSign(p2,p1)
                        cv2.circle(imgFile3,contar.intersectPoint(p2,p1),4,(100,255,255), -1) #intersecting point
                        track.p.p[idx].contado=True
                        track.p.p[idx].contadores[contar.linecount]=1
                        contar.addToLineCounter(str(track.p.p[idx].str),frames,tiempoactual,direct)
                    
        if pintarTrayectos:
            track.drawPaths(imgFile3)
        
    
    #    print(count)
        if count < len_SPT:
            itbuffer = createLineIterator(P1, P2, imgFile4)
            acc_labels[:,int(count)] = itbuffer[:,2]
            
            b,g,r = cv2.split (cv2.cvtColor(imgFile2,cv2.COLOR_BGR2RGB))
            itbufferB = createLineIterator(P1, P2, b)
            acc_B[:,int(count)] = itbufferB[:,2]
            itbufferG = createLineIterator(P1, P2, g)
            acc_G[:,int(count)] = itbufferG[:,2]        
            itbufferR = createLineIterator(P1, P2, r)
            acc_R[:,int(count)] = itbufferR[:,2]        
            count += 1    
    #        print(count)
            
        elif count == len_SPT:
            classes = [0,1,2,3,4,5,6,7,8]
            cv2.imwrite(os.path.join('SPT_images/Labels_init', "SPT_image_labels_"+tail+"_%04i.JPEG" %number_SPT), acc_labels)
            num2clases=['particular', 'bus', 'motociclista', 'minivan', 'peaton', 'camion', 'taxi', 'ciclista', 'tractomula']
            acc_labels_th, labels_coord = edge_detect(acc_labels, classes,P1, P2)
            cv2.imwrite(os.path.join('SPT_images/Labels', "SPT_image_labels_"+tail+"_%04i.JPEG" %number_SPT), acc_labels_th)
            SPT_RGB = np.uint8(cv2.merge([acc_R,acc_G,acc_B]))
            cv2.imwrite(os.path.join('SPT_images/RGB', "SPT_image_RGB_"+tail+"_%04i.JPEG" %number_SPT),SPT_RGB )
            print('endendendendendendendendendendendendendendendendendendendendendendend')
            print(number_SPT)
    
            txt_outpath = os.path.join('SPT_images/labels', "SPT_image_RGB_"+tail+"_%04i.txt" %number_SPT)
            txt_outfile = open(txt_outpath, "w")  
            
            count = 0
            
            n_obj_tot = 0
            for o in range (len(labels_coord)):
                n_obj_tot += len(labels_coord[o])
            txt_outfile.write(str(n_obj_tot)+ '\r\n')
            for j in (range(len(classes))) :
                labels_coord_cl = labels_coord[j]
                obj = len(labels_coord_cl)
    #            colour = (int(random.uniform(0,255)),int(random.uniform(0,255)),int(random.uniform(0,255)))
                colour = [(0,0,0),(0,0,255),(0,255,0),(255,0,0),(255,255,255),(0,255,255),(128,0,0),(255,165,0),(255,255,0)]            
                if obj != 0:
                    for n_obj in range (obj):
                        x= int(labels_coord_cl[n_obj][0])
                        y= int(labels_coord_cl[n_obj][1])
                        u= int(labels_coord_cl[n_obj][2])
                        v= int(labels_coord_cl[n_obj][3])
                        cv2.rectangle(SPT_RGB, (x,y), (u,v), colour[j], thickness=2, lineType=8, shift=0)
                        
#                        w = u-x
#                        h = v-y
#                        xc = int(x+w/2)
#                        yc = int(y+h/2)
#                        txt_outfile.write(str(xc)+' '+str(yc)+' '+str(w)+' '+str(h)+' '+num2clases[j]+'-'+str(j)+ '\r\n')
                        txt_outfile.write(str(x)+' '+str(y)+' '+str(u)+' '+str(v)+' '+num2clases[j]+'-'+str(j)+ '\r\n')
           
            cv2.imwrite(os.path.join('SPT_images/RGB_rectangle', "SPT_image_RGB_rec_"+tail+"_%04i.JPEG" %number_SPT),SPT_RGB )        
            
            cv2.imshow('SPT', SPT_RGB)
            txt_outfile.close()              
            number_SPT += 1
    
            
    #    cv2.imshow('Video', imgFile4)
        k = cv2.waitKey(2)& 0xFF
        if k==ord('q'):    # Esc key=537919515 en linux WTF??? para parar y en mi otro PC 1048689
            print ('interrupcion de usuario...')
            break
    
    for contar in contadores:
        contar.saveFinalCounts(frames)
    cv2.imwrite('ultimofotogramaprocesado.jpg',imgFile3)
    print ('Saliendo...')
    cv2.destroyAllWindows()
    cam.release()
exit()