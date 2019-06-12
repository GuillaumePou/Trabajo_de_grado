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
from Lineiterator import *


#SDM YOLO TRAIN 80000 iterations
net = load_net("../data/yolo-obj.cfg", "../weight_SDM/yolo-obj_final.weights", 0)
meta = load_meta("../data/obj.data")
    
charlador=False
framesttl=5
deCamara=False
MAXW=550
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
    print ('Couldn t open the camera')
    exit()

imgFile3 = cv2.cvtColor(imgFile2, cv2.COLOR_BGR2RGB)
tama=imgFile2.shape
imgImported=make_image(tama[1],tama[0],tama[2])

imgFileptr,cv_img=get_iplimage_ptr(imgFile3)    
ipl_in2_image(imgFileptr,imgImported)
rgbgr_image(imgImported)

track=tr.tracking(verbose=charlador,mindist=mindist,framesttl=framesttl)

lineasDeConteo=1 # only one couting line
print ("Only ",lineasDeConteo," line for spatio-temporal accumulation")

contadores=[]
for cc in range(lineasDeConteo):
    sleep(1)
    lineaDeConteo=lc.selectLine(imgFile2,ownString='Select the line for the spatio-temporal accumulation #' +str(cc+1),filename=archsal,linecount=cc+1)
# parameters of each line of counting    
    sleep(1)
    contadores.append(lc.counter(lineaDeConteo.pt1,lineaDeConteo.pt2,filename=archsal,linecount=cc+1,fps=20))


count = 0 #counter for the length of the image
len_SPT = 416 # length of the STI could me more
number_SPT = 1 #number of the STI processed

#points of the ST line
P1 = lineaDeConteo.pt1 
P2 = lineaDeConteo.pt2

#initialisation of the STI
itbuffer = createLineIterator(P1, P2, cv2.cvtColor(imgFile2,cv2.COLOR_BGR2GRAY))
acc_labels = np.ones((len(itbuffer[:,2]),len_SPT))
acc_labels[:,int(count)] = itbuffer[:,2]

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
    
colour = [(0,0,0),(0,0,255),(0,255,0),(255,0,0),(255,255,255),(0,255,255),(128,0,0),(255,165,0),(255,255,0)]            
num2clases=['particular', 'bus', 'motociclista', 'minivan', 'peaton', 'camion', 'taxi', 'ciclista', 'tractomula']
classes = [0,1,2,3,4,5,6,7,8]

while True:
    ret_val, imgFile2 = cam.read()
    frames+=1 
    if not ret_val:#end process at the end of the video
        print ("End of the video, exit")
        break
    
    segframes=cam.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    tiempoactual=cam.get(cv2.cv.CV_CAP_PROP_POS_MSEC)*20.0/30.0
    imgFile3 = cv2.cvtColor(imgFile2,cv2.COLOR_BGR2RGB)
    imgFile4 = np.zeros((tama[0],tama[1]), np.uint8)
    tama=imgFile3.shape
    imgFileptr=copy_iplimage_ptr(imgFile3,imgFileptr,cv_img)
    ipl_in2_image(imgFileptr,imgImported)

    r = detect_img(net, meta, imgImported) 
    if charlador:
        print ('Detections: '+str(len(r)))
        print (r)
        
        for i in range(len(r)):
            w=int(r[i][2][2])
            h=int(r[i][2][3])
            x=int(r[i][2][0])-w/2
            y=int(r[i][2][1])-h/2
    
    for i in range(len(r)):
        if r[i][2][2]<MAXW:
            track.insertNewObject(r[i][2][0],r[i][2][1],r[i][2][2],r[i][2][3],strFeature=r[i][0])
        else:
            print (" Object deleted because of size = ",r[i][2][2])
            
#    if charlador:
#        print('Antes de procesar')
#        track.printObjets()
#        track.printPaths()
        
    track.processObjectstoPaths()

    # show rectangle 
    for j in (range(len(track.p.p[:]))):
        x=int(track.p.p[j].rect.x)
        y=int(track.p.p[j].rect.y)
        u=int(track.p.p[j].rect.u)
        v=int(track.p.p[j].rect.v)
        cv2.rectangle(imgFile4, (x,y), (u,v),track.p.p[j].colour, thickness=cv2.cv.CV_FILLED, lineType=8, shift=0)
    
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
        
    elif count == len_SPT:
        cv2.imwrite(os.path.join('SPT_images/Labels_init', "SPT_image_labels_"+tail+"_%04i.jpg" %number_SPT), acc_labels)
        acc_labels_th, labels_coord = edge_detect(acc_labels, classes,P1, P2)
        cv2.imwrite(os.path.join('SPT_images/Labels', "SPT_image_labels_"+tail+"_%04i.jpg" %number_SPT), acc_labels_th)
        SPT_RGB = np.uint8(cv2.merge([acc_R,acc_G,acc_B]))
        cv2.imwrite(os.path.join('SPT_images/RGB', "SPT_image_RGB_"+tail+"_%04i.jpg" %number_SPT),SPT_RGB )
        print(number_SPT)        
        print('------------------------------------------------')
        

        txt_outpath = os.path.join('SPT_images/labels', "SPT_image_RGB_"+tail+"_%04i.txt" %number_SPT)
        txt_outfile = open(txt_outpath, "w")  
        
        count = 0
        
        #create the labeling file
        n_obj_tot = 0
        for o in range (len(labels_coord)):
            n_obj_tot += len(labels_coord[o])
        txt_outfile.write(str(n_obj_tot)+ '\r\n')
        for j in (range(len(classes))) :
            labels_coord_cl = labels_coord[j]
            obj = len(labels_coord_cl)
            if obj != 0:
                for n_obj in range (obj):
                    x= int(labels_coord_cl[n_obj][0])
                    y= int(labels_coord_cl[n_obj][1])
                    u= int(labels_coord_cl[n_obj][2])
                    v= int(labels_coord_cl[n_obj][3])
                    cv2.rectangle(SPT_RGB, (x,y), (u,v), colour[j], thickness=2, lineType=8, shift=0)
                    txt_outfile.write(str(x)+' '+str(y)+' '+str(u)+' '+str(v)+' '+num2clases[j]+'-'+str(j)+ '\r\n')
                    
        #to keep the different images of the process           
        cv2.imwrite(os.path.join('SPT_images/RGB_rectangle', "SPT_image_RGB_rec_"+tail+"_%04i.jpg" %number_SPT),SPT_RGB )        
        cv2.imwrite(os.path.join('reportusefull', "spt"+tail+"_%04i.jpg" %number_SPT),SPT_RGB )
        cv2.imwrite(os.path.join('reportusefull', "algo"+tail+"_%04i.jpg" %number_SPT),imgFile3 )
        cv2.imwrite(os.path.join('reportusefull', "algo-gray"+tail+"_%04i.jpg" %number_SPT),imgFile4 )
        cv2.imwrite(os.path.join('reportusefull', "spt-gray"+tail+"_%04i.jpg" %number_SPT),acc_labels )
        cv2.imwrite(os.path.join('reportusefull', "spt-rect"+tail+"_%04i.jpg" %number_SPT),acc_labels_th )
        
        cv2.imshow('SPT',SPT_RGB)# Show the STI with the bounding box
        txt_outfile.close()              
        number_SPT += 1
        
#        to show the creation of the STI
#    SPT_RGB_build = np.uint8(cv2.merge([acc_R,acc_G,acc_B]))    
#    cv2.line(imgFile2,P1,P2,1)
#    cv2.imshow('video', imgFile2)     
#    cv2.imshow('STI-build', SPT_RGB_build) 
    
    k = cv2.waitKey(2)& 0xFF
    if k==ord('q'):    # Esc key=537919515 en linux WTF??? para parar y en mi otro PC 1048689
        print ('Interuption by the user...')
        break

print ('Exit')
cv2.destroyAllWindows()
cam.release()
exit()
