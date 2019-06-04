# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:55:43 2015

This script is to convert the txt annotation files to appropriate format needed by YOLO 

@author: Guanghan Ning
Email: gnxr9@mail.missouri.edu




Clases Alejandro iteracion #1

placa-0


"""

import os
from os import walk, getcwd
from PIL import Image
from shutil import copyfile


classes = ['particular-0', 'bus-1', 'motociclista-2', 'minivan-3', 'peaton-4', 'camion-5', 'taxi-6', 'ciclista-7', 'tractomula-8']

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
    
    
"""-------------------------------------------------------------------""" 

""" Configure Paths"""   
mypath = "labels/original/"
outpath = "labels/out/"
reppath = "images/repetidas/"

labseppath = "labels/claseaparte/"
imgseppath = "images/claseaparte/"


wd = getcwd()
print ('Trabajando en :')
print (wd)
list_file = open('%s/training_list.txt'%(wd), 'w')

""" Get input text file list """
txt_name_list = []
for (dirpath, dirnames, filenames) in walk(mypath):
    txt_name_list.extend(filenames)
    break
#print(txt_name_list)
#txt_name_list.remove('desktop.ini')

""" Process """
for txt_name in txt_name_list:
    # txt_file =  open("Labels/stop_sign/001.txt", "r")
    
    """ Open input text files """
    txt_path = mypath + txt_name
    print("Input:" + txt_path)
    txt_file = open(txt_path, "r")
    lines = txt_file.read().split('\n')   #for ubuntu, use "\r\n" instead of "\n"
    
    
            #
    img_path = str('%s/images/%s.jpg'%(wd, os.path.splitext(txt_name)[0]))
    if (not os.path.isfile(img_path)):
        print ('No se encontró la imagen correspondiente pasando al siguiente archivo de texto.')
        print (img_path)
        os.rename(mypath+txt_name, reppath+txt_name)
        print ('Moviendo archivo.')
        continue
    
    """ Open output text files """
    txt_outpath = outpath + txt_name
    print("Output:" + txt_outpath)
    txt_outfile = open(txt_outpath, "w")
    
    
    """ Convert the data to YOLO format """
    ct = 0
    
    separateClass=0
    for line in lines:
        print('lenth of line is: ')
        print(len(line))
        print('\n')
        if(len(line) > 4):
            ct = ct + 1
            print(line + "\n")
            elems = line.split(' ')
            print(elems)
            print('ok')
            xmin = elems[0]
            xmax = elems[2]
            ymin = elems[1]
            ymax = elems[3]
            classname = elems[4].rstrip()
            namec,splitc=classname.split("-")
            splitc.rstrip()
            
            #t = magic.from_file(img_path)
            #wh= re.search('(\d+) x (\d+)', t).groups()
            im=Image.open(img_path)
            w= int(im.size[0])
            h= int(im.size[1])
            #w = int(xmax) - int(xmin)
            #h = int(ymax) - int(ymin)
            # print(xmin)
            print(w, h)
            b = (float(xmin), float(xmax), float(ymin), float(ymax))
            bb = convert((w,h), b)
            print(bb)
            txt_outfile.write(str(splitc) + " " + " ".join([str(a) for a in bb]) + '\r\n')
            
            print ("mostrando splitc")
            print(splitc)
            type(splitc)            
            
            if int(splitc) == 8:
                print ("Se encontró tractomula...")
                print ("se levanta bandera para copiar archivos resultantes a carpetas correspondientes")
                separateClass=1
                

        
    """ Save those images with bb into list"""
    if(ct != 0):
        list_file.write('%s/images/%s.jpg\n'%(wd,os.path.splitext(txt_name)[0]))
    separateClass=0  
    if(separateClass):
        imgseppath = str('%s/images/claseaparte/%s.jpg'%(wd, os.path.splitext(txt_name)[0]))
        copyfile(mypath+txt_name, labseppath+txt_name)
        copyfile(img_path, imgseppath)
        
print('finish')               
list_file.close()       
