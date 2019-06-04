# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:26:00 2019

@author: gpoullain
"""

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
    img_path = str('%s/images/%s.JPG'%(wd, os.path.splitext(txt_name)[0]))
    
    
    """ Open output text files """
    txt_outpath = outpath + txt_name
    print("Output:" + txt_outpath)
    txt_outfile = open(txt_outpath, "w")
    

        
                
list_file.close()       
