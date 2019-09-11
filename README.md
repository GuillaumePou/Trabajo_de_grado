# Trabajo de Grado

This repository presents the code of my "Trabajo de Grado" realized to obtain an undergraduate degree in electronic engineering from the "Pontificia Universidad Javeriana" of Bogota and the master from the french engineering school INP-ENSEEIHT. The project received the congratulations of the jury from the Pontificia Universidad Javeriana.

This thesis presents a new method using spatio-temporal images to count and to classify in nine objects-classes objects crossing a defined line in the street using convolutional neural networks. The nine classes are : pedestrian, bicycle, motorcycle, peronnal car, taxi, minivan, truck, trucking-rigs, bus. Firstly, it presents an automatic labeling algorithm of the spatio-temporal images based on  an opensource algorithm of the "secretaria de movilidad de Bogota". The labeling algorithm allows creating a data set, to train different convolutional networks, in particular, YOLO networks. The results and parameters of training are presented. The trained models are then used to count the numbers of objects in each spatio-temporal image to realize the counting for the whole video test. This counting method is compared with manual counting and the open-source algorithm in terms of counting and also in terms of processing time.

The PDF file is the thesis of the project.


The repository is divided into two parts.
 - The first part called "Labeling_algorithm" presents a solution to label spatio-temporal images based on an open-source algorithm presented here http://urban-dataset.com/
 - The second part called "Inference" presents the counting algorithm using spatio-temporal images to count and classify vehicles in real-time. It uses darknet and YOLO models developped by https://github.com/pjreddie and available here https://github.com/pjreddie/darknet
 
 How to use it?
 
 - Labeling_algorithm
 Download weights available here:
https://www.mediafire.com/file/ucoqdmly04tbybo/weight.rar/file
After downloading, unzip then copy the folder in Labeling_algorithm folder of the repository.

 - Inference
Download weights available here:
https://www.mediafire.com/file/plqcc2sw4pclj3x/weight_SDM.tar.gz/file
After downloading, unzip then copy the folder in inference folder of the repository.

First, you should use the make file of darknet to set up darknet, more details here https://pjreddie.com/darknet/install/.
Then, the following libraries need to be installed:
- openCV
- EasyGui
- numpy

Then to run these lines in the cmd
PATH_to_repo/inference/python python inference.py


Do not hesitate to contact me or to report an issue.
