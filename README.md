# Trabajo de Grado

This repository presents the code of my "Trabajo de Grado" realized to obtain an undergraduate degree in electronic engineering from the "Pontificia Universidad Javeriana" of Bogota and the master from the french engineering school INP-ENSEEIHT. The project received the congratulations of the jury from the Pontificia Universidad Javeriana.

This thesis presents a new method using spatio-temporal images to count and to classify in nine objects-classes objects crossing a defined line in the street using convolutional neural networks. The nine classes are pedestrian, bicycle, motorcycle, personal car, taxi, minivan, truck, trucking-rigs, bus. Firstly, it presents an automatic labeling algorithm of the spatio-temporal images based on an opensource algorithm of the "Secretaria de movilidad de Bogota". The labeling algorithm allows creating a data set, to train different convolutional networks, in particular, YOLO networks. The results and parameters of training are presented. The trained models are then used to count the numbers of objects in each spatio-temporal image to realize the counting for the whole video test. This counting method is compared with manual counting and the open-source algorithm in terms of counting and also in terms of processing time.
The PDF file is the thesis of the project.

The repository is divided into two parts.
- The first part called "Labeling_algorithm" presents a solution to label spatio-temporal images based on an open-source algorithm presented here http://urban-dataset.com/
- The second part called "Inference" presents the counting algorithm using spatio-temporal images to count and classify vehicles in real-time. It uses darknet and YOLO models developed by https://github.com/pjreddie and available here https://github.com/pjreddie/darknet

Use that repository

- How to use the labeling algorithm?

Download weights here: https://www.mediafire.com/file/ucoqdmly04tbybo/weight.rar/file.
After downloading, unzip then copy the folder (weight_SDM) in Labeling_algorithm folder of the repository.
Then setup darknet for your computer and useful libraries, see below.
To try the code, you can use the following video: 
then, you should run the following command: PATH_to_repo/Labeling_algo/python python Aforo.py
If you want to run it for several videos without the need to reload models and video, place all the videos in the same folder and then run: PATH_to_repo/Labeling_algo/python python Aforo-auto.py
In the folder Labeling_algo/python/SPT_images you will obtain different images representing the different steps of the process. (see read.me of the folder for more explanation)
To convert to YOLO input the labels you can use the folder Labeling_algo/python/Convertion_labels
It is then possible to train YOLO models using the Inference folder or directly try it with the available weights.

- How to use the Inference algorithm?

Download weights here: https://www.mediafire.com/file/plqcc2sw4pclj3x/weight_SDM.tar.gz/file.
After downloading, unzip then copy the folder in Inference folder of the repository.
Then setup darknet for your computer and useful libraries, see below.
To try the code run the following command: PATH_to_repo/Inference/python Inference.py
A graphic interface will guide you in different steps. PS: The threshold of detection is not the same for each model, that is why there are different darknet files in that folder.
After processing the whole video, a sum-up file is available in the video folder.

- Darknet and other libraries

First, you should use the make file of darknet to set up darknet, more details here https://pjreddie.com/darknet/install/. Then, the following libraries need to be installed:
openCV
EasyGui
numpy
Then to run these lines in the cmd PATH_to_repo/inference/python python inference.py
Do not hesitate to contact me or to report an issue.
