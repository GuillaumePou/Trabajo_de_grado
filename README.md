# Trabajo de Grado

This repository presents the code of my "trabajo de grado" realized to obtain the undergraduate degree in electronic engineering from the "Pontificia Universidad Javeriana" of Bogota and the master from the french engineering school INP-ENSEEIHT.

The work is in two part.
 - The fisrt part called "Labeling_algorithm" presents a solution to label spatio temporal images.
 - The second part called "Inference" presents the counting algorithm using a spatio-temporal images to count and classify vehicles in real time.
 
 How to use?
 

All the weights necessary to run the trained model are avaible at the following link:
https://www.mediafire.com/file/plqcc2sw4pclj3x/weight_SDM.tar.gz/file

You shoul download then unzipp and copy the folder in inference folder of the repository.

First you should use the make file of darknet to set up darknet (depending of your configuration).
Then, the following librairies need to be installed:
- openCV
- EasyGui
- numpy

Then to run these lines in the cmd
PATH_to_repo/inference/python python inference.py

For the labeling algorithm weights are availaible here:
https://www.mediafire.com/file/ucoqdmly04tbybo/weight.rar/file

