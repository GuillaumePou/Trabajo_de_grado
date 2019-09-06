# Trabajo de Grado

This repository presents the code of my "trabajo de grado" realized to obtain an undergraduate degree in electronic engineering from the "Pontificia Universidad Javeriana" of Bogota and the master from the french engineering school INP-ENSEEIHT. The project received the congratulations of the jury from the Pontificia Universidad Javeriana.

The PDF file is the thesis of the project.

The repository is divided into two parts.
 - The first part called "Labeling_algorithm" presents a solution to label spatio-temporal images using an open-source algorithm.
 - The second part called "Inference" presents the counting algorithm using spatio-temporal images to count and classify vehicles in real-time.
 
 How to use it?
 

All the weights necessary to run the trained model are available at the following link:
https://www.mediafire.com/file/plqcc2sw4pclj3x/weight_SDM.tar.gz/file

You should download then unzip and copy the folder in inference folder of the repository.

First, you should use the make file of darknet to set up darknet (depending on your configuration).
Then, the following libraries need to be installed:
- openCV
- EasyGui
- numpy

Then to run these lines in the cmd
PATH_to_repo/inference/python python inference.py

For the labeling algorithm weights are available here:
https://www.mediafire.com/file/ucoqdmly04tbybo/weight.rar/file

