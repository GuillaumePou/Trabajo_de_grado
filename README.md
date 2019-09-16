# Trabajo de Grado

This repository presents the code of my "Trabajo de Grado" realized to obtain an undergraduate degree in electronic engineering from the "Pontificia Universidad Javeriana" of Bogota and the master from the french engineering school INP-ENSEEIHT. The project received the congratulations of the jury from the Pontificia Universidad Javeriana.

## Project description

This thesis presents a new method using spatio-temporal images to count and to classify in nine objects-classes objects crossing a defined line in the street using convolutional neural networks. The nine classes are pedestrian, bicycle, motorcycle, personal car, taxi, minivan, truck, trucking-rigs, bus. Firstly, it presents an automatic labeling algorithm of the spatio-temporal images based on an opensource algorithm of the "Secretaria de movilidad de Bogota". The labeling algorithm allows creating a data set, to train different convolutional networks, in particular, YOLO networks. The trained models are then used to count the numbers of objects in each spatio-temporal image to realize the counting for the whole video test. It reacches in some condition arround 90% acurracy and run at 40 frames per second. For more detail about the thesis and review the results, take a look at the thesis available in the PDF file.


These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.


### Installing

The repository is divided into two parts.
- The first part called "Labeling_algorithm" presents a solution to label spatio-temporal images based on an open-source algorithm presented here http://urban-dataset.com/
- The second part called "Inference" presents the counting algorithm using spatio-temporal images to count and classify vehicles in real-time. It uses darknet and YOLO models developed by https://github.com/pjreddie and available here https://github.com/pjreddie/darknet

Use that repository

- How to use the labeling algorithm?

Download weights here: https://www.mediafire.com/file/ucoqdmly04tbybo/weight.rar/file.
After downloading, unzip then copy the folder (weight_SDM) in Labeling_algorithm folder of the repository.
Then setup darknet for your computer and useful libraries, see below.
To try the code, you can use the following video: https://www.mediafire.com/file/e3xnqpobyl2lojm/CAM11_20190314072959_7.avi/file
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
You can use the following video: https://www.mediafire.com/file/e3xnqpobyl2lojm/CAM11_20190314072959_7.avi/file
A graphic interface will guide you in different steps. PS: The threshold of detection is not the same for each model, that is why there are different darknet files in that folder.
After processing the whole video, a sum-up file is available in the video folder.

- Darknet and other libraries

First, you should use the make file of darknet to set up darknet, more details here https://pjreddie.com/darknet/install/. Then, the following libraries need to be installed:
openCV
EasyGui
numpy
Then to run these lines in the cmd PATH_to_repo/inference/python python inference.py


A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo


### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

## Running the tests

Explain how to run the automated tests for this system

## Issues

I will be pleased to answer you for any questions or issues.

## Authors

* **Guillaume Poullain** [LinkedIn](https://www.linkedin.com/in/guillaume-poullain/?locale=en_US)

## License

This project is not licensed.

## Acknowledgments

* **Francisco Carlos Calderon** - *Thesis director* - [GitHub](https://github.com/calderonf)
