# Trabajo de Grado

This repository presents the code of my "Trabajo de Grado" realized to obtain an undergraduate degree in electronic engineering from the [Pontificia Universidad Javeriana](https://www.javeriana.edu.co/home) of Bogota and a master degree from the french engineering school [INP-ENSEEIHT](http://www.enseeiht.fr/en/index.html). The project received the congratulations of the jury from the Pontificia Universidad Javeriana.

## Project description

This thesis presents a new method using spatio-temporal images to count and to classify in nine objects-classes objects crossing a defined line in the street using convolutional neural networks. The nine classes are pedestrian, bicycle, motorcycle, personal car, taxi, minivan, truck, trucking-rigs, bus. Firstly, it presents an automatic labeling algorithm of the spatio-temporal images based on an opensource algorithm of the "Secretaria de movilidad de Bogota". The labeling algorithm allows creating a data set, to train different convolutional networks, in particular, YOLO networks. The trained models are then used to count the numbers of objects in each spatio-temporal image to realize the counting for the whole video test. It reaches in some condition around 90% accuracy and runs at 40 frames per second. For more detail about the thesis and review the results, take a look at the thesis available in the PDF file.


## Presentation of the repo

The repository is divided into two parts.
- The first part called [Labeling_algo](https://github.com/GuillaumePou/Trabajo_de_grado/tree/master/Labeling_algo) presents a solution to label spatio-temporal images based on an open-source algorithm presented here: [urban-dataset](http://urban-dataset.com/).
- The second part called [Inference](https://github.com/GuillaumePou/Trabajo_de_grado/tree/master/Inference) presents the counting algorithm using spatio-temporal images to count and classify vehicles in real-time. It uses [darknet](https://pjreddie.com/darknet/) and YOLO models developed by [pjreddie](https://github.com/pjreddie).

### Installing the labeling algorithm
#### Prerequisites

Download weights [here](http://www.mediafire.com/file/plqcc2sw4pclj3x/weight_SDM.tar.gz/file).
After downloading, unzip then copy the folder (weight_SDM) in Labeling_algorithm folder of the repository.

Then setup darknet for your computer and useful libraries, see below (Prerequisites).

#### Recommended video for test
To try the code, you can use the following [video](https://www.mediafire.com/file/e3xnqpobyl2lojm/CAM11_20190314072959_7.avi/file). It is one of the video used for the project.

#### Running the tests
To run the labeling algorithm with one video, and get the label run the following line.
```
PATH_TO_REPO/Labeling_algo/python python Aforo.py
```
If you want to run it for several videos without the need to reload models and video for each video, place all the videos in the same folder and then run: 
```
PATH_TO_REPO/Labeling_algo/python python Aforo-auto.py
```

In the folder Labeling_algo/python/SPT_images, you will obtain different images representing the different steps of the process and also the labels in a textfile. (see read.me of the folder for more explanation)

To convert to YOLO input the labels you can use the folder:
```
PATH_TO_REPO/Labeling_algo/python/Convertion_labels
```

It is then possible to train YOLO models using the Inference folder or directly try it with the available weights.

### Installing the Inference algorithm
#### Prerequisites
Download weights [here](https://www.mediafire.com/file/ucoqdmly04tbybo/weight.rar/file).
After downloading, unzip then copy the folder in Inference folder of the repository.

Then setup darknet for your computer and useful libraries, see below.

#### Recommended video for test
To try the code, you can use the following [video](https://www.mediafire.com/file/e3xnqpobyl2lojm/CAM11_20190314072959_7.avi/file). (Same as above)

#### Running the tests
To try the code run the following command: 
```
PATH_TO_REPO/Inference/python python Inference.py
```
A graphic interface will guide through. PS: The threshold of detection is not the same for each model, that is why there are different darknet files in that folder.

After processing the whole video, a sum-up file is available in the video folder.


### Prerequisites

- [Python 2.7](https://www.python.org/downloads/release/python-2715/)
- [Darknet](https://pjreddie.com/darknet/install/). You can directly use the makefile in the folders but you probably need to change it depending of your configuration.
- Python lybrairies: 

[OpenCV](https://opencv.org/), [EasyGui](https://pypi.org/project/easygui/), [numpy](https://pypi.org/project/numpy/), [glob](https://pypi.org/project/glob2/).

## Issues

I will be pleased to answer you for any questions or issues.

## Authors

* **Guillaume Poullain** [LinkedIn](https://www.linkedin.com/in/guillaume-poullain/?locale=en_US)

## License

This project is not licensed.

## Acknowledgments

* **Francisco Carlos Calderon** - *Thesis director* - [GitHub](https://github.com/calderonf)
