# NIST-Handwritten-Alphabets-Lowercase
Googlenet trained Caffemodel for Alphabets (lower case) from NIST dataset 

## NIST Dataset

NIST special database 19 contains iages of handwritten characters. There are around 810,000 character images in the dataset. Nist dataset is available for download from the following url

* [NIST SD 19 home page](https://www.nist.gov/srd/nist-special-database-19)
* [Download the dataset segregated and used for training](https://drive.google.com/file/d/0B0LDJX3BuAYkSjA1VFk3M2tEYjA/view?usp=sharing)

The second link contains the data segregated into Digits, Capital letters and Lower case letters used for training. 

## Googlenet trained caffemodel

The alphabets (lower case letters) from NIST dataset is trained in caffe using Googlenet architecture. The images are 128 x 128 in the dataset and are upsampled to 256 x 256 to be trained on googlenet. For training purpose, 4000 images (on average) per class is randomly chosen to out of which 2000 images per class is used for training and 1000 images each are used to create testing set and validation set.

![Alt text](https://github.com/vj-1988/NIST-Handwritten-Alphabets-Lowercase/blob/master/Images/lowercase.png "Training Accuracy and loss")


The caffemodel can be downloaded from the below link

[Download the caffemodel from Google Drive](https://drive.google.com/file/d/0B0LDJX3BuAYkaUcySnNpVFFMWVE/view?usp=sharing)

The snapshot is from the 30th epoch.


## Sample validation scripts

The sample validation scripts to validate the model on test set is given in validation folder. The confusion_matrix.py should be able to generate the confusion matrix in csv format (albeit with minimal changes)
