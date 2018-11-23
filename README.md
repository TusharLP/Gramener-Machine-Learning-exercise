# Gramener-Machine-Learning-exercise

## Introduction
This is the exercise of the classification of 46 different Devanagari Characters from the 32x32 images of handwritten characters. It uses data from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset) , a popular repository for machine learning datasets. A total of 92000 images is used as training and test sets, of which 78200 images were used as training images and 13800 images were used as test images. Here, for the classification of the images 'Convolutional Neural Networks' is used.

## Dataset
[Devanagari Handwritten Character Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00389/DevanagariHandwrittenCharacterDataset.zip) [76.6MB]

## Installing packages
This exercise requires the use of TensorFlow package and Keras package. Please make sure the required packages are installed. For installing TensorFlow package visit [TensorFlow website](https://www.tensorflow.org/) . For installing Keras package run `pip install --upgrade keras`.

## Importing required libraries
The Convolutional Neural Networks needs to have following libraries to work:
* Convolutional Layer: Convolutional layer is added using Conv2D method of Keras library. This method is imported using `from keras.layers import Conv2D`
* Max Pooling: Max pooling is done using MaxPooling2D method of Keras library. This method is imported using `from keras.layers import MaxPooling2D`
* Flattening: The max pooled layer is flattened using Flatten method of Keras library. This method is imported using `from keras.layers import Flatten`
* Hidden Layer: The hidden layers are added using Dense method of Keras library. This method is imported using `from keras.layers import Dense`
* Initialising CNN: The CNN is initialised using Sequential method of Keras library. This method is imported using `from keras.models import Sequential`

## Building the CNN
### Initialising CNN
The CNN is initialised using the Sequential method.
### Convolutional layer
Convolutional layer with a kernel size of (3, 3), ReLU activation and 64 filters is added using Conv2D method.
### Max Pooling
The convolutional layer is max pooled with a pool size of (2, 2) using MaxPooling2D method.
### Adding more convolutional layers and max pooling
One more convolutional layer with a kernel size of (2, 2), ReLU activation and 32 filters is added and is max pooled using a pool size of (2, 2).
### Flattening
The final max pooled layer is flattened using Flatten method.
### Full Connection
One hidden layer with 64 units and ReLU activation is added along with the output layer with 46 units as the number of characters is 46 and Softmax activation is added using Dense method.
### Compiling the CNN
All the layers are compiled using 'adam' optimezer and loss function of Cross Entropy.

## Fitting the CNN to the image
### Image Preprocessing
The images are preprocessed to rescale the pixel values between 0 and 1 for training and test sets. The training set data is augmented using zoom, shear and horizontal flip features of ImageDataGenerator method of Keras library.
### Creating Training and Test sets
`training_set` and `test_set` is created using images from Train and Test folders of the dataset respectively. The images are read in Grayscale.
### Training the Classifier
The CNN is trained using `training_set` with 34 epochs and 2300 steps per epoch. The CNN is validated using `test_set` with 6900 validation steps.

## Conclusion
After training the CNN by `training_set` and validating against `test_set` an accuracy of 0.9704 is obtained on `test_set`.