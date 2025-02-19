# MelanomaDetectionAssignmentCNN
> Melanoma Detection Assignment CNN Tensorflow

## Problem statement
> To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

## Table of Contents
* [General Info](#general-information)
* [Technologies or Platform Used](#technologies-or-platform-used)
* [Model Building](#model-building)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

## General Information

- Importing libraries and the dataset
- Data preparation: Train-validation split, test set, specifying the shape of the input data etc.
- Without Augmentation strategy CNN Model
  A. Building and understanding the CNN architecture
  B. Fitting and evaluating the model
- Use Augmentation strategy in building CNN Model
  A. Create Augmentation strategy
  B. Building and understanding the CNN architecture
  C. Fitting and evaluating the model
- Use Augmentor lib in building CNN Model
  A. Add more samples using Augmentor lib
  B. Building and understanding the CNN architecture
  C. Fitting and evaluating the model
- Predict on image from Test dataset

## Technologies or Platform Used
Project is executed on

- Ubuntu 24.04
- Jupiter notebook
- Python 3.10.x
- Kaggle Platform
- Google Colab

#### Libraries used and their versions
Please install below libs before executing notebook

- matplotlib 3.9.x
- seaborn 0.13.x
- scipy 1.14.x
- Tensorflow 2.17
- Augmentor 0.2.12


## Model Building
The above analysis and model on bike renting gives below data points-

### Model 1:
- Built with layers
    - Rescaling input image to (180,180,3)
    - 3 convolutional layers having 32, 64, 128 filters respectively,
    - followed by a max pooling layer after each convolution layer
    - and then Flatten the output of the pooling layer to give us a long vector,
    - then add a fully connected Dense layer with 128 neurons
    - and finally, add a softmax layer with 9 neurons
    - padding is not used, stride is with default value 1
    - ReLU activation function is used in all neurons except last output layer

#### Observations from Model 1:
1. The model demonstrates a consistent increase in training accuracy, reaching up to 83%, while the validation accuracy remains stable at approximately 54% in 20 epochs.
2. The substantial gap between training and validation accuracy indicates that the model has likely learned to fit the noise present in the training data, rather than generalizing well to unseen data. This suggests that overfitting may be occurring.

### Model 2:
- Built with layers
    - Data Augmentation layer with random rotate, flip, contrast variations
    - Rescaling input image to (180,180,3)
    - 3 convolutional layers having 32, 64, 128 filters respectively,
    - followed by a max pooling layer after each convolution layer
    - and then Flatten the output of the pooling layer to give us a long vector,
    - then add a fully connected Dense layer with 128 neurons
    - and finally, add a softmax layer with 9 neurons
    - padding is not used, stride is with default value 1
    - ReLU activation function is used in all neurons except last output layer

#### Observations from Model 2:
1. The implementation of data augmentation has effectively mitigated overfitting, improving generalization.
2. Both the training and validation accuracy are currently within a similar range
3. Accuracy value is low and stable in the range of 50-53%, making it underfit. Model is not performing on either the training or validation data. This suggests that the model is too simplistic or lacks the capacity to learn complex patterns in the data, which is typical in cases of underfitting.

### Model 3:
- Built with layers
    - Rescaling each input to (180,180,3) for normlaization
    - 4 convolutional layers having 32, 64, 128, 256 filters respectively,
    - followed by a max pooling layer after each convolution layer
    - and then Flatten the output of the pooling layer to give us a long vector,
    - then add 2 fully connected Dense layer with 256, 128 neurons
    - and finally, add a softmax layer with 9 neurons
    - The generic way to build a model in Keras is to instantiate a Sequential model and keep adding keras.layers to it.
    - We will also use some dropouts in few of the layers to overcome overfitting
    - padding is not used, stride is with default value 1
    - ReLU activation function is used in all neurons except last output layer

#### Observations from Model 3:
1. Class rebalance has improved train and validation accuracy to great extent as both training and validation accuracy is more than 75%, hence not underfit
2. Model validation and train accuracy is very close, hence not overfit
3. Model has come out of underfitting problem and accuracy is much higher than random guessing
4. Adding more training data set with distorted/flipped/zoomed/cropped images like real life images helped improve the model

### Observations from Predicting on Test data set
1. Predicting model output on completely unseed test data set showing similar results as training accuracy.
2. Ranomly selected image with Melanoma symptoms correctly predicted with model
3. Prediction on 1 sample of each symptom is very close to actual classification.

## Conclusions
- 3 models are built
- Because of limited images, simple model was overfitted
- After applying augmentation strategy in CNN model layer, overfitting issue was overcome, but model was underfit
- After adding sample variation of images of each classification, model performed better and both overfit and underfit problems were resolved
- Model predictions on test unseen data also working as per training accuracy
- Hence melanoma deteciton can be better predicted with final last model with much higher accuracy than random guessing.

## Acknowledgements
Project is done by

- Nikhil Lunawat

## Contact
Created by [@nikhil-lunawat] - feel free to contact me!
