# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/ModelSummary.png "Model Visualization"
[image2]: ./writeup_images/Center_image_example.jpg "Center"
[image3]: ./writeup_images/Left_image_example.jpg "Left"
[image4]: ./writeup_images/Right_image_example.jpg "Right"
[image5]: ./writeup_images/Center_image_example_flipped.jpg "CenterFlipped"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md  summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model architecture that I used here was the NVIDIA architecture which gives very good performance for self driving autonomous cars.

The model consists of 5 layers of  convolutional neural network with 5x5 filter sizes for the first 2 layers and 3x3 for the next 3 layers  and depths between 24 and 64. It has 4 fully connected (Dense)layers.

The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer using the formula
x= (x/255.0) - 0.5 to normalize the data in the range 0.5 to -0.5

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model uses an Adam optimizer with mean squared error loss , the learning rate was not modified manually.

#### 4. Appropriate training data

The training data consisted of images which had center lane driving , the samples of the left and right camera view and also the flipped images of the center lane camera view were added in order to make sure that the training set was versatile enough for the model to learn and traverse the road accordingly. This ensured that after simulation the car was able to traverse along the road whithout deviating much from the center and without hitting the side pavements. Since the training data size was large, generators were used to generate batches of sample image data for training as well as validation as seen in the code with a batch size of 32.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try out various models as suggested in the tutorials and then work on the best fit.

My first step was to use a simple model with only a Flatten and a fully connected layer and then testing the performance of the model by training it for 3-4 epochs. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set(80% and 20% of the sample data respectively).

Since the performance of the car on the simulator was not satisfactory for the basic model I decided to switch to the Lenet model. I also added preprocessing such as the  normalization function mentioned above to normalize the sample data in the range of 0.5 to -0.5. I also added a dropout layer  after the first convolutional layer hoping to improve the performance. This worked better than the simple model as the car tried to stay at the center of the road but still failed to complete a lap since it was still hitting the pavements on the side. 

After this I decided to go for slightly complex models like comma.ai or NVIDIA hoping that the performance would be better in them. I tried both models and found the NVIDIA architecture working better when combined with Image augumentation techniques like adding the left and right camera view images and flipping the center camera image and adding it to the data set and training the model on that.The image was cropped to remove 50 rows from the top and 20 layers from the bottom to omit the hood of the car and the sky, tree and other elements in the image which were not useful for calculating the steering angle. A correction factor of 0.2 was also used when appending the left and right camera images so that the model could learn to correct itself when it deviates to the left or right side.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I changed the throttle speed to 6 instead of 9 so that the car would be able to correct its steering angle according to the correction as mentioned above factoring the lag in the simulator due to limitations in hardware of the machine(Graphics card).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of the NVIDIA model with 5 layers of  convolution with 5x5 filter sizes for the first 2 layers and 3x3 for the next 3 layers and depths between 24 and 64.It is followed by a flattening layer and has 4 fully connected (Dense)layers and uses RELU activation for its layers.

Here is a screenshot  visualizing  the architecture 

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

Due to limitations of the hardware as mentioned above I was unable to add additional sample data to the existing data set.

However I have implemented few image augumentation techniques by adding the left and right camera view images  to the sample data set and used a correction factor of 0.2 along with the steering angle of the center image on them to train the model to adjust itself accordingly and correct its path if it were to deviate.

Here is an example showing the images of 

##### The Center View 
![alt text][image2]

##### The Left View 

![alt text][image3]

##### The Right View 

![alt text][image4]

I have also flipped the image of the center view and also its steering angle and added it to the data set so that the model is not biased towards a particular direction and is able to perform equally well on roads which are winding in the opposite direction compared to the one in the dataset.

##### The Flipped image

![alt text][image5]


 I then preprocessed this data by cropping the Information such as trees, hood of the car etc which were unnecessary for calculation of steering angle by cropping 50 rows from top and 20 rows from bottom and normalized the data set to the range 0.5 to -0.5 as mentioned earlier.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I started with 5 epochs for training my model and increased in steps of 5 to check performance and found that 15 epochs were sufficient for proper working of the model. I used an adam optimizer so that manually training the learning rate wasn't necessary.
