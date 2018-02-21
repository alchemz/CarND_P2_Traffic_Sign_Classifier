# **Traffic Sign Recognition** 

## Writeup


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
 
---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/alchemz/Project2_Traffic_Sign_Classifier/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ：34799
* The size of the validation set is ：4410
* The size of test set is ：12630
* The shape of a traffic sign image is （32，32，3）
* The number of unique classes/labels in the data set is :43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram of label frequency.

![alt text](https://github.com/alchemz/Project2_Traffic_Sign_Classifier/blob/master/CarND-Traffic-Sign-Classifier-Project/data-images/datavisualization.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to convert the images to grayscale because it will help reduce the training time, and it is hard to justify if the additional color information isn't helpful for applocations of interest.

Here is an example of a traffic sign image before and after grayscaling.

![alt text](https://github.com/alchemz/Project2_Traffic_Sign_Classifier/blob/master/CarND-Traffic-Sign-Classifier-Project/data-images/grayscale.png)

As a last step, I normalized the image data to the range(-1, 1). This is done by using the code X_train_normalized = (X_train-128)/128. I choose it mainly becase it is suggested by the course.

Here is an example of an original image and an augmented image:

![alt text](https://github.com/alchemz/Project2_Traffic_Sign_Classifier/blob/master/CarND-Traffic-Sign-Classifier-Project/data-images/normalized.png)[image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model has been set up with the following steps

1. 5x5 convolution (32x32x1 in, 28x28x6 out)
2. ReLU
3. 2x2 max pool (28x28x6 in, 14x14x6 out)
4. 5x5 convolution (14x14x6 in, 10x10x16 out)
5. ReLU
6. 2x2 max pool (10x10x16 in, 5x5x16 out)
7. 5x5 convolution (5x5x6 in, 1x1x400 out)
8. ReLu
9. Flatten layers from numbers 8 (1x1x400 -> 400) and 6 (5x5x16 -> 400)
10. Concatenate flattened layers to a single size-800 layer
11. Dropout layer
12. Fully connected layer (800 in, 43 out)


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer, which is also implemented in the original LeNet model. The final settings are the following:
- batch size: 100
- epochs: 60
- mu: 0
- sigma: 0.1
- dropout keep probability: 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
First attempt:
- batch size: 100
- epochs: 60
- rate: 0.0009
- mu: 0
- sigma: 0.1
- dropout keep probability: 0.5
- validation set accuracy : 99.4% 
- test set accuracy : 94.0%
- new image, test set accuracy : 12.5%

Second attempt:
- batch size: 128
- epochs: 10
- rate: 0.0009
- mu: 0
- sigma: 0.1
- dropout keep probability: 0.75
- validation set accuracy : 98.6% 
- test set accuracy : 92.1%
- new image, test set accuracy : 0

Third attempt:
- batch size: 256
- epochs: 20
- rate: 0.001
- mu: 0
- sigma: 0.1
- dropout keep probability: 0.75
- validation set accuracy : 98.8%
- test set accuracy : 91.8%
- new image, test set accuracy : 0.0

Fourth attempt:
- batch size: 256
- epochs: 20
- rate: 0.00075
- mu: 0
- sigma: 0.3
- dropout keep probability: 0.55
- validation set accuracy : 90.8%
- test set accuracy : 78.7%
- new image, test set accuracy : 0

Fifth attempt:
- batch size: 256
- epochs: 20
- rate: 0.0009
- mu: 0
- sigma: 0.1
- dropout keep probability: 0.4
- validation set accuracy : 99.2%
- test set accuracy : 92.9%
- new image, test set accuracy : 0
My approach is a modified version of the original LetNet model.


The final approach: 
- batch size: 100
- epochs: 60
- rate: 0.0009
- mu: 0
- sigma: 0.1
- dropout keep probability: 0.5
- validation set accuracy : 99.4% 
- test set accuracy : 94.0%
- new image, test set accuracy : 12.5%

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text](https://github.com/alchemz/Project2_Traffic_Sign_Classifier/blob/master/CarND-Traffic-Sign-Classifier-Project/test-images/11.jpg) ![alt text](https://github.com/alchemz/Project2_Traffic_Sign_Classifier/blob/master/CarND-Traffic-Sign-Classifier-Project/test-images/22.jpg) ![alt text](https://github.com/alchemz/Project2_Traffic_Sign_Classifier/blob/master/CarND-Traffic-Sign-Classifier-Project/test-images/33.jpg) ![alt text](https://github.com/alchemz/Project2_Traffic_Sign_Classifier/blob/master/CarND-Traffic-Sign-Classifier-Project/test-images/44.jpg) 

![alt text](https://github.com/alchemz/Project2_Traffic_Sign_Classifier/blob/master/CarND-Traffic-Sign-Classifier-Project/test-images/55.jpg) 

For Image1:
Image Accuracy = 0.000, which means this image is hard to be classified, mainly because the original image has many noises.

For Image2:
Image Accuracy = 0.500, compared to the other 4 images, the image 2 obtained the best classified
result, mainly because of the high contrast between the sign and the background.

For Image3:
Image Accuracy = 0.333, the test accuracy of image 3 is affected by some characters on the image.

For Image4:
Image Accuracy = 0.250, and image 4 has the same issue with image 1.

For Image5:
Image Accuracy = 0.200, for image 5, there is an object blocked up the traffic sign, and this could be the reason why the tested image accuracy is low for image 5.





