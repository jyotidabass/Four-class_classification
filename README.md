#  Image Classification using Convolutional Neural Networks & TensorFlow.

Problem:
Classification of images specifically chair, kitchen, knife and saucepan using Convolutional Neural Networks and Tensorflow API.

Data:
Dataset on Kaggle containing 5214 training images of 4 classes and 1267 testing images. The four classes are chair, kitchen, knife and saucepan.
https://www.kaggle.com/mbkinaci/chair-kitchen-knife-saucepan

Model:
We will use Alex Net architecture with 5 convolution layers and 3 fully connected layers  with max pooling in between them. For implementing a CNN, we will stack up Convolutional Layers, followed by Max Pooling layers. We will also include Dropout to avoid overfitting. Finally, we will add a fully connected ( Dense ) layer followed by a softmax layer. 

Prediction:
90% accuracy on 30 epochs.




