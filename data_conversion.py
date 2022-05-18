# -*- coding: utf-8 -*-
"""
Created on Sun Nov  23 13:32:42 2018

@author: Surya
"""

#Importing libraries
import cv2                 
import numpy as np         
import os                  
from random import shuffle 
from tqdm import tqdm      

training_data_path = './training'
testing_data_path = './testing'
# our images are 80x80x3
IMG_SIZE = 80 

#Converting the output images into one-hot format
def label_img(img):
    word_label = img.split('_')[0]
    if word_label == 'chair': return [1,0,0,0]
    elif word_label == 'kitchen': return [0,1,0,0]
    elif word_label == 'knife': return [0,0,1,0]
    elif word_label == 'saucepan': return [0,0,0,1]
    

#function to read  training data images and convert it np array    
def process_train_data():
    training_data = []
    for img in tqdm(os.listdir(training_data_path)):
        label = label_img(img)
        path = training_data_path + '\\' + img
        img = cv2.imread(path,1)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            training_data.append([np.array(img),np.array(label)])
        else:
            pass
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

process_train_data()

#function to read test data images and convert it np array
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(testing_data_path)):
        label = label_img(img)
        path = testing_data_path + '\\' + img
        img = cv2.imread(path,1)
        
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            testing_data.append([np.array(img),np.array(label)])
        else:
             pass
    #        testing_data.append([np.array(img),label])       
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

process_test_data()

##loading the training and testing data to verify if images are loaded properly
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')


