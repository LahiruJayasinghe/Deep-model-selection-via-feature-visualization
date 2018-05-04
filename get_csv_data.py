from numpy import genfromtxt
import numpy as np
import os
import sys
import matplotlib.image as mpimg
import timeit


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def load_custom_input(name,dtype,width):
    img = rgb2gray(mpimg.imread(name))
    img = np.resize(img, (img.shape[0], width))
    # print(img.shape)
    img = np.reshape(img, (np.product(img.shape)))
    img = np.array(img)[np.newaxis]
    img = np.asarray(img, dtype=dtype)
    # print(img.shape)
    return (img)

class HandleData(object):

    def __init__(self, num_classes):
        self.total_data=0
        self.input_vector_size=0
        self.num_classes = num_classes
        self.current_point = 0


    def onehot_encode(self,number):
        encoded_no = np.zeros(self.num_classes, dtype=np.float32)
        if number < self.num_classes:
            encoded_no[number] = 1
        return encoded_no

    def next_batch(self,batch_size):
        # print("start : " + str(self.current_point))
        if self.current_point >= self.total_data:
            self.current_point = 0
        start = self.current_point
        end = start + batch_size
        return_data = self.data_set[start:end]
        return_label = self.label_set[start:end]
        self.current_point=end
        # print(return_data)
        # print("end : " + str(self.current_point))
        return return_data,return_label