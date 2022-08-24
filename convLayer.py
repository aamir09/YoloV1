import tensorflow as tf 
import numpy as np

class Convolve2D(tf.keras.layers.Layer):
    def __init__(self,kernels, filters, padding,  strides=(1,1)) -> None:
        super(Convolve2D,self).__init__()
        self.padding = padding
        self.pads = [self.padding,self.padding] 
        self.paddings =[[0, 0], self.pads, self.pads , [0,0]]
        self.kernels = kernels
        self.filters = filters
        self.strides = strides

    def build(self,inputs):
        channels = inputs[3]
        self.kernel =  tf.Variable(lambda: tf.random.normal([self.kernels,self.kernels, channels,self.filters]), trainable=True)

    def call(self,inputs):
        print(inputs.shape[1:])
        x = tf.nn.conv2d(inputs, filters=self.kernel, strides=[1,self.strides, self.strides, 1], padding = self.paddings)   
        return x  
