import tensorflow as tf 

class Convolve2D(tf.keras.layers.Layer):
    def __init__(self,padding,kernels, filters, strides=(1,1)) -> None:
        super(Convolve2D).__init__()
        self.padding = padding 
        self.convLayer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernels, strides=strides)

    def call(self,input):
        pads = [self.padding,self.padding]
        paddings =[pads, pads , [0,0]]
        padded = tf.pad(input, paddings , 'CONSTANT')
        return self.convLayer(padded)
        
