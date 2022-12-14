import tensorflow as tf 
from YoloV1.convLayer import Convolve2D
import json

tf.keras.backend.clear_session()
# with open('./YoloV1/architecture.json','r') as f:
#     arc = json.load(f)
    
    
def buildModel():
    with open('./YoloV1/architecture.json','r') as f:
        arc = json.load(f)
        
        
        
      
    inp = tf.keras.Input((448,448,3))

    firstLayer = True

    for key in arc:
        repetitions = int(key.split('-')[-1])
        i=0
        while i<repetitions:
            structure = arc[key]
            for struct in structure:
                if struct['pooling']!=1:
                    kernels, filters, paddings, strides = struct['kernel'], struct['filters'], struct['padding'], struct['strides']
                    # print(strides)
                    if firstLayer:
                        x =  Convolve2D(kernels, filters, paddings, strides)(inp)
                        x = tf.keras.layers.BatchNormalization()(x)
                        x = tf.keras.layers.LeakyReLU(0.1)(x) 
                        firstLayer = False
                    else:
                        x =  Convolve2D(kernels, filters, paddings, strides)(x)
                        x = tf.keras.layers.BatchNormalization()(x)
                        x = tf.keras.layers.LeakyReLU(0.1)(x) 
                else:
                    kernels, filters, paddings, strides = struct['kernel'], struct['filters'], struct['padding'], struct['strides']
                    x = tf.keras.layers.MaxPooling2D(
                                    pool_size=(strides, strides),
                                    strides=strides, padding="same",
                                    )(x)
            i+=1
    # x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    # x = tf.keras.layers.Reshape((7,7,30))(x)

    x = tf.keras.layers.Dense(30,activation='linear')(x)
    # x = tf.keras.layers.LeakyReLU(0.1)(x)
    # x = tf.keras.layers.Reshape((7,7,30))(x)

    model = tf.keras.Model(inp,x)
    
    return model

# model = buildModel()

# print(model.summary())

