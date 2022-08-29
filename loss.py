import tensorflow as tf


def iou(true, pred):
    '''Returns the Intersection Over Union Score between [0,1] for two bounding boxes. The returned type will be similar 
     to the inputs; tensor/numpy.ndarray. 
     
     Parameters:
     true: It is a tensor of last of shape [...,4] where the last four elements are x, y, w, h. (x,y) represents midpoint
     of the object and w & h represents the width and height of the image respectively.
     
     pred: It has the same shape and structure as true, it is the output of the model.
     
     Formula for corner points = (x +/- w/2, y +/- h/2)
  '''
    
    #Bounding Box Coordinates for true box
    x1_t = true[...,0] - true[...,2]/2
    y1_t = true[...,1] - true[...,3]/2
    x2_t = true[...,0] + true[...,2]/2
    y2_t = true[...,1] + true[...,3]/2
    
    #Bouniding Box Coordinates for predicted Box
    x1_p = pred[...,0] - pred[...,2]/2
    y1_p = pred[...,1] - pred[...,3]/3
    x2_p = pred[...,0] + pred[...,2]/2
    y2_p = pred[...,1] + pred[...,3]/2
    
    
    #Bounding Box Coordinates for the intersection
    x1 = tf.math.maximum(x1_t, x1_p)
    y1 = tf.math.maximum(y1_t, y1_p)
    x2 = tf.math.minimum(x2_t, x1_p)
    y2 = tf.math.minimum(y2_t, y2_p)
    
    area_true = abs((x2_t - x1_t)*(y2_t-y1_t))
    area_pred = abs((x2_p - x1_p)*(y2_p-y1_p))
    area_intersection = abs((x2 - x1)*(y2-y1))
    
    iou = area_intersection / (area_true + area_pred - area_intersection)
    
    iou = tf.clip_by_value(iou, 0, 1) 

    return iou
  
def loss(target, predictions):
    
    '''Returns a scalar
    
    The loss have been calculated as mentioned in the paper, You Only Look Once.
    '''
    targetBbox = target[...,21:]
    bbox1 = predictions[...,21:25]
    bbox2 = predictions[...,26:]
    targetClasses = target[...,:20]
    predictedClasses = target[...,:20]

    pObject = target[...,20]

    iou1, iou2 = iou(targetBbox, bbox1), iou(targetBbox, bbox2)
    iou1, iou2 = tf.expand_dims(iou1,axis=-1), tf.expand_dims(iou2,axis=-1)
    concatenated = tf.concat([iou1,iou2],axis = -1)
    print('Concatenated', concatenated.shape)
    maxIou = tf.argmax(concatenated, axis = -1)
    print('MaxIou', maxIou.shape)
    #maxIou = tf.cast(tf.expand_dims(maxIou, axis = -1),tf.float32)
    maxIou = tf.cast(maxIou,tf.float32)
    maxIou = tf.expand_dims(maxIou,axis=-1)

    responsibleBox = bbox1*(tf.cast(1-maxIou,tf.float32)) + maxIou*bbox2
    print('ResponsibleBox',responsibleBox.shape)
    
    ########################################
    # Loss 1: Bounding Box Mid Points Loss #
    ########################################
    bboxLoss = pObject*(tf.keras.losses.mse(targetBbox, responsibleBox))
    
    ########################################
    # Loss 2: Bounding Box Width and Height#
    # Points Loss                          #
    ########################################
    
    hWLoss = pObject*(tf.keras.losses.mse(tf.sqrt(tf.abs(targetBbox[...,-2])), tf.sqrt(tf.abs(responsibleBox[...,-2])))
                     
                     + tf.keras.losses.mse(tf.sqrt(tf.abs(targetBbox[...,-1])), tf.sqrt(tf.abs(responsibleBox[...,-1]))))
    
    
    
    ########################################
    # Loss 3: Loss for probability finding #
    # an object in the bounding box respon-#
    # for identifying the object           #
    ########################################
    
    bbox1 = predictions[...,20]
    bbox2 = predictions[...,25]
    
    responsibleBox = (1-maxIou)*bbox1 + maxIou*bbox2
    
    pcObjectLoss = pObject*tf.keras.losses.mse(1, responsibleBox[:,:,:,0])
    
    ########################################
    # Loss 4: Loss for probability not fin-#
    # ding an object in the bounding box   #
    # responsible for identifying the object 
    ########################################
    
    
    minIou = tf.argmin(concatenated, axis = -1)
#     minIou = tf.cast(tf.expand_dims(minIou, axis = -1),tf.float32)
    minIou = tf.cast(minIou, tf.float32)
    
    bbox1 = predictions[...,20]
    bbox2 = predictions[...,25]
    
    
    responsibleBox = (1-minIou)*bbox1 + minIou*bbox2
#     print(responsibleBox[:,:,0].shape)
#     print(target.shape)
    
    pcNoObjectLoss = (1-pObject)*tf.keras.losses.mse(target[...,20], responsibleBox[...,0])
    
    ########################################
    # Loss 4: Class Score Loss for each cell
    ########################################
    
    classLoss = tf.keras.losses.mse(targetClasses, predictedClasses)
    
    return 5*tf.reduce_sum(bboxLoss)  + 5*tf.reduce_sum(hWLoss) + tf.reduce_sum(pcObjectLoss) + 0.5*tf.reduce_sum(pcNoObjectLoss) + tf.reduce_sum(classLoss)
    




    
    
    
    

