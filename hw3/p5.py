from keras.models import load_model
from keras import backend as K
import os
import argparse
#from termcolor import colored,cprint
import keras.backend as K
from keras.utils import *
import numpy as np
import matplotlib.pyplot as plt

Classifier = load_model('../../model-00022-0.64176.h5')
Classifier.summary()


def deprocess_image(x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1
    
        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)
    
        # convert to RGB array
        x *= 255
        x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

layer_name = "conv2d_1"
plt.figure()
for filter_index in range(32):


    layer_dict = dict([(layer.name, layer) for layer in Classifier.layers])
    
    for layer in Classifier.layers:
        print(layer.name)

    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])
    
    # compute the gradient of the input picture wrt this loss
    input_img = Classifier.input
    grads = K.gradients(loss, input_img)[0]
    
    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])
    
    
    input_img_data = np.random.random((1, 48, 48, 1)) * 20 + 128.
    # run gradient ascent for 20 steps
    step = 0.3
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    
    
    
    
    img = input_img_data[0]
    img = deprocess_image(img)
    plt.subplot(4,8,filter_index + 1) 
    plt.imshow(img.reshape(48,48),cmap='gray')
    plt.axis('off') 

plt.show()