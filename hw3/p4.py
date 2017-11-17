import os
import argparse
from keras.models import load_model
#from termcolor import colored,cprint
import keras.backend as K
from keras.utils import *
import numpy as np
import matplotlib.pyplot as plt

def deprocessimage(x):

    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    #x += 0.5
    x = np.clip(x, 0, 1)
    
    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

base_dir = os.path.dirname(os.path.dirname(os.path.realpath("~/Documents/ML2017FALL/hw3")))
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
model_dir = os.path.join(base_dir, 'model')

def read_data(filename, label=True, width=48, height=48):
    width = height = 48
    with open(filename, 'r') as f:
        data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
        data = np.array(data)
        X = np.delete(data, range(0, len(data), width*height+1), axis=0).reshape((-1, width, height, 1)).astype('float')
        Y = data[::width*height+1].astype('int')

        X /= 255

        if label:
            return X, Y
        else:
            return X

def main():
    print("Loaded model from {}")

    private_pixels = read_data('../../train拷貝.csv', label=False)
    Classifier = load_model('../../model-00022-0.64176.h5')
    input_img = Classifier.input
    img_ids = [796]
    

    for idx in img_ids:
        plt.figure()
        plt.imshow(private_pixels[idx].reshape(48,48),cmap='gray')
        val_proba = Classifier.predict(private_pixels[idx].reshape((1,48,48,1)))
        pred = val_proba.argmax(axis=-1)
        target = K.mean(Classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        ##
        heatmap = deprocessimage(private_pixels[idx].reshape(48,48))
       
        ##
        
        

        thres = 15
        see = private_pixels[idx].reshape(48, 48)
        # for i in range(48):
            # for j in range(48):
                # print heatmap[i][j]
        print(heatmap[25,25])
        see[np.where(heatmap[:,:] <= thres)] = 0#np.mean(see)
       
        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        test_dir = os.path.join(cmap_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        fig.savefig(os.path.join(test_dir, '{}.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        test_dir = os.path.join(partial_see_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        fig.savefig(os.path.join(test_dir, '{}.png'.format(idx)), dpi=100)

main()