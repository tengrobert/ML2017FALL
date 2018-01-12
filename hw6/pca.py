import os
import argparse
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage import io, img_as_float, img_as_ubyte
import sys

def read_data(dir):
    X = []
    size = None
    for file in sorted(os.listdir(sys.argv[1])):
        if file.endswith('.jpg'):
            img = io.imread(os.path.join(sys.argv[1], file))
            img = img_as_float(img)
            size = img.shape
            X.append(img.flatten())

    return np.array(X), size

def average_face(X, size=(64, 64)):
    mean = np.mean(X, axis=0)
    io.imsave('avgtest.png', mean.reshape(size))

def eigenface(X, size=(64, 64), top=9):
    mean = np.mean(X, axis=0)
    X_center = X - mean
    U, S, V = np.linalg.svd(X_center.T, full_matrices=False)

    for i in range(top):
        image = U[:,i].reshape(size)
        image = img_as_ubyte(image)
        io.imsave(str(i)+'eigenface.png', image)


def reconstruct_face(X, size=(64, 64), top=4):
    mean = np.mean(X, axis=0)
    X_center = X - mean
    U, S, V = np.linalg.svd(X_center.T, full_matrices=False)
    weights = np.dot(X_center, U)

    target = io.imread(os.path.join(sys.argv[1], sys.argv[2]))
    target = target.flatten()
    target = target - mean

    weights2 = np.dot(target, U[:, :top].T)
    pic = np.dot(weights2, U[:, :top].T)
    pic += mean

    recon = pic
    recon -= np.min(recon)
    recon /= np.max(recon)
    recon = (recon*255).astype(np.uint8)
    io.imsave('reconstruction.jpg', recon.reshape(size)) 

X, size = read_data(sys.argv[1])

reconstruct_face(X, size, 4)