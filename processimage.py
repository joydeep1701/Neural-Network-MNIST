import numpy as np
import os.path
import base64
from PIL import Image
import sys
import matplotlib.pyplot as plt

target_shape = [28,28]

def show_a_single_mnist_digit(data):
    pixels = data.copy().reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


def rgb2gray(rgb):
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return gray


def naiveInterp2D(M, newx, newy):
	result = np.zeros((newx,newy))
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			indx = i*newx / M.shape[0]
			indy = j*newy / M.shape[1]
			result[indx,indy] +=M[i,j]
	return result


def preprocess(jpgtxt):
    # data = base64.decodestring(data)
    data = jpgtxt.split(',')[-1]
    data = base64.b64decode(data.encode('ascii'))

    g = open("temp.jpg", "wb")
    g.write(data)
    g.close()

#    infile = "bigtemp.jpg"
#    outfile = "temp.jpg"
#    size = 28, 28
#    im = Image.open(infile)
#    im.thumbnail(size, Image.ANTIALIAS)
#    im.save(outfile, "JPEG")

    pic = Image.open("temp.jpg")
    M = np.array(pic) #now we have image data in numpy
    M = rgb2gray(M)
    M = naiveInterp2D(M,target_shape[0],target_shape[0])
    n = M/3000
    n = n.reshape(-1)
    if np.isnan(np.sum(n)):
        n = np.zeros(n.shape)
    return n

def latest_image_data():
    pic = Image.open("temp.jpg")
    M = np.array(pic) #now we have image data in numpy
    M = rgb2gray(M)
    M = naiveInterp2D(M,target_shape[0],target_shape[0])
    n = M/3000
    n = n.reshape(-1)
    if np.isnan(np.sum(n)):
        n = np.zeros(n.shape)
    return n
