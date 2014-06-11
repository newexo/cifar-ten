import numpy
from numpy import fft
from PIL import Image
import pywt
import os

def load(name):
    # don't want name to be a number or any data type other than string
    name = str(name)
    # construct the path assuming the environment variable is set to the location
    # of the CIFAR-10 data directory, which has a subdirectory train
    path = os.path.join(os.environ['CIFIR_DATA_DIR'], 'train', str(name) + '.png')
    return Image.open(path)

def reconstruct(arr, size):
    n = size[0] * size[1]
    imr = Image.fromstring('L', size, arr[0:n].tostring(), 'raw')
    img = Image.fromstring('L', size, arr[n:2*n].tostring(), 'raw')
    imb = Image.fromstring('L', size, arr[2*n:3*n].tostring(), 'raw')
    return Image.merge('RGB', (imr, img, imb))

def muSigma(x):
    # compute the means and standard deviations of each of the features
    mu = numpy.mean(x, axis=0)
    sigma = numpy.std(x, axis=0)
    
    return mu, sigma

def normalize(x, mu, sigma):
    return numpy.array([(row - mu) / sigma for row in x])

def normalizeImage(image):
    # find the dimensions of image, which is assumed to be square
    n = image.size[0]
    
    # construct a list of three arrays, one for each channel RGB, RGBA, etc.
    arr = [numpy.fromstring(i.tostring(), numpy.uint8) for i in image.split()]
    
    # mean normalize the arrays
    flat = numpy.concatenate(arr)
    mu = flat.mean()
    sigma = flat.std()
    arr = [(a - mu)/sigma for a in arr]
    
    # reshape the arrays to square arrays and return the results
    return [numpy.reshape(a, (n, n)) for a in arr]    

def conflatten(arr):
    return numpy.concatenate([a.flatten() for a in arr])

def conflattenDwt(a):
    dw = pywt.dwt2(a, 'db3', 'sym')
    b = conflatten(dw[1])
    return conflatten([dw[0], b])

def dwtFftFeatures(image):
    # find the dimensions of image, which is assumed to be square
    n = image.size[0]

    arr = normalizeImage(image)
    
    # compute the 2d Fourier transform of each channel and flatten into features
    f = conflatten([abs(fft.fft2(a)) / (n * n) for a in arr])
    
    # compute a wavelet transform of the channels
    # flatten the results to get features
    dw = conflatten([conflattenDwt(a) for a in arr])
    
    return numpy.concatenate([dw, f])

def loadImageFeatures(name):
    return dwtFftFeatures(load(name))

def sigmoid(t):
    return 1 / (1 + numpy.exp(-t))

