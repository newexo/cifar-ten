import numpy
from preprocess import *
from PIL import Image

def rotateFlip(img, n):
    if n > 3:
        n = n % 4
        img = img.flipHorizontal()
        
    if n == 0:
        return img
     
	return img.rotate(45 * n)

def eightByEight(img, n):
    def box(n):
        i = (n & 3) * 8
        j = (n & 12) * 2
        return i, j, i + 8, j + 8

    return img.crop(box(n))

def datasetEightByEight(dataset):
    imgsize = (32, 32)
    def transformBatch(batch):
        def normalizeImage(img):
            return numpy.array(image2narray(eightByEight(img, i)), dtype=numpy.float32)
        def transformImage(arr):
            img = reconstruct(arr, imgsize)
            return [image2narray(eightByEight(img, i)) for i in range(0, 16)]
        data = []
        for arr in batch[0]:
            data += transformImage(arr)
        labels = [i % 16 for i in range(0, len(data))]
        return [numpy.array(data), numpy.array(labels)]
    dataset.test = transformBatch(dataset.test)
    dataset.train = transformBatch(dataset.train)
    dataset.valid = transformBatch(dataset.valid)
    
