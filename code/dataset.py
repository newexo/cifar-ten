import random
import numpy
import csv
import os

import theano
import theano.tensor as T

import cifar10
import mnist
import cifarDirectories

class Dataset(object):
    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test
        self.initN()
        self.initSharedData()

    def initSharedData(self):
        def sharedPair(data):
            def shared(z):
                return theano.shared(numpy.asarray(z, dtype=theano.config.floatX), borrow=True) 
                
            x, y = data
            return shared(x), T.cast(shared(y), 'int32')
            
        self.sharedTrain = sharedPair(self.train)
        self.sharedValid = sharedPair(self.valid)
        self.sharedTest = sharedPair(self.test)
        
    def initN(self):
        self.n_in = len(self.train[0][0])
        self.n_out = max(self.train[1]) + 1

class Cifar10Part(Dataset):
    def __init__(self):
        batch = cifar10.batch1()
        train = batch['data'] / 256.0, batch['labels']
        batch = cifar10.batch2()
        valid = batch['data'] / 256.0, batch['labels']
        batch = cifar10.batch3()
        test = batch['data'] / 256.0, batch['labels']
        Dataset.__init__(self, train, valid, test)

class CifarFeatures(Dataset):
    def __init__(self, cifarDataset):
        def transform(data):
            from preprocess import dwtFftFeatures as feat
            from preprocess import reconstruct as recon
            x, y = data
            z = [feat(recon(a, (32, 32))) for a in x]
            return z, y
            
        train = transform(cifarDataset.train)
        valid = transform(cifarDataset.valid)
        test = transform(cifarDataset.test)
        Dataset.__init__(self, train, valid, test)
    
class Mnist(Dataset):
    def __init__(self):
        train, valid, test = mnist.mnist()
        Dataset.__init__(self, train, valid, test)

class Iris(Dataset):
    def __init__(self):
        def irisType(iris):
            if iris == 'Iris-setosa': return 0
            if iris == 'Iris-versicolor': return 1
            if iris == 'Iris-virginica': return 2
            return 0
        
        maxSepalLength = 7.9
        maxSepalWidth = 4.4
        maxPetalLength = 6.9
        maxPetalWidth = 2.5

        xTrain = []
        yTrain = []
        xTest = []
        yTest = []
        xValid = []
        yValid = []

        randomState = random.getstate()
        random.seed(42)
        irisPath = os.path.join(cifarDirectories.data(), 'iris.data')
        f = open(irisPath, 'r')
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if len(row) > 0:
                sepalLength = float(row[0]) / maxSepalLength
                sepalWidth = float(row[1]) / maxSepalWidth
                petalLength = float(row[2]) / maxPetalLength
                petalWidth = float(row[3]) / maxPetalWidth
                x = [sepalLength, sepalWidth, petalLength, petalWidth]
                y = irisType(row[4])
                r = random.random()
                if r < 0.8:
                    xTrain.append(x)
                    yTrain.append(y)
                elif r < 0.9:
                    xTest.append(x)
                    yTest.append(y)
                else:
                    xValid.append(x)
                    yValid.append(y)
        f.close()
        random.setstate(randomState)
        
        train = numpy.array(xTrain, dtype=numpy.float32), numpy.array(yTrain)
        test = numpy.array(xTest, dtype=numpy.float32), numpy.array(yTest)
        valid = numpy.array(xValid, dtype=numpy.float32), numpy.array(yValid)
        Dataset.__init__(self, train, valid, test)

