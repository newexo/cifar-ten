import random
import numpy
import csv
import os

import theano
import theano.tensor as T

import cifar10
import mnist
import cifarDirectories
import preprocess

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

def xy(batch):
    return batch['data'], batch['labels']

class Cifar10PartRaw(object):
    def __init__(self):
        self.train = xy(cifar10.batch1())
        self.valid = xy(cifar10.batch4())
        self.test = xy(cifar10.batch5())

class Cifar10AllRaw(Dataset):
    def __init__(self):
        batch1 = cifar10.batch1()
        batch2 = cifar10.batch2()
        batch3 = cifar10.batch3()
        x = numpy.vstack([batch1['data'], batch2['data'], batch3['data']])
        y = numpy.concatenate([batch1['labels'], batch2['labels'], batch3['labels']])
        self.train = x, y
        self.valid = xy(cifar10.batch4())
        self.test = xy(cifar10.batch5())
        
class CifarData(Dataset):
    def __init__(self, raw):
        def ng(data):
            x, y = data
            return numpy.array(x / 255.0, dtype = numpy.float32), y

        Dataset.__init__(self, ng(raw.train), ng(raw.valid), ng(raw.test))
        
class CifarTransformedData(Dataset):
    def __init__(self, raw):
        train = self.mix(raw.train)
        test = self.mix(raw.test)
        valid = self.mix(raw.valid)

        self.mu, self.sigma = preprocess.muSigma(train[0])

        Dataset.__init__(self, self.normalize(train), 
                            self.normalize(valid),
                            self.normalize(test))

    def mix(self, data):
        def mixRow(row):
            return preprocess.dwtFftFeatures(preprocess.reconstruct(row, (32, 32)))
                
        x, y = data
        z = [mixRow(row) for row in x]
        return z, y

    def normalize(self, data):
        x, y = data
        z = preprocess.sigmoid(preprocess.normalize(x, self.mu, self.sigma))
        return numpy.array(z, dtype = numpy.float32), y        
    
class Mnist(Dataset):
    def __init__(self):
        train, valid, test = mnist.mnist()
        Dataset.__init__(self, train, valid, test)

class Iris(Dataset):
    def __init__(self):
        class Subdata(object):
            def __init__(self):
                self.x = []
                self.y = []
                
            def append(self, x, y):
                self.x.append(x)
                self.y.append(y)
                
            def array(self):
                return numpy.array(self.x, dtype=numpy.float32), numpy.array(self.y)

        def irisType(iris):
            if iris == 'Iris-setosa': return 0
            if iris == 'Iris-versicolor': return 1
            if iris == 'Iris-virginica': return 2
            return 0
        
        maxSepalLength = 7.9
        maxSepalWidth = 4.4
        maxPetalLength = 6.9
        maxPetalWidth = 2.5

        train = Subdata()
        test = Subdata()
        valid = Subdata()

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
                    train.append(x, y)
                elif r < 0.9:
                    test.append(x, y)
                else:
                    valid.append(x, y)
        f.close()
        random.setstate(randomState)
        
        Dataset.__init__(self, train.array(), valid.array(), test.array())

