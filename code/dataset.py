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

def xy(batch):
    return batch['data'], batch['labels']

def ng(data):
    x, y = data
    return x / 255.0, y

def cifar10ThreeBatch():
    batch1 = cifar10.batch1()
    batch2 = cifar10.batch2()
    batch3 = cifar10.batch3()
    x = numpy.vstack([batch1['data'], batch2['data'], batch3['data']])
    y = numpy.concatenate([batch1['labels'], batch2['labels'], batch3['labels']])
    return x, y

def mix(data):
    from preprocess import dwtFftFeatures as feat
    from preprocess import reconstruct as recon
    x, y = data
    z = [feat(recon(a, (32, 32))) for a in x]
    return z, y

class Cifar10Part(Dataset):
    def __init__(self):
        train = xy(cifar10.batch1())
        valid = xy(cifar10.batch4())
        test = xy(cifar10.batch5())
        Dataset.__init__(self, ng(train), ng(valid), ng(test))
        
class Cifar10All(Dataset):
    def __init__(self):
        train = cifar10ThreeBatch()
        valid = xy(cifar10.batch4())
        test = xy(cifar10.batch5())
        Dataset.__init__(self, ng(train), ng(valid), ng(test))
        
class Cifar10PartTransform(Dataset):
    def __init__(self):
        train = xy(cifar10.batch1())
        valid = xy(cifar10.batch4())
        test = xy(cifar10.batch5())
        Dataset.__init__(self, mix(train), mix(valid), mix(test))
    
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

