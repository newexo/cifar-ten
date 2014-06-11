import csv
import os
import numpy

import cifarDirectories
import dataset

def loadTenImages():
    tenImages = os.path.join(cifarDirectories.tests(), 'tenImages.csv')            
    f = open(tenImages, 'r')
    reader = csv.reader(f)
    x = [[int(field) for field in row] for row in reader]
    f.close()
    return x
    
tenImages = loadTenImages()

class TestCifar10Datasets():
    class RawCifar10Data(object):
        def __init__(self):
            x = tenImages
            self.train = numpy.array(x[0:8], dtype=numpy.uint8), range(8)
            self.test = numpy.array([x[8]], dtype=numpy.uint8), range(1)
            self.valid = numpy.array([x[9]], dtype=numpy.uint8), range(1)
    
    def setUp(self):
        self.rawCifar10Data = self.RawCifar10Data()

    def testRawData(self):
        def testDataConsistency(data):
            x, y = data            
            # there should be the same number of datapoints as labels 
            self.assertEqual(len(x), len(y))
            
            # the datapoints represent 32 x 32 RGB images
            self.assertEqual(32 * 32 * 3, len(x[0]))
            self.assertTrue(x.max() <= 255)
            self.assertTrue(x.min() >= 0)
            
            for row in x:
                self.assertEqual(32 * 32 * 3, len(row))
        
        testDataConsistency(self.rawCifar10Data.train)
        testDataConsistency(self.rawCifar10Data.test)
        testDataConsistency(self.rawCifar10Data.valid)
        
        # there should be 8 training examples and one each of testing and validation
        self.assertEqual(8, len(self.rawCifar10Data.train[0]))
        self.assertEqual(1, len(self.rawCifar10Data.test[0]))
        self.assertEqual(1, len(self.rawCifar10Data.valid[0]))
        
        # test a few known entries to make sure we have the dataset we think we should
        self.assertEqual(59, self.rawCifar10Data.train[0][0][0])
        self.assertEqual(177, self.rawCifar10Data.train[0][4][17])
        self.assertEqual(126, self.rawCifar10Data.test[0][0][99])
        self.assertEqual(206, self.rawCifar10Data.valid[0][0][74])
        
    def testNormalizedCifarData(self):
        def testDataConsistency(data):
            x, y = data            
            # there should be the same number of datapoints as labels 
            self.assertEqual(len(x), len(y))
            
            # the datapoints represent 32 x 32 RGB images normalized with values
            # between zero and one
            self.assertEqual(32 * 32 * 3, len(x[0]))
            self.assertTrue(x.max() <= 1.0)
            self.assertTrue(x.min() >= 0.0)
            
            for row in x:
                self.assertEqual(32 * 32 * 3, len(row))
        
        normal = dataset.CifarData(self.rawCifar10Data)
        
        testDataConsistency(normal.train)
        testDataConsistency(normal.test)
        testDataConsistency(normal.valid)
        
        # there should be 8 training examples and one each of testing and validation
        self.assertEqual(8, len(normal.train[0]))
        self.assertEqual(1, len(normal.test[0]))
        self.assertEqual(1, len(normal.valid[0]))
        
        # test a few known entries to make sure we have the dataset we think we should
        self.assertAlmostEqual(59.0 / 255.0, normal.train[0][0][0])
        self.assertAlmostEqual(177.0 / 255.0, normal.train[0][4][17])
        self.assertAlmostEqual(126.0 / 255.0, normal.test[0][0][99])
        self.assertAlmostEqual(206.0 / 255.0, normal.valid[0][0][74])
        
    def testTransformedCifarData(self):
        def testDataConsistency(data):
            x, y = data            
            # there should be the same number of datapoints as labels 
            self.assertEqual(len(x), len(y))
            
            # the datapoints represent 32 x 32 RGB images expanded by two transforms
            self.assertEqual(6960, len(x[0]))
            self.assertTrue(x.max() <= 1.0)
            self.assertTrue(x.min() >= 0.0)
            
            for row in x:
                self.assertEqual(6960, len(row))
        
        transformed = dataset.CifarTransformedData(self.rawCifar10Data)
        
        testDataConsistency(transformed.train)
        testDataConsistency(transformed.test)
        testDataConsistency(transformed.valid)
        
        # there should be 8 training examples and one each of testing and validation
        self.assertEqual(8, len(transformed.train[0]))
        self.assertEqual(1, len(transformed.test[0]))
        self.assertEqual(1, len(transformed.valid[0]))
        
        # test a few known entries to make sure we have the dataset we think we should
        self.assertAlmostEqual(-0.33383461702211, transformed.mu[0])
        self.assertAlmostEqual(1.75921669049030, transformed.sigma[0])
        self.assertAlmostEqual(0.25848758344751, transformed.train[0][0][0])
        
        self.assertAlmostEqual(0.90850525506560, transformed.mu[49])
        self.assertAlmostEqual(1.32527435828925, transformed.sigma[49])
        self.assertAlmostEqual(0.52393654991588, transformed.train[0][3][49])
        
        self.assertAlmostEqual(0.24115763600947, transformed.test[0][0][49])
        self.assertAlmostEqual(0.72794633501417, transformed.valid[0][0][49])
        
        
