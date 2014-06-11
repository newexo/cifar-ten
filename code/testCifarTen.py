import unittest
import sys

import cifarDirectories
sys.path.append(cifarDirectories.tests())

import testExample
import testGeneticAlgorithm
import testDirectories
import testIris
import testPreprocess
import testCifar10Datasets

class TestCifarTen(unittest.TestCase, 
    testExample.TestSequenceFunctions,
    testGeneticAlgorithm.TestGeneticAlgorithm,
    testDirectories.TestDirectories,
    testIris.TestIris,
    testPreprocess.TestPreprocess,
    testCifar10Datasets.TestCifar10Datasets):

    def setUp(self):
        testExample.TestSequenceFunctions.setUp(self)
        testGeneticAlgorithm.TestGeneticAlgorithm.setUp(self)
        testDirectories.TestDirectories.setUp(self)
        testIris.TestIris.setUp(self)
        testPreprocess.TestPreprocess.setUp(self)
        testCifar10Datasets.TestCifar10Datasets.setUp(self)

    def testTest(self):
        self.assertEqual(2 + 2, 4)
            
if __name__ == '__main__':
    unittest.main()

