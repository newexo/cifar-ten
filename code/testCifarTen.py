import unittest
import sys

import cifarDirectories
sys.path.append(cifarDirectories.tests())

import testExample
import testDirectories
import testIris
import testPreprocess
import testCifar10Datasets
import testNsga2
import testChromosome

class TestCifarTen(unittest.TestCase, 
    testExample.TestSequenceFunctions,
    testDirectories.TestDirectories,
    testIris.TestIris,
    testPreprocess.TestPreprocess,
    testCifar10Datasets.TestCifar10Datasets,
    testNsga2.TestNSGA2,
    testChromosome.TestChromosome):

    def setUp(self):
        testExample.TestSequenceFunctions.setUp(self)
        testDirectories.TestDirectories.setUp(self)
        testIris.TestIris.setUp(self)
        testPreprocess.TestPreprocess.setUp(self)
        testCifar10Datasets.TestCifar10Datasets.setUp(self)
        testNsga2.TestNSGA2.setUp(self)
        testChromosome.TestChromosome.setUp(self)

    def tearDown(self):
        testNsga2.TestNSGA2.tearDown(self)
              

    def testTest(self):
        self.assertEqual(2 + 2, 4)
            
if __name__ == '__main__':
    unittest.main()

