import random
import chromosome

class ChromosomeTestImplementation(chromosome.Chromosome):
    def __init__(self, genes):
        chromosome.Chromosome.__init__(self, genes)

    def numberOfGenes(self):
        return 32

class TestChromosome():

    def setUp(self):
        self.chrome = chromosome.ChromosomeMnist([], [0.2, 0.4, 0.5, 0.5, 0.9, 0.8, 0.3])
        self.zeros = ChromosomeTestImplementation([0] * 32)
        self.ones = ChromosomeTestImplementation([1] * 32)

    def testLearningRate(self):
        self.assertAlmostEqual(0.001, self.chrome.computeLearningRate(0))
        self.assertAlmostEqual(0.01, self.chrome.computeLearningRate(1.0 / 3.0))
        self.assertAlmostEqual(0.1, self.chrome.computeLearningRate(2.0 / 3.0))
        self.assertAlmostEqual(1.0, self.chrome.computeLearningRate(1))

    def testEpochs(self):
        self.assertEqual(1, self.chrome.computeEpochs(0))
        # there is a 10% chance of zero epochs
        self.assertEqual(5, self.chrome.computeEpochs(0.1))
        self.assertEqual(25, self.chrome.computeEpochs(0.5))
        self.assertEqual(50, self.chrome.computeEpochs(1))

    def testLayersize(self):
        self.assertEqual(100, self.chrome.computeLayersize(0))
        self.assertEqual(2600, self.chrome.computeLayersize(0.5))
        self.assertEqual(5100, self.chrome.computeLayersize(1))

    def testRegularization(self):
        self.assertEqual(0.0, self.chrome.computeRegularization(0))
        # there is a 10% chance of a zero regularization factor
        self.assertEqual(0.0, self.chrome.computeRegularization(0.1))
        self.assertAlmostEqual(0.01, self.chrome.computeRegularization(1.0 / 3.0))
        self.assertAlmostEqual(0.1, self.chrome.computeRegularization(2.0 / 3.0))
        self.assertAlmostEqual(1.0, self.chrome.computeRegularization(1))

    def testCopy(self):
        oldvalue = self.chrome.genes[0]
        cp = self.chrome.copy()
        self.assertEqual(self.chrome.genes, cp.genes)
        cp.genes[0] += 0.1
        self.assertNotEqual(self.chrome.genes, cp.genes)
        self.assertEqual(oldvalue, self.chrome.genes[0])
        
    def testMutate(self):
        chrome = self.chrome.copy()
        chrome.mutate()
        # just testing that the function does not crash anything
        
    def testCrossover(self):
        parent0 = chromosome.ChromosomeMnist([], [0, 1, 2, 3, 4, 5, 6])
        parent1 = chromosome.ChromosomeMnist([], [7, 8, 9, 10, 11, 12, 13])
        child0, child1 = parent0.crossover(parent1)
        
        # Chromosomes together should still contain all of the original genes.
        allgenes = child0.genes + child1.genes
        allgenes.sort()
        self.assertEqual(range(14), allgenes)

        # There is only 1 in 2^32 chance that two chromosomes of length 32 will 
        # remain changed after crossover. Consider that probability zero.
        child2, child3 = self.zeros.crossover(self.ones)
        self.assertNotEqual(child2.genes, [0] * 32)

    def testHyperparameters(self):
        hyper = self.chrome.hyper()
        self.assertEqual(hyper.learningRate, self.chrome.learningRate())
        self.assertEqual(hyper.numberEpochs, self.chrome.numberEpochs())
        self.assertEqual(hyper.batchSize, self.chrome.batchSize())
        self.assertEqual(hyper.patience, self.chrome.patience())
        self.assertEqual(hyper.patienceIncrease, self.chrome.patienceIncrease())
        self.assertEqual(hyper.improvementThreshold, self.chrome.improvementThreshold())
        self.assertEqual(hyper.pretrainingEpochs, self.chrome.pretrainingEpochs())
        self.assertEqual(hyper.pretrainingLearningRate, self.chrome.pretrainingLearningRate())
        self.assertEqual(hyper.k, self.chrome.k())
        self.assertEqual(hyper.nHidden, self.chrome.nHidden())
        
