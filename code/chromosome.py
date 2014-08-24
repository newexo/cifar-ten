import random
import hyperparameter
import DBNClassifier

class Chromosome(object):
    def __init__(self, genes = []):
        if len(genes) == self.numberOfGenes():
            self.genes = list(genes)
        else:
            self.genes = [random.random() for i in range(self.numberOfGenes())]

    def copy(self):
        return ChromosomeMnist(self.genes)

    def mutationRate(self):
        return self.genes[0]
        
    def mutate(self):
        for i in range(len(self.genes)):
            if random.random() < self.mutationRate():
                self.genes[i] = random.random()

    def crossover(self, other):
        child0 = self.copy()
        child1 = other.copy()
        for i in range(len(child0.genes)):
		if random.random() < 0.5:
			child0.genes[i] = other.genes[i]
			child1.genes[i] = self.genes[i]
        return child0, child1
    
class ChromosomeMnist(Chromosome):
    def __init__(self, data, genes = []):
        self.data = data
        Chromosome.__init__(self, genes)

    def numberOfGenes(self):
        return 7

    def copy(self):
        return ChromosomeMnist(self.data, self.genes)

    def computeLearningRate(self, gene):
        return 0.001 * (10 ** (gene * 3.0))

    def computeEpochs(self, gene):
        return max(int(50 * gene), 1)
        
    def computeLayersize(self, gene):
        return 100 + int(5000 * gene)

    def computeRegularization(self, gene):
        if gene <= 0.1:
            return 0.0
        return 0.001 * (10 ** (gene * 3.0))

    def learningRate(self):
        return self.computeLearningRate(self.genes[1])
        
    def numberEpochs(self):
        return self.computeEpochs(self.genes[2])
        
    def batchSize(self):
        return 100

    def patience(self):
        return 5000
        
    def patienceIncrease(self):
        return 2
         
    def improvementThreshold(self):
        return 0.995
        
    def pretrainingEpochs(self):
        return self.computeEpochs(self.genes[3])
        
    def pretrainingLearningRate(self):
        return self.computeLearningRate(self.genes[4])
        
    def k(self):
        return 1
    
    def nHidden(self):
        return [self.computeLayersize(self.genes[5]), self.computeLayersize(self.genes[6])]
    
    def hyper(self):
        return hyperparameter.HyperparametersDBN(
            learningRate = self.learningRate(),
            numberEpochs = self.numberEpochs(),
            batchSize = self.batchSize(),
            patience = self.patience(),
            patienceIncrease = self.patienceIncrease(),
            improvementThreshold = self.improvementThreshold(),
            pretrainingEpochs = self.pretrainingEpochs(),
            pretrainingLearningRate = self.pretrainingLearningRate(),
            k = self.k(),
            nHidden = self.nHidden())

    def getObjectives(self):
	    objectives = DBNClassifier.test_DBN(self.data, self.hyper())
	    return DBNClassifier.interpretObjectives(objectives)

    def display(self):
        print self.genes
        print "Mutation rate = %f" % self.mutationRate()
        print "Pretraining learning rate = %f" % self.pretrainingLearningRate()
        print "Maximum number of pretraining epochs per layer = %d" % self.pretrainingEpochs()
        print "Fine tuning learning rate = %f" % self.learningRate()
        print "Maximum number of fine tuning epochs = %d" % self.numberEpochs()
        print "Batch size = %d" % self.batchSize()
        print "Patience = %f" % self.patience()
        print "Patience increase = %f" % self.patienceIncrease()
        print "Improvement threshold = %f" % self.improvementThreshold()
        print "k = %d" % self.k()
        print "Size of hidden layers = ", self.nHidden()
    
