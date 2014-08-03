import random
import hyperparameter

class ChromosomeMnist(object):
    def __init__(self, genes):
        self.genes = list(genes)

    def copy(self):
        return ChromosomeMnist(self.genes)

    def computeLearningRate(self, gene):
        return 0.001 * (10 ** (gene * 3.0))

    def computeEpochs(self, gene):
        if gene <= 0.1:
            return 0
        return int(50 * gene)
        
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

    def mutationRate(self):
        return self.genes[0]

    def display(self):
        print self.genes
        print "Mutation rate = %f" % self.mutationRate()
        print "Fine tuning learning rate = %f" % self.learningRate()
        print "Maximum number of Epochs = %d" % self.numberEpochs()
        print "Batch size = %d" % self.batchSize()
        print "Patience = %f" % self.patience()
        print "Patience increase = %f" % self.patienceIncrease()
        print "Improvement threshold = %f" % self.improvementThreshold()
        print "Number of pretraining epochs per layer = %f" % self.pretrainingEpochs()
        print "Pretraining learning rate = %f" % self.pretrainingLearningRate()
        print "k = %f" % self.k()
        print "Size of hidden layers = ", self.nHidden()
        
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

