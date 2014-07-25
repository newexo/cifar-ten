from dataset2 import Mnist, Dataset
from hyperparameter import HyperparametersDBN
from DBNClassifier import test_DBN, interpretObjectives

def get_objectives(hyperparameters):
#takes a list of 8 hyperparameters and returns the four objectives: pretraining time, post-pretrain error, finetuning time, test score
	hyper = HyperparametersDBN(learningRate=hyperparameters[1],
                               numberEpochs=hyperparameters[2],
                               pretrainingEpochs=hyperparameters[3],
                               pretrainingLearningRate=hyperparameters[4],
                               nHidden=[hyperparameters[5], hyperparameters[6], hyperparameters[7]])
	objectives = test_DBN(Mnist(), hyper)
	return [interpretObjectives(objectives)]
	
get_objectives([0.5, 0.1, 4, 1, 0.1, 3000, 3000, 3000])