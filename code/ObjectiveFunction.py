from dataset import Mnist, Dataset
from hyperparameter import HyperparametersDBN
from DBNClassifier import test_DBN, interpretObjectives

def get_objectives(chromosome):
# takes a list of 8 hyperparameters and returns the four objectives: pretraining time, post-pretrain error, finetuning time, test score
	objectives = test_DBN(Mnist(), chromosome.hyper())
	return [interpretObjectives(objectives)]

