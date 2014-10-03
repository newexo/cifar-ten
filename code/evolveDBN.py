import chromosome
import NSGAII
import sys

def evolveMnist(seed = 42):
    logger = NSGAII.log_file('mnistgenerations.log')
    pop = chromosome.initialMnistPopulation(40)
    sys.stdout.flush()
    result = NSGAII.nsgaii(pop, 100, logger.log)
    print result

if __name__ == '__main__':
    evolveMnist()

