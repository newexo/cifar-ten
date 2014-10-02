import random
import sys

import cifarDirectories
sys.path.append(cifarDirectories.tests())

from testNsga2 import ChromosomeTestImplementation
import NSGAII

def demonstrate(seed = 42):
    random.seed(seed)
    logger = NSGAII.log_file('demonsga.log')
    pop = [ChromosomeTestImplementation() for i in range(10)]
    return NSGAII.nsgaii(pop, 100, logger.log)
    
if __name__ == '__main__':
    demonstrate()

