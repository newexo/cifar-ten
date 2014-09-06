import chromosome
import NSGAII
import sys

pop = chromosome.initialMnistPopulation(40)
sys.stdout.flush()
result = NSGAII.nsgaii(pop, 100)
print result

