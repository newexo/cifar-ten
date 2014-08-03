import random

def mutate(chromosome):
	mutation_rate = chromosome(0)
	new_chromosome = []
	for i in range(len(chromosome)):
		if random.random() < mutation_rate:
			new_chromosome.append(random.random())
		else:
			new_chromosome.append(chromosome(i))
	return new_chromososome

