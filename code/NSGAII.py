import random

def domination_count_and_set(population, element):
	# gives the domination count and dominating set of the element of the population
	
	domination_count = 0
	dominated_set = [] 
	for chromosome in population:
		dominated_sum = 0 # will equal number of objectives if chromosome is dominated by element
		dominating_sum = 0 # will equal number of objectives if element is dominated by chromosome
		for i in range(len(population[element])):
			if population[element][i] < population[chromosome][i]:
				dominated_sum += 1
			elif population[element][i] == population[chromosome][i]:
				dominated_sum += 1
				dominating_sum += 1
			else:
				dominating_sum += 1
		if dominated_sum == len(population[element]) and population[element] != population[chromosome]:
			dominated_set.append(chromosome)
		if dominating_sum == len(population[element]) and population[element] != population[chromosome]:
			domination_count += 1
	return [domination_count, dominated_set]
# what should we do if two chromosomes have the exact some objectives? here neither dominates the other	

def fast_nondominated_sort(population):
	# gives the list of the domination fronts, in order, and a dictionary with the ranks of the chromosomes

	domination_dict = {}
	for chromosome in population:
		domination_dict[chromosome] = domination_count_and_set(population, chromosome)
	i = 0
	domination_fronts = []
	domination_fronts.append([])
	rank_list = {}
	for chromosome in domination_dict:
		if domination_dict[chromosome][0] == 0:
			domination_fronts[0].append(chromosome)
			rank_list[chromosome] = 1
	while domination_fronts[i] != []:
		domination_fronts.append([])
		for chromosome in domination_fronts[i]:
			for chrom in domination_dict[chromosome][1]:
				domination_dict[chrom][0] -= 1
				if domination_dict[chrom][0] == 0:
					domination_fronts[i + 1].append(chrom)
					rank_list[chrom] = i + 2
		i += 1
	return [rank_list, domination_fronts[:len(domination_fronts) - 1]]
	
def crowding_distance(population):
# gives a dictionary in which the value of each chromosome is its crowding distance

	l = len(population)
	num_objectives = len(population.values()[0])
	crowding_distances = {}
	for chromosome in population:
		crowding_distances[chromosome] = 0
	for i in range(num_objectives):
		ordered_by_objective = sorted(population, key = lambda x:population[x][i])
		if population[ordered_by_objective[0]] == population[ordered_by_objective[l-1]]:
			for j in range(1, l - 1):
				crowding_distances[ordered_by_objective[j]] -= 1
		else:
			crowding_distances[ordered_by_objective[0]] -= float("inf") 
			crowding_distances[ordered_by_objective[l - 1]] -= float("inf")
			for j in range(1, l - 1):
				crowding_distance = (float(population[ordered_by_objective[j + 1]][i]) - float(population[ordered_by_objective[j - 1]][i]))/(float(population[ordered_by_objective[l-1]][i]) - float(population[ordered_by_objective[0]][i]))
				crowding_distances[ordered_by_objective[j]] -= crowding_distance
	return crowding_distances
	
def new_population(parents_hyperparameters, parents_objectives, children_hyperparameters, children_objectives):
	# combines the two new populations and then selects the best chromosome with respect to the crowded comparison order
	
	combined_population_hyperparameters = dict(parents_hyperparameters.items() + children_hyperparameters.items())
	combined_population_objectives = dict(parents_objectives.items() + children_objectives.items())
	fns = fast_nondominated_sort(combined_population_objectives)
	fronts = fns[1]
	if len(fronts) == 1:
		return "stop"
	else:
		new_pop_hyperparameters = {}
		new_pop_objectives = {}
		spaces_remaining = len(parents_hyperparameters)
		front = 0
		while spaces_remaining > 0:
			if spaces_remaining >= len(fronts[front]):
				for chromosome in fronts[front]:
					new_pop_objectives[chromosome] = combined_population_objectives[chromosome]
					new_pop_hyperparameters[chromosome] = combined_population_hyperparameters[chromosome]
					spaces_remaining -= 1
				front += 1
			else:
				crowding_distances = crowding_distance(combined_population_objectives)
				front_with_crowding = dict((key, crowding_distances[key]) for key in fronts[front])
				sorted_by_crowding = sorted(front_with_crowding, key = lambda x:front_with_crowding[x])
				for i in range(spaces_remaining):
					new_pop_objectives[sorted_by_crowding[i]] = combined_population_objectives[sorted_by_crowding[i]]
					new_pop_hyperparameters[sorted_by_crowding[i]] = combined_population_hyperparameters[sorted_by_crowding[i]]
				spaces_remaining = 0
		return [new_pop_hyperparameters, new_pop_objectives]	
		
def crossover(chromosome0, chromosome1):
	# creates two children that are a mix of their parents
	new_chromosome0 = [0] * len(chromosome0)
	new_chromosome1 = [0] * len(chromosome0)
	for i in range(len(chromosome0)):
		if random.random() > 0.5:
			new_chromosome0[i] = chromosome1[i]
			new_chromosome1[i] = chromosome0[i]
		else:
			new_chromosome0[i] = chromosome0[i]
			new_chromosome1[i] = chromosome1[i]
	return[new_chromosome0, new_chromosome1]
	
def make_children(get_objectives, parents_hyperparementers, parents_objectives, generation):
	# creates a generation of offspring using the genetic algorithm 
	
	children_hyperparameters = {}
	children_objectives = {}
	list_of_chromosomes = list(parents_objectives.keys())
	while len(list_of_chromosomes) > 1:
		chromosome1 = random.choice(list_of_chromosomes)
		list_of_chromosomes.remove(chromosome1)
		chromosome2 = random.choice(list_of_chromosomes)
		list_of_chromosomes.remove(chromosome2)
		children = crossover(parents_hyperparementers[chromosome1], parents_hyperparementers[chromosome2])
		# should add mutation here
		new_key1 = chromosome1 + "-%s" % (generation + 1)
		new_key2 = chromosome2 + "-%s" % (generation + 1)
		children_hyperparameters[new_key1] = children[0]
		children_hyperparameters[new_key2] = children[1]
		children_objectives[new_key1] = get_objectives(children_hyperparameters[new_key1])
		children_objectives[new_key2] = get_objectives(children_hyperparameters[new_key2])
	return [children_hyperparameters, children_objectives]
	
def nsgaii(get_objectives, mutate, initial_population_hyperparameters):
	# implement NSGA-II
	
	population_hyperparameters = initial_population_hyperparameters
	population_objectives = {}
	for chromosome in population_hyperparameters:
		population_objectives[chromosome] = get_objectives(population_hyperparameters[chromosome])
	population = [population_hyperparameters, population_objectives]
	num_generations = 100
	generation = 0
	while generation < num_generations: 
		children = make_children(get_objectives, population_hyperparameters, population_objectives, generation)
		population = new_population(population_hyperparameters, population_objectives, children[0], children[1])
		if population == "stop":
			break
		else:
			population_hyperparameters = population[0]
			population_objectives = population[1]
			generation += 1
	return population_hyperparameters.values()
	
