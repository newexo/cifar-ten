#we assume that the population is a dictionary whose keys are the members of the population of potential solutions, with the value of each key containing a list of the objective functions evaluated at that key

def domination_count_and_set(population, element):
	#gives the domination count and dominating set of the element of the population
	domination_count = 0
	dominated_set = []
	for chromisome in population:
		sum = 0
		for i in range(len(population[element])):
			if population[element][i] <= population[chromisome][i]:
				sum += 1
		if sum == len(population[element]) and element != chromisome:
			dominated_set.append(chromisome)
		elif sum == 0 and element != chromisome:
			domination_count += 1
	return [domination_count, dominated_set]
	
def fast_nondominated_sort(population):
	#gives the list of the domination fronts, in order, and a dictionary with the ranks of the chromisomes
	domination_dict = {}
	for chromisome in population:
		domination_dict[chromisome] = domination_count_and_set(population, chromisome)
	i = 0
	domination_fronts = []
	domination_fronts.append([])
	rank_list = {}
	for chromisome in domination_dict:
		if domination_dict[chromisome][0] == 0:
			domination_fronts[0].append(chromisome)
			rank_list[chromisome] = 1
	while domination_fronts[i] != []:
		domination_fronts.append([])
		for chromisome in domination_fronts[i]:
			for chrom in domination_dict[chromisome][1]:
				domination_dict[chrom][0] -= 1
				if domination_dict[chrom][0] == 0:
					domination_fronts[i + 1].append(chrom)
					rank_list[chrom] = i + 2
		i += 1
	return [rank_list, domination_fronts]
	
def crowding_distance(population):
#gives a dictionary in which the value of each chromisome is its  crowding distance
	l = len(population)
	num_objectives = len(population.values()[0])
	crowding_distances = {}
	for chromisome in population:
		crowding_distances[chromisome] = 0
	for i in range(num_objectives):
		ordered_by_objective = sorted(population, key = lambda x:population[x][i])
		crowding_distances[ordered_by_objective[0]] += float("inf") # would num_objectives be better?
		crowding_distances[ordered_by_objective[l - 1]] += float("inf")
		for j in range(1, l - 1):
			crowding_distance = (population[ordered_by_objective[j + 1]][i] - population[ordered_by_objective[j - 1]][i])/(population[ordered_by_objective[l-1]][i] - population[ordered_by_objective[0]][i])
			crowding_distances[ordered_by_objective[j]] += crowding_distance
	return crowding_distances
	

def new_population(parents, children):
#combines the two new populations and then selects the best chromisomes with respect to the crowded comparison order
	combined_population = dict(parents.items() + children.items())
	fns = fast_nondominated_sort(combined_population)
	fronts = fns[1]
	new_pop = {}
	spaces_remaining = len(parents)
	front = 0
	while spaces_remaining > 0:
		if spaces_remaining >= len(fronts[front]):
			for chromisome in fronts[front]:
				new_pop[chromisome] = combined_population[chromisome]
				spaces_remaining -= 1
			front += 1
		else:
			front_dict = dict((key, combined_population[key]) for key in fronts[front])
			crowding_distances = crowding_distance(front_dict)
			sorted_by_crowding = sorted(crowding_distances, key = lambda x:crowding_distances[x])
			for i in range(spaces_remaining):
				new_pop[sorted_by_crowding[i]] = combined_population[sorted_by_crowding[i]]
			spaces_remaining = 0
	return new_pop	


# def crowded_comparison(population, chrom1, chrom2):
# # returns whichever chromisome is lower according to the operator; returns nothing if they are tied
# 	fronts = fast_nondominated_sort(population)[0]
# 	rank1 = fronts[chrom1]
# 	rank2 = fronts[chrom2]
# 	if rank1 < rank2:
# 		return chrom1
# 	elif rank2 < rank1:
# 		return chrom2
# 	else:
# 		crowding_distance = crowding_distance(population)
# 		crowding_distance_1 = crowding_distance[chrom1]
# 		crowding_distance_2 = crowding_distance[chrom2]
# 		if crowding_distance_1 < crowding_distance_2:
# 			return chrom2
# 		elif crowding_distance_2 < crowding_distance_1:
# 			return chrom1