def get_objectives(hyperparameters):
	#this is a stand in, should either run a DBN with the given hyperparameters or estimate the outcome
	
	x = hyperparameters[0]
	y = hyperparameters[1]
	return [(x + y - 10) ** 2, (x + y - 20) ** 2]