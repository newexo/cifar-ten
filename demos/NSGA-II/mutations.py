def learning_rate_mutation(learning_rate):
	mutation = 10 ** (0.5 * random.random() - 0.25)
	learning_rate = learning_rate * mutation

def epochs_mutation(num_epochs):
	if random.random() < 0.5:
		num_epochs += 1
	else: 
		num_epochs -= 1
		
def layer_size_mutation(layer_size):
	layer_size = int(abs(layer_size + random.normalvariate(0.5, 200)))