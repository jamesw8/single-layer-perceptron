import random
import numpy as np
# Perceptron class that includes fitting and predicting
class Perceptron:
	def __init__(self):
		# Store weights
		self.weights = []
		# Store answers
		self.answers = []
		# Set learning rate
		self.learning_rate = 0.5

	def fit(self, input_):
		# Initialize weights to random values from [0, 1]
		self.initializeWeights(len(input_[0]['data']),len(input_))
		# Store answers and assign answer arrays that represent them
		self.answers[:] = [item['target'] for item in input_]
		for answer_index in range(len(self.answers)):
			answer_array = [0]*len(self.answers)
			answer_array[answer_index] = 1
			self.answers[answer_index] = (answer_array, self.answers[answer_index])
		output = []
		for i in range(len(input_)):
			input_[i]['data'].append(-1)

		# Epoch loops that breaks early if perceptron learned the training data completely
		epochs = int(input('How many epochs? '))
		for epoch in range(epochs):
			correct = True
			print('\nEpoch', epoch, '\n')
			# Check all input combinations
			for possible_output in range(len(self.answers)):
				local_output = []
				print('Next combination')
				# Check each output node for current input combination
				for output_node in range(len(self.weights)):
					# Compare each output node from current weights with targets and adjust
					target = self.answers[possible_output][0][output_node]
					dot_prod = np.dot(input_[possible_output]['data'], self.weights[output_node])
					threshold = self.weights[output_node][-1]
					print('Inputs', input_[possible_output]['data'])
					print('Dot product', dot_prod, 'vs Threshold', threshold)
					print('Weights', self.weights)
					# If activated
					if dot_prod > 0 and dot_prod > threshold:
						print('Target', target)
						print('Passed threshold')
						# If target is incorrect, adjust weights
						if target != 1:
							correct = False
							self.adjustWeights(input_, output_node, possible_output, error=-1)
						local_output.append(1)
					# If not activated
					else:
						# If target is incorrect, adjust weights
						if target != 0:
							correct = False
							self.adjustWeights(input_, output_node, possible_output, error=1)
						print('Did not pass threshold')
						local_output.append(0)
					print()
				output.append(local_output)
			# End training
			if correct:
				print('Finished in %d epochs'%(epoch+1))
				break
		
		return output

	# Use perceptron to predict an output
	def predict(self, input_):
		# Check if perceptron has been trained
		if len(self.weights) == 0:
			print('Perceptron is not trained')
			return
		# Keep track of output to match with answers
		output_nodes = []
		for output_node in range(len(self.weights)):
			# Calculate dot product of inputs and weights
			dot_prod = np.dot(input_, self.weights[output_node])
			threshold = self.weights[output_node][-1]
			# Check if neuron fires
			if dot_prod > 0 and dot_prod > threshold:
				output_nodes.append(1)
			else:
				output_nodes.append(0)

		# Match answer arrays with the actual answers
		try:
			retval_index = [answer[0] for answer in self.answers].index(output_nodes)
			retval_answers = [self.answers[retval_index][1]]
		except:
			retval_answers = []
			for bit in range(len(output_nodes)):
				if output_nodes[bit] == 1:
					retval_answers.append(self.answers[bit][1])
		return retval_answers

	# Initialize weights with floats between -1 and 1
	def initializeWeights(self, size, output_size):
		self.weights = [[random.random()*random.choice([-1, 1]) for num in range(size + 1)] for output in range(output_size)]

	# Adjust weights if output missed the target
	def adjustWeights(self, input_, output_node, possible_output, error):
		dWeight_without_x = self.learning_rate * error
		# Adjust weights by learning rate and input
		for weight in range(len(self.weights[output_node])):
			old_weight = self.weights[output_node][weight]
			self.weights[output_node][weight] = old_weight + (dWeight_without_x * input_[possible_output]['data'][weight])

	# Getter function for weights
	def getWeights(self):
		return self.weights

	# Getter function for answers
	def getAnswers(self):
		return self.answers