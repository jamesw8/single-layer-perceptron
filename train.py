import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron

# Training sets
words = [ 
	{
		'data':	[	0, 0, 1, 0, 0, 0, 0, 0,
					0, 0, 1, 1, 0, 0, 0, 0,
					0, 0, 0, 1, 1, 0, 0, 0,
					0, 0, 0, 1, 1, 0, 0, 0,
					0, 0, 1, 1, 1, 1, 0, 0,
					0, 1, 1, 0, 1, 1, 1, 0,
					1, 1, 1, 0, 0, 1, 1, 1,
					1, 1, 0, 0, 0, 0, 1, 1
				],
		'target': ['enter, come in(to), join']
	},
	{
		'data': [	0, 0, 1, 1, 1, 1, 0, 0,
					0, 0, 1, 1, 1, 0, 0, 0,
					0, 0, 1, 1, 1, 1, 0, 0,
					0, 0, 1, 1, 1, 1, 0, 0,
					0, 0, 1, 0, 0, 1, 0, 0,
					0, 1, 1, 0, 0, 1, 1, 0,
					0, 1, 1, 0, 0, 1, 1, 1,
					1, 1, 0, 0, 0, 0, 1, 1],
		'target': ['eight']
	},
	{
		'data': [	0, 0, 0, 1, 1, 0, 0, 0,
					0, 0, 0, 1, 1, 0, 0, 0,
					0, 0, 0, 1, 1, 0, 0, 0,
					0, 0, 0, 1, 1, 0, 0, 0,
					0, 0, 1, 1, 1, 1, 0, 0,
					0, 0, 1, 1, 1, 1, 0, 0,
					0, 1, 1, 0, 0, 1, 1, 1,
					1, 1, 0, 0, 0, 0, 1, 1],
		'target': ['Man, people, mankind, someone else']
	},
	{
		'data': [	1, 1, 1, 1, 1, 0, 0, 0,
					1, 1, 0, 1, 1, 0, 0, 0,
					1, 1, 0, 1, 1, 1, 0, 0,
					1, 1, 0, 1, 1, 1, 0, 0,
					1, 1, 0, 1, 0, 1, 1, 0,
					1, 1, 1, 0, 0, 1, 1, 1,
					1, 1, 1, 0, 0, 0, 1, 1,
					1, 1, 1, 1, 1, 1, 1, 1],
		'target': ['destruction']
	},
	{
		'data': [	0, 0, 0, 1, 1, 0, 0, 0,
					0, 0, 0, 1, 1, 0, 0, 0,
					0, 0, 0, 1, 1, 0, 0, 0,
					0, 0, 1, 1, 1, 1, 0, 0,
					0, 0, 1, 1, 1, 1, 1, 0,
					0, 1, 1, 0, 0, 1, 1, 1,
					1, 1, 0, 0, 0, 0, 1, 1,
					1, 1, 1, 1, 1, 1, 1, 1],
		'target': ['to assemble, to gather together']
	},
	{
		'data': [	0, 0, 0, 0, 1, 0, 0, 0,
					1, 1, 0, 1, 1, 0, 0, 0,
					1, 1, 0, 1, 1, 0, 0, 0,
					1, 1, 0, 1, 1, 1, 0, 0,
					1, 1, 0, 1, 1, 1, 0, 0,
					1, 1, 1, 1, 0, 1, 1, 1,
					1, 1, 1, 0, 0, 0, 1, 1,
					1, 1, 1, 1, 1, 1, 1, 1],
		'target': ['perish']
	}
]

# Verify 64 bit vectors
word_train = []
def_train = []
for word in words:
	assert len(word['data']) == 64
	word_train.append(word['data'])
	def_train.append(word['target'])

# Smaller training set for NAND function
nand = [
	{
		'data': [0, 0],
		'target': 1
	},
	{
		'data': [0, 1],
		'target': 1
	},
	{
		'data': [1, 0],
		'target': 1
	},
	{
		'data': [1, 1],
		'target': 0
	},

]



# Display output iterations from fitting
def displayOutput(outputs):
	for output in range(len(outputs)):
		if output%len(train_set) == 0:
			print('\n\n\n','Epoch',output/len(train_set),'\n\n\n')
		print('Output node', output%len(train_set), 'should be', output%len(train_set))
		print(outputs[output])

# Set training set
train_set = words

# Create perceptron and train
clf = Perceptron()
outputs = clf.fit(train_set)

# Display output iterations
if 'y' in input('Display output? ').lower():
	displayOutput(outputs)

print()

# Check accuracy with training set
total_correct = 0
correct_list = []
for item in range(len(train_set)):
	predict = clf.predict(train_set[item]['data'])
	if len(predict) == 1:
		print(predict[0], '==', train_set[item]['target'], '?', end=' ')
		if predict[0] == train_set[item]['target']:
			print('Yes')
			total_correct += 1
			correct_list.append(item)
		else:
			print('No')
	else:
		print('Potential answers', predict)
print('Total Correct: %d out of %d'%(total_correct, len(train_set)))
print('Correct', correct_list)


