from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

words = [ ([0, 0, 1, 0, 0, 0, 0, 0,
0, 0, 1, 1, 0, 0, 0, 0,
0, 0, 0, 1, 1, 0, 0, 0,
0, 0, 0, 1, 1, 0, 0, 0,
0, 0, 1, 1, 1, 1, 0, 0,
0, 1, 1, 0, 1, 1, 1, 0,
1, 1, 1, 0, 0, 1, 1, 1,
1, 1, 0, 0, 0, 0, 1, 1], ['enter, come in(to), join']),
	([0, 0, 1, 1, 1, 1, 0, 0,
		0, 0, 1, 1, 1, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 0, 0,
		0, 0, 1, 1, 1, 1, 0, 0,
		0, 0, 1, 0, 0, 1, 0, 0,
		0, 1, 1, 0, 0, 1, 1, 0,
		0, 1, 1, 0, 0, 1, 1, 1,
		1, 1, 0, 0, 0, 0, 1, 1], ['eight']),
	([0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 0, 0,
		0, 0, 1, 1, 1, 1, 0, 0,
		0, 1, 1, 0, 0, 1, 1, 1,
		1, 1, 0, 0, 0, 0, 1, 1], ['Man, people, mankind, someone else']),
	([1, 1, 1, 1, 1, 0, 0, 0,
		1, 1, 0, 1, 1, 0, 0, 0,
		1, 1, 0, 1, 1, 1, 0, 0,
		1, 1, 0, 1, 1, 1, 0, 0,
		1, 1, 1, 1, 0, 1, 1, 0,
		1, 1, 1, 0, 0, 1, 1, 1,
		1, 1, 1, 0, 0, 0, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1], ['destruction']),
	([0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 0, 0,
		0, 0, 1, 1, 1, 1, 1, 0,
		0, 1, 1, 0, 0, 1, 1, 1,
		1, 1, 0, 0, 0, 0, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1], ['to assemble, to gather together']),
	([0, 0, 0, 0, 1, 0, 0, 0,
	1, 1, 0, 1, 1, 0, 0, 0,
	1, 1, 0, 1, 1, 0, 0, 0,
	1, 1, 0, 1, 1, 1, 0, 0,
	1, 1, 0, 1, 1, 1, 0, 0,
	1, 1, 1, 1, 0, 1, 1, 1,
	1, 1, 1, 0, 0, 0, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1], ['perish'])]

clf = MLPClassifier()

# Verify 64 bit vectors
word_train = []
def_train = []
for word in words:
	assert len(word[0]) == 64
	word_train.append(word[0])
	def_train.append(word[1][0])

clf.fit(word_train, def_train)

print(type(clf.coefs_))
for coef in clf.coefs_:
	print(coef.shape)

import random
total_correct = 0
correct_list = []
iterations = int(input('How many testing iterations? '))
for i in range(iterations):
	correct = 0
	for word in words:
		# Add random number of marks on characters to make test data harder
		test = word[0]
		rand_marks = random.randrange(2)
		for mark in range(rand_marks):
			r_index = random.randrange(0, 64)
			test[r_index] = 0 if word[0][r_index] == 1 else 1
		print(rand_marks, 'random marks made')
		# Test prediction
		prediction = clf.predict(np.array(test).reshape(1, -1))[0]
		if word[1][0] == prediction:
			correct += 1
		print(word[1][0], '==', prediction, '?', word[1][0] == prediction)
	print('***', correct, 'out of', len(words), 'predictions correct', '***')
	total_correct += correct
	correct_list.append(correct)
print(total_correct, 'out of', iterations*len(words), 'total predictions correct')

plt.plot(list(range(iterations)), correct_list)
plt.show()
# for i in range(1000):
# 	print(clf.predict(np.array(words[3][0]).reshape(1, -1))[0] == 'destruction')