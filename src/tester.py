import numpy as np
def te():
	Wei = np.arange(20)
	Wei = Wei.reshape((2,10))
	K = 2
	print (Wei)
	Y_train = np.array([0, 1, 2, 3, 2, 5, 6, 7, 8, 2])
	print (Y_train)

	categories = np.unique(Y_train) # categories -> [label1, label2, ..., labelk]
	W = np.zeros((K, len(categories))) # W -> [W_vct_label1, W_vct_label2, ..., ]
	for i in range(len(categories)):
		category = categories[i]
		truevals = (Y_train==category)
		occurances = np.sum(truevals) # number of times
		W[:,i] = np.sum(Wei * truevals, axis=1)/occurances

	print (W)
