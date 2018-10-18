'''
Author:
Apoorv Agnihotri
16110020

This class tries to implement the tutorial
linked here.
https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#preparing-the-data
'''
import numpy as np
from numpy.linalg import norm 
from scipy.linalg import eigh
from sklearn.preprocessing import normalize
import sys

'''
params:
	K:
	Dimension of the face space, should be less
	than no of test images
'''
class EigenFaces(object):
	def __init__(self, K=10, debug=False):
		self.K = K
		self.debug=debug # for debugging purposes

	'''
	brief:
		This method trains the class and
		saves the faces provided to it 
		for future recognition.
		
	params:
		X_train: numpy array of 2d array(images) (greyscale only)
		Y_train: numpy array of label
	'''
	def train(self, X_train, Y_train):
		K = self.K
		M = len(X_train)

		# assume same height and width
		height, width = X_train[0].shape
		r = np.zeros((height*width, M))

		# reshaping into vectors, col vectors in flatten
		for i in range(M):
			r[:, i] = X_train[i].flatten()
		Nsq = r.shape[0] # length of one vec
		assert height*width == Nsq, "bad length of img vctr"

		# find the mean image
		self.avgImg = r.mean(1).reshape((Nsq,1)) # avg across rows

		# find the offsets
		A = r - self.avgImg
		L = np.matmul(A.T, A) # MxM cov. matrix
		vals, vi = eigh(L) # since L is PSD

		# we want bigger values first
		vals = np.flip(vals)
		vi = np.flip(vi, axis=1)

		# keeping only top K eigenvectors
		vals = vals[:K]
		vi = vi[:, :K]

		# getting ui(eigenfaces) and corresp uvals
		uvals = vals
		ui = np.matmul(A, vi)
		del vals, vi

		# normalize
		ui = normalize(ui, axis=0, norm='l2')

		# will store weigth vectors as cols
		Wei = np.matmul(ui.T, A) # KxM Matrix

		# finding avg weight vector for each class
		categories = np.unique(Y_train) # categories -> [label1, label2, ..., labelk]
		W = np.zeros((K, len(categories))) # W -> [W_vct_label1, W_vct_label2, ..., ]
		for i in range(len(categories)):
			category = categories[i]
			truevals = (Y_train==category)
			occurances = np.sum(truevals) # number of times
			W[:,i] = np.sum(Wei * truevals, axis=1)/occurances

		# saving into class variables
		self.categories = categories
		self.uvals = uvals
		self.ui = ui
		self.W = W

	'''
	params:
		@param te
		 euclidian threshold used to classify 
		 images as unkown-face
		@param tf
		 euclidian threshold used to classify 
		 images as non-face
		@param unknownface
		 index to return if unkown-face detected
		@param nonface
		 index to return if non-face detected
	'''
	def predict(self, X, te, tf, unknownface=-1, nonface=-2):
		categories = self.categories
		K = self.K
		W = self.W
		ui = self.ui
		y = []

		for x in X:
			# finding offset w.r.t. meanface
			h,w = x.shape
			Nsq = h*w
			x = x.reshape((Nsq, 1))
			offset = x - self.avgImg

			# finding the weight vector of test img
			weiVct = np.matmul(ui.T, offset)
			assert weiVct.shape[0]==K and weiVct.shape[1]==1, \
						'Weight for test images is bad shape'

			# comparing with other class representatives
			temp = norm((W - weiVct), axis=0) # euclidian norms
			matched_category = categories[np.argmin(temp)] # closest match
			min_dist_bw_faces = np.min(temp)

			# calculating the distance from facespce
			recon = np.matmul(ui, weiVct)
			min_dist_bw_faceSpace = np.min(norm(x - recon, axis=0))

			if min_dist_bw_faceSpace <= tf and min_dist_bw_faces <= te:
				y.append(matched_category)
			elif min_dist_bw_faceSpace <= tf and min_dist_bw_faces > te:
				y.append(unknownface)
				if self.debug:
					print('dist_bw_faces', min_dist_bw_faces)
			else: # min_dist_bw_faceSpace > tf
				y.append(nonface)
				if self.debug:
					print('dist_bw_faceSpace', min_dist_bw_faceSpace)

		return np.array(y)

# find the nearest neigbour of the with the given weights
# test img provided