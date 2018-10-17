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
import sys

'''
params:
	K:
	Dimension of the face space, should be less
	than no of test images
'''
class EigenFaces(object):
	def __init__(self, K=10):
		self.K = K

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

		# getting ui and corresp uvals
		uvals = vals
		ui = np.matmul(A, vi)

		assert (ui[:,0] == np.matmul(A,vi[:,0])).all(), 'not equal' #random check :P

		del vals, vi
		# print ('final eigs', ui)
		# print ('final eigvals', uvals)
		# normlize
		for i in range(K):
			ui[:,i] = ui[:,i] / norm(ui[:,i])
		# print ('normalized eigs', ui)
		# we got the top K ui eigenvectors.

		# will store weigth vectors as cols
		Wei = np.zeros((K, M))
		for i in range(M):
			im = A[:, i]
			for j in range(K):
				Wei[j,i] = np.matmul(im.T, ui[:, j])
		# print(Wei)

		# using the weigths of only a 
		# representative member of a class (can 
		# also use avg of all members of a class, but 
		# following this convention)
		self.uvals = uvals
		self.ui = ui
		self.Wei = Wei
		u, self.indices = np.unique(Y_train, return_index=True)
		# print (self.indices)

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
	def predict(self, X, te=2, tf=100000, unknownface=-1, nonface=-2):
		K = self.K
		Wei = self.Wei
		ui = self.ui
		y = []

		for x in X:
			# reshaping the image and finding offset
			h,w = x.shape
			Nsq = h*w
			xr = x.reshape((Nsq, 1))
			offset = xr - self.avgImg.reshape((Nsq,1))
			# print (xr, x, offset)
			# sys.exit()

			# finding the weight vecotr of test img
			weiVct = np.zeros((K, 1))
			for j in range(K):
				weiVct[j,0] = np.matmul(offset.T, ui[:, j])
			# print (weiVct)

			# comparing with other class representatives
			dist_bw_faces = -1
			indx = -1
			for i in self.indices:
				df = norm(Wei[:, i] - weiVct)
				if (dist_bw_faces == -1):
					dist_bw_faces = df
				else:
					if (dist_bw_faces != min(dist_bw_faces, df)):
						dist_bw_faces = df
						indx = i

			# print('indx',indx, '\n wiegths:', Wei, '\nweight', weiVct)
			# sys.exit()

			recon = np.zeros((Nsq, 1))
			for i in range(K):
				recon += weiVct[i,0]*ui[:,i].reshape((Nsq, 1))
			# print (recon)

			dist_bw_faceSpace = norm(recon)

			if dist_bw_faceSpace <= tf and dist_bw_faces <= te:
				y.append(indx)
			elif dist_bw_faceSpace <= tf and dist_bw_faces > te:
				print(dist_bw_faces)
				y.append(unknownface)
			else: # dist_bw_faceSpace > tf
				print(dist_bw_faceSpace)
				y.append(nonface)

		return y

# find the nearest neigbour of the with the given weights
# test img provided