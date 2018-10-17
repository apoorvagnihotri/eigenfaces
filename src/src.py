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
		X_train: numpy array of 2d array(images)
		Y_train: numpy array of label
	'''
	def train(self, X_train, Y_train):
		self.X_train = X_train
		self.Y_train = Y_train
		K = self.K
		# print ('M = ',len(X_train))
		M = len(X_train)

		height, width = X_train[0].shape
		r = np.zeros((height*width, M))

		# reshaping into vectors, col vectors in flatten
		for i in range(M):
			r[:, i] = X_train[i].flatten()
		Nsq = r.shape[0] # length of one vec

		# print ('flattened train', r)
		# find the mean image, images stored as rows
		self.avgImg = r.mean(1)
		assert len(self.avgImg) == Nsq, "Length of the mean formed bad"
		# print (self.avgImg)

		# find the offsets
		A = r - self.avgImg.reshape((Nsq,1))
		#the offsets are stored as cols for each img
		# print (A)
		L = np.matmul(A.T, A)
		# print(L)
		vals, vi = eigh(L)
		
		uvals = np.zeros(K)
		ui = np.zeros((Nsq, K))
		# we choose only top K
		for i in range(K):
			ui[:,i] = np.matmul(A, vi[M-i-1])
			uvals[i] = vals[M-i-1]

		del vals, vi
		print ('final eigs', ui)
		# print ('final eigvals', uvals)
		# normlize
		for i in range(K):
			ui[:,i] = ui[:,i] / norm(ui[:,i])
		print ('normalized eigs', ui)
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

	def predict(self, x, tf, te):
		K = self.K
		Wei = self.Wei
		ui = self.ui

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

		print('indx',indx, '\n wiegths:', Wei, '\nweight', weiVct)
		# sys.exit()

		recon = np.zeros((Nsq, 1))
		for i in range(K):
			recon += weiVct[i,0]*ui[:,i].reshape((Nsq, 1))
		print (recon)

		dist_bw_faceSpace = norm(recon)

		if dist_bw_faceSpace <= tf and dist_bw_faces <= te:
			print("yes it is a face, indx", indx)
		elif dist_bw_faceSpace <= tf and dist_bw_faces > te:
			print("Unknown Face.")
		else: # dist_bw_faceSpace > tf
			print("Non-Face")

# find the nearest neigbour of the with the given weights
# test img provided