import src
import numpy as np
import cv2 as cv


import sys

# Variables
##########################################################
path = '../att_faces/'
result_dir = '../result/'
subjects = range(1, 41)
subject_images = range(1,11)
M = len(subject_images) * len(subjects) # total no of images
K = int(M / 2) # dimention of face_space


# Reading data
##########################################################
# data -> person -> pose (one of the pics)
print("Reading data")
data = {}
for subject in subjects:
	data[subject] = {}
	for subject_image in subject_images:
		i_path = path +'s'+ str(subject) + \
		         '/' + str(subject_image) + '.pgm'
		temp = cv.imread(i_path)
		data[subject][subject_image] = temp
del temp
print ('done')


# Train the classfier
##########################################################
e = src.EigenFaces(K)

print(type(data[1][1])) 
print(data[1][1].shape)

sys.exit()


i1 = np.array(i1)
i2 = np.array(i2)
i3 = np.array(i3)

e.train(np.array([i1, i2, i3]), np.array([1, 1, 3]))
e.predict(i3+1, 33, 12)