import numpy as np
import cv2 as cv
import sys
import src
from sklearn.model_selection import train_test_split


# Variables
##########################################################
path = '../att_faces/'
result_dir = '../result/'
subjects = range(1, 41)
subject_images = range(1,11)
M = len(subject_images) * len(subjects) # total no of images
K = 30 # dimension of face_space
b = 2 # number of classes to keep unseen
te = 2000 # Threshold for the L2 distance for training weight vectors
tf = 15000 # Threshold for the L2 distance from the face space
unknownface = -1 # label to denote an unknownface
nonface = -2 # label to denote an nonface

# Reading data
##########################################################
# X -> [pose1, pose2, ..., poses]
# y -> [class, class, ..., class]
print("Reading data")
X = []
y = []
for subject in subjects:
    for subject_image in subject_images:
        i_path = path +'s'+ str(subject) + \
                 '/' + str(subject_image) + '.pgm'
        temp = cv.imread(i_path)
        temp = cv.cvtColor(temp, cv.COLOR_BGR2GRAY) # B&W
        X.append(temp)
        y.append(subject)
del temp
X = np.array(X)
y = np.array(y)
print ('done')


# Keeping 'b' number of classes unseen by the algo.
##########################################################
b_ = M - b*len(subject_images)
X_knownFaces = X[:(b_)]
y_knownFaces = y[:(b_)]
X_unknownFaces = X[(b_):]
y_unknownFaces = y[(b_):]
# print (len(y_knownFaces), len(X_unknownFaces))


# Split Train-Test
##########################################################
X_train, X_test, y_train, y_test = train_test_split(X_knownFaces,
	                                                y_knownFaces, 
	                                                random_state=42)


# Train the classfier
##########################################################
print('Taking K:', K)
e = src.EigenFaces(K, debug=True)
e.train(X_train, y_train)


## TESTING
# Testing known faces
##########################################################
predictions_knownFaces = e.predict(X_test, te=te, tf=tf,
                                   nonface=nonface,
                                   unknownface=unknownface)
print(predictions_knownFaces*(predictions_knownFaces != y_test))
print('Accuracy_knownFaces:', np.sum(predictions_knownFaces
	  == y_test) / len(predictions_knownFaces))


# Testing unknown faces
##########################################################
predictions_unknownFaces = e.predict(X_unknownFaces, te=te, tf=tf,
                                     nonface=nonface,
                                     unknownface=unknownface)
print(predictions_unknownFaces*(predictions_unknownFaces != y_unknownFaces))
print('Accuracy_unknownFaces:', np.sum(predictions_unknownFaces
	  == unknownface) / len(predictions_unknownFaces))

# Testing non faces
##########################################################


# print(predictions == y_test)
# print(predictions*(predictions != y_test))

# h,w = X_train[0].shape
# for i in range(10):
#   nonfaceimg = (np.random.rand(h,w))
#   print(e.predict([nonfaceimg], te=3000, tf=10000,\
#                         nonface=-2, unknownface=-1))

# # print(type(data[1][1])) 
# print(data[1][1].shape)

sys.exit()
