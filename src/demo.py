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

# Hyperparameters
K = 30 # dimension of face_space
b = 2 # number of classes to keep unseen
te = 2100 # Threshold for the L2 distance for training weight vectors
tf = 13000 # Threshold for the L2 distance from the face space
unknownface = -1 # label to denote an unknownface
nonface = -2 # label to denote an nonface
oz = 3 # top 'oz' eigenfaces to show

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


# Split Train-Test
##########################################################
X_train, X_test, y_train, y_test = train_test_split(X_knownFaces,
	                                                y_knownFaces, 
	                                                random_state=42)


# Train the classfier
##########################################################
print('Training on M:', len(X_train))
print('Taking K:', K)
print('Taking te:', te)
print('Taking tf:', tf)
e = src.EigenFaces(K, debug=False)
e.train(X_train, y_train)


## TESTING
# Testing known faces
##########################################################
predictions_knownFaces = e.predict(X_test, te=te, tf=tf,
                                   nonface=nonface,
                                   unknownface=unknownface)
# print(predictions_knownFaces*(predictions_knownFaces != y_test))
print('Accuracy_knownFaces:', np.sum(predictions_knownFaces
	  == y_test) / len(predictions_knownFaces))


# Testing unknown faces
##########################################################
predictions_unknownFaces = e.predict(X_unknownFaces, te=te, tf=tf,
                                     nonface=nonface,
                                     unknownface=unknownface)
# print(predictions_unknownFaces*(predictions_unknownFaces != y_unknownFaces))
print('Accuracy_unknownFaces:', np.sum(predictions_unknownFaces
	  == unknownface) / len(predictions_unknownFaces))

# Testing non faces
##########################################################
h,w = X_train[0].shape
nonfaces=[]
for i in range(10):
  temp = (np.random.rand(h,w))*255
  nonfaces.append(temp)
nonfaces = np.array(nonfaces)
predictions_nonFaces = e.predict(nonfaces, te=te, tf=tf, nonface=nonface,
                                 unknownface=unknownface)
print('Accuracy_nonFaces:', np.sum(predictions_nonFaces
	  == nonface) / len(predictions_nonFaces))


# Showing the top 'oz' eigenfaces
for ox in range(oz):
	cv.imshow('eigenface', (255*e.ui[:,ox].reshape(h,w)).astype(np.uint8))
	cv.waitKey(0)
	cv.destroyAllWindows()
sys.exit()
