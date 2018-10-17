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
K = int(M / 3) # dimention of face_space


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
print ('done')
# img = np.array(X_train[0])
# print (img)
# print (img.dtype)
# print (img.shape)


# Split Train-Test
##########################################################
X_train, X_test, y_train, y_test = train_test_split(
								X, y, random_state=42)


# Train the classfier
##########################################################
e = src.EigenFaces(K)
e.train(X_train, y_train)

print(e.predict(X_train))
# print(type(data[1][1])) 
# print(data[1][1].shape)

sys.exit()

# e = src.EigenFaces(2)

# i1 = [[[1, 1, 1], [2, 2, 2], [4, 4, 4]],
# 	[[23, 23, 23],[23, 23, 23],[53, 53, 53]],
# 	[[23, 23, 23],[2, 2, 2],[1, 1, 1]]]

# i2 = [[[222, 222, 222], [0, 0, 0], [4, 4, 4]],
# 	[[33, 33, 33],[24, 24, 24],[3, 3, 3]],
# 	[[2, 2, 2],[9, 9, 9],[0, 0, 0]]]

# i3 = [[[2, 2, 2], [0, 0, 0], [4, 4, 4]],
# 	[[3, 3, 3],[4, 4, 4],[3, 3, 3]],
# 	[[2, 2, 2],[9, 9, 9],[10, 10, 10]]]

# i1 = np.array(i1)
# i2 = np.array(i2)
# i3 = np.array(i3)

# e.train(np.array([i1, i2, i3]), np.array([1, 1, 3]))
# e.predict(i3+1, 33, 12)