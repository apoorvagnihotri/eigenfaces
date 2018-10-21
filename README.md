# Intro
This repository implements [EigenFaces paper](http://www.face-rec.org/algorithms/pca/jcn.pdf). This is a algorithm that tries to recognize and detec faces using PCA.

# Usage
The `src.py` file contains the class EigenFaces, whose usage is given in `demo.py` over the [ATnT face dataset](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html).
Run `demo.py` for a sample run.

# Requirements
```
python 3.7.0
opencv-python 3.4.2.16
opencv-contrib-python 3.4.2.16
numpy 1.15.2
scipy 1.1.0
sklearn 0.19.2
```

# Results
We used the ATnT face data set to train our algorithm and then we tested the implementation using some known faces, some unknown faces and some non faces, in each of the case we recived accuracies of ` ` ` ` ` `. The hyperparameters that we worked with were as below. One can change the hyperparameter in the start of demo.py to get different results.

We print the accuracies to the console and the main class that has been implemented resides in `src.py`. Have a look at that file to know the implementation details.

Hyperparameters used:
```python
K = 30 # dimension of face_space
b = 2 # number of classes to keep unseen
te = 3500 # Threshold for the L2 distance for training weight vectors
tf = 15000 # Threshold for the L2 distance from the face space
unknownface = -1 # label to denote an unknownface
nonface = -2 # label to denote an nonface
```








# Reference
*  F. Samaria and A. Harter 
  "Parameterisation of a stochastic model for human face identification"
  2nd IEEE Workshop on Applications of Computer Vision
  December 1994, Sarasota (Florida).
