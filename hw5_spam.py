from scipy.io import loadmat
from numpy import array
import matplotlib.pyplot as plt
import numpy as np
import random

train_set = loadmat(file_name="spam-dataset/spam_data.mat")
labels = train_set['training_labels'].ravel()
X = train_set['training_data']
X = np.float64(np.copy(X))
print X.shape
print labels.shape
X_p = np.concatenate((X,labels[:,None]), 1)
xmax= X_p.max(axis=0)
print xmax
t = "i8"
for i in range(1,32):
	t+= ",i8"
class Tree:
	def __init__(self, feature, split, left_node, right_node):
		self.feature = feature
		self.split = split
		self.left_node = left_node
		self.right_node = right_node


	def build_tree(self, elements):
		for i in range(0,32):
			mat = elements[elements[:,i].argsort()]
			uniq = np.unique(mat.T[i])
			for j in range(0,len(uniq)-1):
				res = np.nonzero(mat.T[i] == uniq[j])
				last_index = res[len(res)-1][0]
				split1,split2 = X_p[:last_index+1], X_p[last_index+1:]
				print split1.shape
				print split2.shape
				print res
				1/0
			