from scipy.io import loadmat
from numpy import array
import matplotlib.pyplot as plt
import numpy as np
from math import log
import random

train_set = loadmat(file_name="spam-dataset/spam_data.mat")
labels = train_set['training_labels'].ravel()
X = train_set['training_data']
X = np.int64(np.copy(X))
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
		if 0 in np.bincount(elements.T[32]) or len(np.bincount(elements.T[32])) == 1:
			print "=============END REACHED=========="
			return Tree(elements.T[32][0], 0, None, None)
		feature, split = -1,-1
		info = -10000000
		X_1, X_2 = None,None
		for i in range(0,32):
			mat = elements[elements[:,i].argsort()]
			uniq = np.unique(mat.T[i])
			for j in range(0,len(uniq)-1):
				res = np.nonzero(mat.T[i] == uniq[j])
				last_index = res[0][len(res[0])-1]
				split1,split2 = mat[:last_index+1], mat[last_index+1:]
				count1 = np.bincount(split1.T[32])
				count2 = np.bincount(split2.T[32])
				total = 0
				pre_split_bins = np.bincount(mat.T[32])
				p_c_0 = 1.0*pre_split_bins[0]/(pre_split_bins[0] + pre_split_bins[1])
				p_c_1 = 1.0*pre_split_bins[1]/(pre_split_bins[0] + pre_split_bins[1]) 
				H_S = -1* p_c_0 * log(p_c_0,2) + -1 * p_c_1 * log(p_c_1,2)
				H_S_R = 0
				H_S_L = 0
				if len(count1) != 1 and count1[0] != 0 and count1[1] != 0:
					p_c_0 = 1.0*count1[0]/(count1[0]+count1[1])
					p_c_1 = 1.0*count1[1]/(count1[0]+count1[1])
					H_S_L = -1* p_c_0 * log(p_c_0,2) + -1 * p_c_1 * log(p_c_1,2)
				if len(count2) != 1 and count2[0] != 0 and count2[1] != 0:
					p_c_0 = 1.0*count2[0]/(count2[0]+count2[1])
					p_c_1 = 1.0*count2[1]/(count2[0]+count2[1])
					H_S_R = -1* p_c_0 * log(p_c_0,2) + -1 * p_c_1 * log(p_c_1,2)
				H_SS = 1.0* (len(split1) * H_S_L + len(split2) * H_S_R)/(len(split1) + len(split2))
			#	print count1, count2
			#	print H_S
			#	print H_S_R
			#	print H_S_L
			#	print H_SS
			#	print "------------"
				if info < H_S - H_SS:
					info = H_S - H_SS
					feature = i
					split = uniq[j]
					X_1 = split1
					X_2 = split2
		if feature == -1:
			if np.average(elements.T[32]) > 0.5:
				return Tree(1, 0, None, None)
			return Tree(0,0,None,None)
		#print np.unique(X_1.T[feature])
		#print np.unique(X_2.T[feature])
		#print "@@@@@@@@"
		temp = Tree(1,2,3,4)
		left = temp.build_tree(X_1)
		right = temp.build_tree(X_2)
		print feature, split
		return Tree(feature, split, left, right)	
