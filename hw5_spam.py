from collections import Counter
from scipy.io import loadmat
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
from numpy import array
import matplotlib.pyplot as plt
import numpy as np
from math import log
import random
import csv

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
		#	print "=============END REACHED=========="
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
		#print feature, split
		return Tree(feature, split, left, right)	

def lookup(tree, item):
	if tree.right_node == None and tree.left_node == None:
		return tree.feature	
	if item[tree.feature] <= tree.split:
		return lookup(tree.left_node,item)
	return lookup(tree.right_node,item)

def lookup2(tree, item):
	if tree.right_node == None and tree.left_node == None:
		print "Feature: " + str(tree.feature) + " Split Val: " + str(tree.split)
		return tree.feature	
	if item[tree.feature] <= tree.split:
		print "Feature: " + str(tree.feature) + " Split Val: " + str(tree.split)
		return lookup2(tree.left_node,item)
	print "Feature: " + str(tree.feature) + " Split Val: " + str(tree.split)
	return lookup2(tree.right_node,item)

def validation_spam_regTree():
	np.random.shuffle(X_p)
	train = X_p[0:4000]
	validation = X_p[4000:]
	t = Tree(1,2,3,4).build_tree(train)
	correct = 0
	total = 0
	for el in validation:
		total +=1 
		correct += (lookup(t,el) == el[32])
	print correct, total
	print "Validation Rate: " + str(correct*1.0/total)
	print "Path of 1 sample"
	lookup2(t,validation[0])

def kaggle_spam_regTree_csv():
	t = Tree(1,2,3,4).build_tree(X_p)
	X = train_set['test_data']
	print "Id,Category"
	for i in range(0,X.shape[0]):
		print str(i+1)+ ","  + str(lookup(t,X[i]))
#kaggle_spam_regTree_csv()
D = []
labels = []
with open('census_data/train_data.csv') as csvfile:
	reader = csv.DictReader(csvfile)

	for row in reader:
		labels.append(row['label'])
		row.pop('label', None)
		row.pop('fnlwgt', None)
		row.pop('education-num', None)
		for key,item in row.items():
			if item == '?':
				row[key] = np.nan
		D.append(row)
labels = np.array(labels)
v = DictVectorizer(sparse=False)
R = v.fit_transform(D)
print R.shape
I = Imputer()
R = I.fit_transform(R)
R_p = np.concatenate((R,labels[:,None]), 1)
D = []
with open('census_data/test_data.csv') as csvfile:
	reader = csv.DictReader(csvfile)

	for row in reader:
		row.pop('fnlwgt', None)
		row.pop('education-num', None)
		for key,item in row.items():
			if item == '?':
				row[key] = np.nan
		D.append(row)
L = v.transform(D)
I = Imputer()
L_p = I.fit_transform(L)
class Tree2:
	def __init__(self, feature, split, left_node, right_node):
		self.feature = feature
		self.split = split
		self.left_node = left_node
		self.right_node = right_node


	def build_tree(self, elements):
		if 0 in np.bincount(elements.T[elements.shape[1]-1]) or len(np.bincount(elements.T[elements.shape[1]-1])) == 1:
		#	print "=============END REACHED=========="
			return Tree2(elements.T[elements.shape[1]-1][0], 0, None, None)
		feature, split = -1,-1
		info = -10000000
		X_1, X_2 = None,None
		for i in range(0,elements.shape[1]-1):
			mat = elements[elements[:,i].argsort()]
			uniq = np.unique(mat.T[i])
			for j in range(0,len(uniq)-1):
				res = np.nonzero(mat.T[i] == uniq[j])
				last_index = res[0][len(res[0])-1]
				split1,split2 = mat[:last_index+1], mat[last_index+1:]
				count1 = np.bincount(split1.T[elements.shape[1]-1])
				count2 = np.bincount(split2.T[elements.shape[1]-1])
				total = 0
				pre_split_bins = np.bincount(mat.T[elements.shape[1]-1])
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
			if np.average(elements.T[elements.shape[1]-1]) > 0.5:
				return Tree2(1, 0, None, None)
			return Tree2(0,0,None,None)
		#print np.unique(X_1.T[feature])
		#print np.unique(X_2.T[feature])
		#print "@@@@@@@@"
		temp = Tree2(1,2,3,4)
		left = temp.build_tree(X_1)
		right = temp.build_tree(X_2)
		#print feature, split
		return Tree2(feature, split, left, right)	

def lookup3(tree, item):
	if tree.right_node == None and tree.left_node == None:
		print "Feature: " + str(v.feature_names_[tree.feature])
		return tree.feature	
	if item[tree.feature] <= tree.split:
		print "Feature: " + str(v.feature_names_[tree.feature])
		return lookup3(tree.left_node,item)
	print "Feature: " + str(v.feature_names_[tree.feature])
	return lookup3(tree.right_node,item)
def validation_census_regTree():
	R_pp = np.int64(R_p)
	np.random.shuffle(R_pp)
	train = R_pp[0:5000]
	validation = R_pp[25000:]
	t = Tree2(1,2,3,4).build_tree(train)
	correct = 0
	total = 0
	for el in validation:
		total +=1 
		correct += (lookup(t,el) == el[488])
	print correct, total
	print "Validation Rate: " + str(correct*1.0/total)
	print lookup3(t,validation[0])
#validation_census_regTree()

def kaggle_census_regTree_csv():
	R_pp = np.int64(R_p)
	np.random.shuffle(R_pp)
	t = Tree2(1,2,3,4).build_tree(R_pp[0:10000])
	print "Id,Category"
	for i in range(0,L_p.shape[0]):
		print str(i+1)+ ","  + str(lookup(t,L_p[i]))
	return t
#M = kaggle_census_regTree_csv()
class Tree3:
	def __init__(self, feature, split, left_node, right_node):
		self.feature = feature
		self.split = split
		self.left_node = left_node
		self.right_node = right_node


	def build_tree(self, elements):
		if 0 in np.bincount(elements.T[32]) or len(np.bincount(elements.T[32])) == 1:
		#	print "=============END REACHED=========="
			return Tree3(elements.T[32][0], 0, None, None)
		feature, split = -1,-1
		info = -10000000
		X_1, X_2 = None,None
		rand_features = random.sample(list(range(0,31)),8)
		for i in rand_features:
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
				return Tree3(1, 0, None, None)
			return Tree3(0,0,None,None)
		#print np.unique(X_1.T[feature])
		#print np.unique(X_2.T[feature])
		#print "@@@@@@@@"
		temp = Tree3(1,2,3,4)
		left = temp.build_tree(X_1)
		right = temp.build_tree(X_2)
		#print feature, split
		return Tree3(feature, split, left, right)	

def validation_spam_forestTree():
	forest = []
	np.random.shuffle(X_p)
	train = X_p[0:4000]
	validation = X_p[4000:]
	for i in range(0,500):
		forest.append(Tree3(1,2,3,4).build_tree(train))
	correct = 0
	total = 0
	for el in validation:
		guess = []
		for i in range(0,500):
			guess.append(lookup(forest[i],el))
		data = Counter(guess)
		mode = data.most_common(1)[0][0]  # Returns the highest occurring item
		correct += (mode == el[32])
		total +=1 
	print correct, total
	print "Validation Rate: " + str(correct*1.0/total)

def kaggle_spam_forestTree_csv():
	forest = []
	for i in range(0,100):
		forest.append(Tree3(1,2,3,4).build_tree(X_p))
	X = train_set['test_data']
	print "Id,Category"
	for i in range(0,X.shape[0]):
		guess = []
		for j in range(0,100):
			guess.append(lookup(forest[j], X[i]))
		data = Counter(guess)
		mode = data.most_common(1)[0][0]  # Returns the highest occurring item
		print str(i+1)+ ","  + str(mode)

class Tree4:
	def __init__(self, feature, split, left_node, right_node):
		self.feature = feature
		self.split = split
		self.left_node = left_node
		self.right_node = right_node


	def build_tree(self, elements):
		if 0 in np.bincount(elements.T[elements.shape[1]-1]) or len(np.bincount(elements.T[elements.shape[1]-1])) == 1:
		#	print "=============END REACHED=========="
			return Tree4(elements.T[elements.shape[1]-1][0], 0, None, None)
		feature, split = -1,-1
		info = -10000000
		X_1, X_2 = None,None
		rand_features = random.sample(list(range(0,elements.shape[1]-1)),25)
		for i in rand_features:
			mat = elements[elements[:,i].argsort()]
			uniq = np.unique(mat.T[i])
			for j in range(0,len(uniq)-1):
				res = np.nonzero(mat.T[i] == uniq[j])
				last_index = res[0][len(res[0])-1]
				split1,split2 = mat[:last_index+1], mat[last_index+1:]
				count1 = np.bincount(split1.T[elements.shape[1]-1])
				count2 = np.bincount(split2.T[elements.shape[1]-1])
				total = 0
				pre_split_bins = np.bincount(mat.T[elements.shape[1]-1])
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
			if np.average(elements.T[elements.shape[1]-1]) > 0.5:
				return Tree4(1, 0, None, None)
			return Tree4(0,0,None,None)
		#print np.unique(X_1.T[feature])
		#print np.unique(X_2.T[feature])
		#print "@@@@@@@@"
		temp = Tree4(1,2,3,4)
		left = temp.build_tree(X_1)
		right = temp.build_tree(X_2)
		#print feature, split
		return Tree4(feature, split, left, right)	
def validation_census_forestTree():
	R_pp = np.int64(R_p)
	np.random.shuffle(R_pp)
	train = R_pp[0:3000]
	validation = R_pp[25000:]
	forest = []
	for i in range(0,25):
		forest.append(Tree4(1,2,3,4).build_tree(train))
		print i
	correct = 0
	total = 0
	for el in validation:
		total +=1 
		guess = []
		for i in range(0,25):
			guess.append(lookup(forest[i],el))
		data = Counter(guess)
		mode = data.most_common(1)[0][0]  # Returns the highest occurring item
		correct += (mode == el[488])
	print correct, total
	print "Validation Rate: " + str(correct*1.0/total)
	

def kaggle_census_forestTree_csv():
	R_pp = np.int64(R_p)
	np.random.shuffle(R_pp)
	forest = []
	for i in range(0,25):
		forest.append(Tree4(1,2,3,4).build_tree(R_pp[0:3000]))
	print "Id,Category"
	for i in range(0,L_p.shape[0]):
		guess = []
		for j in range(0,25):
			guess.append(lookup(forest[j],L_p[i]))
		data = Counter(guess)
		mode = data.most_common(1)[0][0]  # Returns the highest occurring item
		print str(i+1)+ ","  + str(mode)
#validation_spam_regTree()
#kaggle_spam_regTree_csv()
validation_census_regTree()
#kaggle_census_regTree_csv()
#validation_spam_forestTree()
#kaggle_spam_forestTree_csv()
#validation_census_forestTree()
#kaggle_census_forestTree_csv()
