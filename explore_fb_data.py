# -*- coding: utf-8 -*-
"""
Explore the pattern of facebook user data
With objective of unveiling the patterns explain
user-user relationship with an unified model which
is a mixture of two distinguished set of logics

@author: beingzy
@date: Feb/17/2015
"""
## ########################## ##
## MODULE IMPORTS             ##
## ########################## ##
import os
import glob

import re
from itertools import combinations

import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk

import matplotlib.pyplot as plt


#%matplotlib inline
## ######################### ##
## DEFINE ENVINRONMENT       ##
## ######################### ##
ROOT_PATH = "/Users/beingzy/Documents/Projects/learning_dist_metrics"
DATA_PATH = ROOT_PATH + "/datasets/data/"

os.chdir(ROOT_PATH)

## ######################### ##
## FUNCTION DEFINITION       ##
## ######################### ##
def nonull_ptg(x):
	"""Calculate the percentage of features bearing un-missing value
	   for given instances
	"""
	return round(sum(x != 0) * 1.0 / len(x), 2)

## ######################### ##
## LOAD DATA                 ##
## ######################### ##
"""
Load Friendship Information
"""
edges = [i for i in glob.glob(DATA_PATH + "facebook/*.edges")]
chucks = []
for f in edges:
    chuck = pd.read_csv(f, sep = " ", header = None, names = ["x1", "x2"])
    chucks.append(chuck)
friends = pd.concat(chucks)

del edges, chucks, chuck, i, f

"""
Integrate .csv of various fields into a single DataFrame
"""
# 1st Loop: file --> df
# 2st Loop: df --> row
# 3st Loop: row --> item
# if len(item) >= threshold: keep in list
# pd.concat(list, axis = 1).T
csvfiles = [ i for i in glob.glob(DATA_PATH + "facebook/csv/*.csv") ]

rows = []
min_info_length = 5
for fcsv in csvfiles:
	df = pd.read_csv(fcsv, header = 0)
	for idx, row in df.iterrows():
		encode = dict()
		for i, j in zip(row.index, row.values):
			if not np.isnan(j):
				encode[i] = j
		if len(encode) >= min_info_length:
			rows.append(pd.Series(encode))

users = pd.concat(rows, axis = 1).T

del fcsv, df, idx, row, i, j, encode, rows

"""
Calculate the sparsity of users(DataFrame)
"""
n_obs, n_feats = users.shape
n_null = users.isnull().apply(sum, axis = 0).sum()
sparsity = n_null * 1.0 / (n_obs * n_feats)
print "****** Sparsity(DataFrame: users) is {0} *************".format(np.round(sparsity, 3))
"""
Examine the percentage of user in users(DataFrame) having connections
"""
linked_ids = set.union(set(friends["x1"].values),
				 set(friends["x2"].values))
n_linked_users = users["user_id"].apply(lambda x: 1 if x in linked_ids else 0).sum()
print "******** {0}% users have defined friendships ***********".format(np.round(n_linked_users * 1.0 / n_obs, 3) * 100)


