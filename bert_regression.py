import torch
import os
import sys
import csv
import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_squared_error
import re
import numpy as np
import time
from sklearn.model_selection import cross_val_score
from multiscorer import MultiScorer
from sklearn.kernel_ridge import KernelRidge

csv.field_size_limit(sys.maxsize)

start = time.time()

operation = 'concat' #'sum', 'max', 'mean', 'mean_max', 'max_mean', 'concat'
d = 3072 #768, 1536, 3072
years_list = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016']
X_vect = np.empty((0,d))
y = np.empty((0,1))

#Load X, y tensors
for i in range(len(years_list)):
    t = 'X_' + operation + '_' + years_list[i] + '.pt'
    a = torch.load(t)
    a_ = a.cpu()
    #print("a_ shape: ", a_.shape)
    X_vect = np.append(X_vect, a_, axis=0)

    t_ = 'y_train_' + years_list[i] + '.pt'
    a1 = torch.load(t_)
    a1_ = a1.cpu()
    a1_ = a1_.unsqueeze(1)
    #print("a1_ shape: ", a1_.shape)
    y = np.append(y, a1_, axis=0)


#Normalize X_vect matrix for 'sum'
xmax, xmin = X_vect.max(), X_vect.min()
X_vect = (X_vect - xmin)/(xmax - xmin)

scorer = MultiScorer({'mse' : (mean_squared_error, {})})

#KernelRidgeRegression model - default degree=3
model = KernelRidge(kernel='poly', alpha=0.1, gamma=0.1)

# Perform 10-fold cross validation
scores = cross_val_score(model, X_vect, y, cv=10, scoring=scorer)
results = scorer.get_results()

final_scores = []

for metric_name in results.keys():
	average_score = np.average(results[metric_name])
	print('%s : %f' % (metric_name, average_score))
	final_scores.append(average_score)

print("Total execution time: ", time.time() - start)