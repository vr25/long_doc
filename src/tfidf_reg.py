import os
import sys
import csv
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import re
import numpy as np
from sklearn.svm import SVR
import time
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics
import copy
from multiscorer import MultiScorer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from scipy.sparse import hstack
from sklearn import linear_model
csv.field_size_limit(sys.maxsize)

start = time.time()

mse_file = open('new_mse_scores_8K_Q2_common.txt', 'w')

def tokenizer(text):
    if text:
        result = re.findall('[a-z]{2,}', text.lower())
    else:
        result = []
    return result

def tfidf_vect(X):
	vect = TfidfVectorizer(tokenizer=tokenizer, stop_words='english')
	v = vect.fit(X)
	X_vect  = v.transform(X)
	return X_vect

def compute(X_vect,y):

	scorer = MultiScorer({
	    'mse' : (mean_squared_error, {})	    
	    })

	#KernelRidgeRegression model
	model = KernelRidge(kernel='poly', alpha=0.1, gamma=0.1)

	# Perform 10-fold cross validation
	scores = cross_val_score(model, X_vect, y, cv=10, scoring=scorer)
	results = scorer.get_results()

	final_scores = []

	for metric_name in results.keys():
		average_score = np.average(results[metric_name])
		print('%s : %f' % (metric_name, average_score))
		final_scores.append(average_score)

	mse_file.write(str(final_scores[0]) + '\n')

word_type = ['words', 'sentiment_words', 'expanded_syntactic_words'] 
target = ['roa'] 
vect_shape = 0


df_10K_t_ = pd.read_csv("data_2436.csv", engine='python')
df_10K_t_ = df_10K_t_.loc[:, ~df_10K_t_.columns.str.contains('^Unnamed')]
df_10K_t_ = df_10K_t_.dropna()

df_prev = pd.read_csv("roa_data_scaled.csv")
df_merge = pd.merge(df_10K_t_, df_prev, on='cik_year')
print("df_merge len: ", len(df_merge), df_merge.columns)
#sys.exit(0)

df_roa = df_merge['prev_roa']


for w in word_type:

	for t in target:

		print("w: ", w, "t: ", t)

		#All 10K		
		X_10K = df_merge[w].astype('U').values  
		y_10K = df_merge[t] 
		X_vect_10K = tfidf_vect(X_10K)
		X_vect_concat = csr_matrix(pd.concat([pd.DataFrame(X_vect_10K.todense()), pd.DataFrame(df_roa)], axis=1))
		compute(X_vect_concat, y_10K)
		
	mse_file.write(w + " " + t + " " + str(vect_shape) + "\n")
					
mse_file.close()

print("Total execution time: ", time.time() - start)
