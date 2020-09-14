from scipy.sparse import csr_matrix
import collections
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics
from multiscorer import MultiScorer
from sklearn.kernel_ridge import KernelRidge
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import pandas as pd
import os
import sys
import csv
import gensim
import gensim.models as g
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
import matplotlib
import matplotlib.pyplot as plt
import time

start = time.time()

df1 = pd.read_csv("doc_sim_roa.csv")
#df1 = pd.read_csv('roa_data_2006_2005.csv')  #'doc_sim_roa.csv')
df1_cik_year_list = df1['cik_year']

'''
#data_stats = pd.DataFrame(columns=['cik', 'year', 'cik_year', 'doc_len', 'sent_count', 'all_sent_len'])

#Uncomment this!
#Create a train docs corpus

f = open('kdd_train_long_docs_sec7.txt', 'w+')
f1 = open('2438_files_list.txt', 'w')

basepath1 = "/data/ftm/xgb_regr/ch_an_data/cleaned_sec7/"
files_list = os.listdir(basepath1)
files_list.remove("2007-02-09_1057706_10-K.txt")
files_list.remove("2013-01-04_1435508_10-K.txt")

print("files_list: ", files_list[:2])
files_list_cik_year = [i.split("_")[1] + "_" + i.split("_")[0].split("-")[0] for i in files_list]

common_files = set(df1_cik_year_list).intersection(set(files_list_cik_year))

print("common_files: ", len(common_files))
#sys.exit(0)

c = 0

fi_list = []

for fi in set(files_list):
    
    cik = fi.split("_")[1] 
    year = fi.split("_")[0].split("-")[0]
    file_cik_year = cik + "_" + year #fi.split("_")[0].split("-")[0]

    if file_cik_year in common_files:
        fi_list.append(file_cik_year)
        f1.write(fi + "\n")
        print("c: ", c, "fi: ", fi)
        i_con = open(basepath1 + fi).read()
        f.write(i_con.rstrip('\n'))
        f.write("\n")
        #sent_len = [len(s) for s in sent_tokenize(i_con)]
        #max_sent_len = max(sent_len)
        #token_count = len(word_tokenize(i_con))

        
        data_stats.at[c, 'cik'] = cik
        data_stats.at[c, 'year'] = year
        data_stats.at[c, 'cik_year'] = file_cik_year
        data_stats.at[c, 'doc_len'] = len(word_tokenize(i_con))
        data_stats.at[c, 'sent_count'] = len(sent_len)
        data_stats.at[c, 'all_sent_len'] = sent_len
        data_stats.at[c, 'max_sent_len'] = max_sent_len
        

        c = c + 1

f.close()
f1.close()

print("# files written: ", c)

print("len: ", len(fi_list))

print("duplicate: ", [item for item, count in collections.Counter(fi_list).items() if count > 1])

#data_stats = pd.merge(data_stats, df1, on='cik_year')
#print(data_stats.head(2))

#data_stats.to_csv('data_stats1.csv')
'''

'''
#doc2vec parameters
vector_size = 300
window_size = 15
min_count = 1 
sampling_threshold = 1e-5
negative_size = 5 
train_epoch = 100
dm = 0
worker_count = 120 #number of parallel processes

#pretrained word embeddings
pretrained_emb = "/data/ftm/xgb_regr/doc2vec_toy/toy_data/pretrained_word_embeddings.txt"

#input corpus
train_corpus = open("kdd_train_long_docs_sec7.txt", 'r').readlines()

#output model
saved_path = "/data/ftm/xgb_regr/kdd_mlf_copy/long_doc_sec7_dm_model.bin"

#enable logging  
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#train doc2vec model
docs = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(train_corpus)]

#Train doc2vec model
model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1,pretrained_emb=pretrained_emb, iter=train_epoch)

#save model
model.save(saved_path)
'''


model = Doc2Vec.load("long_doc_sec7_model.bin")

#print(type(model.docvecs[1])) 
#print("doc2vec: ", model.docvecs[1].shape)

b = np.empty((0,300)) #(model.docvecs[1]).shape[1])) 

#print("b: ", type(b), b.shape)

for i in range(2436): #len(list(docs))):
    
    dv = (model.docvecs[i]).reshape(1,-1)

    #print("dv: ", dv)

    b = np.append(b, dv, axis=0)


X_vect = b

#Historic values
#X_vect = csr_matrix(pd.concat([pd.DataFrame(b), pd.DataFrame(df1['prev_roa'])], axis=1)) 

y = df1['roa'].to_numpy()

scorer = MultiScorer({'mse' : (mean_squared_error, {})})
model1 = KernelRidge(kernel='poly', alpha=0.1, gamma=0.1)

# Perform 10-fold cross validation
scores = cross_val_score(model1, X_vect, y, cv=10, scoring=scorer)
results = scorer.get_results()

final_scores = []

for metric_name in results.keys():
    average_score = np.average(results[metric_name])
    print('%s : %f' % (metric_name, average_score))
    final_scores.append(average_score)

print("Total execution time: ", time.time() - start)
