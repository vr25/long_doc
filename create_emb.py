import os
import sys
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import logging

#Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('/gpfs/u/home/HPDM/HPDMrawt/scratch/ptl_sample/bert-base-uncased')

#text = "I like Python"

#basepath = "/gpfs/u/home/HPDM/HPDMrawt/scratch/ptl_sample/pts/2006-03-03_879635_10-K/"

basepath1 = "/gpfs/u/home/HPDM/HPDMrawt/scratch/ptl_sample/pts/"

basepath2 = "/gpfs/u/home/HPDM/HPDMrawt/scratch/ptl_sample/all_cik_year_roa_scaled.csv"

cik_year = open(basepath2, "r").readlines()

cik_year_dict = dict([(i.split(",")[0], float(i.split(",")[2].strip("\n"))) for i in cik_year[1:]])

pts_list = sorted(os.listdir(basepath1))

years_list = ['2011'] #, '2007', '2008', '2009', '2010', '2011'] 
#print("years_list: ", years_list[:5])

X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []

for pt in pts_list:
    #print("pt: ", pt)
    
    cik = pt.split("_")[1]
    year = pt.split("_")[0].split("-")[0]
    cik_year = cik + "_" + year
    #print("cik_year: ", cik_year)

    if year in years_list:
        print("TRAIN DATASET")

        if cik_year in list(cik_year_dict.keys()):
            
            basepath = "/gpfs/u/home/HPDM/HPDMrawt/scratch/ptl_sample/pts/" + pt + "/"
        
            y_train_list.append(cik_year_dict[cik_year])

            sent_list = sorted(os.listdir(basepath), key=lambda x: x.split("_")[-4])
            #print(sent_list)    

            all_sent_embedding_t = []

            for s in sent_list:

                #word embedding
                all_word_embedding_t = []
                
                encoded_layers = torch.load(basepath + "/" + s)
                encoded_layers = encoded_layers.permute(1, 0, 2 )  #Swap dimension 1 (layer) and 2 (word)

                i = encoded_layers.shape[0]

                for w in range(0, i):
                    #print("i", i)
                    token_vecs_8 = encoded_layers[w][8]
                    token_vecs_9 = encoded_layers[w][9]
                    token_vecs_10 = encoded_layers[w][10]
                    token_vecs_11 = encoded_layers[w][11]

                    token_vecs = torch.stack([token_vecs_8, token_vecs_9, token_vecs_10, token_vecs_11], 0)
                    #token_vecs = torch.cat([token_vecs_8, token_vecs_9, token_vecs_10, token_vecs_11], 0)  #concat
                    #print("tokens vecs all 4 layers shape: ", token_vecs.shape)

                    #word_embedding = torch.sum(token_vecs, 0)  #sum
                    #word_embedding = torch.max(token_vecs, 0)  #max
                    #word_embedding = torch.mean(token_vecs, 0)  #mean
                    
                    #mean_max
                    word_embedding_mean = torch.mean(token_vecs, 0)  #mean
                    word_embedding_max = torch.max(token_vecs, 0)  #max
                    word_embedding = torch.cat([word_embedding_mean, word_embedding_max[0]], 0)

                    #max_mean
                    #word_embedding_max = torch.max(token_vecs, 0)  #max
                    #word_embedding_mean = torch.mean(token_vecs, 0)  #mean
                    #word_embedding = torch.cat([word_embedding_max[0], word_embedding_mean], 0)

                    #print("word embedding: ", word_embedding.shape)
                    #word_embedding = token_vecs  #concat
                    all_word_embedding_t.append(word_embedding)  #sum, mean, max_mean, concat
                    #all_word_embedding_t.append(word_embedding[0])  #max

                #sentence embedding
                sentence_embedding = torch.stack(all_word_embedding_t, 0)
                #sentence_embedding = torch.sum(sentence_embedding, 0)  #sum
                sentence_embedding = torch.mean(sentence_embedding, 0)  #mean
                #sentence_embedding = torch.max(sentence_embedding, 0)  #max
                #print("sentence embedding: ", sentence_embedding.shape)

                all_sent_embedding_t.append(sentence_embedding) #sum, mean
                #all_sent_embedding_t.append(sentence_embedding[0])   #max

            #document embedding
            document_embedding = torch.stack(all_sent_embedding_t, 0)
            #document_embedding = torch.sum(document_embedding, 0)  #sum
            document_embedding = torch.mean(document_embedding, 0)  #mean
            #document_embedding = torch.max(document_embedding, 0)  #max
            #print("document embedding: ", document_embedding[0].shape)

            X_train_list.append(document_embedding)  #sum, mean
            #X_train_list.append(document_embedding[0])  #max

X_train = torch.stack(X_train_list, 0)
y_train = torch.FloatTensor(y_train_list)

print(X_train.shape, y_train.shape)  

i = years_list[-1]

torch.save(X_train, "X_mean_max_" + str(i) + ".pt")
#torch.save(y_train, "y_" + str(i) + ".pt")
