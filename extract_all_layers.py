import torch
 
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
 
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)
import pickle
#import matplotlib.pyplot as plt
#outfile = open('hid_weights.pkl', wb)
import nltk
from nltk import word_tokenize, sent_tokenize
nltk.data.path.append("/gpfs/u/home/HPDM/HPDMrawt/nltk_data/packages/")
import time
import os
import sys
from multiprocessing import Pool, current_process
from pathlib import Path
 
start = time.time()
 
def foo(filename):
    # Hacky way to get a GPU id using process name (format "ForkPoolWorker-%d")
     gpu_id = (int(current_process().name.split('-')[-1]) - 1) % 4
 
     # run processing on GPU <gpu_id>
     ident = current_process().ident
     print('{}: starting process on GPU {}'.format(ident, gpu_id))
     # ... process filename
     print('{}: finished'.format(ident))
 
 
def f_mp(l):
     pool = Pool(processes=4*2)
 
     #files = ['{}.txt'.format(x) for x in range(10)]
 
     result = pool.imap_unordered(foo, l)
     return 0
 
     pool.close()
     pool.join()

'''
files_list = os.listdir('docs_7')
#print("files_list: ", files_list)
 
basepath = "/gpfs/u/home/HPDM/HPDMrawt/scratch/ptl_sample/docs_7/"
 
f = open(basepath + files_list[0], 'r').read()
 
n = 2
 
files_list1 = [files_list[i:i + n] for i in range(0, len(files_list), n)]
 
print("files_list1: ", files_list1[1])
'''

def sent_bert(l): #, cuda_id):

    #basepath = "/gpfs/u/home/HPDM/HPDMrawt/scratch/ptl_sample/pts/"
   
    for l_ in l:
        
        #print("l_: ", l_)
        
        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = BertTokenizer.from_pretrained('/gpfs/u/home/HPDM/HPDMrawt/scratch/ptl_sample/bert-base-uncased')
       
        f = open(l_, 'r').read()
        #print("f: ", f)
        text = sent_tokenize(f)
        #print("text: ", text)
        #text = ["After stealing money", "The bank robber was seen"] # from the bank vault, the bank robber was seen " \
        #"fishing on the Mississippi river bank."
 #marked_text = "[CLS] " + text + " [SEP]"
 
        i = 1

        for marked_text in text:
            #marked_text = text

            # Tokenize our sentence with the BERT tokenizer.
            tokenized_text = tokenizer.tokenize(marked_text)
 
            # Print out the tokens.
            #print (tokenized_text)
 
            # Add the special tokens.
            #marked_text = "[CLS] " + text + " [SEP]"

            # Split the sentence into tokens.
            tokenized_text = tokenizer.tokenize(marked_text)
 
            # Map the token strings to their vocabulary indeces.
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
 
            # Display the words with their indeces.
            #for tup in zip(tokenized_text, indexed_tokens):
                #print('{:<12} {:>6,}'.format(tup[0], tup[1]))
 
            # Mark each of the 22 tokens as belonging to sentence "1".
            segments_ids = [1] * len(tokenized_text)
 
            #print (segments_ids)
 
            #device = torch.device(cuda_id)  
 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens]).to(device)
            segments_tensors = torch.tensor([segments_ids]).to(device)
 
            # Load pre-trained model (weights)
            model = BertModel.from_pretrained('/gpfs/u/home/HPDM/HPDMrawt/scratch/ptl_sample/bert-base-uncased')
 
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

            # Put the model in "evaluation" mode, meaning feed-forward operation.
            model.to(device)#cuda()
            model.eval()
 
            # Predict hidden states features for each layer
            with torch.no_grad():
                encoded_layers, _ = model(tokens_tensor, segments_tensors)
 
            #print ("Number of layers:", len(encoded_layers))
            layer_i = 0
 
            #print ("Number of batches:", len(encoded_layers[layer_i]))
            batch_i = 0
 
            #print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
            token_i = 0
 
            #print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))

            # `encoded_layers` is a Python list.
            #print('     Type of encoded_layers: ', type(encoded_layers))
 
            # Each layer in the list is a torch tensor.
            #print('Tensor shape for each layer: ', encoded_layers[0].size())
 
            token_embeddings = torch.stack(encoded_layers, dim=0)
 
            #print("token_embeddings_size: ", token_embeddings.size())
 
            # Remove dimension 1, the "batches".
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
 
            #print("After removing batches dimension: ", token_embeddings.size())
            #print("First layer: ", token_embeddings)
            #print("Second layer: ", token_embeddings[1])
            #print("Third layer: ", token_embeddings[2])
            #print("Twelfth layer: ", token_embeddings[11])
 
            basepath = "/gpfs/u/home/HPDM/HPDMrawt/scratch/ptl_sample/pts/"
            f_name = l_.split('/')[-1] + '_cuda_' + str(i) + '_hid_weights_tensor.pt'
            dir_name = l_.split('/')[-1].split('.')[0]
            basepath1 = basepath + dir_name
            if not os.path.exists(basepath1):
                os.mkdir(basepath1)
            
            basepath2 = os.path.join(basepath1, f_name)
            
            #path1 = "/pts/"

            #basepath1 = os.path.join(basepath, path1)

            #print("Before: ", os.getcwd())
            #print("basepath1: ", basepath1)
            #print("After: ", os.getcwd())
            
            #f9 = open(basepath1)
            #f9.close()
            
            torch.save(token_embeddings, basepath2)
            i = i + 1

l = sys.argv[1:]
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)
import pickle
#print("l: ", l)
sent_bert(l) #[0:4], "cuda:0")
#sent_bert(l[4:8], "cuda:1")
#sent_bert(l[8:12], "cuda:2")
#sent_bert(l[12:16], "cuda:3")
 
#n1 = 4
#l1 = [l[i:i+n1] for i in range(0, len(l), n1)]
 
#f_mp(l1)
 
print("Total execution time: ", time.time() - start, " sec")
