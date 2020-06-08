import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import logging
import pickle
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


def sent_bert(l):
   
    for l_ in l:
                
        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = BertTokenizer.from_pretrained('/gpfs/u/home/HPDM/HPDMrawt/scratch/ptl_sample/bert-base-uncased')
       
        f = open(l_, 'r').read()
        text = sent_tokenize(f)
        
        i = 1

        for marked_text in text:

            # Tokenize our sentence with the BERT tokenizer.
            tokenized_text = tokenizer.tokenize(marked_text)
 
            # Split the sentence into tokens.
            tokenized_text = tokenizer.tokenize(marked_text)
 
            # Map the token strings to their vocabulary indeces.
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
 
            # Mark each of the 22 tokens as belonging to sentence "1".
            segments_ids = [1] * len(tokenized_text)
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
 
 
            token_embeddings = torch.stack(encoded_layers, dim=0)
 
            #print("token_embeddings_size: ", token_embeddings.size())
 
            # Remove dimension 1, the "batches".
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
 
            basepath = "/gpfs/u/home/HPDM/HPDMrawt/scratch/ptl_sample/pts/"
            f_name = l_.split('/')[-1] + '_cuda_' + str(i) + '_hid_weights_tensor.pt'
            dir_name = l_.split('/')[-1].split('.')[0]
            basepath1 = basepath + dir_name
            if not os.path.exists(basepath1):
                os.mkdir(basepath1)
            
            basepath2 = os.path.join(basepath1, f_name)
                        
            torch.save(token_embeddings, basepath2)
            i = i + 1

l = sys.argv[1:]

sent_bert(l) 
 
print("Total execution time: ", time.time() - start, " sec")