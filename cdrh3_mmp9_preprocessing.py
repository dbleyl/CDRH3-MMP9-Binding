import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate

import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

#Data preprocessing
def df_data_preprocessing(df_Positive, df_Negative):

  # Dropping the rows with any missing (no CDRH3) or nan or na values.
  df_Positive = df_Positive.dropna()
  print("Total postive sequences without missing values: ", len(df_Positive))
  df_Negative = df_Negative.dropna()
  print("Total negative sequences without missing values : ", len(df_Negative))

  df_sort_cdrh3_count = df.sort_values(by=['CDRH3', 'Count'], ascending = [True, False], inplace = False, ignore_index= True)
  df_CDRH3_Unique = df_sort_cdrh3_count.drop_duplicates(subset=["CDRH3"], keep='first',ignore_index = True)
  df_AA_Binding = df_CDRH3_Unique

  print("Positive labels: ", len(df_AA_Binding[(df_AA_Binding['Binding_Labels'] == 1)]))
  print("Negative labels: ", len(df_AA_Binding[(df_AA_Binding['Binding_Labels'] == 0)]))

  #Reordering the columns for convenience
  df_Binding_AA = df_AA_Binding[['Binding','Binding_Labels', 'Full_AA_Sequence','Count','CDRH3','Start_Idx', 'CDRH3_Length']]

  df_Binding_AA = df_Binding_AA[df_Binding_AA['CDRH3'].str.count('-').le(0)]

  print("Positive labels after removing CDRH3 with hyphens: ", len(df_Binding_AA[(df_Binding_AA['Binding_Labels'] == 1)]))
  print("Negative labels after removing CDRH3 with hyphens: ", len(df_Binding_AA[(df_Binding_AA['Binding_Labels'] == 0)]))

  df_shuffled = df_Binding_AA.sample(frac=1,random_state=200)

  #Shuffled list for ESM2 token generations

  df_shuffled_list = list(df_shuffled.to_records(index=False))

  df_AA_shuffled= df_shuffled[['Binding', 'Full_AA_Sequence']]
  df_binding_AA_shuffled_list = list(df_AA_shuffled.to_records(index=False))
  df_sequence_shuffled_list= df_shuffled['Full_AA_Sequence'].tolist()

  return df_shuffled, df_shuffled_list, df_binding_AA_shuffled_list, df_sequence_shuffled_list


def representations_650MB(df_shuffled_list,df_binding_AA_shuffled_list,max_CDRH3_length):

  import torch
  import esm

  # Load ESM-2 model
  model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
  #model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
  #model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
  device = torch.device("cuda")
  model = model.eval().cuda()
  model.to(device)
  batch_converter = alphabet.get_batch_converter()

  sequence_representations_650MB = []

  batch_size=200
  no_steps= int(len(df_binding_AA_shuffled_list)/batch_size)

  for i in range(no_steps+1):
    #setting the lower and upper boundaries for the batch
    lower= i*batch_size
    upper= lower+batch_size
    upper=upper if upper<len(df_binding_AA_shuffled_list) else len(df_binding_AA_shuffled_list)
    print(f"lower:{lower},upper:{upper}")

    #Creating the amino acid and sample data lists for the batch based on the lower and upper boundaries
    AA_data=[df_binding_AA_shuffled_list[i] for i in range(lower,upper)]
    batch_data = [df_shuffled_list[i] for i in range(lower,upper)]
    batch_labels, batch_strs, batch_tokens = batch_converter(AA_data)
    #batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    #print(batch_lens)

    #print(batch_strs)
    #print(batch_labels)
    #print(batch_tokens)

    batch_tokens = batch_tokens.to(next(model.parameters()).device)

    with torch.no_grad():
      results = model(batch_tokens, repr_layers=[33], return_contacts=False)
      token_representations = results["representations"][33]
      #print(token_representations)

    # Generate CDRH3 per-sequence representations via averaging over CDRH3
    for i, (_, _, _, _, _, start_idx, cdrh3_len ) in enumerate(batch_data):
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1. It means that we need to add 1 to CDRH3 start index.
    #sequence_representations1.append(token_representations[i, start_idx+1: (start_idx + 1 + cdrh3_len)].mean(0))
    #Get the each aa in CDRH3 and get the average token representations for that aminoacid
      for j in range(cdrh3_len):
        if (j==0):
          #Initializing for each CDRH3
          seq_rep = []
        seq_rep.append(token_representations[i, start_idx+j+1: (start_idx + j + 2)].mean(0).cpu())
      #torch.stack((tensor_stack,T1))
      sequence_representations_650MB.append(seq_rep)

    #Since each aminoacid in CDRH3 has a separate tesnsor, we need to stack them together to create sequence of tensors for a given CDRH3.

  embedding_list_650MB = []

  for i in range(len(sequence_representations_650MB)):
    embedding_list_650MB.append(torch.stack((sequence_representations_650MB[i])))

  #Padding sequences to the maximum CDRH3 length to make all the embeddings' length equal

  seq_rep_padded_650MB = embedding_list_650MB.copy()

  import torch
  import torch.nn as nn
  padding = (0,0,0,1)
  pad = nn.ZeroPad2d(padding)
  #max_CDRH3_length = 10
  #The max_CDRH3 length in the training file is 17. Hence hardcoded here.
  #max_CDRH3_length = 17
  for i in range(len(embedding_list_650MB)):
    number_of_padding_required = max_CDRH3_length - len(embedding_list_650MB[i])
    for j in range(number_of_padding_required):
      seq_rep_padded_650MB[i] = pad(seq_rep_padded_650MB[i])

  x_650MB_array=np.array([seq_rep_padded_650MB[i].cpu().numpy() for i in range(len(seq_rep_padded_650MB))] )

  num_features = list(seq_rep_padded_650MB[0][-1].shape)[0]
  #list(var.size())
  print(num_features)

  return x_650MB_array, num_features

def representations_3B(df_shuffled_list,df_binding_AA_shuffled_list,max_CDRH3_length):

    import torch
    import esm
    
    # Load ESM-2 model
    #model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm2_3B, alphabet1 = esm.pretrained.esm2_t36_3B_UR50D()
    #model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    
    import torch
    torch.cuda.empty_cache()
    
    # Load ESM-2 model
    #model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    #model1, alphabet1 = esm.pretrained.esm2_t36_3B_UR50D()
    #model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    device = torch.device("cuda")
    esm2_3B = esm2_3B.eval().cuda()
    esm2_3B.to(device)
    batch_converter1 = alphabet1.get_batch_converter()
    
    #Creating batches to avoid any out of memory error
    #model.eval()  # disables dropout for deterministic results

    seq_rep_aa_in_cdrh3_average_3B = []
    batch_size=200
    
    no_steps= int(len(df_binding_AA_shuffled_list)/batch_size)
    for i in range(no_steps+1):
      #setting the lower and upper boundaries for the batch
      lower= i*batch_size
      upper= lower+batch_size
      upper=upper if upper<len(df_binding_AA_shuffled_list) else len(df_binding_AA_shuffled_list)
      print(f"lower:{lower},upper:{upper}")
    
     #Creating the amino acid and sample data lists for the batch based on the lower and upper boundaries
      AA_data=[df_binding_AA_shuffled_list[i] for i in range(lower,upper)]
      batch_data = [df_shuffled_list[i] for i in range(lower,upper)]
    
      batch_labels, batch_strs, batch_tokens = batch_converter1(AA_data)
      #batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
      #print(batch_lens)
    
      #print(batch_strs)
      #print(batch_labels)
      #print(batch_tokens)
    
      batch_tokens = batch_tokens.to(next(esm2_3B.parameters()).device)
    
      with torch.no_grad():
        results = esm2_3B(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
       #print(token_representations)
    
      # Generate CDRH3 per-sequence representations via averaging over CDRH3
    
      for i, (_, _, _, _, _, start_idx, cdrh3_len ) in enumerate(batch_data):
       # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1. It means that we need to add 1 to CDRH3 start index.
        for j in range(cdrh3_len):
          if (j==0):
            seq_rep = []
          seq_rep.append(token_representations[i, start_idx+j+1: (start_idx + j + 2)].mean(0).cpu())
        seq_rep_aa_in_cdrh3_average_3B.append(seq_rep)

    embedding_list_3B = []
    for i in range(len(seq_rep_aa_in_cdrh3_average_3B)):
      embedding_list_3B.append(torch.stack((seq_rep_aa_in_cdrh3_average_3B[i])))

    seq_rep_padded_3B = embedding_list_3B.copy()
    
    import torch
    import torch.nn as nn
    padding = (0,0,0,1)
    pad = nn.ZeroPad2d(padding)
    #max_CDRH3_length = 10
    #The max_CDRH3 length in the training file is 17. Hence hardcoded here.
    #max_CDRH3_length = 17
    for i in range(len(embedding_list_3B)):
      number_of_padding_required = max_CDRH3_length - len(embedding_list_3B[i])
      for j in range(number_of_padding_required):
        seq_rep_padded_3B[i] = pad(seq_rep_padded_3B[i])
    
    X_3B_array=np.array([seq_rep_padded_3B[i].cpu().numpy() for i in range(len(seq_rep_padded_3B))] )

    num_features = list(seq_rep_padded_3B[0][-1].shape)[0]
    #list(var.size())
    print(num_features)
    
    return X_3B_array, num_features

def representations_15B(df_shuffled_list,df_sequence_shuffled_list,max_CDRH3_length):

    import transformers
    checkpoint = "Rocketknight1/esm2_t48_15B_UR50D"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    import torch
    torch.cuda.empty_cache()
    
    import torch
    from transformers import pipeline
    
    esm_feature_extractor = pipeline(task="feature-extraction",framework="pt",model=checkpoint, tokenizer=tokenizer, device='cuda', torch_dtype=torch.bfloat16)

    seq_rep_avg_aa_in_cdrh3_15B = []
    batch_size=200
    #Ifthi: Modified this for test ab
    #df_AA_Sequences_List = list(df_shuffled['Full_AA_Sequence'])
    #df_AA_Sequences_List = list(df_test_ab_shuffled['Full_AA_Sequence'])
    
    no_steps= int(len(df_sequence_shuffled_list)/batch_size)
    for i in range(no_steps+1):
      #setting the lower and upper boundaries for the batch
      lower= i*batch_size
      upper= lower+batch_size
      upper=upper if upper<len(df_sequence_shuffled_list) else len(df_sequence_shuffled_list)
      print(f"lower:{lower},upper:{upper}")
    
      #Creating the amino acid and sample data lists for the batch based on the lower and upper boundaries
      AA_data=[df_sequence_shuffled_list[i] for i in range(lower,upper)]
      batch_data = [df_shuffled_list[i] for i in range(lower,upper)]
    
      with torch.autocast("cuda"):
        features1=esm_feature_extractor(AA_data,return_tensors = "pt")
    
      # Generate CDRH3 per-sequence representations via averaging over CDRH3
      for i, (_, _, _, _, _, start_idx, cdrh3_len ) in enumerate(batch_data):
       # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1. It means that we need to add 1 to CDRH3 start index.
       # Similarly, since we have to take average over CDRH3, we need to add 1 to CDRH3 length to mark the end of slcing in the tensor.
       #sequence_representations6.append(features1[i][0,(start_idx+1):(start_idx + 1 + cdrh3_len + 1)].mean(0))
       #sequence_representations6.append(features1[i][0,(start_idx+1):(start_idx + 1 + cdrh3_len)].mean(0))
        for j in range(cdrh3_len):
          if (j==0):
            seq_rep = []
          #embedding_tensor_stack = token_representations[i, start_idx+j+1: (start_idx + j + 2)].mean(0).cpu
        #sequence_representations_650MB.append(token_representations[i][(start_idx+j+1): (start_idx + j+ 2)].mean(0))
        #seq_rep.append(token_representations[i, start_idx+j+1: (start_idx + j + 2)].mean(0))
        #seq_rep.append(token_representations[i, start_idx+j+1: (start_idx + j + 2)].mean(0).cpu())
          seq_rep.append(features1[i][0,(start_idx+j+1):(start_idx + j+2)].mean(0).cpu())
        seq_rep_avg_aa_in_cdrh3_15B.append(seq_rep)

    embedding_list_15B = []
    for i in range(len(seq_rep_avg_aa_in_cdrh3_15B)):
      embedding_list_15B.append(torch.stack((seq_rep_avg_aa_in_cdrh3_15B[i])))

    seq_rep_padded_15B = embedding_list_15B.copy()
    
    import torch
    import torch.nn as nn
    padding = (0,0,0,1)
    pad = nn.ZeroPad2d(padding)
    #max_CDRH3_length = 10
    #The max_CDRH3 length in the training file is 17. Hence hardcoded here.
    #max_CDRH3_length = 17
    for i in range(len(embedding_list_15B)):
      number_of_padding_required = max_CDRH3_length - len(embedding_list_15B[i])
      for j in range(number_of_padding_required):
        seq_rep_padded_15B[i] = pad(seq_rep_padded_15B[i])
    
    X_15B_array=np.array([seq_rep_padded_15B[i].cpu().numpy() for i in range(len(seq_rep_padded_15B))] )

    num_features = list(seq_rep_padded_15B[0][-1].shape)[0]
    #list(var.size())
    print(num_features)
    
    return X_15B_array, num_features

def antiberty_representations(df_shuffled_list,df_sequence_shuffled_list, max_CDRH3_length):

    from antiberty import AntiBERTyRunner

    antiberty = AntiBERTyRunner()

    sequence_representations_antiberty = []
    batch_size=200
    no_steps= int(len(df_sequence_shuffled_list)/batch_size)
    
    for i in range(no_steps+1):
      #setting the lower and upper boundaries for the batch
      lower= i*batch_size
      upper= lower+batch_size
      upper=upper if upper<len(df_sequence_shuffled_list) else len(df_sequence_shuffled_list)
      print(f"lower:{lower},upper:{upper}")
    
      #Creating the amino acid and sample data lists for the batch based on the lower and upper boundaries
      AA_data=[df_sequence_shuffled_list[i] for i in range(lower,upper)]
      batch_data = [df_shuffled_list[i] for i in range(lower,upper)]
      #batch_labels, batch_strs, batch_tokens = batch_converter(AA_data)
      embeddings = antiberty.embed(AA_data)
      #print(embeddings)
      #batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
      #print(batch_lens)
    
      #print(batch_strs)
      #print(batch_labels)
      #print(batch_tokens)
    
    
      embeddings_cpu=[embeddings[i].cpu() for i in range(len(embeddings))]
      # Generate  per-sequence GDRH3 representations
      for i, (_, _, _, _, _, start_idx, cdrh3_len ) in enumerate(batch_data):
       # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1. It means that we need to add 1 to CDRH3 start index.
       sequence_representations_antiberty.append(embeddings_cpu[i][start_idx+1: (start_idx + 1 + cdrh3_len)])

    antiberty_seq_rep_padded = sequence_representations_antiberty.copy()
    
    import torch
    import torch.nn as nn
    padding = (0,0,0,1)
    pad = nn.ZeroPad2d(padding)
    #max_CDRH3_length = 10
    #The max_CDRH3 length in the training file is 17. Hence hardcoded here.
    #max_CDRH3_length = 17
    for i in range(len(sequence_representations_antiberty)):
      number_of_padding_required = max_CDRH3_length - len(sequence_representations_antiberty[i])
      for j in range(number_of_padding_required):
        antiberty_seq_rep_padded[i] = pad(antiberty_seq_rep_padded[i])
    
    X_antiberty_array=np.array([antiberty_seq_rep_padded[i].cpu().numpy() for i in range(len(antiberty_seq_rep_padded))] )

    num_features = list(antiberty_seq_rep_padded[0][-1].shape)[0]
    #list(var.size())
    print(num_features)
    
    return X_antiberty_array, num_features