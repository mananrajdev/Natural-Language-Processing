#!/usr/bin/env python
# coding: utf-8

# # MANAN RAJDEV - CSCI 544 - HW4

# ## Libraries

# In[342]:


import pandas as pd
import matplotlib.pyplot as plt
import copy
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")

# In[384]:


import torch
from torch.utils import data
import torch.nn as nn


task_name, model_filename, dev_filename, test_filename=sys.argv[1:]

# ## Task 1 - Simple Bidirectional LSTM Model

# ### Data Preprocessing

# In[3]:


#create a dictionary for mapping word to integer and label to integer for creating their embeddings
df_train=pd.read_csv("data/train", sep="\s", names=["idx","word","tag"])


# In[4]:


word_count=df_train['word'].value_counts()
word_count=word_count[word_count>1]
word_set=set(word_count.index)


# In[5]:


#create a dictionary for mapping word to integer and label to integer for creating their embeddings
#convert training into list of lists of sentences of integers (mapped through the dictionary)
train_sentences = []        
train_labels = []
temp_sentence=[]
temp_label=[]
word2idx={}
label2idx={}

word_idx=1
label_idx=1
with open("data/train") as f:
    for sentence in f.read().splitlines():
        if sentence=="":
            if temp_sentence.count("UNK")!=len(temp_sentence):
                train_sentences.append(temp_sentence)
                train_labels.append(temp_label)
            temp_sentence=[]
            temp_label=[]
            continue
        _,word,label=sentence.split()
        
        word=word if word in word_set else "UNK"
        if word not in word2idx:
            word2idx[word]=word_idx
            temp_sentence.append(word_idx)
            word_idx+=1
        else:
            temp_sentence.append(word2idx[word])

        if label not in label2idx:
            label2idx[label]=label_idx
            temp_label.append(label_idx)
            label_idx+=1
        else:
            temp_label.append(label2idx[label])
    
    train_sentences.append(temp_sentence)
    train_labels.append(temp_label)   
            


# In[6]:


# Find max length for padding
# max_train_sentences = max([len(s) for s in train_sentences])
# max_train_sentences
# list_len=[len(s) for s in train_sentences]


# # In[53]:


# plt.hist(list_len)
# plt.title("Histogram of Length of Sentences")
# plt.xlabel("Length")
# plt.ylabel("Count")
# print("Final Length of sentences will be 30")


# In[11]:


# Add pad token and unknown token in dictionary
word2idx["PAD"]=0
label2idx["PAD"]=0


# In[12]:


#convert a reverse dictionary of labels for mapping it back while creating the dev file output
idx2label={label2idx[k] : k for k in label2idx}


# In[51]:


#pad sentences to the maximum length
def padding_sentences(final_length, sentences, labels):
    for i in range(len(sentences)):
        lenn=len(sentences[i])
        if lenn>=final_length:
            sentences[i]=sentences[i][:final_length]
            labels[i]=labels[i][:final_length]
        else:
            sentences[i]+=[0]*(final_length-lenn)
            labels[i]+=[0]*(final_length-lenn)
    return sentences, labels

train_sentences, train_labels = padding_sentences(30, train_sentences, train_labels)


# In[13]:


#pad sentences to the maximum length
max_train_sentences=30
for i in range(len(train_sentences)):
    lenn=len(train_sentences[i])
    if lenn>=max_train_sentences:
        train_sentences[i]=train_sentences[i][:max_train_sentences]
        train_labels[i]=train_labels[i][:max_train_sentences]
    else:
        train_sentences[i]+=[0]*(max_train_sentences-lenn)
        train_labels[i]+=[0]*(max_train_sentences-lenn)


# In[64]:


def gen_sentences(filename, word2idx, label2idx):
#Same operation for dev data
    val_sentences = []        
    val_labels = []
    temp_sentence=[]
    temp_label=[]
    unk_set=set()


    with open(filename) as f:
        for sentence in f.read().splitlines():
            if sentence=="":
                val_sentences.append(temp_sentence)
                val_labels.append(temp_label)
                temp_sentence=[]
                temp_label=[]
                continue
            _,word,label=sentence.split()

            if word not in word2idx:
                unk_set.add(word)
                temp_sentence.append(word2idx["UNK"])
            else:
                temp_sentence.append(word2idx[word])

            temp_label.append(label2idx[label])

        val_sentences.append(temp_sentence)
        val_labels.append(temp_label)
    return val_sentences, val_labels, unk_set

val_sentences, val_labels, dev_unk_set= gen_sentences("data/dev", word2idx, label2idx)


# In[59]:


#padding of dev data
padded_val_sentences= copy.deepcopy(val_sentences)
padded_val_labels = copy.deepcopy(val_labels)
padded_val_sentences, padded_val_labels = padding_sentences(30, padded_val_sentences, padded_val_labels)


# ### Model 

# In[16]:


class myDataset(data.Dataset):
    def __init__(self, features, labels):
        self.features=features
        self.labels=labels
        self.len = len(features)
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        row=self.features[index]
        row_label=self.labels[index]
        return torch.tensor(row),torch.tensor(row_label)


# In[17]:


training_set=myDataset(train_sentences, train_labels)
training_generator = data.DataLoader(training_set, batch_size=32, shuffle=True)


# In[18]:


padded_val_set=myDataset(padded_val_sentences, padded_val_labels)
padded_val_generator = data.DataLoader(padded_val_set, batch_size=32, shuffle=False)


# In[19]:



class Model_LSTM(nn.Module):
    
    def __init__(self, output_size, hidden_dim, embed_size, dropout_rate,  n_layers):
        
        super(Model_LSTM, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        self.embedding=nn.Embedding(len(word2idx),embed_size, padding_idx=0)
        
        self.dropout = nn.Dropout(dropout_rate)
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_dim, batch_first=True, bidirectional=True)   
        
        #dense layer
        self.fc = nn.Linear(2*hidden_dim, output_size)
        self.elu=nn.ELU()
        self.fc1 = nn.Linear(output_size, len(label2idx))
        

        
    def forward(self, x):
        
        s=self.embedding(x)
        s=self.dropout(s)
        s, _ = self.lstm(s)
        s=self.dropout(s)

        s = self.fc(s)          

        s=self.elu(s)

        s=self.fc1(s)

        
        
        
        return s

    
    def init_weights(self):
    # to initialize all parameters from normal distribution
    # helps with converging during training
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)




def write_out(in_filename, out_filename, y_pred, idx2label):
    open(out_filename, 'w').close()
    f1 = open(out_filename, "a")

    i=0
    j=0
    with open(in_filename) as f:
        for sentence in f.read().splitlines():
            if sentence=="":
                i+=1
                j=0
                f1.write("\n")
                continue
    #         _,word,label=sentence.split()

            f1.write(f'{sentence} {idx2label[y_pred[i][j]]}\n')
            j+=1 


    f1.close()


def write_out_dev(in_filename, out_filename, y_pred, idx2label):
    open(out_filename, 'w').close()
    f1 = open(out_filename, "a")

    i=0
    j=0
    with open(in_filename) as f:
        for sentence in f.read().splitlines():
            if sentence=="":
                i+=1
                j=0
                f1.write("\n")
                continue
            idx,word,label=sentence.split()

            f1.write(f'{idx} {word} {idx2label[y_pred[i][j]]}\n')
            j+=1 


    f1.close()
    
    

    
test_sentences = []        

temp_sentence=[]

test_unk_set=set()

with open("data/test") as f:
    for sentence in f.read().splitlines():
        if sentence=="":
            test_sentences.append(temp_sentence)

            temp_sentence=[]
 
            continue
        _,word=sentence.split()
        
        if word not in word2idx:
            test_unk_set.add(word)
            temp_sentence.append(word2idx["UNK"])
        else:
            temp_sentence.append(word2idx[word])
        

            
    test_sentences.append(temp_sentence)
        
# In[20]:

if task_name.lower()=="task1":
    
    # Instantiate the model with hyperparameters
    model = Model_LSTM(output_size=128, hidden_dim=256, embed_size=100, dropout_rate=0.33,  n_layers=1)
    model.init_weights()
    
    
    
    
    # Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    
    task1_model = torch.load(model_filename)
    
    
    
    
    
    val_set=myDataset(val_sentences, val_labels)
    val_generator = data.DataLoader(val_set, batch_size=1, shuffle=False)
    
    
    
    
    y_pred=[]
    for test_batch in val_generator:
        pred=task1_model(test_batch[0]).squeeze().topk(1)[1].T[0].numpy()
        y_pred.append(pred.reshape(-1))
    
    
    
    
    
    write_out_dev("data/dev", dev_filename, y_pred, idx2label)
    
    

    
                
                
    
    y_pred=[]
    for test_sentence in test_sentences:
        pred=task1_model(torch.tensor(test_sentence).view(1,len(test_sentence))).squeeze().topk(1)[1].T[0].numpy()
        y_pred.append(pred.reshape(-1))
    
    
    
    write_out("data/test", test_filename, y_pred, idx2label)
    
    sys.exit()
    


# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

# ## Task 2 - Using GloVe word embeddings

# ### Data Preprocessing

# In[65]:


word2idx_new=copy.deepcopy(word2idx)


# In[66]:


embeddings_dict = {}
with open("glove.6B.100d.txt", 'r', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float64")
        embeddings_dict[word] = vector


# In[67]:


# Adding 0 or 1 as the first element of the embedding deciding on whether the first letter is uppercase or lowercase.
# This will help in providing case sensitivity as Glove embeddings are case insensitive
glove_emb={}
for word,idx in word2idx_new.items():
    if word=='PAD' or word=='UNK':
        pass
    if word in embeddings_dict:
        glove_emb[idx]=np.append(embeddings_dict[word],0.0)
    elif word.lower() in embeddings_dict:
        glove_emb[idx]=np.append(embeddings_dict[word.lower()],1.0)
    else:
        glove_emb[idx]=np.random.normal(scale=0.6, size=(101, ))


# In[68]:


val_arr=np.array(list(glove_emb.values()))
meann=np.mean(val_arr,axis=0)
glove_emb[word2idx_new['PAD']]=np.zeros((101,))
glove_emb[word2idx_new['UNK']]=meann


# In[69]:


l=len(glove_emb)

for word in test_unk_set.union(dev_unk_set):
    if word in embeddings_dict:
        glove_emb[l]=np.append(embeddings_dict[word],0.0)
    elif word.lower() in embeddings_dict:
        glove_emb[l]=np.append(embeddings_dict[word.lower()],1.0)
    else:
        continue
    word2idx_new[word]=l
    l+=1


# In[70]:


emb_matrix=[]
for i in range(len(glove_emb)):
    emb_matrix.append(glove_emb[i])
emb_matrix=np.array(emb_matrix, dtype='float64')


# In[76]:


train_sentences, train_labels, _= gen_sentences("data/train", word2idx_new, label2idx)
train_sentences, train_labels = padding_sentences(30, train_sentences, train_labels)


# In[81]:


val_sentences, val_labels, _= gen_sentences("data/dev", word2idx_new, label2idx)
padded_val_sentences= copy.deepcopy(val_sentences)
padded_val_labels = copy.deepcopy(val_labels)
padded_val_sentences, padded_val_labels = padding_sentences(30, padded_val_sentences, padded_val_labels)


# ### Model - Task 2

# In[83]:


training_set=myDataset(train_sentences, train_labels)
training_generator = data.DataLoader(training_set, batch_size=32, shuffle=True)


# In[84]:


padded_val_set=myDataset(padded_val_sentences, padded_val_labels)
padded_val_generator = data.DataLoader(padded_val_set, batch_size=32, shuffle=False)


# In[85]:



class Model_Glove_LSTM(nn.Module):
    
    def __init__(self, output_size, hidden_dim, embed_size, dropout_rate,  n_layers):
        
        super(Model_Glove_LSTM, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers



        self.embedding = nn.Embedding(len(word2idx_new),embed_size, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix).float())

        
        self.dropout = nn.Dropout(dropout_rate)
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_dim, batch_first=True, bidirectional=True)   
        
        #dense layer
        self.fc = nn.Linear(2*hidden_dim, output_size)
        self.elu=nn.ELU()
        self.fc1 = nn.Linear(output_size, len(label2idx))
        

        
    def forward(self, x):
        
        s=self.embedding(x)
        s=self.dropout(s)
        s, _ = self.lstm(s)
        s=self.dropout(s)

        s = self.fc(s)          

        s=self.elu(s)

        s=self.fc1(s)

        
        
        
        return s

    
    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)


# In[89]:


# Instantiate the model with hyperparameters
model = Model_Glove_LSTM(output_size=128, hidden_dim=256, embed_size=101, dropout_rate=0.33,  n_layers=1)
# model.init_weights()


test_sentences = []        
temp_sentence=[]
with open("data/test") as f:
    for sentence in f.read().splitlines():
        if sentence=="":
            test_sentences.append(temp_sentence)

            temp_sentence=[]
 
            continue
        _,word=sentence.split()
        
        if word not in word2idx_new:
            test_unk_set.add(word)
            temp_sentence.append(word2idx_new["UNK"])
        else:
            temp_sentence.append(word2idx_new[word])
        

            
    test_sentences.append(temp_sentence)
    
    
# In[90]:



if task_name.lower()=="task2":
    task2_model = torch.load(model_filename)
    
    
    
    
    val_set=myDataset(val_sentences, val_labels)
    val_generator = data.DataLoader(val_set, batch_size=1, shuffle=False)
    
    
    
    y_pred=[]
    for test_batch in val_generator:
        pred=task2_model(test_batch[0]).squeeze().topk(1)[1].T[0].numpy()
        y_pred.append(pred.reshape(-1))
    
    
    
    
    write_out_dev("data/dev", dev_filename, y_pred, idx2label)
    
    
    
    y_pred=[]
    for test_sentence in test_sentences:
        pred=task2_model(torch.tensor(test_sentence).view(1,len(test_sentence))).squeeze().topk(1)[1].T[0].numpy()
        y_pred.append(pred.reshape(-1))
    
    
    
    write_out("data/test", test_filename, y_pred, idx2label)
    
    sys.exit()

    
