#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from collections import defaultdict
import torch.optim as optim
import warnings
import sys
warnings.filterwarnings("ignore")

model_filename, dev_filename, test_filename=sys.argv[1:]
# In[2]:


word_train_sentences = []        
train_labels = []
temp_sentence=[]
temp_labels=[]
tag_set=set()

with open("data/train") as f:
    for sentence in f.read().splitlines():
        if sentence=="":
            word_train_sentences.append(temp_sentence)
            train_labels.append(temp_labels)
            temp_sentence=[]
            temp_labels=[]
            continue
        _,word,label=sentence.split()
        temp_sentence.append(word)
        temp_labels.append(label)
        tag_set.add(label)    
    word_train_sentences.append(temp_sentence)
    train_labels.append(temp_labels)


# In[128]:


word_dev_sentences = []        
dev_labels = []
temp_sentence=[]
temp_labels=[]


with open("data/dev") as f:
    for sentence in f.read().splitlines():
        if sentence=="":
            word_dev_sentences.append(temp_sentence)
            dev_labels.append(temp_labels)
            temp_sentence=[]
            temp_labels=[]
            continue
        _,word,label=sentence.split()
        temp_sentence.append(word)
        temp_labels.append(label)
            
    word_dev_sentences.append(temp_sentence)
    dev_labels.append(temp_labels)


# In[129]:


word_test_sentences = []        
temp_sentence=[]


with open("data/test") as f:
    for sentence in f.read().splitlines():
        if sentence=="":
            word_test_sentences.append(temp_sentence)
            temp_sentence=[]
            continue
        _,word=sentence.split()
        temp_sentence.append(word)
            
    word_test_sentences.append(temp_sentence)


# In[130]:


pad_token = 'PAD'
unk_token = 'UNK'
pad_id = 0
unk_id = 1

label2idx = {'PAD': 0}
word2idx = {'PAD': 0, 'UNK': 1}


# In[131]:


#create a dictionary for mapping word to integer and label to integer for creating their embeddings
df_train=pd.read_csv("../data/train", sep="\s", names=["idx","word","tag"])


# In[132]:


word_count=df_train['word'].value_counts()
word_count=word_count[word_count>1]
word_set=list(set(word_count.index))
tag_set=list(tag_set)
for i in range(len(word_set)):
    word2idx[word_set[i]]=i+2

for i in range(len(tag_set)):
    label2idx[tag_set[i]]=i+1


# In[133]:


idx2label={label2idx[k] : k for k in label2idx}



#pad sentences to the maximum length
def padding_sentences(final_length, sentences, labels):
    train_sentences=[]
    train_labels=[]
    for i in range(len(sentences)):
        lenn=len(sentences[i])
        temp_words=[]
        temp_labels=[]
        for j in range(final_length):
            if j<lenn:
                word = word2idx[sentences[i][j]] if sentences[i][j] in word2idx else 1
                label = label2idx[labels[i][j]] if labels[i][j] in label2idx else 1
                temp_words.append(word)
                temp_labels.append(label)
            else:
                temp_words.append(0)
                temp_labels.append(0)
        train_sentences.append(temp_words)
        train_labels.append(temp_labels)

    return torch.tensor(train_sentences, dtype=torch.long), torch.tensor(train_labels, dtype=torch.long)

train_sentences, train_labels = padding_sentences(28, word_train_sentences, train_labels)


# In[135]:


char_set=set()
for seq in word_train_sentences:
    for word in seq:
        for c in word:
            char_set.add(c)
        
char2idx={'PAD': 0}      
char_set=list(char_set)
for i in range(len(char_set)):
    char2idx[char_set[i]]=i+1

idx2char={label2idx[k] : k for k in label2idx}


# In[209]:


idx2word={word2idx[k] : k for k in word2idx}


# In[215]:


max_char_len=19
def gen_char_sentences(dataset):
    char_sentences=[]
    for sentence in dataset:
        word_temp=[]
        for idx in sentence:
            word=idx2word[int(idx.numpy())]
            l=len(word)
            temp=[]
            if l<max_char_len:
                s=(max_char_len-l)//2
                e= max_char_len-l-s
                temp.extend([0]*s)
                for c in word:
                    temp.append(char2idx[c])
                temp.extend([0]*e)
            else:
                for i in range(max_char_len):
                    temp.append(char2idx[word[i]])
            word_temp.append(temp)
        char_sentences.append(word_temp)
    return torch.tensor(char_sentences, dtype=torch.long)


# In[216]:


train_data_chars=gen_char_sentences(train_sentences)


# ## Task 2

# In[139]:


filepath_glove = 'glove.6B.100d.txt'


# In[140]:


embeddings_dict = {}
with open(filepath_glove, 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector


# In[141]:


for sentence in word_dev_sentences:
    for word in sentence:
        if word not in word2idx:
            word2idx[word]=len(word2idx)
            
for sentence in word_test_sentences:
    for word in sentence:
        if word not in word2idx:
            word2idx[word]=len(word2idx)
            
            


# In[142]:


emb_matrix=np.random.normal(0, 0.1, (len(word2idx), 100))

for word,index in word2idx.items():
    word = word if word in embeddings_dict else word.lower()
   
    if word in embeddings_dict:
        emb_matrix[index]=torch.as_tensor(embeddings_dict[word])    
    
emb_matrix=torch.tensor(emb_matrix, dtype=torch.float32)


# In[143]:


class Model_Glove_LSTM_CNN(nn.Module):

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 char_emb_dim,  
                 char_input_dim,  
                 char_cnn_filter_num,  
                 char_cnn_kernel_size,  
                 hidden_dim,
                 linear_output_dim,
                 lstm_layers,
                 emb_dropout,
                 cnn_dropout,  
                 fc_dropout,
                 word_pad_idx,
                 char_pad_idx):  
        super().__init__()

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
            padding_idx=word_pad_idx
        )
        self.emb_dropout = nn.Dropout(emb_dropout)


        self.char_emb_dim = char_emb_dim
        self.char_emb = nn.Embedding(
            num_embeddings=char_input_dim,
            embedding_dim=char_emb_dim,
            padding_idx=char_pad_idx
        )
        self.char_cnn = nn.Conv1d(
            in_channels=char_emb_dim,
            out_channels=char_emb_dim * char_cnn_filter_num,
            kernel_size=char_cnn_kernel_size,
            groups=char_emb_dim  
        )
        self.cnn_dropout = nn.Dropout(cnn_dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim + (char_emb_dim * char_cnn_filter_num),
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True, batch_first=True)
      
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_dim * 2, linear_output_dim)  
        
        
      
        self.elu = nn.ELU(alpha=1.0, inplace=False)
        
      
        self.linear_classifier = nn.Linear(linear_output_dim, len(label2idx))
        
        
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def forward(self, words, chars):
      
        embedding_out = self.emb_dropout(self.embedding(words))
     
        char_emb_out = self.emb_dropout(self.char_emb(chars))
        batch_size, sent_len, word_len, char_emb_dim = char_emb_out.shape
        char_cnn_max_out = torch.zeros(batch_size, sent_len, self.char_cnn.out_channels)
     
        for sent_i in range(sent_len):
            
            sent_char_emb = char_emb_out[:, sent_i, :, :]  
           
            sent_char_emb_p = sent_char_emb.permute(0, 2, 1)  
            
            char_cnn_sent_out = self.char_cnn(sent_char_emb_p)
            char_cnn_max_out[:, sent_i, :], _ = torch.max(char_cnn_sent_out, dim=2) 
        char_cnn = self.cnn_dropout(char_cnn_max_out)

    
        word_features = torch.cat((embedding_out, char_cnn), dim=2)

        lstm_out, _ = self.lstm(word_features)
 
        s = self.fc(self.fc_dropout(lstm_out))
        
        s = self.elu(s)
        s = self.linear_classifier(s)
        return s

    def init_embeddings(self, char_pad_idx, word_pad_idx, pretrained=None, freeze=True):
      
        self.embedding.weight.data[word_pad_idx] = torch.zeros(self.embedding_dim)
        self.char_emb.weight.data[char_pad_idx] = torch.zeros(self.char_emb_dim)
        if pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=torch.as_tensor(pretrained),
                padding_idx=word_pad_idx,
                freeze=freeze
            )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# In[144]:


model = Model_Glove_LSTM_CNN(
    input_dim=len(word2idx),
    embedding_dim=100,
    char_emb_dim=30,
    char_input_dim=len(char2idx),
    char_cnn_filter_num=5,
    char_cnn_kernel_size=3,
    hidden_dim=256,
    linear_output_dim=128,
    lstm_layers=1,
    emb_dropout=0.33,
    cnn_dropout=0.25,
    fc_dropout=0.33,
    word_pad_idx=0,
    char_pad_idx=0
)
model.init_embeddings(
    char_pad_idx=0,
    word_pad_idx=0,
    pretrained=emb_matrix,
    freeze=True
)






model = torch.load(model_filename)




def infer(sentence, true_tags=None):
    model.eval()
    word_list=[]
    encoded_word_list=[]
    for word in sentence:
        word_list.append(word)
        encoded_word_list.append(word2idx[word] if word in word2idx else 1)
        



    
    word_temp=[]
    for word in sentence:
        l=len(word)
        temp=[]
        if l<max_char_len:
            s=(max_char_len-l)//2
            e= max_char_len-l-s
            temp.extend([0]*s)
            for c in word:
                temp.append(char2idx[c] if c in char2idx else 0)
            temp.extend([0]*e)
        else:
            for i in range(max_char_len):
                temp.append(char2idx[word[i]] if word[i] in char2idx else 0)
        word_temp.append(temp)
    
    
 
    pred = model(torch.as_tensor(encoded_word_list).unsqueeze(0), torch.as_tensor(word_temp).unsqueeze(0)).argmax(-1)


    return word_list, pred


# ### Dev Output

# In[264]:


out_filename=dev_filename
open(out_filename, 'w').close()
f1 = open(out_filename, "a")
for i in range(len(word_dev_sentences)):
    word_list, pred = infer(word_dev_sentences[i], dev_labels[i])
    for j in range(len(word_list)):
        # f1.write(f'{j+1} {word_list[j]} {dev_labels[i][j]} {idx2label[int(pred[0][j].numpy())]}\n')
        f1.write(f'{j+1} {word_list[j]} {idx2label[int(pred[0][j].numpy())]}\n')
    f1.write("\n")
f1.close()


# ### Test Output

# In[265]:


out_filename=test_filename
open(out_filename, 'w').close()
f1 = open(out_filename, "a")
for i in range(len(word_test_sentences)):
    word_list, pred = infer(word_test_sentences[i])
    for j in range(len(word_list)):
        f1.write(f'{j+1} {word_list[j]} {idx2label[int(pred[0][j].numpy())]}\n')
    f1.write("\n")
f1.close()






