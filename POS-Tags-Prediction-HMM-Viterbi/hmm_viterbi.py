#!/usr/bin/env python
# coding: utf-8

# # MANAN RAJDEV - CSCI 544 - HW3

# ## Libraries

# In[63]:


import pandas as pd
import json


# ## Task 1 - Vocabulary Creation

# Data Preprocessing -> converted all number tags to a token (<num>) and words with a low frequency to a special token (<unk>).

# In[64]:


df_train=pd.read_csv("data/train", sep="\t", names=["idx","word","tag"])
# df_train['word']=df_train['word'].str.lower()


# In[65]:


df_dev=pd.read_csv("data/dev", sep="\t", names=["idx","word","tag"])
# df_dev["word"]=df_dev['word'].str.lower()


# In[66]:


df_test=pd.read_csv("data/test", sep="\t", names=["idx","word"])


# In[67]:


unk_token = "< unk >"
unk_num_token = "< num >"


# In[68]:


df_train['word'] = df_train['word'].str.replace(r'^(\d*\.?\d+|\d{1,3}(,\d{3})*(\.\d+)?)$',unk_num_token)


# In[69]:


df_dev['word'] = df_dev['word'].str.replace(r'^(\d*\.?\d+|\d{1,3}(,\d{3})*(\.\d+)?)$',unk_num_token)


# In[70]:


df_test['word'] = df_test['word'].str.replace(r'^(\d*\.?\d+|\d{1,3}(,\d{3})*(\.\d+)?)$',unk_num_token)


# In[71]:


#add unk
threshold=2
df_vocab=pd.DataFrame(df_train['word'].value_counts())
v_size=df_vocab.shape[0]
unk=sum(df_vocab['word'][df_vocab['word']<threshold])
df_vocab=df_vocab[df_vocab['word']>=threshold]


# In[72]:


df_vocab.reset_index(inplace=True)


# In[73]:


df_vocab.loc[-1]=[unk_token,unk]
df_vocab.index+=1
df_vocab.sort_index(inplace=True)


# In[74]:


df_vocab.reset_index(inplace=True)
df_vocab.rename(columns={"word":"frequency","index":"word","level_0":"index"},inplace=True)


# In[75]:


df_vocab=df_vocab[["word","index","frequency"]]


# In[76]:


df_vocab.to_csv("vocab.txt",sep="\t", header=False, index=None)


# In[77]:


print(f"""
What is the selected threshold for unknown words replacement? 
Ans. {threshold}

What is the total size of your vocabulary?
Ans. {v_size}

What is the total occurrences of the special token '< unk >' after replacement?
Ans. {unk}

What is the final size of your vocabulary?
Ans. {df_vocab.shape[0]}

""")


# In[78]:


df_vocab.head()


# ## Task 2 - Model Learning

# In[79]:


count_s=df_train["tag"].value_counts().to_dict()


# ### Emission

# In[80]:


df_train['normalized']=df_train['word'].where(df_train['word'].isin(df_vocab['word']).astype(int)==1, unk_token)


# In[81]:


df_train["s->x"]=list(zip(df_train["tag"],df_train["normalized"]))


# In[82]:


count_em=df_train["s->x"].value_counts().to_dict()


# In[83]:


df_temp=df_train.drop_duplicates(subset=["s->x"])


# In[84]:


emission={}
for i in range(df_temp.shape[0]):
    em=df_temp.iloc[i,-1]

    s=df_temp.iloc[i,2]

    emission[em]=count_em[em]/count_s[s]


# In[85]:


print("Emission Parameters: ",len(emission))


# In[86]:


word_tag_dict = df_train.groupby('normalized')['tag'].apply(set).apply(list).to_dict()
# set(a.get_group(unk_token)['tag'])


# ### Transition

# In[87]:


df_train.drop(["s->x","normalized"], axis=1, inplace=True)


# In[88]:


temp=list(df_train["tag"][:-1])
temp=["."]+temp


# In[89]:


df_train["s_dash"]=temp


# In[90]:


df_train["s->s_dash"]=list(zip(df_train["s_dash"],df_train["tag"]))


# In[91]:


count_tn=df_train["s->s_dash"].value_counts().to_dict()


# In[92]:


df_temp=df_train.drop_duplicates(subset=["s->s_dash"])


# In[93]:


transition={}
for i in range(df_temp.shape[0]):
    tn=df_temp.iloc[i,-1]
    s=df_temp.iloc[i,-2]
    transition[tn]=count_tn[tn]/count_s[s]
#     transition[tuple(tn.split("###"))]=count_tn[tn]/count_s[s]


# In[94]:


print("Transition Parameters: ",len(transition))


# ### Storing the result

# In[95]:


keys_values = emission.items()
new_emission = {str(key): value for key, value in keys_values}
keys_values = transition.items()
new_transition= {str(key): value for key, value in keys_values}


# In[96]:


result={}
result['emission']=new_emission
result['transition']=new_transition
f = open("hmm.json", mode = 'w', encoding = 'UTF-8')
f.write(json.dumps(result))
f.close()


# ## Task 3 - Greedy Decoding with HMM

# In[97]:


train_bag=list(count_s.keys())
valid_words=set(df_vocab['word'])


# In[98]:


def greedy(words, train_bag = train_bag, valid_words=valid_words):
    state = []
    T = train_bag
    W=valid_words
     
    for key, word in enumerate(words):
        word = word if word in W else unk_token
        p = [] 
        for tag in word_tag_dict[word]:
            if key == 0:
                transition_p = transition.get(('.',tag),0)
            else:
                transition_p = transition.get((state[-1],tag),0)
                 

            
            emission_p = emission.get((tag,word),0)
           
            state_probability = emission_p * transition_p    
            p.append(state_probability)
             
        pmax = max(p)

        state_max = word_tag_dict[word][p.index(pmax)] 
        state.append(state_max)
#     return list(zip(words, state))
    return state


# ### Dev Data

# In[99]:


df_dev['y_pred']=greedy(list(df_dev['word']))


# In[100]:


df_dev["bool"]=df_dev['tag']==df_dev['y_pred']


# In[101]:


accuracy_greedy=sum(df_dev["bool"])/len(df_dev)
print("Accuracy of Greedy Approach: ",accuracy_greedy*100)


# In[102]:


df_dev.drop(["y_pred","bool"],axis=1,inplace=True)


# ### Test Data

# In[103]:


df_test_new=pd.read_csv("data/test", sep="\t", names=["idx","word"])


# In[104]:


df_test['y_pred']=greedy(list(df_test['word']))


# In[105]:


df_test_new['tag']=df_test['y_pred']


# In[106]:


open("greedy.out", 'w').close()
f = open("greedy.out", "a")
for i in range(len(df_test_new)):
    f.write(f'{df_test_new.iloc[i,0]}\t{df_test_new.iloc[i,1]}\t{df_test_new.iloc[i,2]}\n')
    if df_test_new.iloc[i,1]==".":
        pass
    elif df_test_new.iloc[i+1,0]==1:
        f.write("\n")
    else:
        pass
f.close()


# In[107]:


df_test.drop("y_pred",axis=1,inplace=True)


# ## Task 4 - Viterbi Decoding with HMM

# In[108]:


def Viterbi(words):
    
    result_dict = {}  # 3d - matrix to store the data..
    for i in range(0, len(words) + 1):
        result_dict[i] = {}
        if i == len(words):
            maxValue = -float("inf")
            result = ''
            for previousTag in result_dict[i - 1].keys():
                probablity = result_dict[i - 1][previousTag]['probablity'] *  transition.get((previousTag,'.'),0)
                if probablity > maxValue:
                    maxValue = probablity
                    result = previousTag
            result_dict[i]['end'] = {}
            result_dict[i]['end']['probablity'] = maxValue
            result_dict[i]['end']['backpointer'] = result
            continue
        
        word = words[i]
        word = word if word in valid_words else unk_token
        
        if i == 0:
            for tag in word_tag_dict[word]:
                result_dict[i][tag] = {}
                result_dict[i][tag]['probablity'] = emission.get((tag,word),0) * transition.get(('.',tag),0)
                result_dict[i][tag]['backpointer'] = 'start'
 
            continue

       
        for tag in word_tag_dict[word]:
            result_dict[i][tag] = {}
            maxValue = -float("inf")
            result = ''
            for previousTag in result_dict[i - 1].keys():
                probablity = result_dict[i - 1][previousTag]['probablity'] * emission.get((tag,word),0) * transition.get((previousTag,tag),0)
                if probablity > maxValue:
                    maxValue = probablity
                    result = previousTag
            result_dict[i][tag] = {}
            result_dict[i][tag]['probablity'] = maxValue
            result_dict[i][tag]['backpointer'] = result

    

    tag_sentence_list = []
    startTag = 'end'
    i = len(result_dict) - 1;
    j = len(result_dict) - 2;
    while i - 1 >= 0:
        tag = result_dict[i][startTag]['backpointer']
        tag_sentence_list.append(tag)
        startTag = tag
        i = i - 1
        j = j - 1
    return tag_sentence_list[::-1]


# ### Dev Data

# In[109]:


temp=[]
sentences=[]
y_pred=[]
for i in range(len(df_dev)):
    temp.append(df_dev.iloc[i,1])
    if df_dev.iloc[i,1]==".":
        if i==len(df_dev)-1:
            y_pred+=Viterbi(temp)
            temp=[]
        elif df_dev.iloc[i+1,0]==1:
            y_pred+=Viterbi(temp)
            temp=[]
        else:
            pass


# In[110]:


df_dev['y_pred']=y_pred


# In[111]:


df_dev['bool']=df_dev['tag']==df_dev['y_pred']


# In[112]:


accuracy_viterbi=sum(df_dev["bool"])/len(df_dev)
print("Accuracy of Viterbi Approach: ",accuracy_viterbi*100)


# ### Test Data

# In[113]:


temp=[]
sentences=[]
y_pred=[]
for i in range(len(df_test)):
    temp.append(df_test.iloc[i,1])
    if df_test.iloc[i,1]==".":
        if i==len(df_test)-1:
            y_pred+=Viterbi(temp)
            temp=[]
        elif df_test.iloc[i+1,0]==1:
            y_pred+=Viterbi(temp)
            temp=[]
        else:
            pass


# In[114]:


df_test_new['tag']=y_pred


# In[115]:


open("viterbi.out", 'w').close()
f = open("viterbi.out", "a")
for i in range(len(df_test_new)):
    f.write(f'{df_test_new.iloc[i,0]}\t{df_test_new.iloc[i,1]}\t{df_test_new.iloc[i,2]}\n')
    if df_test_new.iloc[i,1]==".":
        pass
    elif df_test_new.iloc[i+1,0]==1:
        f.write("\n")
    else:
        pass
f.close()


# In[ ]:





# In[ ]:




