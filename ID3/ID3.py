
# coding: utf-8

# ## Test Dataset

# In[1]:

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[2]:

dataSet = pd.read_csv('tree_data.csv')
labels = list(dataSet.columns)


# In[3]:

dataSet2 = dataSet.copy()

names = ['workclass','education','marrrital_status', 'occupation', 'relationship', 'race', 'sex','native_country','high_income']
for name in names:
    col = pd.Categorical.from_array(dataSet2[name])
    dataSet2[name] = col.codes
    
dataSet2.head()  


# In[4]:

dataSet2.describe()


# ## Entropy Calculation

# In[5]:

from math import log

def cal_entrophy(dataset):
    # dataset: pandas.DataFrame object
    dataset = np.array(dataset).tolist()
    num = len(dataset)
    labelcount = {}
    
    for sample in dataset:
        label = sample[-1]
        if not label in labelcount.keys():
            labelcount[label] = 0
        labelcount[label] += 1
        
    entropy = 0
    for label in labelcount.keys():
        prob = float(labelcount[label])/num
        entropy -= prob*log(prob,2)
    
    return entropy

def entrophy_gain_consecutive(dataset,fea,baseline):
    # calculate entrophy gain for consecutive features
    # dataset: pandas.DataFrame
    # fea: string, feature name
    # baseline: float
    num = dataset.shape[0]
    median_value = float(dataset[fea].median())
    subset1 = dataset[dataset[fea] > median_value]
    subset2 = dataset[dataset[fea] <= median_value]
    num1 = subset1.shape[0]
    num2 = subset2.shape[0]
    entrophy = (num1/num)*cal_entrophy(subset1) + (num2/num)*cal_entrophy(subset2)
    entrophy_gain = baseline - entrophy
    return entrophy_gain

def entrophy_gain_discrete(dataset,fea,baseline):
    levels = list(np.unique(dataset[fea].values))
    entrophy = 0
    num_total = dataset.shape[0]
    for level in levels:
        subset = dataset[dataset[fea] == level]
        num = subset.shape[0]
        entrophy += (num/num_total)*cal_entrophy(subset)
    entrophy_gain = baseline - entrophy
    return entrophy_gain  

baseline = cal_entrophy(dataSet2)
print(baseline)


# In[6]:

discrete_fea = ['workclass','education','marrrital_status','occupation', 'relationship', 'race', 'sex','native_country']
consecutive_fea = ['age','fnlwgt','education_num','capital_gain', 'capital_loss', 'hours_per_week']

def chooseFeature(dataset,discrete_fea,consecutive_fea):
    # dataset: pandas.DataFrame
    # discrete_fea & consecutive_fea: list
    
    entrophy_gains = {}

    for fea in discrete_fea:
        entrophy_gain = entrophy_gain_discrete(dataset,fea,baseline)
        entrophy_gains[fea] = entrophy_gain

    for fea in consecutive_fea:
        entrophy_gain = entrophy_gain_consecutive(dataset,fea,baseline)
        entrophy_gains[fea] = entrophy_gain

    engrophy_gain_list = []
    for fea, eg in sorted(entrophy_gains.items(), key=lambda item: item[1], reverse=True):
        row = {'feature':fea,'entrophy_gain':eg}
        engrophy_gain_list.append(row)

    entrophy_gain_df = pd.DataFrame(engrophy_gain_list)
    best_fea = entrophy_gain_df.feature[0]
    
    return best_fea, entrophy_gain_df

def barChart(entrophy_gain_df):
    plt.ylim(-1,1)
    plt.bar(entrophy_gain_df.index,entrophy_gain_df.entrophy_gain,tick_label=entrophy_gain_df.feature,color='green')
    plt.show()

best_fea, entrophy_gain_df = chooseFeature(dataSet2,discrete_fea,consecutive_fea)
print(best_fea)
barChart(entrophy_gain_df)
print(entrophy_gain_df)


# In[7]:

np.unique(dataSet[best_fea].values)


# In[8]:

discrete_fea = ['workclass','education','occupation', 'race', 'sex','native_country']
consecutive_fea = ['age','fnlwgt','education_num','capital_gain', 'capital_loss', 'hours_per_week']

g_labels = list(np.unique(dataSet2[best_fea].values))
g_best_feas = {}
g_entrophy_gain_df = {}
g_num = {}
for lab in g_labels:
    subset = dataSet2[dataSet2[best_fea] == lab]
    num = subset.shape[0]
    this_fea, this_entrophy_gain_df = chooseFeature(subset,discrete_fea,consecutive_fea)
    
    print(str(lab) + ':(' +str(num)+')' + this_fea)
    barChart(this_entrophy_gain_df)
    print(this_entrophy_gain_df)
    print('\n')
    
    g_best_feas[lab] = this_fea
    g_entrophy_gain_df[lab] = this_entrophy_gain_df
    g_num[lab] = num


# In[9]:

print(g_best_feas)


# In[10]:

print(g_entrophy_gain_df)


# In[11]:

print(g_num)


# In[12]:

dataSet


# In[13]:

dataSet2


# In[36]:

col = pd.Categorical.from_array(dataSet['education_num'])


# In[37]:

col.categories


# In[38]:

col.describe()


# In[ ]:



