#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[76]:


data = pd.read_csv('data.csv')


# In[77]:


data.head()


# In[78]:


data['cluster'].value_counts()


# In[79]:


x_ = data['x']
y_ = data['y']
label = data['cluster']
plt.scatter(x_, y_, c = label, cmap = 'plasma')


# In[80]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[81]:


x = data[['x','y']]
y = data['cluster']
x.head()


# In[82]:


# preprocessing
scaler = StandardScaler()
x = scaler.fit_transform(x)
x[:5]


# In[92]:


kmeans = KMeans(n_clusters=3, random_state=30)
kmeans.fit(x)


# In[84]:


kmeans.labels_


# In[93]:


x_ = data['x']
y_ = data['y']
label = kmeans.labels_
plt.scatter(x_, y_, c = label, cmap = 'plasma')


# In[88]:


# evaluasi jika tidak mengetahui label yang awal
# 1. sum of squared error

sse= []
index = range(1,10)
for i in index:
    kmeans = KMeans(n_clusters=i, random_state=30)
    kmeans.fit(x)
    sse_ = kmeans.inertia_
    sse.append(sse_)
    print(i, sse_)


# In[94]:


plt.plot(index, sse)
plt.xlabel('n_clusters')
plt.ylabel('SSE')
plt.show()

# 


# In[110]:


# evaluasi rand score jika sudah diketahui label dan clusternya

from sklearn import metrics
rand = []
index = range(1,10)
for i in index:
    kmeans = KMeans(n_clusters=i, random_state=30)
    kmeans.fit(x)
    rand_ = metrics.adjusted_rand_score(y, kmeans.labels_)
    rand.append(rand_)
    print(i, rand_)


# In[111]:


plt.plot(index, rand)
plt.xlabel('n_clusters')
plt.ylabel('rand score')
plt.show()

