#!/usr/bin/env python
# coding: utf-8

# In[89]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris_datasets=load_iris()


# In[19]:


print('keys of iris_datasets is:\n{}'.format(iris_datasets.keys()))


# In[65]:


print(iris_datasets['DESCR'][:193]+"\n")


# In[23]:


print(iris_datasets['feature_names'])


# In[25]:


print(iris_datasets['target'])


# In[26]:


print(iris_datasets['data'])


# In[28]:


print(iris_datasets['target_names'])


# In[32]:


print('shape of data{}'.format(iris_datasets['data'].shape))


# In[33]:


print('first five row of datas\n{}'.format(iris_datasets['data'][:5]))


# In[35]:


print(iris_datasets['feature_names'])


# In[36]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris_datasets['data'],iris_datasets['target'],random_state=0)


# In[45]:


print("X_train:{}".format(X_train.shape))


# In[57]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)


# In[58]:


knn.fit(X_train,y_train)


# In[59]:


y_pred=knn.predict(X_test)
print("test set predictions:\n{}".format(y_pred))


# In[60]:


print("test set score:{:.5f}".format(np.mean(y_pred==y_test)))


# In[61]:


print("test set score:{:.5f}".format(knn.score(X_test,y_test)))


# In[62]:


ak=np.array([[2,2.9,1,0.2]])
print("input:{}".format(knn.predict(ak)))


# In[68]:


s=knn.predict(ak)
print("花的种类{}".format(iris_datasets["target_names"][s]))


# In[ ]:




