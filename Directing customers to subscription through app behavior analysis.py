#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from dateutil import parser
import warnings
warnings.filterwarnings('ignore')


# In[2]:


dataset = pd.read_csv('appdata10.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.describe()


# In[5]:


dataset['hour'] = dataset.hour.str.slice(1,3).astype(int)


# In[6]:


dataset['hour'].head() 


# In[7]:


dataset2 = dataset.copy().drop(columns=['user','screen_list','enrolled_date','first_open','enrolled'])


# In[8]:


dataset2.head()


# In[9]:


plt.suptitle('Histogram of numerical columns',fontsize=20)
for i in range(1,dataset2.shape[1]+1):
    plt.subplot(3,3,i)
    f =plt.gca()
    f.set_title(dataset2.columns.values[i-1])
    vals = np.size(dataset2.iloc[:,i-1].unique())
    plt.hist(dataset2.iloc[:,i-1],bins=vals,color='#3f5D7D')


# In[10]:


dataset2.corrwith(dataset.enrolled).plot.bar(figsize=(20,10),title ='Correlation with response varaible',fontsize=15,rot=45,grid=True)


# In[11]:


#correlation matrix
sn.set(style='white',font_scale=2)
corr = dataset2.corr()
#generate a mask for uppper traingle
mask = np.zeros_like(corr,dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
#set up to matplot fig
f, ax = plt.subplots(figsize =(18,15))
f.suptitle("Correlation Matrix",fontsize=40)
#generate a custom diverging color-map
cmap  = sn.diverging_palette(220,10,as_cmap = True)
#draw the heatmap with the maskup and correct aspect ratio
sn.heatmap(corr,mask=mask,cmap=cmap,vmax=.3,center=0,square=True,linewidths=.5,cbar_kws={"shrink":.5})


# In[12]:


###Feature engineering types ###
dataset.dtypes


# In[13]:


dataset['first_open'] = [parser.parse(row_data) for row_data in dataset['first_open']]
dataset['enrolled_date'] = [parser.parse(row_data) if isinstance(row_data,str) else row_data for row_data in dataset['enrolled_date']]
dataset.dtypes


# In[14]:


dataset['difference'] = (dataset.enrolled_date - dataset.first_open).astype('timedelta64[h]')


# In[15]:


plt.hist(dataset['difference'].dropna(),color='#3F5D7D')
plt.title('Distributionof Time-since-Enrolled')
plt.show()


# In[16]:


plt.hist(dataset['difference'].dropna(),color='#3F5D7D',range=[0,100])
plt.title('Distributionof Time-since-Enrolled')
plt.show()


# In[17]:


dataset.loc[dataset.difference> 48,'enrolled'] =0
dataset = dataset.drop(columns=['difference','enrolled_date','first_open'])


# In[18]:


dataset.dtypes


# In[19]:


top_screens = pd.read_csv('top_screens.csv').top_screens.values


# In[20]:


dataset['screen_list']  = dataset.screen_list.astype(str)+','


# In[21]:


for sc in top_screens:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
    dataset['screen_list'] = dataset.screen_list.str.replace(sc+",","")


# In[22]:


dataset['Other']= dataset.screen_list.str.count(',')
dataset = dataset.drop(columns= ['screen_list'])


# In[23]:


savings_screens = ['Saving1','Saving2','Saving2Amount','Saving4','Saving5','Saving6','Saving7','Saving8','Saving9','Saving10']


# In[24]:


dataset['SavingsCount']  = dataset[savings_screens].sum(axis=1)
dataset = dataset.drop(columns= savings_screens)


# In[25]:


cc_screens = ['CC1','CC1Category','CC3']
dataset['CCCount']  = dataset[cc_screens].sum(axis=1)
dataset = dataset.drop(columns= cc_screens)


# In[26]:


loan_screens = ['Loan','Loan2','Loan3','Loan4']
dataset['LoanCount']  = dataset[loan_screens].sum(axis=1)
dataset = dataset.drop(columns= loan_screens)


# In[27]:


dataset.head(10)


# In[28]:


dataset.to_csv('new_appdate.csv',index=False)


# In[29]:


import time


# In[30]:


dataset3 = pd.read_csv('new_appdate.csv')


# In[31]:


response = dataset3['enrolled']
datasettt = dataset3.drop(columns='enrolled')


# In[32]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(datasettt,response,test_size =0.2,random_state=0)


# In[33]:


train_identifier = X_train['user']
X_train = X_train.drop(columns='user')
test_identifier = X_test['user']
X_test = X_test.drop(columns='user')


# In[34]:


from sklearn.preprocessing import StandardScaler


# In[46]:


sc_X  =  StandardScaler()
X_train2  = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2  = pd.DataFrame(sc_X.fit_transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index= X_train.index.values
X_test2.index= X_test.index.values
X_train = X_train2
X_test = X_test2


# In[47]:


from sklearn.linear_model import LogisticRegression


# In[64]:


classfier = LogisticRegression(random_state=0,penalty='l1',solver='saga')


# In[65]:


classfier.fit(X_train,y_train)
y_pred =  classfier.predict(X_test)


# In[66]:


from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score


# In[67]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[68]:


accuracy_score(y_test,y_pred)


# In[69]:


precision_score(y_test,y_pred)


# In[70]:


recall_score(y_test,y_pred)


# In[71]:


f1_score(y_test,y_pred)


# In[74]:


df_cm = pd.DataFrame(cm,index=(0,1),columns=(0,1))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm,annot=True,fmt='g')
print("Test data Accuracy: %0.4f" %accuracy_score(y_test,y_pred))


# In[75]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classfier,X=X_train,y=y_train,cv=10)
print('Logistic Accuracy: %0.3f (+/- %0.3f)' % (accuracies.mean(),accuracies.std() *2))


# In[76]:


#Formating final results
final_results  =pd.concat([y_test,test_identifier],axis=1).dropna()
final_results['predicted_results'] = y_pred
final_results[['user','enrolled','predicted_results']].reset_index(drop=True)


# In[ ]:




