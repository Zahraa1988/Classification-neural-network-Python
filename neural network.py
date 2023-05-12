#!/usr/bin/env python
# coding: utf-8

# In[97]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


# In[98]:


dataset= pd.read_csv('Shill Bidding Dataset.csv')


# In[99]:


dataset


# In[100]:


len(dataset) == len(dataset['Record_ID'].unique())


# In[101]:


len(dataset) == len(dataset['Auction_ID'].unique())


# In[102]:


dataset = dataset.drop(['Record_ID','Bidder_ID'],axis=1)


# In[103]:


dataset.head()


# In[104]:


dataset.tail()


# In[105]:


dataset.shape


# In[106]:


dataset.size


# In[107]:


dataset.describe()


# In[108]:


dataset.describe(include= 'all')


# In[109]:


dataset['Class'].value_counts()


# In[110]:


sns.countplot(dataset['Class'])


# In[111]:


dataset.info()


# In[112]:


count= dataset.isnull().sum().sort_values(ascending=False)
percentage = ((dataset.isnull().sum()/len(dataset))*100).sort_values(ascending=False)
missing_data = pd.concat([count,percentage],axis=1,keys=['Count','Percentage'])
print ('Count and Percentage of missing values for the columns:')
missing_data


# In[113]:


dataset.isna().sum().sum()


# In[114]:


dataset['Class'].value_counts()


# In[115]:


print ('Percentage for defult\n')
print (round(dataset.Class.value_counts(normalize=True)*100,2))
round (dataset.Class.value_counts(normalize=True)*100,2).plot(kind='bar')
plt.title('percentage Distribution by Class type')
sns.countplot(dataset['Class'])


# In[116]:


X = dataset.drop('Class',axis=1)
y = dataset ['Class'] -1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
stratify=y, random_state=99)


# In[117]:


from imblearn.over_sampling import RandomOverSampler
resample = RandomOverSampler(random_state=0)
X_train_oversampled , y_train_oversampled = resample.fit_resample(X_train,y_train)
sns.countplot(x=y_train_oversampled)


# In[118]:


from imblearn.under_sampling import RandomUnderSampler
resample = RandomUnderSampler(random_state=0)
X_train_undersampled , y_train_undersampled = resample.fit_resample(X_train,y_train)
sns.countplot(x=y_train_undersampled)


# In[119]:


from imblearn.over_sampling import SMOTE
resampler = SMOTE(random_state=0)
X_train_smote , y_train_smote = resampler.fit_resample(X_train,y_train)
sns.countplot(x=y_train_smote)


# In[120]:


features=['Auction_ID','Bidder_Tendency','Bidding_Ratio','Successive_Outbidding',
         'Last_Bidding','Auction_Bids','Starting_Price_Average','Early_Bidding','Winning_Ratio',
         'Auction_Duration']


# In[121]:


features


# In[122]:


#separating out the features
X=dataset.loc[:, features].values


# In[123]:


features


# In[124]:


target= dataset.loc[:,'Class'].values


# In[125]:


target


# In[126]:


from sklearn.model_selection import train_test_split
X_train,X_test,target_train,target_test= train_test_split(X,target,test_size=0.2,random_state=0)


# In[127]:


print (X_train.shape)
print (target_test.shape)


# In[128]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[129]:


X_train


# In[130]:


# keras model object created from sequential class.
#this will be the container that contains all layers
model = Sequential()


# In[131]:


model.add(tf.keras.layers.Dense(12,activation='relu',input_shape=(10,)))
model.add(tf.keras.layers.Dense(2,activation='softmax')) 


# In[132]:


model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics='accuracy')


# In[133]:


model.compile


# In[134]:


model.summary()


# In[135]:


get_ipython().run_cell_magic('time', '', 'EPOCHS = 50\nbatch_size = 10')


# In[136]:


class_weights = {0:1, 1:10}
history = model.fit(X_train, target_train,  batch_size = batch_size , epochs= EPOCHS ,  verbose=2, class_weight=class_weights,  validation_split=0.2)


# In[137]:


_, accuracy = model.evaluate(X_train, target_train)
print ('Train Accuracy: %2f' % (accuracy*100))


# In[138]:


#test accuracy
from sklearn.metrics import accuracy_score
y_predict = np.argmax(model.predict(X_test), axis=-1)
accuracy_score(target_test,y_predict)


# In[139]:


print('Summary of the results after each epoch:')
hist=pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.head(10)


# In[140]:


accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
plt.plot(accuracy, label='Training Set Accuracy')
plt.plot(validation_accuracy, label='Validation Set Accuracy')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy Across Epochs')
plt.legend()


# In[141]:


loss = history.history['loss']
validation_loss = history.history['val_loss']
plt.plot(loss, label='Training Set Loss')
plt.plot(validation_loss, label='Validation Set Loss')
plt.ylabel('Loss')
plt.title('Training and Validation Accuracy Across Epochs')
plt.legend()


# In[142]:


target_pred = model.predict(X_test)
target_pred = target_pred.argmax(axis=1)


# In[143]:


confusion_matrix = confusion_matrix(target_test,target_pred)

ax = sns.heatmap(confusion_matrix, cmap='flare',annot=True, fmt='d')
plt.xlabel("Predicted Class",fontsize=12)
plt.ylabel("True Class",fontsize=12)
plt.title("Confusion Matrix",fontsize=12)
plt.show()


# In[144]:


print(classification_report(target_test,target_pred))

