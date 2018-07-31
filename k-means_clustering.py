
# coding: utf-8

# In[152]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn import preprocessing,cross_validation

df=pd.read_excel('C:\\Users\\Amit George\\Downloads\\titanic.xls')
df.drop(['name','body'],1,inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)
print(df.columns.values)
df.head()


# In[172]:


def handle_non_numericaldata(df):
  columns = df.columns.values
  for column in columns:
        text_digit_val={}
        def convert_to_int(vals):
            return text_digit_val[vals]
        
        if df[column].dtype !=np.int64 and df[column].dtype!=np.float64:
            column_contents=df[column].values.tolist()
            column_set=set(column_contents)
            x=0
            for unique in column_set:
                if unique not in text_digit_val:
                    text_digit_val[unique]=x                
                    x+=1
            df[column]=list(map(convert_to_int,df[column]))
 
  return df

df= handle_non_numericaldata(df)


df.head()


# In[170]:


X=np.array(df.drop(['survived'], 1).astype(float))
print(X)
X=preprocessing.scale(X)
y=np.array(df['survived'])
clf= KMeans(n_clusters=2)
clf.fit(X)
correct=0
for i in range(len(X)):
    predictm= np.array(X[i].astype(float))
    predictm= predictm.reshape(-1, len(predictm))
    prediction=clf.predict(predictm)
    
    if prediction[0]==y[i]:
        correct+=1
print("accuracy",correct/len(X))

    

