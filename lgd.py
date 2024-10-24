#!/usr/bin/env python
# coding: utf-8

# In[21]:


import streamlit as st
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[23]:


model=pickle.load(open('logmodel_pkl','rb'))
model


# In[25]:


st.title('Model Deployment using Logistic Regression')


# In[36]:


def user_input_variables():
    Pclass=st.sidebar.selectbox('Pcalss',[1,2,3])
    Age=st.sidebar.number_input('Age',min_value=0,max_value=100,step=1)
    SibSp = st.sidebar.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, step=1)
    Parch = st.sidebar.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, step=1)
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    Female = 1 if gender == 'Female' else 0
    Male = 1 if gender == 'Male' else 0
    Embarked=st.sidebar.selectbox('Embarked',['C','Q','S'])
    Embarked_C = 1 if Embarked == 'C' else 0
    Embarked_Q = 1 if Embarked == 'Q' else 0
    Embarked_S = 1 if Embarked == 'S' else 0
    data={
        'Pclass': [Pclass],
        'Age': [Age],
        'SibSp': [SibSp],
        'Parch': [Parch],
        'Female': [Female],
        'Male': [Male],
        'Embarked_C': [Embarked_C],
        'Embarked_Q': [Embarked_Q],
        'Embarked_S': [Embarked_S]
    }
    
    features=pd.DataFrame(data,index=[0])
    return features
df=user_input_variables()
st.subheader('user_input_variables')
st.write(df)
pred_prob=model.predict_proba(df)
pred= model.predict(df)

st.subheader('Predicted')
st.write('Yes' if pred_prob [0][1]>0.5 else 'No')
st.subheader('Predict_Prob')
st.write(pred_prob)


# In[38]:


df


# In[40]:


pred_prob


# In[ ]:




