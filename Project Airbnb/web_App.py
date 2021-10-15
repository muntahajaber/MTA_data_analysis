#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import pickle 
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import pickle 
from sklearn.preprocessing import PolynomialFeatures
import streamlit as st
import nest_asyncio
import asyncio
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor


# In[2]:


from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[3]:


# df = pd.read_csv('cleaned_v6.csv', index_col=0)
df = pd.read_csv('data_v8.csv', index_col=0)


# In[4]:


df.head(2)


# In[5]:


# l = df.columns
# for x in l:
#     print(x)

#     mean_per_prov
# rating_count
# id
# guest_num
# bath_num
#id
# province	superhost	price
#Bed_num
# bedroom_num
# tv
# view


# # # LinearRegression

# In[6]:


test = df.copy(deep=True)


# In[7]:



# test.isnull().sum()
test = test.fillna(0)


# In[8]:


y = test['price']
#test.drop(columns = ['id', 'price', 'city', 'province', 'house_type', 'link'], inplace=False)


# In[9]:


test = test[['rating', 'Bed_num', 'bedroom_num', 'guest_num', 'bath_num','pool','tv',
             'view','province']].copy()


# test = test[['mean_per_prov','rating_count','id','guest_num','bath_num','superhost',
# 'price','Bed_num','bedroom_num','tv','view','pool']].copy()


# In[10]:


list_col = test.columns
for x in list_col:
    print(x)


# In[11]:


x = test


# In[12]:


model = LinearRegression()


# In[13]:


X, X_test, y, y_test = train_test_split(x, 
                                        y, test_size=0.2, random_state=40)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=40)


# In[14]:


m = model.fit(X_train, y_train)


# In[15]:


model.score(X_val, y_val)


# In[16]:


y_pred = m.predict(X_test)
# y_pred


# In[17]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_pred, y_test)


# # Feature Engneering

# In[18]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


# In[157]:


Tem_prev = df.copy(deep=True)


# In[158]:


y = Tem_prev["price"]   #target column i.e price range


# In[159]:


#y.shape


# In[160]:


# Tem_prev = Tem_prev.drop(columns=["city","superhost","house_type","link","id"])
Tem_prev = Tem_prev[['rating', 'Bed_num', 'bedroom_num', 'guest_num', 'bath_num','pool','tv','view','province']].copy()


# In[161]:


#Tem_prev = Tem_prev.drop(columns=["price"])


# In[162]:


# Tem_prev


# In[163]:


X = Tem_prev.iloc[:]  #independent columns


# In[164]:


#X.shape


# In[25]:


model = ExtraTreesClassifier()
model.fit(X,y)
#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
plt.figure(figsize=(30,30))
feat_importances = pd.Series(model.feature_importances_, index=X.columns);
feat_importances.nlargest(3).plot(kind='barh');
plt.show();


# In[26]:


# col = Tem_prev.columns


# In[27]:


# for x in col:
#     print(x)


# In[ ]:





# # Inhance The Model

# In[19]:


from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import pickle 


# In[20]:


# X, X_test, y, y_test = train_test_split(x, 
#                                         y, test_size=0.33, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=40)


# In[21]:


#LinearRegression
m = LinearRegression()
m.fit(X_train,y_train)
m.score(X_train,y_train)
m.score(X_test,y_test)


# In[22]:


#PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = m.fit(X_train_poly,y_train)
m.score(X_test_poly,y_test)


# In[23]:


mean_absolute_error(model.predict(X_test_poly), y_test)


# In[24]:


#RandomForestClassifier
model = RandomForestClassifier(max_depth=2, random_state = 42) 
new_model = model.fit(X_train, y_train)


# In[25]:


pred_cv = new_model.predict(X_test)
accuracy_score(y_test,pred_cv)


# In[26]:


pred_train = new_model.predict(X_test)
accuracy_score(y_test,pred_train)


# In[27]:


mean_absolute_error(pred_train, y_test)


# In[28]:


pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(new_model, pickle_out) 
pickle_out.close()


# In[ ]:





# # Website APP

# In[29]:


# !pip install -q pyngrok

# !pip install -q streamlit

# !pip install -q streamlit_ace
#%%writefile app.py
 
#import pickle
import streamlit as st


# In[30]:


# !pip install --upgrade --user hmmlearn
import nest_asyncio
import asyncio


# In[38]:


import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)

st.cache()

st.write("""
# Saudi Arabia House Price Prediction App
This app predicts the **Saudi Arabia House Price for the Airbnb Website**!
""")
st.write('---')

# Loads the Boston House Price Dataset
# X = pd.DataFrame(boston.data, columns=boston.feature_names)
# Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def main():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

        
#////////
data = {}

def plotGraph():
    
    st.subheader("Avrage Prices in Saudi Arabia")
    dff = df.copy(deep=True)
    st.dataframe(dff)
    df_1 = dff.loc[:, ["province", "price"]]
    groupby = df_1.groupby(["province"], axis=0)
    groupby.mean().plot(kind = 'bar')
    df.groupby['province']["price"].avg().plot(kind = 'bar')
    st.pyplot
    
    
def Prediction(bedroom_num,Bed_num,guest_num,bath_num,rating,pool,tv,view,province):   
    # Pre-processing user input 
    
    if pool == "Yes":
        pool = 1
    else:
        pool = 0
        
    if tv == "Yes":
        tv = 1
    else:
        tv = 0
    
    if view == "Yes":
        view = 1
    else:
        view = 0

    if province == "Mecca":
        province = 1
    elif province == "Riyadh":
        province = 2
    if province == "Eastern":
        province = 3
    elif province == "Medina":
        province = 4
    if province == "Asir":
        province = 5
    elif province == "AlQassim":
        province = 6
    if province == "Tabuk":
        province = 7
    elif province == "Ha'il":
        province = 8
    if province == "Al Bahah":
        province = 9
    elif province == "Jizan":
        province = 10
    else:
        province = 2 
    # Making predictions 
    prediction = classifier.predict( 
        [[bedroom_num,Bed_num,guest_num,bath_num,rating,pool,tv,view,province]])
#     plotGraph()
     
#     if prediction == 0:
#         pred = 'Rejected'
#     else:
#         pred = 'Approved'
    
    return prediction




def user_input_features():
    prediction = [-1]
    province = st.selectbox('Province',("Mecca","Riyadh","Eastern","Medina","Asir",
                                      "AlQassim","Tabuk","Ha'il","Al Bahah","Jizan"))     
    bedroom_num = st.sidebar.slider('Number of bedroom', X.bedroom_num.min(), X.bedroom_num.max()) 
    Bed_num = st.sidebar.slider('Number of Bed', X.Bed_num.min(), X.Bed_num.max())
    guest_num = st.sidebar.slider('Number of guest', X.guest_num.min(), X.guest_num.max())
    bath_num = st.sidebar.slider('Number of bath', X.bath_num.min(), X.bath_num.max())
    rating = st.sidebar.slider('Rating', X.rating.min(), X.rating.max())
    
    pool = st.selectbox("Have Pool",("Yes","No"))
    tv = st.selectbox("Have TV",("Yes","No"))
    view = st.selectbox("Have nice View",("Yes","No"))

    
   

    if st.button("Predict"):
        data['Province'] = province
        data['Number of bedroom'] = bedroom_num
        data['Number of Bed'] = Bed_num
        data['Number of guest'] = guest_num
        data['Number of bath'] = bath_num
        data['Rating'] = rating
        data['Have TV'] = tv
        data['Have Pool'] = pool
        data['Have nice View'] = view
        features = pd.DataFrame(data, index=[0])

        prediction = Prediction(bedroom_num,Bed_num,guest_num,bath_num,rating,pool,tv,view,province)
        return prediction
    return prediction

prediction = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
#st.write(data)
features = pd.DataFrame(data, index=[0])
st.table(features)
st.write('---')

# Build Regression Model
# model = RandomForestRegressor()
# model.fit(X, Y)
# Apply Model to Make Prediction
# prediction = model.predict(dff)

#prediction = Prediction(dff)
st.header('Prediction of Price')
if prediction[0] == -1:
    st.write("No Prediction")
else:
    st.write("House Price Prediction Equals ",prediction[0])
#print(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')

# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')
# nest_asyncio.apply()
if __name__ == "__main__":
    main() 


# In[ ]:





# In[39]:


# # dff = df.copy(deep=True)
# # dff.groupby[["province"]=="Riyadh"]["price"].avg().plot(kind = 'bar')

# loc = dff.loc[dff.province == 2] 
 
# # Grouping and couting
# t = df.groupby(["province"])["price"].count


# In[ ]:





# In[ ]:





# In[ ]:




