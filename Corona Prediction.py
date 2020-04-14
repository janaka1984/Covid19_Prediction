#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('E:\\PYTHON TEST PROJECT\\covid\\covid_19_data.csv', parse_dates=['Last Update'])
df.head()


# In[3]:


df.rename(columns={'ObservationDate':'Date','Country/Region':'Country'}, inplace=True)
df_date = df.groupby(["Date"])[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
date_confirmed = df_date[['Date', 'Confirmed']]
date_death = df_date[['Date', 'Deaths']]
date_recovered = df_date[['Date', 'Recovered']]

for index, row in date_confirmed.iterrows():
        if row['Confirmed'] is None:
            row['Confirmed'] = 0.0
            
for index, row in date_death.iterrows():
    if row['Deaths'] is None:
        row['Deaths'] = 0.0

for index, row in date_recovered.iterrows():
    if row['Recovered'] is None:
        row['Recovered'] = 0.0        


# In[4]:


model = Sequential()
model.add(Dense(32, activation='relu', input_dim=1))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[6]:


model.fit(date_confirmed["Confirmed"][:30], date_confirmed['Confirmed'][:30], epochs=20,)
prediction_confirmed = model.predict(date_confirmed["Confirmed"])
final_prediction_confirmed = []

for i in range(0,len(prediction_confirmed)):
    final_prediction_confirmed.append(prediction_confirmed[i]*date_confirmed['Confirmed'][i])


# In[9]:


plt.title('Actual vs Prediction for Confirmed cases')
plt.plot(date_confirmed['Confirmed'][:30], label='Confirmed', color='blue')
plt.plot(date_confirmed['Confirmed'][30:], label='Confirmed unknown', color='green')
plt.plot(final_prediction_confirmed, label='Predicted', linestyle='dashed', color='orange')
plt.legend()
plt.show()


# In[10]:


model.fit(date_death['Deaths'][:30],date_death['Deaths'][:30],epochs=20,)
prediction_death = model.predict(date_death['Deaths'])
final_prediction_death = []
for i in range(0,len(prediction_death)):
    final_prediction_death.append(prediction_death[i]*date_death['Deaths'][i])

#print(final_prediction_death)
plt.title('Actual vs Prediction for Death cases')
plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(date_death['Deaths'][:30], label='Deaths', color='blue')
plt.plot(date_death['Deaths'][30:], label='Deaths unknown', color='green')
plt.plot(final_prediction_death, label='Predicted', linestyle='dashed', color='orange')
plt.legend()
plt.show()


# In[11]:


model.fit(date_recovered['Recovered'][:30],date_recovered['Recovered'][:30],epochs=20,)
prediction_recovered = model.predict(date_recovered['Recovered'])
final_prediction_recovered = []

for i in range(0,len(prediction_recovered)):
    final_prediction_recovered.append(prediction_recovered[i]*date_recovered['Recovered'][i])

#print(final_prediction_recovered)
plt.title('Actual vs Prediction for Recovered cases')
plt.xlabel('Dates')
plt.ylabel('Cases')
plt.plot(date_recovered['Recovered'][:30], label='Recovered', color='blue')
plt.plot(date_recovered['Recovered'][30:], label='Recovered unknown', color='green')
plt.plot(final_prediction_recovered, label='Predicted', linestyle='dashed', color='orange')
plt.legend()
plt.show()


# In[ ]:




