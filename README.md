# Covid19_Prediction

```python
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('E:\\PYTHON TEST PROJECT\\covid\\covid_19_data.csv', parse_dates=['Last Update'])
df.head()
```

    Using TensorFlow backend.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SNo</th>
      <th>ObservationDate</th>
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Last Update</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>01/22/2020</td>
      <td>Anhui</td>
      <td>Mainland China</td>
      <td>2020-01-22 17:00:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>01/22/2020</td>
      <td>Beijing</td>
      <td>Mainland China</td>
      <td>2020-01-22 17:00:00</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>01/22/2020</td>
      <td>Chongqing</td>
      <td>Mainland China</td>
      <td>2020-01-22 17:00:00</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>01/22/2020</td>
      <td>Fujian</td>
      <td>Mainland China</td>
      <td>2020-01-22 17:00:00</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>01/22/2020</td>
      <td>Gansu</td>
      <td>Mainland China</td>
      <td>2020-01-22 17:00:00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```


```python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=1))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

    WARNING:tensorflow:From c:\users\janaka_w\appdata\local\programs\python\python36\lib\site-packages\tensorflow_core\python\ops\resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
    Instructions for updating:
    If using Keras pass *_constraint arguments to layers.
    


```python
model.fit(date_confirmed["Confirmed"][:30], date_confirmed['Confirmed'][:30], epochs=20,)
prediction_confirmed = model.predict(date_confirmed["Confirmed"])
final_prediction_confirmed = []

for i in range(0,len(prediction_confirmed)):
    final_prediction_confirmed.append(prediction_confirmed[i]*date_confirmed['Confirmed'][i])
```

    Epoch 1/20
    30/30 [==============================] - 0s 135us/step - loss: -501786.3438 - accuracy: 0.0000e+00
    Epoch 2/20
    30/30 [==============================] - 0s 67us/step - loss: -501786.3438 - accuracy: 0.0000e+00
    Epoch 3/20
    30/30 [==============================] - 0s 67us/step - loss: -501786.4062 - accuracy: 0.0000e+00
    Epoch 4/20
    30/30 [==============================] - 0s 99us/step - loss: -501786.3750 - accuracy: 0.0000e+00
    Epoch 5/20
    30/30 [==============================] - 0s 70us/step - loss: -501786.3750 - accuracy: 0.0000e+00
    Epoch 6/20
    30/30 [==============================] - 0s 56us/step - loss: -501786.3750 - accuracy: 0.0000e+00
    Epoch 7/20
    30/30 [==============================] - 0s 67us/step - loss: -501786.3438 - accuracy: 0.0000e+00
    Epoch 8/20
    30/30 [==============================] - 0s 67us/step - loss: -501786.3750 - accuracy: 0.0000e+00
    Epoch 9/20
    30/30 [==============================] - 0s 66us/step - loss: -501786.3750 - accuracy: 0.0000e+00
    Epoch 10/20
    30/30 [==============================] - 0s 65us/step - loss: -501786.3438 - accuracy: 0.0000e+00
    Epoch 11/20
    30/30 [==============================] - 0s 67us/step - loss: -501786.3750 - accuracy: 0.0000e+00
    Epoch 12/20
    30/30 [==============================] - 0s 67us/step - loss: -501786.3750 - accuracy: 0.0000e+00
    Epoch 13/20
    30/30 [==============================] - 0s 63us/step - loss: -501786.4062 - accuracy: 0.0000e+00
    Epoch 14/20
    30/30 [==============================] - 0s 67us/step - loss: -501786.3438 - accuracy: 0.0000e+00
    Epoch 15/20
    30/30 [==============================] - 0s 67us/step - loss: -501786.3750 - accuracy: 0.0000e+00
    Epoch 16/20
    30/30 [==============================] - 0s 67us/step - loss: -501786.4062 - accuracy: 0.0000e+00
    Epoch 17/20
    30/30 [==============================] - 0s 67us/step - loss: -501786.4062 - accuracy: 0.0000e+00
    Epoch 18/20
    30/30 [==============================] - 0s 67us/step - loss: -501786.3750 - accuracy: 0.0000e+00
    Epoch 19/20
    30/30 [==============================] - 0s 66us/step - loss: -501786.3750 - accuracy: 0.0000e+00
    Epoch 20/20
    30/30 [==============================] - 0s 92us/step - loss: -501786.3750 - accuracy: 0.0000e+00
    


```python
plt.title('Actual vs Prediction for Confirmed cases')
plt.plot(date_confirmed['Confirmed'][:30], label='Confirmed', color='blue')
plt.plot(date_confirmed['Confirmed'][30:], label='Confirmed unknown', color='green')
plt.plot(final_prediction_confirmed, label='Predicted', linestyle='dashed', color='orange')
plt.legend()
plt.show()
```


![png](output_4_0.png)



```python
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
```

    Epoch 1/20
    30/30 [==============================] - 0s 100us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 2/20
    30/30 [==============================] - 0s 66us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 3/20
    30/30 [==============================] - 0s 67us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 4/20
    30/30 [==============================] - 0s 32us/step - loss: -12187.3701 - accuracy: 0.0000e+00
    Epoch 5/20
    30/30 [==============================] - 0s 33us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 6/20
    30/30 [==============================] - 0s 65us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 7/20
    30/30 [==============================] - 0s 67us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 8/20
    30/30 [==============================] - 0s 67us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 9/20
    30/30 [==============================] - 0s 32us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 10/20
    30/30 [==============================] - 0s 31us/step - loss: -12187.3701 - accuracy: 0.0000e+00
    Epoch 11/20
    30/30 [==============================] - 0s 67us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 12/20
    30/30 [==============================] - 0s 67us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 13/20
    30/30 [==============================] - 0s 65us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 14/20
    30/30 [==============================] - 0s 69us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 15/20
    30/30 [==============================] - 0s 64us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 16/20
    30/30 [==============================] - 0s 67us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 17/20
    30/30 [==============================] - 0s 66us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 18/20
    30/30 [==============================] - 0s 33us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 19/20
    30/30 [==============================] - 0s 67us/step - loss: -12187.3691 - accuracy: 0.0000e+00
    Epoch 20/20
    30/30 [==============================] - 0s 133us/step - loss: -12187.3701 - accuracy: 0.0000e+00
    [array([17.], dtype=float32), array([18.], dtype=float32), array([26.], dtype=float32), array([42.], dtype=float32), array([56.], dtype=float32), array([82.], dtype=float32), array([131.], dtype=float32), array([133.], dtype=float32), array([171.], dtype=float32), array([213.], dtype=float32), array([259.], dtype=float32), array([362.], dtype=float32), array([426.], dtype=float32), array([492.], dtype=float32), array([564.], dtype=float32), array([634.], dtype=float32), array([719.], dtype=float32), array([806.], dtype=float32), array([906.], dtype=float32), array([1013.], dtype=float32), array([1113.], dtype=float32), array([1118.], dtype=float32), array([1371.], dtype=float32), array([1523.], dtype=float32), array([1666.], dtype=float32), array([1770.], dtype=float32), array([1868.], dtype=float32), array([2007.], dtype=float32), array([2122.], dtype=float32), array([2247.], dtype=float32), array([2251.], dtype=float32), array([2458.], dtype=float32), array([2469.], dtype=float32), array([2629.], dtype=float32), array([2708.], dtype=float32), array([2770.], dtype=float32), array([2814.], dtype=float32), array([2872.], dtype=float32), array([2941.], dtype=float32), array([2996.], dtype=float32), array([3085.], dtype=float32), array([3160.], dtype=float32), array([3254.], dtype=float32), array([3348.], dtype=float32), array([3460.], dtype=float32), array([3558.], dtype=float32), array([3803.], dtype=float32), array([3996.], dtype=float32), array([4262.], dtype=float32), array([4615.], dtype=float32), array([4720.], dtype=float32), array([5404.], dtype=float32), array([5819.], dtype=float32), array([6440.], dtype=float32), array([7126.], dtype=float32), array([7905.], dtype=float32), array([8733.], dtype=float32), array([9867.], dtype=float32), array([11299.], dtype=float32), array([12973.], dtype=float32), array([14623.], dtype=float32), array([16497.], dtype=float32), array([18615.], dtype=float32), array([21181.], dtype=float32), array([23970.], dtype=float32), array([27198.], dtype=float32), array([30652.], dtype=float32), array([33925.], dtype=float32), array([37582.], dtype=float32), array([42107.], dtype=float32), array([46809.], dtype=float32), array([52983.], dtype=float32), array([58787.], dtype=float32), array([64606.], dtype=float32), array([69374.], dtype=float32), array([74565.], dtype=float32), array([81865.], dtype=float32), array([88338.], dtype=float32), array([95455.], dtype=float32), array([102525.], dtype=float32), array([108503.], dtype=float32), array([114090.], dtype=float32), array([119483.], dtype=float32)]
    


![png](output_5_1.png)



```python
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
```

    Epoch 1/20
    30/30 [==============================] - 0s 100us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 2/20
    30/30 [==============================] - 0s 100us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 3/20
    30/30 [==============================] - 0s 71us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 4/20
    30/30 [==============================] - 0s 68us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 5/20
    30/30 [==============================] - 0s 67us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 6/20
    30/30 [==============================] - 0s 67us/step - loss: -62943.9609 - accuracy: 0.0000e+00
    Epoch 7/20
    30/30 [==============================] - 0s 66us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 8/20
    30/30 [==============================] - 0s 67us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 9/20
    30/30 [==============================] - 0s 85us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 10/20
    30/30 [==============================] - 0s 69us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 11/20
    30/30 [==============================] - 0s 33us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 12/20
    30/30 [==============================] - 0s 67us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 13/20
    30/30 [==============================] - 0s 33us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 14/20
    30/30 [==============================] - 0s 32us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 15/20
    30/30 [==============================] - 0s 64us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 16/20
    30/30 [==============================] - 0s 68us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 17/20
    30/30 [==============================] - 0s 62us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 18/20
    30/30 [==============================] - 0s 133us/step - loss: -62943.9648 - accuracy: 0.0000e+00
    Epoch 19/20
    30/30 [==============================] - 0s 137us/step - loss: -62943.9609 - accuracy: 0.0000e+00
    Epoch 20/20
    30/30 [==============================] - 0s 133us/step - loss: -62943.9766 - accuracy: 0.0000e+00
    


![png](output_6_1.png)



```python

```
