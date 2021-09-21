# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:08:30 2020

@author: tanusha.goswami
"""


import pandas as pd
import numpy as np 
from scipy.stats import boxcox
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

top_songs_track_ids = pd.read_excel('complete_data_set.xlsx')

df_tracks = top_songs_track_ids[['track_id','year','tempo','danceability', 'energy', 'key', 
                                 'loudness', 'mode', 'speechiness','acousticness', 'liveness',
                                 'valence','duration_ms','lyric_sentiment_tb','lyric_sentiment_vs']]

df_tracks = df_tracks.set_index('track_id')

''' TRANSFORMING DATA '''
df_tracks['speechiness'] = df_tracks['speechiness'].apply(lambda x: 0.060200 if x > 0.060200 else x)
df_tracks['duration_ms'] = df_tracks['duration_ms'].apply(lambda x: 255467 if x > 255467 else x)

tracks_corr = df_tracks.corr()

for i in df_tracks.columns:
    # Normalise
    if i in ['tempo','danceability', 'energy', 'loudness', 'speechiness','lyrics_sentiment', 
              'year','acousticness','valence']:
        max_value = df_tracks[i].max()
        min_value = df_tracks[i].min()
        df_tracks[i] = (df_tracks[i] - min_value) / (max_value - min_value)
    # Log
    elif i == 'liveness':
        df_tracks[i] = np.log(df_tracks[i]) 
        
    elif i == 'duration_ms':
#        df_tracks[i] = boxcox(df_tracks[i])[0]
        sc = StandardScaler()
        df_tracks[i] = sc.fit_transform(np.array(df_tracks[i]).reshape(-1,1))
        
    # Encode (OHE - trial)
    elif i == 'mode':
        df_tracks = pd.get_dummies(df_tracks, columns=["mode"])
    elif i == 'key':
        df_tracks = pd.get_dummies(df_tracks, columns=["key"])
        
    else:
        continue
    
#tracks_corr = df_tracks.corr()
    
#for i in df_tracks.columns[:-14]:
#    df_tracks.plot.scatter(x = 'valence', y = i)
    
''' Baseline model '''
average = df_tracks['valence'].mean()

baseline_error = abs(df_tracks[['valence']] - average)
baseline_error = baseline_error['valence'].mean()
    
''' Linear Regression '''
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.model_selection import train_test_split 
#
#X = df_tracks.drop(columns = ['valence'])
#y = df_tracks[['valence']]
#
## Split data into train and test
#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#regression_model = LinearRegression().fit(x_train, y_train)
#y_pred = regression_model.predict(x_test)
#y_pred_train = regression_model.predict(x_train)
#
## model evaluation
#rmse = mean_squared_error(y_test, y_pred)
#r2 = r2_score(y_test, y_pred)
#residuals = y_test - y_pred
#
#rmse_train = mean_squared_error(y_train, y_pred_train)
#r2_train = r2_score(y_train, y_pred_train)
#
#
#
#
## printing values
#print('Slope:' ,regression_model.coef_)
#print('Intercept:', regression_model.intercept_)
#print('Root mean squared error_test: ', rmse)
#print('R2 score_test: ', r2)
#
#print('Root mean squared error_test: ',rmse_train )
#print('R2 score_test: ', r2_train)

#plt.scatter(residuals,y_pred)

# plotting values

''' Random Forest '''
x = df_tracks.drop(columns = ['valence','lyric_sentiment_tb','lyric_sentiment_vs']).values  
y = df_tracks[['valence']].values   

from sklearn.ensemble import RandomForestRegressor 
 # create regressor object 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
# fit the regressor with x and y data 
regressor.fit(x, y)   

# Use the forest's predict method on the test data
predictions = regressor.predict(x)

# Calculate the absolute errors
errors = abs(y - predictions.reshape(5795,1))
errors = np.mean(errors)

importance = list(regressor.feature_importances_)
feature_names = list(df_tracks.drop(columns = ['valence','lyric_sentiment_tb','lyric_sentiment_vs']).columns)
importance.insert(8, '1')
feature_names.insert(8, 'valence')
# summarize feature importance
feature_coefs = pd.DataFrame(index = range(len(importance)), columns = ['feature','coef_magnitude'])
for i in range(len(importance)):
    feature_coefs.iloc[i,0] = feature_names[i]
    feature_coefs.iloc[i,1] = importance[i]
	

feature_coefs['coef_magnitude'] = feature_coefs['coef_magnitude'].astype(float)



