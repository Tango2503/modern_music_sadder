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
                                 'valence','duration_ms']] # not considering lyric sentiment

df_tracks_processed = df_tracks.set_index('track_id')

''' TRANSFORMING DATA '''
df_tracks_processed['speechiness'] = df_tracks_processed['speechiness'].apply(lambda x: 0.060200 if x > 0.060200 else x)
df_tracks_processed['duration_ms'] = df_tracks_processed['duration_ms'].apply(lambda x: 255467 if x > 255467 else x)


for i in df_tracks_processed.columns:
    # Normalise
    if i in ['tempo','danceability', 'energy', 'loudness', 'speechiness','lyric_sentiment_tb', 
              'year','acousticness','valence']:
        max_value = df_tracks_processed[i].max()
        min_value = df_tracks_processed[i].min()
        df_tracks_processed[i] = (df_tracks_processed[i] - min_value) / (max_value - min_value)
    # Log
    elif i == 'liveness':
        df_tracks_processed[i] = np.log(df_tracks_processed[i]) 
        
    elif i == 'duration_ms':
#        df_tracks_processed[i] = boxcox(df_tracks_processed[i])[0]
        sc = StandardScaler()
        df_tracks_processed[i] = sc.fit_transform(np.array(df_tracks_processed[i]).reshape(-1,1))
        
    # Encode (OHE - trial)
    elif i == 'mode':
        df_tracks_processed = pd.get_dummies(df_tracks_processed, columns=["mode"])
    elif i == 'key':
        df_tracks_processed = pd.get_dummies(df_tracks_processed, columns=["key"])
        
    else:
        continue
 
tracks_corr1 = df_tracks_processed.corr()
      
coefs = tracks_corr1[['valence']]
coefs['sign'] = coefs['valence'].apply(lambda x: -1 if x < 0 else 1)
coefs = coefs.reset_index()
feature_coefs = pd.merge(feature_coefs, coefs, how = 'outer', left_on = 'feature', right_on = 'index' )
feature_coefs['final_weighted_vector'] = feature_coefs['coef_magnitude']*feature_coefs['sign']
coefs = feature_coefs[['feature','final_weighted_vector']]
coefs = coefs.set_index('feature')
sad_scores = df_tracks_processed.dot(coefs)
sad_scores = sad_scores.reset_index(drop = True)
sad_scores.columns = ['sad_scores']
top_songs_track_ids['sad_score'] = sad_scores['sad_scores']

for i in top_songs_track_ids.columns:
    if i not in ['year','s_no', 'mode']:
        try:
            df = top_songs_track_ids.groupby(['year'])[i].agg(['mean', 'median'])#.reset_index()
            df.plot(kind = 'line', title = i)
        except:
            continue


top_songs_track_ids['sad_score_averaged'] = (top_songs_track_ids['sad_score'] + top_songs_track_ids['lyric_sentiment_tb'])/2


df = top_songs_track_ids.groupby(['year'])['sad_score_averaged'].agg(['mean', 'median'])#.reset_index()
df.plot(kind = 'line', title = 'sad_score_averaged')
#'''


