# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:08:30 2020

@author: tanusha.goswami
"""


import pandas as pd
from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 
import numpy as np 
import matplotlib.pyplot as plt  
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
    

#top_songs_40s = pd.read_excel('top_songs_40s.xlsx')

for i in top_songs_track_ids.columns:
    if i not in ['year','s_no', 'mode']:
        try:
            df = top_songs_track_ids.groupby(['year'])[i].agg(['mean', 'median'])#.reset_index()
            df.plot(kind = 'line', title = i)
        except:
            continue
        

        
# As time progresses, music is getting:
            # more danceable
            # more energetic (more instruments and rise of genres like disco and pop)
            # louder (rise of modern tech and music production)
            # more speechy (rise of rap)
            # less valent
            # interestingly tempo has remained similar over the years even though modern music 'feels' 
            # faster with genres like hip-hop, EDM and pop
            # lyrically morose!!!
            
''' Clustering '''

df_tracks = top_songs_track_ids[['track_id','year','tempo','danceability', 'energy', 'key', 
                                 'loudness', 'mode', 'speechiness','acousticness', 'liveness',
                                 'valence','duration_ms','lyrics_sentiment','lyrics_sentiment_vader']]

df_tracks = df_tracks.set_index('track_id')

#for i in df_tracks.columns:
#    df_tracks.hist(column = i)

''' TRANSFORMING DATA '''
df_tracks['speechiness'] = df_tracks['speechiness'].apply(lambda x: 0.060200 if x > 0.060200 else x)
df_tracks['duration_ms'] = df_tracks['duration_ms'].apply(lambda x: 255467 if x > 255467 else x)


for i in df_tracks.columns:
    print(i)
    # Normalise
    if i in ['tempo','danceability', 'energy', 'loudness', 'speechiness','lyrics_sentiment',
              'year','acousticness','valence']:
        max_value = df_tracks[i].max()
        min_value = df_tracks[i].min()
        df_tracks[i] = (df_tracks[i] - min_value) / (max_value - min_value)
        
    # Log
    elif i == 'liveness':
        print(df_tracks[i].skew())
        df_tracks[i] = np.log(df_tracks[i]) 
#        max_value = df_tracks[i].max()
#        min_value = df_tracks[i].min()
#        df_tracks[i] = (df_tracks[i] - min_value) / (max_value - min_value)
        print(df_tracks[i].skew())
        
    elif i == 'duration_ms':
        print(df_tracks[i].skew())
        df_tracks[i] = boxcox(df_tracks[i])[0]
#        max_value = df_tracks[i].max()
#        min_value = df_tracks[i].min()
#        df_tracks[i] = (df_tracks[i] - min_value) / (max_value - min_value)
        print(df_tracks[i].skew())
        
    # Encode (OHE - trial)
#    elif i == 'mode':
#        df_tracks = pd.get_dummies(df_tracks, columns=["mode"])
#    elif i == 'key':
#        df_tracks = pd.get_dummies(df_tracks, columns=["key"])
        
    else:
        continue
    
for i in df_tracks.columns[:-14]:
    df_tracks.hist(column = i)
        


distortions = [] 
inertias = [] 
K = range(1,10) 
  
for k in K: 
    print(k)
    #Building and fitting the model 
    kmeanModel = KMeans(n_clusters=k).fit(df_tracks) 
    kmeanModel.fit(df_tracks)     
      
    distortions.append(sum(np.min(cdist(df_tracks, kmeanModel.cluster_centers_, 
                      'euclidean'),axis=1)) / df_tracks.shape[0]) 
    inertias.append(kmeanModel.inertia_) 
    
plt.plot(K, distortions, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.title('The Elbow Method using Distortion') 
plt.show() 

plt.plot(K, inertias, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method using Inertia') 
plt.show()            
 
for nc in range(2,10):
    kmeans = KMeans(n_clusters=nc, random_state=0, init='k-means++')
    cluster_labels = kmeans.fit_predict(df_tracks)
    silhouette_avg = metrics.silhouette_score(df_tracks, cluster_labels)
    print("For n_clusters =", nc,"The average silhouette_score is :", silhouette_avg)
#    labeled_df['labels_'+str(nc)] = cluster_labels

from sklearn.cluster import AgglomerativeClustering
for nc in range(2,10):
    hierarchical = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
    cluster_labels = hierarchical.fit_predict(df_tracks)
    silhouette_avg = metrics.silhouette_score(df_tracks, cluster_labels)
    print("For n_clusters =", nc,"The average silhouette_score is :", silhouette_avg)
#    labeled_df['labels_'+str(nc)] = cluster_labels
    
for nc in range(2,5):
    print(nc)
#    kmeans = KMeans(n_clusters=nc, random_state=0, init='k-means++')
    k_prototype = KPrototypes(n_clusters=nc)
    cluster_labels = k_prototype.fit_predict(df_tracks, categorical=[4,6])
#    cluster_labels = kmeans.fit_predict(df_tracks)
    #Plot the clusters obtained using k means
    fig = plt.figure()
    ax = fig.add_subplot(111)
#    scatter = ax.scatter(df_tracks['valence'],df_tracks['lyrics_sentiment'],
#                         c=cluster_labels,s=50)
    scatter = ax.scatter(df_tracks['valence'],df_tracks['lyrics_sentiment'],
                         c=cluster_labels.astype(float),s=50)
    ax.set_title(str(nc) + ' clusters')
    ax.set_xlabel('Valence')
    ax.set_ylabel('Lyrics Sentiment')
    plt.colorbar(scatter)


