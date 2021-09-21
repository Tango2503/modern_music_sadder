# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:44:28 2020

@author: tanusha.goswami
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import pandas as pd

cid = 'c637b844a8744a54b81cb491f84a74b2'
secret = 'a385f5d5f75a4a95885dc2da865bcbc2'
username = "uu5g7t2grcblz06h67w0rjfk1"
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace = True

def get_track_id(row):
    global count 
    print(count)
    count +=1
    track_name = row['track_name']
    artist_name = row['artist_name']
    query = "track: " + track_name + " artist: " + artist_name
    results = sp.search(q= query, type="track")
    try:
        track_id = results['tracks']['items'][0]['id']
    except:
        track_id = None
    row['track_id'] = track_id
    return row

def retrieve_feature(features, field):
    try:
        return features[field]
    except:
        return None

def get_audio_features(row):
    global count
    print(count)
    count += 1
    track_id = row['track_id']
    features = sp.audio_features(track_id)[0]
    row['tempo'] = retrieve_feature(features, 'tempo')
    row['danceability'] = retrieve_feature(features, 'danceability')
    row['energy'] = retrieve_feature(features, 'energy')
    row['key'] = retrieve_feature(features, 'key')
    row['loudness'] = retrieve_feature(features, 'loudness')
    row['mode'] = retrieve_feature(features, 'mode')
    row['speechiness'] = retrieve_feature(features, 'speechiness')
    row['acousticness'] = retrieve_feature(features, 'acousticness')
    row['instrumentalness'] = retrieve_feature(features, 'instrumentalness')
    row['liveness'] = retrieve_feature(features, 'liveness')
    row['valence'] = retrieve_feature(features, 'valence')
    row['type'] = retrieve_feature(features, 'type')
    row['duration_ms'] = retrieve_feature(features, 'duration_ms')
    row['time_signature'] = retrieve_feature(features, 'time_signature')
    return row
    

top_songs_data = pd.read_excel('top_songs_1941_2020.xlsx')
top_songs_data['track_name'] = top_songs_data['track_name'].astype(str)
top_songs_data['artist_name'] = top_songs_data['artist_name'].astype(str)
count = 0
top_songs_data = top_songs_data.apply(get_track_id, axis = 1)

no_track_ids = top_songs_data.loc[top_songs_data['track_id'].isna() == True]
top_songs_track_ids = top_songs_data.loc[top_songs_data['track_id'].isna() == False]
songs_per_year = top_songs_track_ids.groupby(['year'])['track_name'].count().reset_index()

count = 0
top_songs_track_ids = pd.read_excel('top_songs_track_ids_raw.xlsx')
#top_songs_track_ids = top_songs_track_ids.sample(n = 2)
top_songs_track_ids = top_songs_track_ids.apply(get_audio_features, axis = 1)

missing_ids = pd.read_csv('Missing IDs.csv')
missing_ids = missing_ids.loc[missing_ids['track_id'].isna() == False]
missing_ids = missing_ids.apply(get_audio_features, axis = 1)

top_songs_track_ids = pd.concat([top_songs_track_ids, missing_ids])

top_songs_track_ids = top_songs_track_ids.sort_values(by = ['year','s_no'])
top_songs_track_ids.to_excel('top_songs_track_ids.xlsx',index = False)
#tempo = get_features(track_id)


''' Ed Sheeran/Taylor Swift Example Data '''
#ed = pd.read_excel('ed_taylor_example.xlsx')
#ed = ed.apply(get_audio_features, axis = 1)




