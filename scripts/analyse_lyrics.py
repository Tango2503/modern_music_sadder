# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:12:15 2020

@author: tanusha.goswami
"""



import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


stop_words = set(stopwords.words('english')) 
count = 0


def spaced_lyrics(lyrics):
    spaces_to_be_added = []
    for i in range(len(lyrics) - 1):
        c1 = lyrics[i]
        c2 = lyrics[i+1]
        if c1.islower() and c2.isupper():
            spaces_to_be_added.append(i)
        else:
            continue
    if len(spaces_to_be_added) > 0:
        spaced_lyrics = ''
        for i in range(0, (len(spaces_to_be_added))):
            if i == 0:
                i0 = 0
            else:
                i0 = spaces_to_be_added[i-1] + 1
            i1 = spaces_to_be_added[i] + 1        
            part = lyrics[i0:i1]
            spaced_lyrics += part + ' '
            
            if i == len(spaces_to_be_added) - 1:
                spaced_lyrics += lyrics[i1:] 
        return spaced_lyrics
    else:
        return lyrics
    
def process_lyrics(lyrics):
    #remove identifiers like chorus, verse, etc
    lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)
    #remove empty lines
    lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])
    return lyrics
    
def clean_lyrics(lyrics):
    lyrics = lyrics.lower()
    word_tokens = word_tokenize(lyrics) 
    cleaned_lyrics = [w for w in word_tokens if not w in stop_words] 
    remove_set = ["'re","'s", "?","!","'ll",'‘', '’',"'d","n't","'ve",",",'[',']','(',')','verse','chorus']
    cleaned_lyrics = [w for w in cleaned_lyrics if not w in remove_set] 
    
    lem = WordNetLemmatizer()
    cleaned_lyrics_stemmed = []
    for w in cleaned_lyrics:
        root_word = lem.lemmatize(w,"v")
        cleaned_lyrics_stemmed.append(root_word)
    return cleaned_lyrics_stemmed 

def get_sentiment(row):
    lyrics = row['lyrics']
    row['lyric_sentiment_tb'] = TextBlob(lyrics).sentiment.polarity
    row['lyric_sentiment_vs'] = analyzer.polarity_scores(lyrics)['compound']
    return row

top_songs_track_ids = pd.read_excel('complete_data_set.xlsx')

top_songs_track_ids['lyrics'] = top_songs_track_ids['lyrics'].apply(spaced_lyrics)
top_songs_track_ids['lyrics'] = top_songs_track_ids['lyrics'].apply(process_lyrics)

top_songs_track_ids['lyrics_clean'] = top_songs_track_ids['lyrics'].apply(clean_lyrics)

#lyrics_explode = top_songs_track_ids[['lyrics_clean','track_name']]
#lyrics_explode = lyrics_explode.explode('lyrics_clean')
#lyrics_explode = lyrics_explode.groupby('lyrics_clean')['track_name'].count().reset_index()
#lyrics_explode = lyrics_explode.sort_values(by = 'track_name', ascending = False)

top_songs_track_ids = top_songs_track_ids.apply(get_sentiment, axis = 1)
top_songs_track_ids.to_excel('complete_data_set.xlsx',index = False)




