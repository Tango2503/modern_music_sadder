# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 12:31:52 2020

@author: tanusha.goswami
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def get_top_songs(year):
    page = requests.get("http://billboardtop100of.com/" + str(year) + "-2/")
    soup = BeautifulSoup(page.content, 'html.parser')
    top_songs = pd.DataFrame()
    
    try:
        table_body=soup.find('tbody')
        rows = table_body.find_all('tr') 
        for row in rows:
            cols=row.find_all('td')
            cols=[x.text.strip() for x in cols]
            row = pd.Series(cols)
            top_songs = top_songs.append(row, ignore_index = True)  
        top_songs.columns = ['s_no','artist_name','track_name']    
        top_songs['year'] = year
        return top_songs
    except:
        
        try:
            rows = soup.find_all('p')
            for row in rows:
                s_no = re.findall('\d',row.text)[0]
                track_name = re.findall('.(.*) by', row.text)[0][1:]
                r = row.text.replace('\n',' ')
#                print(r)
                try:
                    artist_name = re.findall('by (.*) written by', r)[0]
                except:
                    artist_name = re.findall('by (.*)', r)[0]
                    
                cols = [s_no, artist_name, track_name]
                row = pd.Series(cols)
                top_songs = top_songs.append(row, ignore_index = True)  
                
            top_songs.columns = ['s_no','artist_name','track_name']    
            top_songs['year'] = year
            return top_songs
        
        except Exception as e:
            print(str(e))
            print(str(year) + ' not scraped.')
            pass
    
    
    

#top_songs_yearwise = pd.DataFrame()
#for year in range(2015,2020):
#    print(year)
#    top_songs_yearwise = top_songs_yearwise.append(get_top_songs(year))
    
    
# Manual Scraping Bit

top_2019 = pd.read_excel('top_songs_manual_scrape.xlsx', sheet_name = '2019', header = None)

top_songs = pd.DataFrame()
for i in range(0,len(top_2019),3):
    s_no = str(top_2019[0][i])
    artist_name = top_2019[0][i+1]
    track_name = top_2019[0][i+2]
    cols = [s_no, artist_name, track_name]
    row = pd.Series(cols)
    top_songs = top_songs.append(row, ignore_index = True)  

top_songs.columns = ['s_no','artist_name','track_name']    
top_songs['year'] = 2019
    
for i in [1944,2013,2016]:
    print(i)
    year = str(i)
    top_year = pd.read_excel('top_songs_manual_scrape.xlsx', sheet_name = year)
    top_songs_yearwise = top_songs_yearwise.append(top_year)
    
top_songs_yearwise = top_songs_yearwise.drop_duplicates()
top_songs_yearwise['s_no'] = top_songs_yearwise['s_no'].astype(int)
top_songs_yearwise = top_songs_yearwise.sort_values(by = ['year','s_no'])

top_songs_yearwise.to_csv('Top Songs Yearwise (1941 - 2019).csv', index = False)



