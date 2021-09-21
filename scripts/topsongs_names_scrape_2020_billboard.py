# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 12:31:52 2020

@author: tanusha.goswami
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re


page = requests.get('https://www.billboard.com/articles/news/list/9494940/best-songs-2020-top-100/')
soup = BeautifulSoup(page.content, 'html.parser')

my_divs = soup.find_all("div", class_="flex-row flex--wrap flex-lg--no-wrap flex--center-top spacing--md")
text_list = []
for md in my_divs:
    p = md.find_all('p')
    for p_i in p:
        text_list.append(p_i.text)

cleaned_list = []
for line in text_list:
    try:
        int(line[0])
        cleaned_list.append(line)
        print(line)
    except:
        continue
    
    
top_songs = pd.DataFrame(cleaned_list)
top_songs[0] = top_songs[0].str.split(". ", n = 1)
top_songs['Rank'] = top_songs[0].apply(lambda x: x[0])
top_songs['Artist_Track'] = top_songs[0].apply(lambda x: x[1])
top_songs['Artist_Track'] = top_songs['Artist_Track'].str.split(", ", n = 1)
top_songs['Artist'] = top_songs['Artist_Track'].apply(lambda x: x[0])
top_songs['Track'] = top_songs['Artist_Track'].apply(lambda x: x[1])
top_songs['Track'] = top_songs['Track'].apply(lambda x: x.replace('"',""))

    
    



