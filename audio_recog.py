#!/usr/bin/env python
# coding: utf-8

# ## transcribe audio

# In[1]:


import speech_recognition as sr
from os import path
from pydub import AudioSegment
from termcolor import colored


# In[2]:


def transcribe(audio_file):
    # transcribe audio file                                                         
    AUDIO_FILE = audio_file

    # use the audio file as the audio source                                        
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
            audio = r.record(source)  # read the entire audio file
            
            text = r.recognize_google(audio)
            l1 = ['kill', 'die', 'regret', 'buried', 'dead', 'suicide', 'annoying', 'annoy', 'cold-hearted', 
                  'ignore', 'loser', 'stress', 'stressed']
            l2 = ["awesome", "happy", "good", 'yay', 'beautiful', 'pretty', 'handsome', 'fabulous', 'great', 'best']
            
            formattedText = []
            for t in text.lower().split():
                if t in l1:
                    formattedText.append(colored(t, 'white', 'on_red'))
                elif t in l2:
                    formattedText.append(colored(t, 'white', 'on_green'))
                else:
                    formattedText.append(t)

            print ("text: " + " ".join(formattedText) + "\n")
            return text

#             print("transcription: " + r.recognize_google(audio) + "\n")
#             return r.recognize_google(audio)


# In[3]:


import spacy
import re

def text_sentence_audioF(audio_file):

    nlp = spacy.load("en_core_web_sm")

    text  = transcribe(audio_file)
    token = nlp(text)

    PRP_lst = []
    for i in range (len(token)):
        if token[i].tag_ == 'PRP':
            PRP_lst.append(str(token[i]))

    # print(PRP_lst)


    # \b means word boundaries.
    regex = r"\b(?:{})\b".format("|".join(PRP_lst))


    res = re.split(regex, text)
    
    return res


# In[90]:


from afinn import Afinn
import pandas as pd

def sentiment_analyzer_audioF(audio_file):
    text  = text_sentence_audioF(audio_file)

    af = Afinn()

    # compute sentiment scores (polarity) and labels
    sentiment_scores = [af.score(element) for element in text]
    sentiment = ['positive' if score > 0 
                              else 'negative' if score < 0 
                                  else 'neutral' 
                                      for score in sentiment_scores]
    
    
    df = pd.DataFrame() 
    df['text'] =  text 
    df['sentiments'] = sentiment 
    df['scores'] = sentiment_scores
    df = pd.DataFrame(df.groupby('sentiments')['scores'].sum()).reset_index()


    output = df.values.tolist()
    emotion = []
    sentim = ["positive", "negative", "neutral"]
    
    for i in range (len(output)):
        emotion.append(output[i][0])
    
    for sm in sentim:
        if sm not in emotion:
            output.append([sm, 0.0])
            

    return output

    


# In[ ]:





# ## extracting audio from video

# In[91]:


import moviepy.editor as mp


# In[92]:


def video2audio(video_clip):
    my_clip = mp.VideoFileClip(video_clip)
    my_clip.audio.write_audiofile(video_clip[:-4] + "_result.wav")
    
    # return the name of the audio clip 
    return video_clip[:-4] + "_result.wav"


# ## main

# In[93]:


import os 
from pathlib import Path


# In[99]:


def main():
#     video_clip = input("enter video clip name (.mp4): ")
    video_clip = "happy.mp4"
    path = os.getcwd() + "/" + video_clip
    audio_file = video2audio(video_clip)
    
    
    res =  sentiment_analyzer_audioF(audio_file)
    return json.dumps(res)


# In[100]:


print(main())


# In[ ]:





# In[ ]:




