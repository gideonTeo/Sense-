# import sys
# import subprocess

# implement pip as a subprocess:
# subprocess.check_call([sys.executable, '-m', 'pip', 'install',
# 'speech_recognition'])

import requests


import json

# packages for audio recog
import speech_recognition as sr
from os import path
from pydub import AudioSegment
from termcolor import colored
import spacy
import re
from afinn import Afinn
import pandas as pd
import moviepy.editor as mp
import os
from pathlib import Path
import en_core_web_sm





#########################################################################################################
#                                                                                                       #
#                                           AUDIO RECOG                                                 #
#                                                                                                       #
#########################################################################################################

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

            print ("text: " + " ".join(formattedText))
            return text




def text_sentence_audioF(audio_file):

    # nlp = spacy.load("en_core_web_sm")
    nlp = en_core_web_sm.load()

    text  = transcribe(audio_file)
    token = nlp(text)

    PRP_lst = []
    for i in range (len(token)):
        if token[i].tag_ == 'PRP':
            PRP_lst.append(str(token[i]))


    # \b means word boundaries.
    regex = r"\b(?:{})\b".format("|".join(PRP_lst))


    res = re.split(regex, text)

    return res




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





def video2audio(video_clip):
    my_clip = mp.VideoFileClip(video_clip)
    my_clip.audio.write_audiofile(video_clip[:-4] + "_result.wav")

    # return the name of the audio clip
    return video_clip[:-4] + "_result.wav"




def audio_main(video_clip):
    path = os.getcwd() + "/" + video_clip
    audio_file = video2audio(video_clip)


    res =  sentiment_analyzer_audioF(audio_file)
    return json.dumps(res)



#########################################################################################################
#                                                                                                       #
#                                           FACIAL RECOG                                                #
#                                                                                                       #
#########################################################################################################
# Video snippet
from cv2 import cv2
import os

def snippet(video_clip):
    if not os.path.exists("Out"):
        os.mkdir("Out")
    pathOut = os.getcwd() + """\Out\\"""

    vid = video_clip
    cap = cv2.VideoCapture(vid)
    count = 0
    # counter += 1
    success = True
    while success:
        success, image = cap.read()
        # print('read a new frame:',success)
        if count % 15 == 0:
            cv2.imwrite(pathOut + 'frame%d.jpg'%count, image)
        count += 1

# predicting using model
import numpy as np # linear algebra
import json
import os
import keras
from keras.models import Sequential,model_from_json
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image

def my_model():
    model = Sequential()
    input_shape = (48,48,1)
    model.add(Conv2D(64, (5, 5), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
    return model

def getData(filename):
    Y = []
    X = []
    first = True
    for line in open(filename):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y

def facial_main(video_clip):
    snippet(video_clip)
    model=my_model()
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    model = keras.models.load_model('model_1.h5')

    results = []
    directory = os.getcwd() + "\Out"

    snips = os.listdir(directory)
    for pic in snips:
        img = image.load_img(directory + "\\" + pic, grayscale=True, target_size=(48, 48))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x /= 255

        custom = model.predict(x)
        results.append(custom[0])

    res = [0,0,0,0,0,0,0]
    for i in range(len(results)):
        for j in range(len(results[i])):
            res[j] += results[i][j]

    out = []
    for i in range(len(objects)):
        out.append([objects[i],res[i]])
    return json.dumps(out)





#########################################################################################################
#                                                                                                       #
#                                       MAIN (lambda_handler)                                           #
#                                                                                                       #
#########################################################################################################


def lambda_handler(event, context):
    video_clip = os.getcwd() + "\happy.mp4"
    audio_output = audio_main(video_clip)
    facial_output = facial_main(video_clip)

    # TODO implement
    return audio_output, facial_output

print(lambda_handler(0, 0))
